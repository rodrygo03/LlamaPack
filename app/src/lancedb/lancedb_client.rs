use std::sync::Arc;
use anyhow::Result;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, table, Table};
use lancedb::connection::Connection;
use arrow_schema::{Field, DataType};
use arrow_array::{
    RecordBatch, StringArray, TimestampMicrosecondArray, Int16Array, 
    ListArray, FixedSizeListArray, Float32Array, ArrayRef, Array
};
use arrow_buffer::{OffsetBuffer, Buffer};
use arrow_array::RecordBatchIterator;
use futures::TryStreamExt;

use crate::lancedb::schema::{self, verify_embeddings_table, EMBEDDING_DIM};

/// mirrors schema def
#[derive(Clone, Debug)]
pub struct EmbeddingRecord {
    pub path: String,
    pub hash: String,
    pub embedding: Vec<f32>,
    pub language: String,
    pub last_modified: i64,
    pub last_accessed: i64,
    pub line_count: i16,
    pub imported_by: Vec<String>,
    pub content_preview: Option<String>,
}

/// LanceDbClient is the main interface for reading and writing code embeddings.
pub struct LanceDbClient {
    table: Arc<Table>,
}

impl LanceDbClient {
    /// Connect to the LanceDB database at the given path.
    /// Creates the `embeddings` table if it doesn't exist.
    pub async fn connect(path: &str) -> Result<Self> {
        let db: Connection = connect(path).execute().await?;
        let table = verify_embeddings_table(&db).await?;
        Ok(Self { table })
    }

    pub async fn insert_embeddings(&self, records: Vec<EmbeddingRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let arrays = Self::create_arrow_arrays(&records)?;
        let batch = Self::create_record_batch(arrays, &self.table).await?;
        
        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
        self.table.add(batches).execute().await?;

        Ok(())
    }

    pub async fn delete_embedding(&self, path: &str) -> Result<()> {
        self.table
            .delete(&format!("path = '{}'", path.replace("'", "''"))) // Escape single quotes
            .await?;
        
        Ok(())
    }

    pub async fn update_embedding(&self, path: &str, record: EmbeddingRecord) -> Result<()> {
        if record.path != path {
            return Err(anyhow::anyhow!(
                "Path mismatch: method parameter '{}' != record.path '{}'", 
                path, 
                record.path
            ));
        }
        
        // Delete existing record with the same path (ignore if it doesn't exist)
        let _ = self.delete_embedding(path).await; // Don't fail if record doesn't exist
        
        self.insert_embeddings(vec![record]).await?;
        Ok(())
    }

    pub async fn get_embedding(&self, path: &str) -> Result<Option<EmbeddingRecord>> {
        let query = format!("path = '{}'", path.replace("'", "''"));
        let mut stream = self.table
            .query()
            .only_if(query)
            .limit(1)
            .execute()
            .await?;

        if let Some(batch) = stream.try_next().await? {
            if batch.num_rows() > 0 {
                return Ok(Some(Self::record_batch_to_embedding_record(&batch, 0)?));
            }
        }
        Ok(None)
    }

    pub async fn query_similar(&self, embedding: &[f32], limit: usize) -> Result<Vec<EmbeddingRecord>> {
        if embedding.len() != EMBEDDING_DIM as usize {
            return Err(anyhow::anyhow!(
                "Invalid embedding dimension: expected {}, got {}", 
                EMBEDDING_DIM, 
                embedding.len()
            ));
        }
    
        let mut stream = self.table
            .vector_search(embedding)?
            .limit(limit)
            .execute()
            .await?;

        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            for row_index in 0..batch.num_rows() {
                let record = Self::record_batch_to_embedding_record(&batch, row_index)?;
                results.push(record);
            }
        }

        Ok(results)
    }

    pub async fn query_similar_to_file(&self, file_path: &str, limit: usize) -> Result<Vec<EmbeddingRecord>> {
        let file_record = self.get_embedding(file_path).await?
            .ok_or_else(|| anyhow::anyhow!("No embedding found for file: {}", file_path))?;
        
        let mut similar_records = self.query_similar(&file_record.embedding, limit + 1).await?;
        similar_records.retain(|record| record.path != file_path);
        similar_records.truncate(limit);
        
        Ok(similar_records)
    }

    // private helpers:

    fn create_arrow_arrays(records: &[EmbeddingRecord]) -> Result<Vec<ArrayRef>> {
        Self::validate_embeddings(records)?;
        
        let (paths, hashes, languages, last_modified, last_accessed, line_counts, content_previews) = 
            Self::extract_basic_fields(records);

        let path_array = Arc::new(StringArray::from(paths)) as ArrayRef;
        let hash_array = Arc::new(StringArray::from(hashes)) as ArrayRef;

        let inner_field = Arc::new(Field::new("item", DataType::Float32, true));
        let embedding_values = Self::extract_embeddings(records);
        let embedding_data = Arc::new(Float32Array::from(embedding_values));
        let embedding_array = Arc::new(FixedSizeListArray::new(
            inner_field,
            EMBEDDING_DIM,
            embedding_data,
            None,
        )) as ArrayRef;   
        
        let language_array = Arc::new(StringArray::from(languages)) as ArrayRef;
        let last_modified_array = Arc::new(TimestampMicrosecondArray::from(last_modified)) as ArrayRef;
        let last_accessed_array = Arc::new(TimestampMicrosecondArray::from(last_accessed)) as ArrayRef;
        let line_count_array = Arc::new(Int16Array::from(line_counts)) as ArrayRef;
            
        let (imported_by_values, imported_by_offsets) = Self::extract_imported_by(records);
        let imported_by_string_array = Arc::new(StringArray::from(imported_by_values));
        let imported_by_offsets = OffsetBuffer::new(imported_by_offsets.into());
        let imported_by_array = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Utf8, false)),
            imported_by_offsets,
            imported_by_string_array,
            None,
        )) as ArrayRef;

        let content_preview_array = Arc::new(StringArray::from(content_previews)) as ArrayRef;
        
        let arrays = vec![
            path_array,
            hash_array,
            embedding_array,
            language_array,
            last_modified_array,
            last_accessed_array,
            line_count_array,
            imported_by_array, 
            content_preview_array,
        ];

        Ok(arrays)
    }

    async fn create_record_batch(arrays: Vec<ArrayRef>, table: &Arc<Table>) -> Result<RecordBatch> {
        let schema = table.schema().await?;
        RecordBatch::try_new(schema, arrays)
            .map_err(|e| anyhow::anyhow!("Failed to create record batch: {}", e))
    }

    fn record_batch_to_embedding_record(batch: &RecordBatch, row_index: usize) -> Result<EmbeddingRecord> {
        if row_index >= batch.num_rows() {
            return Err(anyhow::anyhow!("Row index {} out of bounds", row_index));
        }

        let path_array = batch.column(0)
            .as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Failed to cast path column"))?;
        let path = path_array.value(row_index).to_string();

        let hash_array = batch.column(1)
            .as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Failed to cast hash column"))?;
        let hash = hash_array.value(row_index).to_string();

        let embedding_array = batch.column(2)
            .as_any().downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| anyhow::anyhow!("Failed to cast embedding column"))?;
        let embedding_list = embedding_array.value(row_index);

        let float_array = embedding_list
            .as_any().downcast_ref::<Float32Array>()
            .ok_or_else(|| anyhow::anyhow!("Failed to cast embedding values"))?;
        let embedding: Vec<f32> = (0..float_array.len())
            .map(|i| float_array.value(i))
            .collect();

        let language_array = batch.column(3)
            .as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Failed to cast language column"))?;
        let language = language_array.value(row_index).to_string();

        let last_modified_array = batch.column(4)
            .as_any().downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| anyhow::anyhow!("Failed to cast last_modified column"))?;
        let last_modified = last_modified_array.value(row_index);

        let last_accessed_array = batch.column(5)
            .as_any().downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| anyhow::anyhow!("Failed to cast last_accessed column"))?;
        let last_accessed = last_accessed_array.value(row_index);

        let line_count_array = batch.column(6)
            .as_any().downcast_ref::<Int16Array>()
            .ok_or_else(|| anyhow::anyhow!("Failed to cast line_count column"))?;
        let line_count = line_count_array.value(row_index);

        let imported_by_array = batch.column(7)
            .as_any().downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow::anyhow!("Failed to cast imported_by column"))?;
        let imported_by_list = imported_by_array.value(row_index);
        let string_array = imported_by_list
            .as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Failed to cast imported_by values"))?;
        let imported_by: Vec<String> = (0..string_array.len())
            .map(|i| string_array.value(i).to_string())
            .collect();

        let content_preview_array = batch.column(8)
            .as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Failed to cast content_preview column"))?;
        let content_preview = if content_preview_array.is_null(row_index) {
            None
        } else {
            Some(content_preview_array.value(row_index).to_string())
        };

        Ok(EmbeddingRecord {
            path,
            hash,
            embedding,
            language,
            last_modified,
            last_accessed,
            line_count,
            imported_by,
            content_preview,
        })
    }

    // more func helpers

    fn validate_embeddings(records: &[EmbeddingRecord]) -> Result<()> {
        for record in records {
            if record.embedding.len() != EMBEDDING_DIM as usize {
                return Err(anyhow::anyhow!(
                    "Invalid embedding dimension: expected {}, got {}", 
                    EMBEDDING_DIM, 
                    record.embedding.len()
                ));
            }
        }
        Ok(())
    }

    fn extract_basic_fields(records: &[EmbeddingRecord]) -> (Vec<String>, Vec<String>, Vec<String>, Vec<i64>, Vec<i64>, Vec<i16>, Vec<Option<String>>) {
        let paths = records.iter().map(|r| r.path.clone()).collect();
        let hashes = records.iter().map(|r| r.hash.clone()).collect();
        let languages = records.iter().map(|r| r.language.clone()).collect();
        let last_modified = records.iter().map(|r| r.last_modified).collect();
        let last_accessed = records.iter().map(|r| r.last_accessed).collect();
        let line_counts = records.iter().map(|r| r.line_count).collect();
        let content_previews = records.iter().map(|r| r.content_preview.clone()).collect();
        
        (paths, hashes, languages, last_modified, last_accessed, line_counts, content_previews)
    }

    fn extract_embeddings(records: &[EmbeddingRecord]) -> Vec<f32> {
        let mut embedding_values = Vec::new();
        for record in records {
            embedding_values.extend_from_slice(&record.embedding);
        }
        embedding_values
    }

    fn extract_imported_by(records: &[EmbeddingRecord]) -> (Vec<String>, Vec<i32>) {
        let mut imported_by_values = Vec::new();
        let mut imported_by_offsets = vec![0i32];
        
        for record in records {
            imported_by_values.extend(record.imported_by.iter().cloned());
            imported_by_offsets.push(imported_by_values.len() as i32);
        }
        
        (imported_by_values, imported_by_offsets)
    }

}

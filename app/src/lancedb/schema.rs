use std::sync::Arc;
use anyhow::Result;
use ndarray::Data;
use std::iter;
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use arrow_array::{RecordBatchIterator};
use lancedb::connection::Connection;
use lancedb::Table;

pub const EMBEDDING_DIM: i32 = 768;

fn build_embeddings_schema() -> Schema {
    Schema::new(vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("hash", DataType::Utf8, false),
        Field::new("embedding", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                EMBEDDING_DIM,
            ),
            false,
        ),
        Field::new("language", DataType::Utf8, false),
        Field::new("last_modified", DataType::Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("last_accessed", DataType::Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("line_count", DataType::Int16, false),
        Field::new("imported_by", DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))), false),
    ])
}

/// verify the embeddings table exists; create if it does not.
pub async fn verify_embeddings_table(db: &Connection) -> Result<Arc<Table>> {
    let schema = build_embeddings_schema();

    match db.open_table("embeddings").execute().await {
        Ok(table) => Ok(Arc::new(table)),
        Err(_) => {
            let schema_arc = Arc::new(schema.clone());
            let empty_batches = RecordBatchIterator::new(iter::empty(), schema_arc.clone());

            let table = db
                .create_table("embeddings", Box::new(empty_batches))
                .execute()
                .await?;
            Ok(Arc::new(table))
        }
    }
}

use llama_pack::lancedb::{LanceDbClient, EmbeddingRecord};
use anyhow::Result;
use std::fs;
use tempfile::TempDir;

// Test helper to create sample embedding records
fn create_test_record(path: &str, embedding_dim: usize) -> EmbeddingRecord {
    EmbeddingRecord {
        path: path.to_string(),
        hash: format!("hash_{}", path.replace("/", "_")),
        embedding: vec![0.1; embedding_dim], // Create embedding with correct dimension
        language: "rust".to_string(),
        last_modified: 1640995200000, // Jan 1, 2022 in microseconds
        last_accessed: 1640995200000,
        line_count: 42,
        imported_by: vec!["main.rs".to_string(), "lib.rs".to_string()],
        content_preview: Some("fn main() { println!(\"hello\"); }".to_string()),
    }
}

#[tokio::test]
async fn test_insert_single_embedding() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    let record = create_test_record("src/main.rs", 768);
    
    // Test inserting single record
    client.insert_embeddings(vec![record]).await?;
    
    Ok(())
}

#[tokio::test]
async fn test_insert_multiple_embeddings() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Create multiple test records
    let records = vec![
        create_test_record("src/main.rs", 768),
        create_test_record("src/lib.rs", 768),
        create_test_record("src/utils/mod.rs", 768),
    ];
    
    // Test inserting multiple records
    client.insert_embeddings(records).await?;
    
    Ok(())
}

#[tokio::test]
async fn test_insert_empty_records() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Test inserting empty vector (should not fail)
    client.insert_embeddings(vec![]).await?;
    
    Ok(())
}

#[tokio::test] 
async fn test_invalid_embedding_dimension() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    let mut record = create_test_record("src/main.rs", 768);
    
    // Create record with wrong embedding dimension
    record.embedding = vec![0.1; 512]; // Wrong dimension
    
    // Should return error
    let result = client.insert_embeddings(vec![record]).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Invalid embedding dimension"));
    
    Ok(())
}

#[tokio::test]
async fn test_record_with_no_imports() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    let mut record = create_test_record("src/standalone.rs", 768);
    
    // Test record with empty imported_by
    record.imported_by = vec![];
    
    client.insert_embeddings(vec![record]).await?;
    
    Ok(())
}

#[tokio::test]
async fn test_record_with_no_content_preview() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    let mut record = create_test_record("src/binary.rs", 768);
    
    // Test record with no content preview
    record.content_preview = None;
    
    client.insert_embeddings(vec![record]).await?;
    
    Ok(())
}

#[tokio::test]
async fn test_various_languages() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    let mut records = vec![];
    let languages = vec!["rust", "python", "javascript", "go", "java"];
    
    for (i, lang) in languages.iter().enumerate() {
        let mut record = create_test_record(&format!("src/file_{}.{}", i, lang), 768);
        record.language = lang.to_string();
        records.push(record);
    }
    
    client.insert_embeddings(records).await?;
    
    Ok(())
}

#[tokio::test]
async fn test_large_batch_insert() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Create 100 test records
    let records: Vec<EmbeddingRecord> = (0..100)
        .map(|i| create_test_record(&format!("src/file_{}.rs", i), 768))
        .collect();
    
    client.insert_embeddings(records).await?;
    
    Ok(())
}

// Integration test with actual embedder
#[tokio::test]
async fn test_insert_with_real_embeddings() -> Result<()> {
    use llama_pack::embedder::Embedder;
    
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Check if model files exist before running test
    let model_path = "../models/UniXcoder/unixcoder-embedding.onnx";
    let tokenizer_path = "../models/UniXcoder/tokenizer.json";
    
    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(tokenizer_path).exists() {
        println!("Skipping test - model files not found");
        return Ok(());
    }
    
    let mut embedder = Embedder::new(model_path, tokenizer_path)?;
    let code = "fn hello() { println!(\"Hello, world!\"); }";
    let embedding = embedder.embed(code)?;
    
    let record = EmbeddingRecord {
        path: "src/hello.rs".to_string(),
        hash: "real_hash_123".to_string(),
        embedding,
        language: "rust".to_string(),
        last_modified: 1640995200000,
        last_accessed: 1640995200000,
        line_count: 1,
        imported_by: vec![],
        content_preview: Some(code.to_string()),
    };
    
    client.insert_embeddings(vec![record]).await?;
    
    Ok(())
}

// ========== GET_EMBEDDING TESTS ==========

#[tokio::test]
async fn test_get_existing_embedding() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    let original_record = create_test_record("src/main.rs", 768);
    
    // Insert a record
    client.insert_embeddings(vec![original_record.clone()]).await?;
    
    // Retrieve it
    let retrieved = client.get_embedding("src/main.rs").await?;
    
    assert!(retrieved.is_some());
    let retrieved_record = retrieved.unwrap();
    
    // Verify all fields match
    assert_eq!(retrieved_record.path, original_record.path);
    assert_eq!(retrieved_record.hash, original_record.hash);
    assert_eq!(retrieved_record.embedding, original_record.embedding);
    assert_eq!(retrieved_record.language, original_record.language);
    assert_eq!(retrieved_record.last_modified, original_record.last_modified);
    assert_eq!(retrieved_record.last_accessed, original_record.last_accessed);
    assert_eq!(retrieved_record.line_count, original_record.line_count);
    assert_eq!(retrieved_record.imported_by, original_record.imported_by);
    assert_eq!(retrieved_record.content_preview, original_record.content_preview);
    
    Ok(())
}

#[tokio::test]
async fn test_get_nonexistent_embedding() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Try to get a record that doesn't exist
    let result = client.get_embedding("src/nonexistent.rs").await?;
    
    assert!(result.is_none());
    
    Ok(())
}

#[tokio::test]
async fn test_get_embedding_from_multiple_records() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Insert multiple records
    let records = vec![
        create_test_record("src/main.rs", 768),
        create_test_record("src/lib.rs", 768),
        create_test_record("src/utils/mod.rs", 768),
    ];
    client.insert_embeddings(records.clone()).await?;
    
    // Get specific records
    let main_record = client.get_embedding("src/main.rs").await?;
    let lib_record = client.get_embedding("src/lib.rs").await?;
    let utils_record = client.get_embedding("src/utils/mod.rs").await?;
    
    assert!(main_record.is_some());
    assert!(lib_record.is_some());
    assert!(utils_record.is_some());
    
    assert_eq!(main_record.unwrap().path, "src/main.rs");
    assert_eq!(lib_record.unwrap().path, "src/lib.rs");
    assert_eq!(utils_record.unwrap().path, "src/utils/mod.rs");
    
    Ok(())
}

#[tokio::test]
async fn test_get_embedding_with_special_characters_in_path() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Test path with special characters including single quotes
    let special_path = "src/file's_with_apostrophe.rs";
    let mut record = create_test_record(special_path, 768);
    record.path = special_path.to_string();
    
    client.insert_embeddings(vec![record.clone()]).await?;
    
    // Should be able to retrieve by the same path
    let retrieved = client.get_embedding(special_path).await?;
    
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().path, special_path);
    
    Ok(())
}

#[tokio::test]
async fn test_get_embedding_with_empty_imported_by() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    let mut record = create_test_record("src/standalone.rs", 768);
    record.imported_by = vec![]; // Empty imports
    
    client.insert_embeddings(vec![record.clone()]).await?;
    
    let retrieved = client.get_embedding("src/standalone.rs").await?;
    
    assert!(retrieved.is_some());
    let retrieved_record = retrieved.unwrap();
    assert!(retrieved_record.imported_by.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_get_embedding_with_null_content_preview() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    let mut record = create_test_record("src/binary.rs", 768);
    record.content_preview = None; // Null content preview
    
    client.insert_embeddings(vec![record.clone()]).await?;
    
    let retrieved = client.get_embedding("src/binary.rs").await?;
    
    assert!(retrieved.is_some());
    let retrieved_record = retrieved.unwrap();
    assert!(retrieved_record.content_preview.is_none());
    
    Ok(())
}

#[tokio::test]
async fn test_get_embedding_different_languages() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Insert records with different languages
    let languages = vec!["rust", "python", "javascript", "go", "java"];
    let mut records = vec![];
    
    for (i, lang) in languages.iter().enumerate() {
        let mut record = create_test_record(&format!("src/file_{}.{}", i, lang), 768);
        record.language = lang.to_string();
        records.push(record);
    }
    
    client.insert_embeddings(records).await?;
    
    // Retrieve and verify each language
    for (i, expected_lang) in languages.iter().enumerate() {
        let path = format!("src/file_{}.{}", i, expected_lang);
        let retrieved = client.get_embedding(&path).await?;
        
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().language, *expected_lang);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_get_embedding_after_update() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Insert original record
    let original_record = create_test_record("src/main.rs", 768);
    client.insert_embeddings(vec![original_record]).await?;
    
    // Update with new record
    let mut updated_record = create_test_record("src/main.rs", 768);
    updated_record.hash = "updated_hash_123".to_string();
    updated_record.line_count = 99;
    updated_record.content_preview = Some("fn updated() {}".to_string());
    
    client.update_embedding("src/main.rs", updated_record.clone()).await?;
    
    // Get should return the updated record
    let retrieved = client.get_embedding("src/main.rs").await?;
    
    assert!(retrieved.is_some());
    let retrieved_record = retrieved.unwrap();
    assert_eq!(retrieved_record.hash, "updated_hash_123");
    assert_eq!(retrieved_record.line_count, 99);
    assert_eq!(retrieved_record.content_preview, Some("fn updated() {}".to_string()));
    
    Ok(())
}

#[tokio::test]
async fn test_get_embedding_after_delete() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Insert record
    let record = create_test_record("src/main.rs", 768);
    client.insert_embeddings(vec![record]).await?;
    
    // Verify it exists
    let retrieved_before = client.get_embedding("src/main.rs").await?;
    assert!(retrieved_before.is_some());
    
    // Delete it
    client.delete_embedding("src/main.rs").await?;
    
    // Should not be found after deletion
    let retrieved_after = client.get_embedding("src/main.rs").await?;
    assert!(retrieved_after.is_none());
    
    Ok(())
}

#[tokio::test]
async fn test_get_embedding_exact_path_matching() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Insert records with similar but different paths
    let records = vec![
        create_test_record("src/main.rs", 768),
        create_test_record("src/main_test.rs", 768),
        create_test_record("tests/main.rs", 768),
    ];
    client.insert_embeddings(records).await?;
    
    // Should only match exact path
    let result = client.get_embedding("src/main.rs").await?;
    assert!(result.is_some());
    assert_eq!(result.unwrap().path, "src/main.rs");
    
    // Other paths should not match
    let result2 = client.get_embedding("main.rs").await?;
    assert!(result2.is_none());
    
    Ok(())
}

// ========== QUERY_SIMILAR TESTS ==========

#[tokio::test]
async fn test_query_similar_with_valid_embedding() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Insert multiple records with different embeddings
    let mut records = vec![];
    for i in 0..5 {
        let mut record = create_test_record(&format!("src/file_{}.rs", i), 768);
        // Create slightly different embeddings
        record.embedding = vec![0.1 + i as f32 * 0.1; 768];
        records.push(record);
    }
    client.insert_embeddings(records).await?;
    
    // Query with an embedding similar to the first record
    let query_embedding = vec![0.15; 768]; // Close to first record's [0.1; 768]
    let results = client.query_similar(&query_embedding, 3).await?;
    
    // Should return results (order depends on similarity)
    assert!(results.len() <= 3);
    assert!(!results.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_query_similar_with_invalid_dimension() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Try to query with wrong embedding dimension
    let wrong_embedding = vec![0.1; 512]; // Wrong dimension
    let result = client.query_similar(&wrong_embedding, 5).await;
    
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Invalid embedding dimension"));
    
    Ok(())
}

#[tokio::test]
async fn test_query_similar_empty_database() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Query empty database
    let query_embedding = vec![0.1; 768];
    let results = client.query_similar(&query_embedding, 5).await?;
    
    assert!(results.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_query_similar_limit_respected() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Insert 10 records
    let mut records = vec![];
    for i in 0..10 {
        let mut record = create_test_record(&format!("src/file_{}.rs", i), 768);
        record.embedding = vec![i as f32 * 0.1; 768];
        records.push(record);
    }
    client.insert_embeddings(records).await?;
    
    // Query with limit of 3
    let query_embedding = vec![0.5; 768];
    let results = client.query_similar(&query_embedding, 3).await?;
    
    assert_eq!(results.len(), 3);
    
    Ok(())
}

#[tokio::test]
async fn test_query_similar_different_embeddings() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Insert records with very different embeddings
    let mut rust_record = create_test_record("src/main.rs", 768);
    rust_record.embedding = vec![1.0; 768]; // All 1.0s
    rust_record.language = "rust".to_string();
    
    let mut python_record = create_test_record("src/main.py", 768);
    python_record.embedding = vec![-1.0; 768]; // All -1.0s
    python_record.language = "python".to_string();
    
    let mut js_record = create_test_record("src/main.js", 768);
    js_record.embedding = vec![0.0; 768]; // All 0.0s
    js_record.language = "javascript".to_string();
    
    client.insert_embeddings(vec![rust_record, python_record, js_record]).await?;
    
    // Query with embedding closer to rust record
    let query_embedding = vec![0.9; 768];
    let results = client.query_similar(&query_embedding, 2).await?;
    
    assert_eq!(results.len(), 2);
    // Results should be ordered by similarity
    
    Ok(())
}

// ========== QUERY_SIMILAR_TO_FILE TESTS ==========

#[tokio::test]
async fn test_query_similar_to_file_existing_file() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Insert multiple records with different embeddings
    let mut records = vec![];
    for i in 0..5 {
        let mut record = create_test_record(&format!("src/file_{}.rs", i), 768);
        record.embedding = vec![i as f32 * 0.2; 768];
        records.push(record);
    }
    client.insert_embeddings(records).await?;
    
    // Query similar to first file
    let results = client.query_similar_to_file("src/file_0.rs", 3).await?;
    
    // Should return up to 3 similar files, excluding the query file itself
    assert!(results.len() <= 3);
    // Should not include the query file
    assert!(!results.iter().any(|r| r.path == "src/file_0.rs"));
    
    Ok(())
}

#[tokio::test]
async fn test_query_similar_to_file_nonexistent_file() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Try to query similar to non-existent file
    let result = client.query_similar_to_file("src/nonexistent.rs", 5).await;
    
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("No embedding found for file"));
    
    Ok(())
}

#[tokio::test]
async fn test_query_similar_to_file_excludes_self() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Insert records with identical embeddings (would normally be most similar to each other)
    let mut records = vec![];
    for i in 0..3 {
        let mut record = create_test_record(&format!("src/identical_{}.rs", i), 768);
        record.embedding = vec![0.5; 768]; // Identical embeddings
        records.push(record);
    }
    client.insert_embeddings(records).await?;
    
    // Query similar to first file
    let results = client.query_similar_to_file("src/identical_0.rs", 5).await?;
    
    // Should return the other identical files but not the query file itself
    assert_eq!(results.len(), 2);
    assert!(!results.iter().any(|r| r.path == "src/identical_0.rs"));
    assert!(results.iter().any(|r| r.path == "src/identical_1.rs"));
    assert!(results.iter().any(|r| r.path == "src/identical_2.rs"));
    
    Ok(())
}

#[tokio::test]
async fn test_query_similar_to_file_respects_limit() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Insert many records
    let mut records = vec![];
    for i in 0..10 {
        let mut record = create_test_record(&format!("src/file_{}.rs", i), 768);
        record.embedding = vec![i as f32 * 0.1; 768];
        records.push(record);
    }
    client.insert_embeddings(records).await?;
    
    // Query with limit of 2
    let results = client.query_similar_to_file("src/file_0.rs", 2).await?;
    
    assert_eq!(results.len(), 2);
    assert!(!results.iter().any(|r| r.path == "src/file_0.rs"));
    
    Ok(())
}

#[tokio::test]
async fn test_query_similar_to_file_single_record() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Insert only one record
    let record = create_test_record("src/only_file.rs", 768);
    client.insert_embeddings(vec![record]).await?;
    
    // Query similar to the only file
    let results = client.query_similar_to_file("src/only_file.rs", 5).await?;
    
    // Should return empty since there are no other files
    assert!(results.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_query_similar_integration_with_real_embeddings() -> Result<()> {
    use llama_pack::embedder::Embedder;
    
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().to_str().unwrap();
    
    let client = LanceDbClient::connect(db_path).await?;
    
    // Check if model files exist before running test
    let model_path = "../models/UniXcoder/unixcoder-embedding.onnx";
    let tokenizer_path = "../models/UniXcoder/tokenizer.json";
    
    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(tokenizer_path).exists() {
        println!("Skipping test - model files not found");
        return Ok(());
    }
    
    let mut embedder = Embedder::new(model_path, tokenizer_path)?;
    
    // Create embeddings for different code snippets
    let code_snippets = vec![
        ("src/hello.rs", "fn hello() { println!(\"Hello, world!\"); }"),
        ("src/goodbye.rs", "fn goodbye() { println!(\"Goodbye, world!\"); }"),
        ("src/math.rs", "fn add(a: i32, b: i32) -> i32 { a + b }"),
        ("src/string.rs", "fn reverse_string(s: &str) -> String { s.chars().rev().collect() }"),
    ];
    
    let mut records = vec![];
    for (path, code) in code_snippets {
        let embedding = embedder.embed(code)?;
        let record = EmbeddingRecord {
            path: path.to_string(),
            hash: format!("hash_{}", path),
            embedding,
            language: "rust".to_string(),
            last_modified: 1640995200000,
            last_accessed: 1640995200000,
            line_count: 1,
            imported_by: vec![],
            content_preview: Some(code.to_string()),
        };
        records.push(record);
    }
    
    client.insert_embeddings(records).await?;
    
    // Query similar to hello.rs (should find goodbye.rs as most similar)
    let results = client.query_similar_to_file("src/hello.rs", 3).await?;
    
    assert!(!results.is_empty());
    assert!(!results.iter().any(|r| r.path == "src/hello.rs")); // Excludes self
    
    // Should have some similarity ordering
    if results.len() >= 2 {
        // The results should be ordered by similarity (most similar first)
        // println!("Most similar to hello.rs: {}", results[0].path);
    }
    
    Ok(())
}
use std::sync::Arc;
use anyhow::Result;
use lancedb::{connect, Table};
use lancedb::connection::Connection;

use crate::lancedb::schema::verify_embeddings_table;

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
}

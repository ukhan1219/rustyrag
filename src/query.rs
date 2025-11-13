// QUERYING THE VECTOR DATABASE and SEARCHING
use qdrant_client::qdrant::SearchPointsBuilder;
use qdrant_client::Qdrant;
use crate::embedding;

// Result type for search: (filename, chunk_text, score)
pub type SearchResult = (String, String, f32);

/// Embed a query string using the shared embedding function
pub fn embed_query(query: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    embedding::embed_text(query)
}

/// Search Qdrant for similar vectors to the query embedding
pub async fn search(
    query_vector: Vec<f32>,
    limit: usize,
) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
    let client = Qdrant::from_url("http://localhost:6334").build()?;

    // Search for similar vectors
    let search_result = client
        .search_points(
            SearchPointsBuilder::new("rustyrag", query_vector, limit as u64)
                .with_payload(true),
        )
        .await?;

    // Extract results from Qdrant response
    let mut results = Vec::new();
    for point in search_result.result {
        let score = point.score;
        
        // Extract payload (filename and chunk)
        let payload = point.payload;
        let filename = payload
            .get("filename")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let chunk = payload
            .get("chunk")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| String::new());
        
        results.push((filename, chunk, score));
    }

    Ok(results)
}

// EMBEDDING GENERATION
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize)]
struct EmbeddingRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

pub fn embed_chunks(chunks: Vec<(String, String)>) -> Result<Vec<(String, Vec<f32>)>, Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY environment variable not set")?;
    
    let client = reqwest::blocking::Client::new();
    let mut all_embeddings = Vec::new();
    
    // Extract just the text from chunks for API call
    let texts: Vec<String> = chunks.iter().map(|(_, text)| text.clone()).collect();
    
    // Call OpenAI API
    let request = EmbeddingRequest {
        input: texts,
        model: "text-embedding-3-small".to_string(), // or "text-embedding-ada-002"
    };
    
    let response = client
        .post("https://api.openai.com/v1/embeddings")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()?;
    
    let embedding_response: EmbeddingResponse = response.json()?;
    
    // Pair embeddings back with their filenames
    for (i, (filename, _)) in chunks.iter().enumerate() {
        if let Some(embedding_data) = embedding_response.data.get(i) {
            all_embeddings.push((filename.clone(), embedding_data.embedding.clone()));
        }
    }
    
    Ok(all_embeddings)
}

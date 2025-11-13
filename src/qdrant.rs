// QDRANT CLIENT
use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder};
use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder};
use qdrant_client::{Qdrant, Payload};
use serde_json::json;
use uuid::Uuid;


pub async fn create_collection() -> Result<(), Box<dyn std::error::Error>> {
    let client = Qdrant::from_url("http://localhost:6334").build()?;

    match client
        .create_collection(
            CreateCollectionBuilder::new("rustyrag")
                .vectors_config(VectorParamsBuilder::new(768, Distance::Cosine)),
        )
        .await {
            Ok(_) => println!("Collection created successfully"),
            Err(e) => println!("Error creating collection: {}", e),
        }
    Ok(())
}

pub async fn upsert_points(embeddings: Vec<(String, String, Vec<f32>)>) -> Result<(), Box<dyn std::error::Error>> {
    let client = Qdrant::from_url("http://localhost:6334").build()?;

    // Convert embeddings to Qdrant points with UUIDs
    let points: Vec<PointStruct> = embeddings.iter()
        .map(|(filename, chunk, embedding_vec)| {
            // Generate UUID for each point
            let id = Uuid::new_v4().to_string();
            
            PointStruct::new(
                id,                      // UUID as string ID
                embedding_vec.clone(),   // The embedding vector
                Payload::try_from(json!({
                    "filename": filename,
                    "chunk": chunk,
                })).unwrap(),
            )
        })
        .collect();  // Convert iterator to Vec

    let result = client
        .upsert_points(
            UpsertPointsBuilder::new("rustyrag", points)
                .wait(true),
        )
        .await?;
    
    println!("Upserted {} points", result.result.unwrap().status);
    Ok(())
}

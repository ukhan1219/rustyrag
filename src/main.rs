// MAIN CLI ENTRYPOINT, ARG PARSING, ETC.
use clap::{Parser, Subcommand};
mod document;
mod chunking;
mod embedding;
mod qdrant;
mod query;
mod llm;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Index documents (PDFs) from a file or directory
    Index {
        /// Path to a single PDF file or directory containing PDFs
        #[arg(short, long)]
        path: String,
    },
    /// Query the index
    Query {
        /// The query to search for
        #[arg(short, long)]
        query: String,
    },
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // This is a switch statement that will execute the appropriate command based on the user's input
    match args.command {
        Commands::Index { path } => {
            // This is the code that will be executed if the user inputs "index"
            println!("Indexing from path: {}", path);

            // This loads the documents from the path
            let hashmap = document::load_document(&path).unwrap();
            println!("Indexed {} documents\n", hashmap.len());
            
            for (filename, content) in &hashmap {
                println!("=== {} ===", filename);
                let preview: String = content.chars().take(500).collect();
                println!("{}\n", preview);
            }

            // This chunks the documents into 1000 token chunks with the tokenizer library
            let chunks = chunking::chunk_documents(&hashmap).unwrap();
            let chunks_len = chunks.len();
            println!("Chunked {} documents into {} chunks\n", hashmap.len(), chunks_len);

            // This generates embeddings for the chunks
            let embeddings = embedding::embed_chunks(chunks).unwrap();
            println!("Embedded {} chunks into {} embeddings\n", chunks_len, embeddings.len());
            
            // This creates the Qdrant collection if it doesn't exist
            if let Err(e) = qdrant::create_collection().await {
                eprintln!("Warning: Collection creation failed (might already exist): {}", e);
            }
            
            // This stores the embeddings in Qdrant
            match qdrant::upsert_points(embeddings).await {
                Ok(_) => println!("Successfully stored embeddings in Qdrant!"),
                Err(e) => eprintln!("Error storing embeddings: {}", e),
            }
        }
        // This is the code that will be executed if the user inputs "query"
        Commands::Query { query } => {
            println!("Querying: {}", query);
            
            // Embed the query string
            println!("Embedding query...");
            let query_vector = match query::embed_query(&query) {
                Ok(vec) => vec,
                Err(e) => {
                    eprintln!("Error embedding query: {}", e);
                    return;
                }
            };
            
            // Search Qdrant for similar vectors
            println!("Searching vector database...");
            match query::search(query_vector, 5).await {
                Ok(results) => {
                    println!("\n=== Retrieved Context ===\n");
                    for (i, (filename, chunk, score)) in results.iter().enumerate() {
                        println!("[Source {}] File: {} (Score: {:.4})", i + 1, filename, score);
                        println!("Content: {}\n", chunk.chars().take(200).collect::<String>());
                    }
                    
                    // Format context and generate answer with LLM
                    println!("=== Generating Answer ===\n");
                    let context = llm::format_context(&results);
                    let prompt = llm::create_prompt(&query, &context);
                    
                    match llm::generate_answer(&prompt, Some("Qwen/Qwen2.5-0.5B-Instruct")) {
                        Ok(answer) => {
                            println!("Answer:\n{}\n", answer);
                        }
                        Err(e) => {
                            eprintln!("Error generating answer: {}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error searching: {}", e);
                }
            }
        }
    }
}

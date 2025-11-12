// MAIN CLI ENTRYPOINT, ARG PARSING, ETC.
use clap::{Parser, Subcommand};
mod document;
mod chunking;
mod embedding;
mod qdrant;
mod query;

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

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::Index { path } => {
            println!("Indexing from path: {}", path);
            match document::load_document(&path) {
                Ok(hashmap) => {
                    println!("Indexed {} documents\n", hashmap.len());
                    
                    for (filename, content) in &hashmap {
                        println!("=== {} ===", filename);
                        let preview: String = content.chars().take(500).collect();
                        println!("{}\n", preview);
                    }
                    let chunks = chunking::chunk_documents(&hashmap).unwrap();
                    println!("Chunked {} documents into {} chunks\n", hashmap.len(), chunks.len());
                    for (filename, chunk) in &chunks {
                        println!("=== {} ===", filename);
                        println!("{}\n", preview);
                    }
                }
                Err(e) => {
                    println!("Error indexing documents: {}", e);
                }
            }
        }
        Commands::Query { query } => {
            println!("Querying: {}", query);
        }
    }
}
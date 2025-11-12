// TEXT CHUNKING AND TOKENIZATION
use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::Tokenizer;
use std::collections::HashMap;

pub fn chunk_documents(docs: &HashMap<String, String>) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    let mut all_chunks = Vec::new();
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    let max_tokens = 1000;
    let splitter = TextSplitter::new(ChunkConfig::new(max_tokens).with_sizer(tokenizer));

    for (filename, content) in docs {
        let chunks = splitter.chunks(content);
        all_chunks.extend(chunks.map(|chunk| (filename.clone(), chunk.to_string())));
    }
    Ok(all_chunks)
}
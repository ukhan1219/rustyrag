// EMBEDDING GENERATION
use candle_core::{DType, Tensor, Device};
use candle_transformers::models::jina_bert::{BertModel, Config, PositionEmbeddingType};
use candle_nn::{VarBuilder, Module};
use hf_hub::{api::sync::Api, Repo, RepoType};
use anyhow::Error as E;

/// Embed a single text string into a vector
pub fn embed_text(text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let model_name = "jinaai/jina-embeddings-v2-base-en";    
    // Load model and tokenizer from Hugging Face
    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));
    let model_file = repo.get("model.safetensors")?;
    let tokenizer_file = repo.get("tokenizer.json")?;
    
    // Try to use GPU if available (CUDA), otherwise fall back to CPU
    let device = match Device::cuda_if_available(0) {
        Ok(device) => {
            // Verify it's actually a CUDA device
            match &device {
                Device::Cuda(_) => {
                    println!("✓ Using CUDA GPU for embeddings");
                    device
                }
                Device::Cpu => {
                    eprintln!("cuda_if_available returned CPU device, using CPU for embeddings");
                    Device::Cpu
                }
                Device::Metal(_) => {
                    eprintln!("cuda_if_available returned Metal device, using CPU for embeddings (CUDA not available)");
                    Device::Cpu
                }
            }
        }
        Err(e) => {
            eprintln!("Using CPU for embeddings (CUDA not available: {})", e);
            Device::Cpu
        }
    };
    
    // Load tokenizer
    let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    
    // Create model config
    let vocab_size = tokenizer.get_vocab_size(true);
    let config = Config::new(
        vocab_size,
        768,  // hidden_size
        12,   // num_attention_heads
        12,   // num_hidden_layers
        3072, // intermediate_size
        candle_nn::Activation::Gelu,
        8192, // max_position_embeddings
        2,    // type_vocab_size
        0.02, // hidden_dropout_prob
        1e-12, // layer_norm_eps
        0,    // pad_token_id
        PositionEmbeddingType::Alibi,
    );
    
    // Load model weights
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let model = BertModel::new(vb, &config)?;

    // Tokenize the text
    let encoding = tokenizer.encode(text, true).map_err(E::msg)?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    
    // Convert to tensor: [1, seq_len]
    let token_tensor = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;
    
    // Run model forward pass
    let embeddings = model.forward(&token_tensor)?;
    
    // Apply mean pooling
    let pooled = mean_pooling(embeddings)?;

    // Convert to f32: get first (and only) embedding vector
    let embedding_vec: Vec<f32> = pooled.get(0)?.to_vec1()?;

    Ok(embedding_vec)
}

pub fn embed_chunks(chunks: Vec<(String, String)>) -> Result<Vec<(String, String, Vec<f32>)>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();
    for (filename, chunk) in chunks {
        let embedding_vec = embed_text(&chunk)?;
        results.push((filename, chunk, embedding_vec));
    }
    Ok(results)
}

fn mean_pooling(embeddings: Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Get dimensions: [batch_size, seq_len, hidden_dim]
    let (_batch_size, n_tokens, _hidden_size) = embeddings.dims3()?;
    
    // Sum across sequence dimension (dim 1) and divide by number of tokens
    let pooled = (embeddings.sum(1)? / (n_tokens as f64))?;
    
    // Optional: normalize embeddings (L2 normalization)
    // This makes embeddings unit vectors, which is often better for similarity search
    let normalized = normalize_l2(&pooled)?;
    
    Ok(normalized)
}

fn normalize_l2(tensor: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(tensor.broadcast_div(&tensor.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
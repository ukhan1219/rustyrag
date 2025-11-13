// LLM TEXT GENERATION USING CANDLE
use crate::query::SearchResult;
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_transformers::models::qwen2::{Model, Config};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;

/// Format retrieved chunks into context for the LLM
pub fn format_context(results: &[SearchResult]) -> String {
    let mut context = String::from("Context from documents:\n\n");
    
    for (i, (filename, chunk, score)) in results.iter().enumerate() {
        context.push_str(&format!("[Source {}] File: {}\n", i + 1, filename));
        context.push_str(&format!("Relevance Score: {:.4}\n", score));
        context.push_str(&format!("Content: {}\n\n", chunk));
    }
    
    context
}

/// Create a cite-or-deny prompt for the LLM
/// For Qwen2.5-Instruct, we need to use the chat template format
pub fn create_prompt(query: &str, context: &str) -> String {
    let system_message = "You are a helpful assistant that answers questions based ONLY on the provided context documents.\n\n\
        Instructions:\n\
        - If the answer can be found in the context, provide it and cite the source(s) using [Source N] format where N is the source number.\n\
        - If the answer cannot be found in the context, respond with: \"I cannot answer this question based on the provided context.\"\n\
        - Do not make up information or use knowledge outside the provided context.";
    
    let user_message = format!("{}\n\nQuestion: {}", context, query);
    
    // Qwen2.5-Instruct chat template format
    format!(
        "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
        system_message, user_message
    )
}

/// Sample the next token from logits using greedy decoding
/// Expects logits of shape [vocab_size]
fn sample_token(logits: &Tensor, temperature: f64) -> Result<u32, Box<dyn std::error::Error>> {
    let logits = logits.to_dtype(DType::F32)?;
    
    // Apply temperature
    let scaled_logits = if temperature != 1.0 {
        (&logits / temperature)?
    } else {
        logits
    };
    
    // Greedy decoding: take the token with highest probability
    // argmax returns a tensor, need to extract the scalar value
    let argmax_tensor = scaled_logits.argmax(0)?;
    // argmax returns [1] tensor, extract the first (and only) element
    let next_token = argmax_tensor.to_vec1::<u32>()?[0];
    Ok(next_token)
}

/// Generate answer using Qwen2.5-0.5B-Instruct with Candle
/// 
/// Qwen2.5-0.5B is publicly available, small (500M params), and fast
pub fn generate_answer(
    prompt: &str,
    model_name: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    // Default to Qwen2.5-0.5B-Instruct (publicly available, 500M params, fast)
    let model_name = model_name.unwrap_or("Qwen/Qwen2.5-0.5B-Instruct");
    
    println!("Loading model: {}...", model_name);
    println!("Note: This will download the model on first run (may take a while)");
    
    // Try to use GPU if available (CUDA), otherwise fall back to CPU
    let device = match Device::cuda_if_available(0) {
        Ok(device) => {
            // Verify it's actually a CUDA device
            match &device {
                Device::Cuda(_) => {
                    println!("✓ Using CUDA GPU for LLM");
                    device
                }
                Device::Cpu => {
                    eprintln!("cuda_if_available returned CPU device, using CPU for LLM");
                    Device::Cpu
                }
                Device::Metal(_) => {
                    eprintln!("cuda_if_available returned Metal device, using CPU for LLM (CUDA not available)");
                    Device::Cpu
                }
            }
        }
        Err(e) => {
            eprintln!("Using CPU for LLM (CUDA not available: {})", e);
            Device::Cpu
        }
    };
    
    // Add a small test operation to verify GPU is working
    if matches!(device, Device::Cuda(_)) {
        match Tensor::zeros((1, 1), DType::F32, &device) {
            Ok(_t) => {
                println!("✓ GPU tensor creation successful - GPU is active");
            }
            Err(e) => {
                eprintln!("Warning: GPU tensor creation failed: {}, GPU may not be working properly", e);
                // Note: We can't change device here, but at least we've warned the user
            }
        }
    }
    
    // Get token from environment
    let token = std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok();
    
    // Build API with explicit token if available
    // Try ApiBuilder first, fall back to Api::new() if not available
    let api = if let Some(token) = token {
        // Try to use ApiBuilder if available
        ApiBuilder::new()
            .with_token(Some(token))
            .build()
            .map_err(|e| format!("Failed to build API with token: {}", e))?
    } else {
        // Fall back to Api::new() which should read from environment
        Api::new()?
    };
    
    let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));
    
    // Load tokenizer
    let tokenizer_file = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_file)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
    
    // Get EOS token ID (Qwen2 uses <|im_end|> for chat, <|endoftext|> for base)
    let vocab = tokenizer.get_vocab(true);
    let eos_token_id = vocab
        .get("<|im_end|>")  // Qwen2.5-Instruct uses this for chat
        .or_else(|| vocab.get("<|endoftext|>"))
        .or_else(|| vocab.get("<|end_of_text|>"))
        .or_else(|| vocab.get("</s>"))
        .copied()
        .unwrap_or(151643u32); // Default EOS for Qwen2
    
    println!("Using EOS token ID: {}", eos_token_id);
    
    // Load model config from config.json
    // Qwen2 Config implements Deserialize, so we can parse it directly
    let config_file = repo.get("config.json")?;
    let config_json: serde_json::Value = serde_json::from_slice(&std::fs::read(config_file)?)
        .map_err(|e| format!("Failed to parse config.json: {}", e))?;
    let config: Config = serde_json::from_value(config_json.clone())
        .map_err(|e| format!("Failed to deserialize config: {}", e))?;
    
    // Extract max_position_embeddings from JSON (field is private in Config)
    let max_position_embeddings = config_json["max_position_embeddings"]
        .as_u64()
        .unwrap_or(2048) as usize;
    
    // Load model weights
    // Llama models are often split into multiple safetensors files
    // Check for model.safetensors.index.json first, then try individual files
    let model_files = if let Ok(index_file) = repo.get("model.safetensors.index.json") {
        // Model is split into multiple files - read the index
        let index: serde_json::Value = serde_json::from_slice(&std::fs::read(index_file)?)
            .map_err(|e| format!("Failed to parse model index: {}", e))?;
        let weight_map = index["weight_map"].as_object()
            .ok_or("Invalid weight_map in model index")?;
        
        // Get unique file names from weight_map
        let mut unique_files = std::collections::HashSet::new();
        for file_name in weight_map.values() {
            if let Some(name) = file_name.as_str() {
                unique_files.insert(name.to_string());
            }
        }
        
        // Download all unique files
        let mut files_vec = Vec::new();
        for file_name in unique_files {
            files_vec.push(repo.get(&file_name)
                .map_err(|e| format!("Failed to get {}: {}", file_name, e))?);
        }
        files_vec
    } else {
        // Try single model.safetensors file
        match repo.get("model.safetensors") {
            Ok(file) => vec![file],
            Err(_) => {
                // Fallback: try to find split files manually (model-00001-of-00002.safetensors, etc.)
                // This is a last resort - usually the index file should exist
                return Err("Model file not found. Expected model.safetensors or model.safetensors.index.json".into());
            }
        }
    };
    
    println!("Loading {} model file(s)...", model_files.len());
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::F32, &device)? };
    let mut model = Model::new(&config, vb)?;
    
    // Tokenize the prompt
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| format!("Failed to encode: {}", e))?;
    let mut token_ids: Vec<u32> = tokens.get_ids().to_vec();
    
    // Truncate to fit within model's context window
    // Reserve some tokens for generation (e.g., 512 tokens)
    // Also limit prompt size to avoid GPU OOM - use a sliding window of recent context
    let reserved_for_generation = 512;
    let max_prompt_tokens = max_position_embeddings.saturating_sub(reserved_for_generation);
    
    // Limit to 2048 tokens to avoid GPU memory issues
    let max_prompt_for_memory = 2048.min(max_prompt_tokens);
    
    if token_ids.len() > max_prompt_for_memory {
        println!("Warning: Prompt is {} tokens, truncating to {} tokens to avoid GPU OOM", 
                 token_ids.len(), max_prompt_for_memory);
        // Keep the end of the prompt (most recent context) rather than the beginning
        let start_idx = token_ids.len().saturating_sub(max_prompt_for_memory);
        token_ids = token_ids[start_idx..].to_vec();
    }
    
    println!("Prompt tokens: {} (max: {})", token_ids.len(), max_position_embeddings);
    println!("Generating response...");
    
    // Convert prompt tokens to tensor: [batch_size, seq_len]
    let input_tensor = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;
    
    // Process the entire prompt through the model to populate KV cache
    // Qwen2's forward takes seqlen_offset and attention_mask, returns [batch, vocab_size] (logits for last token)
    let seq_len = token_ids.len();
    let prompt_start = std::time::Instant::now();
    let logits = model.forward(&input_tensor, 0, None)?;
    let prompt_time = prompt_start.elapsed();
    println!("Prompt processing time: {:.2}s", prompt_time.as_secs_f64());
    
    // Extract logits for the batch: [batch, vocab_size] -> [vocab_size]
    // Qwen2's forward already returns logits for the last token position
    let mut last_logits = logits.i((0, ..))?;
    
    // Autoregressive generation using KV cache
    let mut generated_tokens = Vec::new();
    let max_new_tokens = 256; // Reduced for faster testing
    let temperature = 0.7;
    let mut seqlen_offset = seq_len; // Track position for KV cache
    
    println!("Starting generation (max {} tokens)...", max_new_tokens);
    println!("Note: To verify GPU usage, run 'nvidia-smi' in another terminal");
    
    let gen_start_time = std::time::Instant::now();
    
    // Generate new tokens using KV cache
    for step in 0..max_new_tokens {
        // Sample next token from the current logits
        let next_token = sample_token(&last_logits, temperature)?;
        
        // Check for EOS token
        if next_token == eos_token_id {
            println!("\nGenerated {} tokens (stopped at EOS token {})", step, eos_token_id);
            break;
        }
        
        generated_tokens.push(next_token);
        
        // Use the newly generated token as input for next iteration
        // The model's KV cache should maintain state from previous tokens
        let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let next_logits = model.forward(&next_input, seqlen_offset, None)?;
        
        // Extract logits: [batch, vocab_size] -> [vocab_size]
        last_logits = next_logits.i((0, ..))?;
        seqlen_offset += 1; // Increment position for next iteration
        
        // Print progress every 5 tokens with token count
        if step % 5 == 0 {
            print!("[{}]", step + 1);
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    
    // Check if we hit max tokens
    if generated_tokens.len() >= max_new_tokens {
        println!("\nGenerated {} tokens (reached max)", generated_tokens.len());
    }
    
    let gen_time = gen_start_time.elapsed();
    println!("Generation time: {:.2}s ({:.2} tokens/sec)", 
             gen_time.as_secs_f64(),
             generated_tokens.len() as f64 / gen_time.as_secs_f64());
    
    println!("\nDecoding {} tokens...", generated_tokens.len());
    
    if generated_tokens.is_empty() {
        return Err("No tokens were generated".into());
    }
    
    // Decode the generated tokens
    let generated_text = tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| format!("Failed to decode: {}", e))?;
    
    // Trim whitespace from the output
    let generated_text = generated_text.trim().to_string();
    
    if generated_text.is_empty() {
        return Err("Generated text is empty after decoding".into());
    }
    
    Ok(generated_text)
}


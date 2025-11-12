// DOCUMENT LOADING AND PARSING
use lopdf::Document;
use std::path::Path;
use std::collections::HashMap;
use std::fs;

pub fn load_document(path: &str) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let path = Path::new(path);

    if path.is_file() {
        parse_pdf(path)
    }
    else if path.is_dir() {
        find_pdfs(path)
    }
    else {
        Err(format!("Invalid path: {}", path.display()).into())
    }
}

fn parse_pdf(path: &Path) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let mut hashmap = HashMap::new();
    let doc = Document::load(path).unwrap();
    let pages = doc.get_pages();
    println!("Number of pages: {}", pages.len());
    
    // Extract text from decrypted document
    let filename = path.file_name().unwrap().to_string_lossy().to_string();
    let page_numbers: Vec<u32> = pages.keys().cloned().collect();
    let text = doc.extract_text(&page_numbers).unwrap();
    println!("Extracted {} characters of text", text.len());
    hashmap.insert(filename, text);
    return Ok(hashmap);
}

fn find_pdfs(dir: &Path) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let mut hashmap = HashMap::new();

    for file in fs::read_dir(dir)? {
        let file = file?;
        let path = file.path();
        if path.is_file() && path.extension() == Some("pdf".as_ref()) {
            let text = parse_pdf(&path)?;
            hashmap.extend(text);
        }
    }
    return Ok(hashmap);
}


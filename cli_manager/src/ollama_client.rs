use reqwest::blocking::Client;
use serde_json::Value;
use std::error::Error;
use std::io::{self, BufRead, BufReader, Write};

/// Queries the Ollama server with a prompt using the specified model
pub fn query_ollama(model: &str, prompt: &str) -> Result<String, Box<dyn Error>> {
    // Create reqwest blocking client
    let client = Client::new();
    
    // Build JSON request body
    let request_body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": true
    });

    // Send POST request to Ollama API
    let response = client
        .post("http://127.0.0.1:11434/api/generate")
        .json(&request_body)
        .send()?;

    // Check response status for success
    if !response.status().is_success() {
        return Err(format!("Ollama API returned status: {}", response.status()).into());
    }

    // Handle streaming response
    let mut full_response = String::new();
    let reader = BufReader::new(response);
    
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        
        // Parse each JSON line
        if let Ok(json) = serde_json::from_str::<Value>(&line) {
            if let Some(response_part) = json.get("response").and_then(|r| r.as_str()) {
                // Print each part as it's received
                print!("{}", response_part);
                io::stdout().flush()?;
                full_response.push_str(response_part);
            }
            
            // Check if this is the final response
            if json.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                break;
            }
        }
    }

    if full_response.is_empty() {
        full_response = "No response received".to_string();
    }

    Ok(full_response)
}
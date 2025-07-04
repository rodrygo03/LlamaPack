use reqwest::blocking::Client;
use serde_json::Value;
use std::error::Error;
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Duration;
use std::thread;

pub struct OllamaClient {
    client: Client,
    base_url: String,
}

impl OllamaClient {
    pub fn new() -> Self {
        OllamaClient { 
            client: Client::new(), 
            base_url: "http://127.0.0.1:11434".to_string(),
        }
    }

    pub fn validate_daemon(&self) -> Result<bool, Box<dyn std::error::Error>> {
        let response = self.client.get(format!("{}/api/tags", self.base_url)).send()?;
        if response.status().is_success() {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn launch_daemon(&self) -> Result<(), Box<dyn Error>> {
        // Start ollama serve in background
        let _child = Command::new("ollama")
            .arg("serve")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start ollama daemon: {}. Make sure 'ollama' is installed and in PATH.", e))?;
        
        for attempt in 1..=10 {
            thread::sleep(Duration::from_secs(2));
            if let Ok(true) = self.validate_daemon() {
                return Ok(());
            }
            if attempt < 10 {
                print!(".");
                io::stdout().flush()?;
            }
        }
        
        Err("Ollama daemon failed to start within timeout period".into())
    }

    pub fn list_available_models(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let response = self.client.get(format!("{}/api/tags", self.base_url)).send()?;
        if !response.status().is_success() {
            return Err(format!("Failed to list models: {}", response.status()).into());
        }
        
        let tags: Value = response.json()?;
        let mut models = Vec::new();
        
        if let Some(model_list) = tags.get("models").and_then(|m| m.as_array()) {
            for model_info in model_list {
                if let Some(name) = model_info.get("name").and_then(|n| n.as_str()) {
                    models.push(name.to_string());
                }
            }
        }
        
        Ok(models)
    }

    pub fn select_model(&self) -> Result<String, Box<dyn Error>> {
        println!("================");
        let models = self.list_available_models()?;
        
        if models.is_empty() {
            println!("No LLMs on this machine");
            println!("Would you like to pull a model? (Y/N)");
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim().to_lowercase();
            
            if input == "y" || input == "yes" {
                return self.prompt_and_pull_model();
            } else {
                return Err("No LLMs available and user declined to pull a model".into());
            }
        }
        
        println!("LLMs on this machine:");
        for (i, model) in models.iter().enumerate() {
            println!("{}. {}", i + 1, model);
        }
        println!("{}. Pull a new model", models.len() + 1);
        
        loop {
            print!("Select a LLM (1-{}): ", models.len() + 1);
            io::stdout().flush()?;
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();
            
            if let Ok(choice) = input.parse::<usize>() {
                if choice >= 1 && choice <= models.len() {
                    return Ok(models[choice - 1].clone());
                } else if choice == models.len() + 1 {
                    return self.prompt_and_pull_model();
                }
            }
            
            println!("Invalid selection. Please try again.");
        }
    }

    pub fn prompt_and_pull_model(&self) -> Result<String, Box<dyn Error>> {
        println!("Enter the name of the model you want to pull:");
        // println!("(Examples: llama2, codellama, mistral, etc.)");
        print!("Model name: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let model_name = input.trim();
        
        if model_name.is_empty() {
            return Err("No model name provided".into());
        }
        
        println!("Pulling model '{}'...", model_name);
        self.pull_model(model_name)?;
        
        Ok(model_name.to_string())
    }

    pub fn pull_model(&self, model: &str) -> Result<(), Box<dyn Error>> {
        println!("Pulling model '{}'... This may take a while.", model);
        
        let request_body = serde_json::json!({
            "name": model,
            "stream": true
        });

        let response = self.client
            .post(format!("{}/api/pull", self.base_url))
            .json(&request_body)
            .send()?;

        if !response.status().is_success() {
            return Err(format!("Failed to pull model: {}", response.status()).into());
        }

        let reader = BufReader::new(response);
        let mut last_status = String::new();
        
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            if let Ok(json) = serde_json::from_str::<Value>(&line) {
                if let Some(status) = json.get("status").and_then(|s| s.as_str()) {
                    if status != last_status {
                        println!("{}", status);
                        last_status = status.to_string();
                    }
                }
                
                if let Some(completed) = json.get("completed").and_then(|c| c.as_u64()) {
                    if let Some(total) = json.get("total").and_then(|t| t.as_u64()) {
                        let progress = (completed as f64 / total as f64) * 100.0;
                        print!("\rProgress: {:.1}%", progress);
                        io::stdout().flush()?;
                    }
                }
            }
        }
        
        println!("\n Model '{}' pulled successfully.", model);
        Ok(())
    }

    pub fn query_model(&self, model: &str, prompt: &str) -> Result<String, Box<dyn Error>> {
        let request_body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": true
        });

        let response = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&request_body)
            .send()?;

        if !response.status().is_success() {
            return Err(format!("Ollama API returned status: {}", response.status()).into());
        }

        let mut full_response = String::new();
        let reader = BufReader::new(response);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            if let Ok(json) = serde_json::from_str::<Value>(&line) {
                if let Some(response_part) = json.get("response").and_then(|r| r.as_str()) {
                    print!("{}", response_part);
                    io::stdout().flush()?;
                    full_response.push_str(response_part);
                }

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
}

  

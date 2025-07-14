use reqwest::blocking::Client;
use serde_json::Value;
use std::error::Error;
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use std::thread;
use std::fs;
use indicatif::{ProgressBar, ProgressStyle};

pub struct OllamaClient {
    client: Client,
    base_url: String,
    daemon_process: Option<Child>
}

impl OllamaClient {
    pub fn new() -> Self {
        OllamaClient { 
            client: Client::new(), 
            base_url: "http://127.0.0.1:11434".to_string(),
            daemon_process: None
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

    pub fn launch_daemon(&mut self) -> Result<(), Box<dyn Error>> {
        // Start ollama serve in background
        let _child = Command::new("ollama")
            .arg("serve")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start ollama daemon: {}. Make sure 'ollama' is installed and in PATH.", e))?;
        
        self.daemon_process = Some(_child);

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

    pub fn select_model(&self) -> Result<String, Box<dyn Error>> {
        println!("================");
        let models = self.list_available_models()?;
        
        if models.is_empty() {
            println!("No LLMs on this machine.");
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

    // Private:

    fn list_available_models(&self) -> Result<Vec<String>, Box<dyn Error>> {
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

    fn prompt_and_pull_model(&self) -> Result<String, Box<dyn Error>> {
        println!("Enter the path of desired model");
        println!("(Examples: codellama, hf.co/TheBloke/CodeLlama-34B-GGUF:Q4_K_M, etc)");
        print!("Model path: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let base_model_name = input.trim();
        
        if base_model_name.is_empty() {
            return Err("No model path provided".into());
        }
        
        Ok(self.pull_model(base_model_name)?)
    }

    fn pull_model(&self, base_model: &str) -> Result<String, Box<dyn Error>> {
        println!("Manifesting '{}'", base_model);

        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner()
            .template("{spinner} {msg}")
            .expect("template valid"));
        pb.set_message("Creating model...");
        pb.enable_steady_tick(Duration::from_millis(100));
        
        // Hardcoded Modelfile template
        let modelfile_template = format!(r#"
            FROM {}

            SYSTEM """
            You are an expert software development assistant. Your job is to help the user understand, write, and debug code across many languages.
            Always clearly separate explanations from code.
            When generating code, use triple backticks with language identifiers (e.g., ```rust).
            Only generate code that is directly related to the user's task and relevant context.
            """

            TEMPLATE """
            {{{{ .System }}}}

            Context:
            {{{{ .Context }}}}

            User:
            {{{{ .Prompt }}}}

            Assistant:
            """
            "#, base_model
        );

        // Create temporary Modelfile 
        let temp_modelfile_path = format!("Modelfile");
        fs::write(&temp_modelfile_path, modelfile_template)?;
        
        let model_name = base_model.to_string();
        
        // Use ollama create to build the custom model
        let output = Command::new("ollama")
            .arg("create")
            .arg(&model_name)
            .arg("-f")
            .arg(&temp_modelfile_path)
            .output()
            .map_err(|e| format!("Failed to run ollama create: {}. Make sure 'ollama' is installed and in PATH.", e))?;
        
        // Clean up temporary file
        let _ = fs::remove_file(&temp_modelfile_path);

        pb.finish_and_clear();
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to create model: {}", stderr).into());
        }
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.is_empty() {
            println!("{}", stdout);
        }
        
        println!("'{}' created successfully.", model_name);
        Ok(model_name)
    }

}

impl Drop for OllamaClient {
    fn drop(&mut self) {
        if let Some(child) = self.daemon_process.as_mut() {
            if let Err(e) = child.kill() {
                eprintln!("Failed to kill Ollama daemon: {}", e);
            } else {
                let _ = child.wait(); // Ensure the process is fully reaped
                println!("Ollama daemon terminated.");
            }
        } else {
            eprintln!("No Ollama daemon to kill.");
        }
    }
}
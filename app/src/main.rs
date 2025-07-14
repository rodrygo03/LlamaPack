use std::env;
use std::error::Error;
use std::io::{self, Write};

use llama_pack::session::SessionManager;
use llama_pack::ollama_client::OllamaClient;
use llama_pack::embedder::Embedder;

fn main() -> Result<(), Box<dyn Error>> {
    // println!("Ollama Code");
    println!("===========================");

    let mut ollama_client = OllamaClient::new();

    match ollama_client.validate_daemon() {
        Ok(true) => {
            // Daemon is running, continue
            println!("Ollama daemon is running.")
        }
        Ok(false) | Err(_) => {
            println!("Launching Ollama daemon.");
            match ollama_client.launch_daemon() {
                Ok(()) => {
                    println!("Daemon started successfully.");
                }
                Err(e) => {
                    eprintln!("Failed to start daemon: {}", e);
                    eprintln!("Please ensure Ollama is installed and try running 'ollama serve' manually.");
                    std::process::exit(1);
                }
            }
        }
    }

    let selected_model = match ollama_client.select_model() {
        Ok(model) => {
            println!("Selected LLM: {}", model);
            model
        }
        Err(e) => {
            eprintln!("Failed to select LLM: {}", e);

            std::process::exit(1);
        }
    };
    
    // Get current working directory
    let current_dir = env::current_dir()?;
    println!("Working directory: {}", current_dir.display());
    
    // Auto-start new session
    let mut session_manager = SessionManager::new_session()?;
    println!("New session started. Type 'exit' to quit.\n");
    
    // Start prompt loop
    prompt_loop(&mut session_manager, &ollama_client, &selected_model)?;

    Ok(())
}

fn prompt_loop(session_manager: &mut SessionManager, ollama_client: &OllamaClient, model: &str) -> Result<(), Box<dyn Error>> {
    loop {
        print!("{}> ", model.split(':').next().unwrap_or(model));
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // Check for exit command
        if input == "exit" || input == "quit" {
            println!("Goodbye!");
            break;
        }

        // Skip empty inputs
        if input.is_empty() {
            continue;
        }

        // Query Ollama with selected model
        println!("Thinking...");
        match ollama_client.query_model(model, input) {
            Ok(response) => {
                println!("\n{}\n", response);
                
                // Save to session log
                if let Err(e) = session_manager.save_log(input, &response) {
                    eprintln!("Warning: Failed to save to session: {}", e);
                }
                

            }
            Err(e) => {
                eprintln!("Error querying Ollama: {}", e);
                eprintln!("Make sure Ollama is running and the '{}' model is available.", model);
            }
        }
    }

    Ok(())
}

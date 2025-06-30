use std::env;
use std::error::Error;
use std::io::{self, Write};

mod session;
mod ollama_client;

use session::SessionManager;
use ollama_client::query_ollama;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Starcoder2-3b Code");
    println!("===========================");
    
    // Get current working directory
    let current_dir = env::current_dir()?;
    println!("Working directory: {}", current_dir.display());
    
    // Auto-start new session
    let mut session_manager = SessionManager::new_session()?;
    println!("New session started. Type 'exit' to quit.\n");
    
    // Start prompt loop
    prompt_loop(&mut session_manager)?;

    Ok(())
}

fn prompt_loop(session_manager: &mut SessionManager) -> Result<(), Box<dyn Error>> {
    loop {
        print!("starcoder2> ");
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

        // Query Ollama with starcoder2-3b-instruct model
        println!("Thinking...");
        match query_ollama("starcoder2-3b-instruct-GGUF:Q4_K_M", input) {
            Ok(response) => {
                println!("\n{}\n", response);
                
                // Save to session log
                if let Err(e) = session_manager.save_log(input, &response) {
                    eprintln!("Warning: Failed to save to session: {}", e);
                }
                

            }
            Err(e) => {
                eprintln!("Error querying Ollama: {}", e);
                eprintln!("Make sure Ollama is running and starcoder2-3b-instruct-GGUF:Q4_K_M model is available.");
            }
        }
    }

    Ok(())
}
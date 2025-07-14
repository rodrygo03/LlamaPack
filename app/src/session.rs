use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::{env, fs};
use std::path::PathBuf;
use uuid::Uuid;


#[derive(Serialize, Deserialize)]
pub struct PromptLog {
    timestamp: String,
    prompt: String,
    response: String,
}

#[derive(Serialize, Deserialize)]
pub struct Session {
    id: String,
    logs: Vec<PromptLog>,
}

pub struct SessionManager {
    session: Session,
    session_dir: PathBuf,
}

impl SessionManager {
    /// Creates a new session with a unique ID
    pub fn new_session() -> std::io::Result<Self> {
        let id = Uuid::new_v4().to_string();
        let session = Session { id, logs: Vec::new() };

        let mut session_dir = env::current_dir()?;
        session_dir.push(".coder_sessions");
        session_dir.push(&session.id);

        fs::create_dir_all(&session_dir)?;

        Ok(SessionManager { session, session_dir })
    }


    /// Loads an existing session by ID
    pub fn load_session(_id: &str) -> Result<Self, Box<dyn Error>> {
        // TODO: Construct path to session file (~/.coder_sessions/<id>.json)
        // TODO: Check if session file exists
        // TODO: Read and parse JSON file into Session struct
        // TODO: Return SessionManager instance or error
        
        todo!("Implement load_session")
    }

    /// Lists all available session IDs
    pub fn list_sessions() -> Result<Vec<String>, Box<dyn Error>> {
        // TODO: Get session directory path
        // TODO: Read directory contents
        // TODO: Filter for .json files
        // TODO: Extract session IDs from filenames
        // TODO: Return sorted list of session IDs
        
        todo!("Implement list_sessions")
    }

    /// Saves a prompt and response to the current session
    pub fn save_log(&mut self, prompt: &str, response: &str) -> Result<(), Box<dyn Error>> {
        let log = PromptLog {
            timestamp: Utc::now().to_rfc3339(),
            prompt: prompt.to_string(),
            response: response.to_string(),
        };
        
        self.session.logs.push(log);
        self.save_session()?;
        Ok(())
    }

    /// Private helper to save session to disk
    fn save_session(&self) -> Result<(), Box<dyn Error>> {
        let session_file = self.session_dir.join("session.json");
        let session_json = serde_json::to_string_pretty(&self.session)?;
        fs::write(session_file, session_json)?;
        Ok(())
    }

    /// Private helper to get session directory path
    fn get_session_dir() -> PathBuf {
        // TODO: Get user home directory
        // TODO: Append .coder_sessions to path
        
        todo!("Implement get_session_dir")
    }
}
use ort::{session::Session, inputs, value::Value,};
use tokenizers::Tokenizer;

use ndarray::{Array, IxDyn};

const MAX_LEN: usize = 768;

pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl Embedder {
    /// Create a new Embedder from ONNX model and tokenizer file paths
    pub fn new(model_path: &str, tokenizer_path: &str) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        tokenizer
            .save("../models/UniXcoder/unixcoder-tokenizer.json", true)
            .map_err(|e| anyhow::anyhow!("Failed to save tokenizer: {}", e))?;

        Ok(Self {
            session,
            tokenizer,
        })
    }

    pub fn embed(&mut self, prompt: &str) -> anyhow::Result<Vec<f32>> {
        let prompt = format!("<encoder-only>{}", prompt);

        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let seq_len = input_ids.len().min(MAX_LEN);

        let input_ids_array = Array::from_shape_vec(
            IxDyn(&[1, seq_len]),
            input_ids.iter().map(|&id| id as i64).collect(),
        )?;
        let attention_mask_array = Array::from_shape_vec(
            IxDyn(&[1, seq_len]),
            attention_mask.iter().map(|&mask| mask as i64).collect(),
        )?;

        let input_tensor = Value::from_array(input_ids_array)?;
        let attention_tensor = Value::from_array(attention_mask_array)?;

        let outputs = self.session.run(inputs![
            "input_ids" => input_tensor,
            "attention_mask" => attention_tensor
        ])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        let output = ndarray::ArrayD::from_shape_vec(shape, data.to_vec())?;
        Ok(output.iter().cloned().collect())
    }

    pub fn embed_batch(&mut self, prompts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(prompts.len());
        
        for prompt in prompts {
            let embedding = self.embed(prompt)?;
            embeddings.push(embedding);
        }
        
        Ok(embeddings)
    }
}

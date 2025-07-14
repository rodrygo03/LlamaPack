use llama_pack::embedder::Embedder; 
use anyhow::Result;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

#[test]
fn test_embed_single_prompt() -> Result<()> {
    let mut embedder = Embedder::new(
        "../models/UniXcoder/unixcoder-embedding.onnx",
        "../models/UniXcoder/tokenizer.json"
    )?;

    let vec = embedder.embed("sort a list of integers")?;
    println!("Embedding vector length: {}", vec.len());
    println!("First 10 values: {:?}", &vec[0..10.min(vec.len())]);
    println!("Vector magnitude: {}", vec.iter().map(|x| x * x).sum::<f32>().sqrt());
    assert!(!vec.is_empty());
    Ok(())
}

#[test]
fn test_embed_retrieval_task() -> Result<()> {
    let mut embedder = Embedder::new(
        "../models/UniXcoder/unixcoder-embedding.onnx",
        "../models/UniXcoder/tokenizer.json"
    )?;

    let query = "sort a list of integers";
    let qv = embedder.embed(query)?;

    let candidates = vec![
        ("bubble_sort", "def bubble_sort(arr): ..."),
        ("factorial", "def factorial(n): ..."),
        ("reverse", "def reverse_string(s): ..."),
    ];

    let mut results = vec![];
    for (label, code) in candidates {
        let cv = embedder.embed(code)?;
        results.push((label, cosine_similarity(&qv, &cv)));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("Query: '{}'", query);
    println!("Query embedding length: {}", qv.len());
    println!("Query embedding magnitude: {}", qv.iter().map(|x| x * x).sum::<f32>().sqrt());
    println!("\nRanked results:");
    for (i, (label, score)) in results.iter().enumerate() {
        println!("  {}. {} (similarity: {:.4})", i + 1, label, score);
    }
    println!("\nBest match: {} with similarity score {:.4}", results[0].0, results[0].1);

    assert_eq!(results[0].0, "bubble_sort");
    Ok(())
}

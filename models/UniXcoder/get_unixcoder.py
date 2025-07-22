import json

def main():
    try:
        import torch
        from torch import nn
        from transformers import AutoTokenizer, AutoModel
        import onnxruntime as ort
    except ImportError as e:
        print(f"Missing package: {e}")
        return False
    
    class UniXCoderEmbedder(nn.Module):
        def __init__(self, model_id="microsoft/unixcoder-base"):
            super().__init__()
            self.model = AutoModel.from_pretrained(model_id)

        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = (hidden_states * mask).sum(1)
            sum_mask = mask.sum(1)
            return sum_hidden / sum_mask

    model_id = "microsoft/unixcoder-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = UniXCoderEmbedder(model_id)
    model.eval()
    
    text = """Write a Python function that reads a file line by line, removes any trailing whitespace from each line, and returns the cleaned list of lines.

```python
def clean_file_lines(file_path):
    with open(file_path, 'r') as f:
        return [line.rstrip() for line in f]
```"""
    
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    torch.onnx.export(
        model,
        (tokens["input_ids"], tokens["attention_mask"]),
        "unixcoder-embedding.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "embedding": {0: "batch_size"},
        },
        opset_version=14
    )
    
    session = ort.InferenceSession("unixcoder-embedding.onnx")
    inputs = {
        "input_ids": tokens["input_ids"].numpy(),
        "attention_mask": tokens["attention_mask"].numpy()
    }
    embedding = session.run(None, inputs)[0]
    
    tokenizer.backend_tokenizer.save("tokenizer.json")
    
    model_config = {
        "model_id": model_id,
        "max_length": 512,
        "embedding_dim": embedding.shape[-1],
        "files": ["unixcoder-embedding.onnx", "tokenizer.json"]
    }
    
    with open("config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    
    return True

if __name__ == "__main__":
    main()
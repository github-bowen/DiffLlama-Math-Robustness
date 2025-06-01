import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path, device=None):
    """Load a model and its tokenizer from the given path."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path} to {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """Generate text based on a prompt using the given model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Paths to the downloaded models
    diffllama_path = "./cache/models--reyllama--DiffLlama-375M/snapshots/8960f22033190f1560537f4932fe649828ef53e2/checkpoint-64434"
    llama_path = "./cache/models--reyllama--Llama_375M/snapshots/416b70824d560b02245268c208ffd5388b4aa056/checkpoint-64434"
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load DiffLlama model
    print("\nLoading DiffLlama-375M model...")
    diffllama_model, diffllama_tokenizer = load_model(diffllama_path, device)
    
    # Example prompt
    prompt = "The quick brown fox jumped over"
    
    # Generate text with DiffLlama
    print("\n===== DiffLlama-375M Generation =====")
    print(f"Prompt: {prompt}")
    diffllama_output = generate_text(diffllama_model, diffllama_tokenizer, prompt)
    print(f"Generated: {diffllama_output}")
    
    # Free up memory
    del diffllama_model
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Load Llama model
    print("\nLoading Llama_375M model...")
    llama_model, llama_tokenizer = load_model(llama_path, device)
    
    # Generate text with Llama
    print("\n===== Llama_375M Generation =====")
    print(f"Prompt: {prompt}")
    llama_output = generate_text(llama_model, llama_tokenizer, prompt)
    print(f"Generated: {llama_output}")

if __name__ == "__main__":
    main()

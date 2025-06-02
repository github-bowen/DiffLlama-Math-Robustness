import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path, device=None):
    """Load a model and its tokenizer from the given path."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path} to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # TODO: Set to float32
        device_map=device
    )
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """Generate text based on a prompt using the given model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

def main():
    parser = argparse.ArgumentParser(description="Interactive text generation with Llama models")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["diffllama", "llama"], 
        required=True,
        help="Which model to use (diffllama or llama)"
    )
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Select model path based on argument
    if args.model == "diffllama":
        model_path = "./cache/models--reyllama--DiffLlama-375M/snapshots/8960f22033190f1560537f4932fe649828ef53e2/checkpoint-64434"
        model_name = "DiffLlama-375M"
    else:  # llama
        model_path = "./cache/models--reyllama--Llama_375M/snapshots/416b70824d560b02245268c208ffd5388b4aa056/checkpoint-64434"
        model_name = "Llama_375M"
    
    # Load the selected model
    model, tokenizer = load_model(model_path, device)
    
    print(f"\nInteractive {model_name} Session")
    print("Enter prompts for text generation. Type 'quit' or 'exit' to end.")
    
    while True:
        prompt = input("\nPrompt: ")
        
        if prompt.lower() in ["quit", "exit"]:
            break
        
        try:
            output = generate_text(model, tokenizer, prompt)
            print(f"\nGenerated: {output}")
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()

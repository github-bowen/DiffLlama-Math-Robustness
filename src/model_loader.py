import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Global variables for model paths (from the download script)
DIFFLAMA_PATH = "./cache/models--reyllama--DiffLlama-375M/snapshots/8960f22033190f1560537f4932fe649828ef53e2/checkpoint-64434"
LLAMA_PATH = "./cache/models--reyllama--Llama_375M/snapshots/416b70824d560b02245268c208ffd5388b4aa056/checkpoint-64434"

def load_model_and_tokenizer(model_type, device=None, for_training=False):
    """
    Load the specified model and tokenizer.
    
    Args:
        model_type: "diffllama" or "llama"
        device: device to load model on, defaults to auto-detect
        for_training: if True, don't set model to eval mode
    
    Returns:
        model, tokenizer
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_type == "diffllama":
        model_path = DIFFLAMA_PATH
        print(f"Loading DiffLlama-375M from: {model_path}")
    elif model_type == "llama":
        model_path = LLAMA_PATH
        print(f"Loading Llama-375M from: {model_path}")
    else:
        raise ValueError("Invalid model_type. Choose 'diffllama' or 'llama'.")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # TODO: Set to float32
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Tokenizer loaded. EOS token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
    print(f"Tokenizer BOS token: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
    print(f"Tokenizer UNK token: '{tokenizer.unk_token}', ID: {tokenizer.unk_token_id}")
    print(f"Tokenizer PAD token before setting: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")

    if not for_training:
        model.eval()  # Set to evaluation mode
    
    # Set pad token if not available
    if tokenizer.pad_token is None:
        print("PAD token is None. Setting PAD token to EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer PAD token after setting: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")

    print(f"{model_type} loaded on {device}.")
    return model, tokenizer

def load_model_from_path(model_path, device=None, for_training=False):
    """
    Load model and tokenizer from a specific path (useful for fine-tuned models).
    
    Args:
        model_path: path to the model directory
        device: device to load model on
        for_training: if True, don't set model to eval mode
    
    Returns:
        model, tokenizer
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    print(f"Loading model from path: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # TODO: Set to float32
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Tokenizer loaded. EOS token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
    print(f"Tokenizer BOS token: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
    print(f"Tokenizer UNK token: '{tokenizer.unk_token}', ID: {tokenizer.unk_token_id}")
    print(f"Tokenizer PAD token before setting (from path): '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
    
    if not for_training:
        model.eval()
    
    if tokenizer.pad_token is None:
        print("PAD token is None when loading from path. Setting PAD token to EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer PAD token after setting (from path): '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
    
    print(f"Model loaded from {model_path} on {device}.")
    return model, tokenizer

def get_model_paths():
    """Return the paths to the original models."""
    return {
        "diffllama": DIFFLAMA_PATH,
        "llama": LLAMA_PATH
    }

if __name__ == "__main__":
    # Test loading both models
    try:
        print("Testing model loading...")
        
        print("\nAttempting to load Llama-375M...")
        llama_model, llama_tokenizer = load_model_and_tokenizer("llama")
        print("Llama-375M loaded successfully.")
        del llama_model, llama_tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\nAttempting to load DiffLlama-375M...")
        diff_model, diff_tokenizer = load_model_and_tokenizer("diffllama")
        print("DiffLlama-375M loaded successfully.")
        del diff_model, diff_tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure model paths are correct and models are downloaded.") 
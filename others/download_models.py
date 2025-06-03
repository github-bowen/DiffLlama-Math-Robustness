from huggingface_hub import snapshot_download


def download_models():
    try:
        #"RedHatAI/Llama-2-7b-gsm8k"  "meta-llama/Llama-3.2-3B-Instruct" "AquaLabs/Llama-3.2-1B-GSM8K"
        diffllama_path = snapshot_download(
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
            cache_dir="./cache",
            use_auth_token=True,
            # allow_patterns=["checkpoint-64434/*"],
            force_download=False
        )
            
        print(f"DiffLlama-1B-patch downloaded to: {diffllama_path}")
        
        return True
    
    except Exception as e:
        print(f"Error downloading models: {e}")
        return False

if __name__ == "__main__":
    download_models()

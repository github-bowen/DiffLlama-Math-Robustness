#!/usr/bin/env python3
"""
Model Download Script - Download Required Models for Experiments

This script is responsible for downloading DiffLlama and Llama models from Hugging Face Hub.
Supports both local environment and Google Colab environment.
"""

import os
from huggingface_hub import snapshot_download

def download_models():
    """Download models required for experiments"""
    
    # Download checkpoint-64434 for DiffLlama-375M modelAdd commentMore actions
    diffllama_path = snapshot_download(
        repo_id="reyllama/DiffLlama-375M",
        cache_dir="./cache",
        allow_patterns=["checkpoint-64434/*"],
        force_download=False
    )

    # Download checkpoint-64434 for Llama_375M model
    llama_path = snapshot_download(
        repo_id="reyllama/Llama_375M",
        cache_dir="./cache",
        allow_patterns=["checkpoint-64434/*"],
        force_download=False
    )

    print(f"DiffLlama-375M checkpoint-64434 downloaded to: {diffllama_path}")
    print(f"Llama_375M checkpoint-64434 downloaded to: {llama_path}")
    return True

if __name__ == "__main__":
    download_models() 
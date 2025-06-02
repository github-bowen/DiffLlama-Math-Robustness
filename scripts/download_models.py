#!/usr/bin/env python3
"""
Model Download Script - Download Required Models for Experiments

This script is responsible for downloading DiffLlama and Llama models from Hugging Face Hub.
Supports both local environment and Google Colab environment.
"""

import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from huggingface_hub import snapshot_download
import argparse

def download_models(quiet=True):
    """Download models required for experiments
    
    Args:
        quiet (bool): If True, suppress download output. Default is True.
    """
    
    if quiet:
        # Suppress output during downloads
        stdout_capture = StringIO()
        stderr_capture = StringIO()
    
    try:
        # Download checkpoint-64434 for DiffLlama-375M model
        if quiet:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                diffllama_path = snapshot_download(
                    repo_id="reyllama/DiffLlama-375M",
                    cache_dir="./cache",
                    allow_patterns=["checkpoint-64434/*"],
                    force_download=False
                )
        else:
            diffllama_path = snapshot_download(
                repo_id="reyllama/DiffLlama-375M",
                cache_dir="./cache",
                allow_patterns=["checkpoint-64434/*"],
                force_download=False
            )

        # Download checkpoint-64434 for Llama_375M model
        if quiet:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                llama_path = snapshot_download(
                    repo_id="reyllama/Llama_375M",
                    cache_dir="./cache",
                    allow_patterns=["checkpoint-64434/*"],
                    force_download=False
                )
        else:
            llama_path = snapshot_download(
                repo_id="reyllama/Llama_375M",
                cache_dir="./cache",
                allow_patterns=["checkpoint-64434/*"],
                force_download=False
            )

        if not quiet:
            print(f"DiffLlama-375M checkpoint-64434 downloaded to: {diffllama_path}")
            print(f"Llama_375M checkpoint-64434 downloaded to: {llama_path}")
        
        return True
    
    except Exception as e:
        print(f"Error downloading models: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models for experiments.")
    parser.add_argument(
        "--quiet", 
        action="store_true", 
        help="Suppress output during model download"
    )
    args = parser.parse_args()
    # Call the download function with quiet mode if specified
    download_models(quiet=args.quiet)

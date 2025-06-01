#!/usr/bin/env python3
"""
Model Download Script - 下载实验所需的模型

此脚本负责从 Hugging Face Hub 下载 DiffLlama 和 Llama 模型。
支持本地环境和 Google Colab 环境。
"""

import os
from huggingface_hub import snapshot_download

def download_models():
    """下载实验所需的模型"""
    
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
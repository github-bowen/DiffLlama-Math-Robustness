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
    
    # 确保缓存目录存在
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    models = {
        "DiffLlama-375M": "reyllama/DiffLlama-375M",
        "Llama_375M": "reyllama/Llama_375M"
    }
    
    print("🔄 开始下载模型...")
    
    for model_name, model_id in models.items():
        print(f"\n📥 下载 {model_name}...")
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                resume_download=True
            )
            print(f"✅ {model_name} 下载完成")
        except Exception as e:
            print(f"❌ {model_name} 下载失败: {e}")
            return False
    
    print("\n🎉 所有模型下载完成！")
    return True

if __name__ == "__main__":
    download_models() 
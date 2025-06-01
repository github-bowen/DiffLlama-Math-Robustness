#!/usr/bin/env python3
"""
Configuration file for Google Colab environment.
Handles Google Drive mounting and path configuration.
"""

import os
from pathlib import Path

# Colab-specific paths
COLAB_PATHS = {
    # Local Colab paths (temporary)
    "local_cache": "/content/cache",
    "local_data": "/content/data", 
    "local_results": "/content/results",
    "local_models_ft": "/content/models_finetuned",
    
    # Google Drive paths (persistent)
    "drive_base": "/content/drive/MyDrive/DiffLlama_Experiment",
    "drive_cache": "/content/drive/MyDrive/DiffLlama_Experiment/models",
    "drive_data": "/content/drive/MyDrive/DiffLlama_Experiment/data",
    "drive_results": "/content/drive/MyDrive/DiffLlama_Experiment/results",
    "drive_models_ft": "/content/drive/MyDrive/DiffLlama_Experiment/models_finetuned"
}

# Model paths within cache directory
MODEL_PATHS = {
    "diffllama": "models--reyllama--DiffLlama-375M/snapshots/8960f22033190f1560537f4932fe649828ef53e2/checkpoint-64434",
    "llama": "models--reyllama--Llama_375M/snapshots/416b70824d560b02245268c208ffd5388b4aa056/checkpoint-64434"
}

# Colab-specific settings
COLAB_SETTINGS = {
    # Memory optimization
    "max_eval_samples_gpu_limited": 50,
    "max_eval_samples_gpu_sufficient": 200,
    "max_sft_samples": 200,
    "sft_batch_size": 2,
    "eval_batch_size": 1,
    
    # Generation settings
    "max_new_tokens": 256,  # Reduced for Colab
    "generation_timeout": 30,  # seconds
    
    # Attention analysis
    "max_attention_samples": 10,
    "attention_layers": [-1],  # Only last layer
    "attention_heads": [0],    # Only first head
    
    # File sizes
    "max_log_file_size": 50 * 1024 * 1024,  # 50MB
    "checkpoint_interval": 50,  # Save every 50 samples
}

def setup_colab_paths(use_drive=True):
    """
    Setup directory structure for Colab.
    
    Args:
        use_drive: Whether to use Google Drive for persistent storage
        
    Returns:
        dict: Configured paths
    """
    if use_drive and os.path.exists("/content/drive/MyDrive"):
        print("📁 Using Google Drive for persistent storage")
        
        # Create Google Drive directories
        for path_key, path_value in COLAB_PATHS.items():
            if path_key.startswith("drive_"):
                os.makedirs(path_value, exist_ok=True)
        
        # Return drive paths
        return {
            "cache": COLAB_PATHS["drive_cache"],
            "data": COLAB_PATHS["drive_data"],
            "results": COLAB_PATHS["drive_results"],
            "models_finetuned": COLAB_PATHS["drive_models_ft"]
        }
    else:
        print("📁 Using local storage (temporary)")
        
        # Create local directories
        for path_key, path_value in COLAB_PATHS.items():
            if path_key.startswith("local_"):
                os.makedirs(path_value, exist_ok=True)
        
        # Return local paths
        return {
            "cache": COLAB_PATHS["local_cache"],
            "data": COLAB_PATHS["local_data"],
            "results": COLAB_PATHS["local_results"],
            "models_finetuned": COLAB_PATHS["local_models_ft"]
        }

def get_model_path(model_type, cache_dir):
    """
    Get full path to model.
    
    Args:
        model_type: "diffllama" or "llama"
        cache_dir: Base cache directory
        
    Returns:
        str: Full path to model
    """
    if model_type not in MODEL_PATHS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return os.path.join(cache_dir, MODEL_PATHS[model_type])

def get_gpu_optimized_settings():
    """
    Get settings optimized for current GPU.
    
    Returns:
        dict: Optimized settings
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {
                "max_samples": 10,
                "batch_size": 1,
                "use_attention": False,
                "message": "CPU mode - very limited"
            }
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_memory < 12:  # T4, etc.
            return {
                "max_samples": COLAB_SETTINGS["max_eval_samples_gpu_limited"],
                "batch_size": 1,
                "use_attention": True,
                "attention_samples": 5,
                "message": f"Limited GPU ({gpu_memory:.1f}GB) - optimized settings"
            }
        else:  # V100, A100, etc.
            return {
                "max_samples": COLAB_SETTINGS["max_eval_samples_gpu_sufficient"], 
                "batch_size": 2,
                "use_attention": True,
                "attention_samples": COLAB_SETTINGS["max_attention_samples"],
                "message": f"Sufficient GPU ({gpu_memory:.1f}GB) - standard settings"
            }
    except:
        return {
            "max_samples": 20,
            "batch_size": 1,
            "use_attention": True,
            "attention_samples": 5,
            "message": "Unknown GPU - conservative settings"
        }

def create_symlinks(target_paths):
    """
    Create symlinks from standard names to target paths.
    
    Args:
        target_paths: Dictionary of target directories
    """
    for link_name, target_path in target_paths.items():
        # Remove existing symlink or directory
        if os.path.exists(link_name):
            if os.path.islink(link_name):
                os.unlink(link_name)
            elif os.path.isdir(link_name):
                import shutil
                shutil.rmtree(link_name)
        
        # Create symlink
        os.symlink(target_path, link_name)
        print(f"  🔗 {link_name} -> {target_path}")

def check_colab_environment():
    """
    Check if running in Colab and return environment info.
    
    Returns:
        dict: Environment information
    """
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            gpu_name = "None"
            gpu_memory = 0
    except:
        gpu_available = False
        gpu_name = "Unknown"
        gpu_memory = 0
    
    drive_available = os.path.exists("/content/drive/MyDrive")
    
    return {
        "in_colab": in_colab,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory,
        "drive_available": drive_available
    }

def print_environment_info():
    """Print comprehensive environment information."""
    env_info = check_colab_environment()
    
    print("🔍 Environment Information:")
    print(f"  Colab: {'✅' if env_info['in_colab'] else '❌'}")
    print(f"  GPU: {'✅' if env_info['gpu_available'] else '❌'} {env_info['gpu_name']}")
    if env_info['gpu_available']:
        print(f"  GPU Memory: {env_info['gpu_memory_gb']:.1f} GB")
    print(f"  Google Drive: {'✅' if env_info['drive_available'] else '❌'}")
    
    # Get optimized settings
    settings = get_gpu_optimized_settings()
    print(f"  Recommended settings: {settings['message']}")

if __name__ == "__main__":
    # Test configuration
    print("🧪 Testing Colab configuration...")
    print_environment_info()
    
    # Test path setup
    print("\n📁 Testing path setup...")
    paths = setup_colab_paths(use_drive=True)
    print("Configured paths:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    # Test optimized settings
    print("\n⚙️  Testing optimized settings...")
    settings = get_gpu_optimized_settings()
    print("Recommended settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}") 
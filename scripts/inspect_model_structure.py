#!/usr/bin/env python3
"""
Model Structure Inspection Script
Uses src/model_loader to load llama or diffllama models and outputs detailed model structure information
"""

import sys
import os
import argparse
import torch

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import load_model_and_tokenizer

def print_model_structure(model, model_name):
    """
    Output detailed model structure information
    
    Args:
        model: Loaded model
        model_name: Model name
    """
    print(f"\n{'='*80}")
    print(f"Model Structure Analysis: {model_name}")
    print(f"{'='*80}")
    
    # 1. Basic information
    print(f"\n1. Basic Information:")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Model config: {model.config}")
    
    # 2. Parameter statistics
    print(f"\n2. Parameter Statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Parameter size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 3. Model architecture
    print(f"\n3. Model Architecture:")
    print(model)
    
    # 4. Layer hierarchy
    print(f"\n4. Detailed Layer Structure:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only show leaf nodes
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                print(f"   {name}: {type(module).__name__} ({param_count:,} parameters)")
    
    # 5. Weight shapes
    print(f"\n5. Main Weight Layer Shapes:")
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"   {name}: {param.shape}")
    
    # 6. Memory usage (if on GPU)
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        print(f"\n6. GPU Memory Usage:")
        print(f"   Allocated memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"   Cached memory: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

def compare_models(model1, model2, name1, name2):
    """
    Compare structural differences between two models
    
    Args:
        model1, model2: Two models to compare
        name1, name2: Model names
    """
    print(f"\n{'='*80}")
    print(f"Model Comparison: {name1} vs {name2}")
    print(f"{'='*80}")
    
    # Parameter count comparison
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    
    print(f"\nParameter Count Comparison:")
    print(f"   {name1}: {params1:,}")
    print(f"   {name2}: {params2:,}")
    print(f"   Difference: {abs(params1 - params2):,}")
    
    # Architecture comparison
    modules1 = set(name for name, _ in model1.named_modules())
    modules2 = set(name for name, _ in model2.named_modules())
    
    common_modules = modules1 & modules2
    unique_to_1 = modules1 - modules2
    unique_to_2 = modules2 - modules1
    
    print(f"\nArchitecture Comparison:")
    print(f"   Common modules: {len(common_modules)}")
    print(f"   {name1} unique: {len(unique_to_1)}")
    print(f"   {name2} unique: {len(unique_to_2)}")
    
    if unique_to_1:
        print(f"\n   {name1} Unique Modules:")
        for module in sorted(unique_to_1):
            if module:  # Skip empty strings
                print(f"     - {module}")
    
    if unique_to_2:
        print(f"\n   {name2} Unique Modules:")
        for module in sorted(unique_to_2):
            if module:  # Skip empty strings
                print(f"     - {module}")

def main():
    parser = argparse.ArgumentParser(description="Inspect model structure")
    parser.add_argument(
        "--model", 
        choices=["llama", "diffllama", "both"], 
        default="both",
        help="Model to inspect (default: both)"
    )
    parser.add_argument(
        "--device", 
        default=None,
        help="Device to run on (default: auto-detect)"
    )
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Compare differences between two models"
    )
    
    args = parser.parse_args()
    
    # Detect device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    try:
        if args.model == "llama":
            print("Loading Llama model...")
            model, tokenizer = load_model_and_tokenizer("llama", device)
            print_model_structure(model, "Llama-375M")
            
        elif args.model == "diffllama":
            print("Loading DiffLlama model...")
            model, tokenizer = load_model_and_tokenizer("diffllama", device)
            print_model_structure(model, "DiffLlama-375M")
            
        elif args.model == "both":
            print("Loading both models for comparison...")
            
            print("\nLoading Llama model...")
            llama_model, llama_tokenizer = load_model_and_tokenizer("llama", device)
            print_model_structure(llama_model, "Llama-375M")
            
            # Clean up some memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("\nLoading DiffLlama model...")
            diff_model, diff_tokenizer = load_model_and_tokenizer("diffllama", device)
            print_model_structure(diff_model, "DiffLlama-375M")
            
            # Compare models
            if args.compare:
                compare_models(llama_model, diff_model, "Llama-375M", "DiffLlama-375M")
            
            # Clean up memory
            del llama_model, llama_tokenizer, diff_model, diff_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure model paths are correct and models are downloaded.")
        return 1
    
    print(f"\n{'='*80}")
    print("Model structure inspection completed!")
    print(f"{'='*80}")
    return 0

if __name__ == "__main__":
    exit(main())

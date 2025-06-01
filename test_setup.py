#!/usr/bin/env python3
"""
Test script to verify that the experimental setup is working correctly.
Run this before executing the full experiment to catch potential issues early.
"""

import os
import sys
import traceback

def test_dependencies():
    """Test if all required dependencies are available."""
    print("🔍 Testing dependencies...")
    
    missing_deps = []
    required_packages = [
        'torch', 'transformers', 'datasets', 'matplotlib', 
        'seaborn', 'pandas', 'numpy', 'tqdm'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing_deps.append(package)
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies found")
    return True

def test_model_paths():
    """Test if model paths exist."""
    print("\n🔍 Testing model paths...")
    
    model_paths = {
        "DiffLlama": "./cache/models--reyllama--DiffLlama-375M/snapshots/8960f22033190f1560537f4932fe649828ef53e2/checkpoint-64434",
        "Llama": "./cache/models--reyllama--Llama_375M/snapshots/416b70824d560b02245268c208ffd5388b4aa056/checkpoint-64434"
    }
    
    all_exist = True
    for model_name, path in model_paths.items():
        if os.path.exists(path):
            print(f"  ✓ {model_name}: {path}")
        else:
            print(f"  ✗ {model_name}: {path}")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some models not found. Please run: python pre_download_models.py")
        return False
    
    print("✅ All model paths found")
    return True

def test_model_loading():
    """Test if models can be loaded successfully."""
    print("\n🔍 Testing model loading...")
    
    try:
        from src.model_loader import load_model_and_tokenizer
        
        # Test Llama loading
        print("  Testing Llama loading...")
        model, tokenizer = load_model_and_tokenizer("llama")
        print(f"    ✓ Llama loaded: {type(model).__name__}")
        del model, tokenizer
        
        # Test DiffLlama loading  
        print("  Testing DiffLlama loading...")
        model, tokenizer = load_model_and_tokenizer("diffllama")
        print(f"    ✓ DiffLlama loaded: {type(model).__name__}")
        del model, tokenizer
        
        # Clear CUDA cache if available
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("✅ Model loading successful")
        return True
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_data_preparation():
    """Test data preparation functions."""
    print("\n🔍 Testing data preparation...")
    
    try:
        # Test GSM8K download function
        from src.utils import download_gsm8k
        print("  ✓ GSM8K download function imported")
        
        # Test noise injection functions
        from src.noise_injection import inject_inf_noise, inject_rcs_noise, inject_sd_noise
        
        test_question = "A store has 10 apples. How many apples are there?"
        
        # Test each noise type
        noisy_inf = inject_inf_noise(test_question)
        noisy_rcs = inject_rcs_noise(test_question)
        noisy_sd = inject_sd_noise(test_question)
        
        print(f"  ✓ INF noise: {len(noisy_inf)} chars")
        print(f"  ✓ RCS noise: {len(noisy_rcs)} chars") 
        print(f"  ✓ SD noise: {len(noisy_sd)} chars")
        
        print("✅ Data preparation functions working")
        return True
        
    except Exception as e:
        print(f"  ❌ Data preparation failed: {e}")
        traceback.print_exc()
        return False

def test_evaluation_functions():
    """Test evaluation functions."""
    print("\n🔍 Testing evaluation functions...")
    
    try:
        from src.evaluation import extract_answer_from_generation
        from src.utils import extract_answer_from_solution
        
        # Test answer extraction
        test_generated = "The calculation is 5 + 3 = 8. The answer is 8."
        test_solution = "Step 1: Add 5 + 3\nStep 2: 5 + 3 = 8\n#### 8"
        
        answer1 = extract_answer_from_generation(test_generated)
        answer2 = extract_answer_from_solution(test_solution)
        
        print(f"  ✓ Generated answer extraction: {answer1}")
        print(f"  ✓ Solution answer extraction: {answer2}")
        
        print("✅ Evaluation functions working")
        return True
        
    except Exception as e:
        print(f"  ❌ Evaluation functions failed: {e}")
        traceback.print_exc()
        return False

def test_attention_functions():
    """Test attention visualization functions."""
    print("\n🔍 Testing attention functions...")
    
    try:
        from src.attention_visualizer import classify_tokens
        import matplotlib.pyplot as plt
        
        # Test token classification
        test_tokens = ["Question", ":", "How", "many", "5", "apples", "?"]
        classifications = classify_tokens(test_tokens, "How many apples?")
        
        print(f"  ✓ Token classification: {len(classifications)} tokens classified")
        
        # Test matplotlib backend
        plt.ioff()  # Turn off interactive mode
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.plot([1, 2, 3], [1, 2, 3])
        plt.close(fig)
        print("  ✓ Matplotlib working")
        
        print("✅ Attention functions working")
        return True
        
    except Exception as e:
        print(f"  ❌ Attention functions failed: {e}")
        traceback.print_exc()
        return False

def test_directories():
    """Test if required directories can be created."""
    print("\n🔍 Testing directory creation...")
    
    test_dirs = ["data", "results", "results/attention_maps", "models_finetuned"]
    
    try:
        for directory in test_dirs:
            os.makedirs(directory, exist_ok=True)
            print(f"  ✓ {directory}")
        
        print("✅ Directory creation successful")
        return True
        
    except Exception as e:
        print(f"  ❌ Directory creation failed: {e}")
        return False

def run_quick_inference_test():
    """Run a quick inference test with both models."""
    print("\n🔍 Running quick inference test...")
    
    try:
        from src.model_loader import load_model_and_tokenizer
        import torch
        
        test_prompt = "Question: What is 2 + 2?\nAnswer:"
        
        for model_type in ["llama", "diffllama"]:
            print(f"  Testing {model_type} inference...")
            
            model, tokenizer = load_model_and_tokenizer(model_type)
            
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            print(f"    ✓ {model_type} generated: {generated_text[:50]}...")
            
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("✅ Quick inference test successful")
        return True
        
    except Exception as e:
        print(f"  ❌ Quick inference test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*80)
    print("🧪 EXPERIMENTAL SETUP TEST")
    print("="*80)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Model Paths", test_model_paths),
        ("Directories", test_directories),
        ("Data Preparation", test_data_preparation),
        ("Evaluation Functions", test_evaluation_functions),
        ("Attention Functions", test_attention_functions),
        ("Model Loading", test_model_loading),
        ("Quick Inference", run_quick_inference_test),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("🏁 TEST SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if passed_test:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to run the full experiment.")
        print("Run: python main.py --quick-test")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before running experiment.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
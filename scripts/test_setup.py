#!/usr/bin/env python3
"""
Setup Test Script - Test Experiment Environment Configuration

This script verifies that all required dependencies, models, and data files are correctly configured.
It is recommended to run this script for verification before running the main experiment.
"""

import os
import sys
import json
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_python_environment():
    """Test Python environment and basic dependencies"""
    print("🔍 Testing Python environment...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("⚠️  CUDA not available, will use CPU (slower)")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets {datasets.__version__}")
    except ImportError:
        print("❌ Datasets not installed")
        return False
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ Data analysis libraries (numpy, pandas, matplotlib, seaborn)")
    except ImportError as e:
        print(f"❌ Missing data analysis library: {e}")
        return False
    
    return True

def test_source_files():
    """Test if source code files exist"""
    print("\n📁 Testing source code files...")
    
    required_files = [
        "src/utils.py",
        "src/model_loader.py",
        "src/noise_injection.py", 
        "src/evaluation.py",
        "src/fine_tuning.py",
        "src/attention_visualizer.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} source files")
        return False
    
    return True

def test_imports():
    """Test core module imports"""
    print("\n🔧 Testing module imports...")
    
    test_modules = [
        ("src.utils", "Download and Processing Tools"),
        ("src.model_loader", "Model Loader"),
        ("src.noise_injection", "Noise Injection"),
        ("src.evaluation", "Evaluation Module"),
        ("src.fine_tuning", "Fine-tuning Module"),
        ("src.attention_visualizer", "Attention Visualization")
    ]
    
    failed_imports = []
    for module_name, description in test_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name} ({description})")
        except ImportError as e:
            print(f"❌ {module_name} ({description}): {e}")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"⚠️  {module_name} ({description}): {e}")
    
    if failed_imports:
        print(f"\n⚠️  {len(failed_imports)} modules failed to import")
        return False
    
    return True

def test_model_availability():
    """Test model file availability"""
    print("\n🤖 Testing model availability...")
    
    model_paths = {
        "DiffLlama": "cache/models--reyllama--DiffLlama-375M",
        "Llama": "cache/models--reyllama--Llama_375M"
    }
    
    missing_models = []
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            print(f"✅ {model_name}: {model_path}")
            
            # Check model contents
            if os.path.isdir(model_path):
                contents = os.listdir(model_path)
                if contents:
                    print(f"   📁 Contains {len(contents)} files/directories")
                else:
                    print(f"   ⚠️  Directory is empty")
        else:
            print(f"❌ {model_name}: {model_path}")
            missing_models.append(model_name)
    
    if missing_models:
        print(f"\n⚠️  Missing {len(missing_models)} models")
        print("Run the following command to download models:")
        print("python scripts/download_models.py")
        return False
    
    return True

def test_model_loading():
    """Test model loading functionality"""
    print("\n🚀 Testing model loading...")
    
    try:
        from src.model_loader import load_model_and_tokenizer
        
        # Test loading DiffLlama (using smaller settings to save memory)
        print("  Loading DiffLlama...")
        model, tokenizer = load_model_and_tokenizer("diffllama")
        print("  ✅ DiffLlama loaded successfully")
        
        # Clean up memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Test loading Llama
        print("  Loading Llama...")
        model, tokenizer = load_model_and_tokenizer("llama")
        print("  ✅ Llama loaded successfully")
        
        # Clean up memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_data_generation():
    """Test data generation functionality"""
    print("\n📊 Testing data generation...")
    
    try:
        from src.utils import download_gsm8k
        from src.noise_injection import inject_inf_noise
        
        # Test GSM8K download (if not exists)
        if not os.path.exists("data/gsm8k_test.jsonl"):
            print("  Downloading GSM8K dataset...")
            download_gsm8k()
        
        if os.path.exists("data/gsm8k_test.jsonl"):
            print("  ✅ GSM8K dataset available")
        else:
            print("  ❌ GSM8K dataset download failed")
            return False
        
        # Test noise injection (small sample)
        print("  Testing noise injection...")
        test_question = "Lisa has 10 apples. She gives 3 to her friend. How many apples does she have left?"
        noisy_question = inject_inf_noise(test_question)
        
        if len(noisy_question) > len(test_question):
            print("  ✅ Noise injection working normally")
        else:
            print("  ⚠️  Noise injection may not be effective")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        traceback.print_exc()
        return False

def test_evaluation_pipeline():
    """Test evaluation pipeline"""
    print("\n🔍 Testing evaluation pipeline...")
    
    try:
        from src.evaluation import extract_answer, evaluate_answer
        
        # Test answer extraction
        test_response = "Let me think step by step. Lisa has 10 apples and gives away 3, so 10 - 3 = 7. The answer is 7."
        extracted = extract_answer(test_response)
        print(f"  Answer extraction test: '{extracted}'")
        
        # Test answer evaluation
        correct = evaluate_answer(extracted, "7")
        print(f"  Answer evaluation test: {correct}")
        
        if correct:
            print("  ✅ Evaluation pipeline working normally")
        else:
            print("  ⚠️  Evaluation pipeline may have issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_directory_structure():
    """Test directory structure"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = ["src", "data", "results", "cache"]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/")
        else:
            print(f"⚠️  {dir_name}/ (will be created automatically)")
            os.makedirs(dir_name, exist_ok=True)
    
    return True

def run_comprehensive_test():
    """Run comprehensive environment test"""
    print("="*80)
    print("🧪 DiffLlama Experiment Environment Test")
    print("="*80)
    
    tests = [
        ("Python Environment", test_python_environment),
        ("Directory Structure", test_directory_structure),
        ("Source Code Files", test_source_files),
        ("Module Imports", test_imports),
        ("Model Availability", test_model_availability),
        ("Data Generation", test_data_generation),
        ("Evaluation Pipeline", test_evaluation_pipeline)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} Test encountered exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("📋 Test Result Summary")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ Passed" if passed_test else "❌ Failed"
        print(f"{test_name:20} {status}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} Tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Environment configuration correct, can run experiment.")
        return True
    else:
        print("⚠️  Some tests failed, please check the error messages above and resolve the issues.")
        return False

def quick_test():
    """Quick test core functionality"""
    print("🚀 Quick Test Mode")
    print("-" * 40)
    
    # Only test the most critical functionality
    critical_tests = [
        test_python_environment,
        test_source_files,
        test_imports,
        test_directory_structure
    ]
    
    all_passed = True
    for test_func in critical_tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✅ Core functionality test passed")
    else:
        print("\n❌ Core functionality test failed")
    
    return all_passed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DiffLlama Experiment Environment Test")
    parser.add_argument("--quick", action="store_true", help="Quick Test Mode")
    parser.add_argument("--models", action="store_true", help="Only test models")
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_test()
    elif args.models:
        success = test_model_availability() and test_model_loading()
    else:
        success = run_comprehensive_test()
    
    sys.exit(0 if success else 1) 
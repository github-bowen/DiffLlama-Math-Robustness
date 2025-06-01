#!/usr/bin/env python3
"""
Setup Test Script - 测试实验环境配置

此脚本验证所有必需的依赖、模型和数据文件是否正确配置。
在运行主实验之前建议先运行此脚本进行验证。
"""

import os
import sys
import json
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_python_environment():
    """测试 Python 环境和基础依赖"""
    print("🔍 测试 Python 环境...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"   GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("⚠️  CUDA 不可用，将使用 CPU（速度较慢）")
    except ImportError:
        print("❌ PyTorch 未安装")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers 未安装")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets {datasets.__version__}")
    except ImportError:
        print("❌ Datasets 未安装")
        return False
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ 数据分析库 (numpy, pandas, matplotlib, seaborn)")
    except ImportError as e:
        print(f"❌ 数据分析库缺失: {e}")
        return False
    
    return True

def test_source_files():
    """测试源代码文件是否存在"""
    print("\n📁 测试源代码文件...")
    
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
        print(f"\n⚠️  缺少 {len(missing_files)} 个源文件")
        return False
    
    return True

def test_imports():
    """测试核心模块导入"""
    print("\n🔧 测试模块导入...")
    
    test_modules = [
        ("src.utils", "下载和处理工具"),
        ("src.model_loader", "模型加载器"),
        ("src.noise_injection", "噪声注入"),
        ("src.evaluation", "评估模块"),
        ("src.fine_tuning", "微调模块"),
        ("src.attention_visualizer", "注意力可视化")
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
        print(f"\n⚠️  {len(failed_imports)} 个模块导入失败")
        return False
    
    return True

def test_model_availability():
    """测试模型文件是否可用"""
    print("\n🤖 测试模型可用性...")
    
    model_paths = {
        "DiffLlama": "cache/models--reyllama--DiffLlama-375M",
        "Llama": "cache/models--reyllama--Llama_375M"
    }
    
    missing_models = []
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            print(f"✅ {model_name}: {model_path}")
            
            # 检查模型内容
            if os.path.isdir(model_path):
                contents = os.listdir(model_path)
                if contents:
                    print(f"   📁 包含 {len(contents)} 个文件/目录")
                else:
                    print(f"   ⚠️  目录为空")
        else:
            print(f"❌ {model_name}: {model_path}")
            missing_models.append(model_name)
    
    if missing_models:
        print(f"\n⚠️  缺少 {len(missing_models)} 个模型")
        print("运行以下命令下载模型:")
        print("python scripts/download_models.py")
        return False
    
    return True

def test_model_loading():
    """测试模型加载功能"""
    print("\n🚀 测试模型加载...")
    
    try:
        from src.model_loader import load_model_and_tokenizer
        
        # 测试加载 DiffLlama（使用较小的设置以节省内存）
        print("  加载 DiffLlama...")
        model, tokenizer = load_model_and_tokenizer("diffllama")
        print("  ✅ DiffLlama 加载成功")
        
        # 清理内存
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 测试加载 Llama
        print("  加载 Llama...")
        model, tokenizer = load_model_and_tokenizer("llama")
        print("  ✅ Llama 加载成功")
        
        # 清理内存
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        return False

def test_data_generation():
    """测试数据生成功能"""
    print("\n📊 测试数据生成...")
    
    try:
        from src.utils import download_gsm8k
        from src.noise_injection import inject_inf_noise
        
        # 测试 GSM8K 下载（如果不存在）
        if not os.path.exists("data/gsm8k_test.jsonl"):
            print("  下载 GSM8K 数据集...")
            download_gsm8k()
        
        if os.path.exists("data/gsm8k_test.jsonl"):
            print("  ✅ GSM8K 数据集可用")
        else:
            print("  ❌ GSM8K 数据集下载失败")
            return False
        
        # 测试噪声注入（小样本）
        print("  测试噪声注入...")
        test_question = "Lisa has 10 apples. She gives 3 to her friend. How many apples does she have left?"
        noisy_question = inject_inf_noise(test_question)
        
        if len(noisy_question) > len(test_question):
            print("  ✅ 噪声注入功能正常")
        else:
            print("  ⚠️  噪声注入可能未生效")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据生成测试失败: {e}")
        traceback.print_exc()
        return False

def test_evaluation_pipeline():
    """测试评估流程"""
    print("\n🔍 测试评估流程...")
    
    try:
        from src.evaluation import extract_answer, evaluate_answer
        
        # 测试答案提取
        test_response = "Let me think step by step. Lisa has 10 apples and gives away 3, so 10 - 3 = 7. The answer is 7."
        extracted = extract_answer(test_response)
        print(f"  答案提取测试: '{extracted}'")
        
        # 测试答案评估
        correct = evaluate_answer(extracted, "7")
        print(f"  答案评估测试: {correct}")
        
        if correct:
            print("  ✅ 评估流程功能正常")
        else:
            print("  ⚠️  评估流程可能有问题")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估流程测试失败: {e}")
        traceback.print_exc()
        return False

def test_directory_structure():
    """测试目录结构"""
    print("\n📁 测试目录结构...")
    
    required_dirs = ["src", "data", "results", "cache"]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/")
        else:
            print(f"⚠️  {dir_name}/ (将自动创建)")
            os.makedirs(dir_name, exist_ok=True)
    
    return True

def run_comprehensive_test():
    """运行全面的环境测试"""
    print("="*80)
    print("🧪 DiffLlama 实验环境测试")
    print("="*80)
    
    tests = [
        ("Python 环境", test_python_environment),
        ("目录结构", test_directory_structure),
        ("源代码文件", test_source_files),
        ("模块导入", test_imports),
        ("模型可用性", test_model_availability),
        ("数据生成", test_data_generation),
        ("评估流程", test_evaluation_pipeline)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} 测试出现异常: {e}")
            results[test_name] = False
    
    # 总结
    print("\n" + "="*80)
    print("📋 测试结果总结")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ 通过" if passed_test else "❌ 失败"
        print(f"{test_name:20} {status}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！环境配置正确，可以运行实验。")
        return True
    else:
        print("⚠️  部分测试失败，请检查上述错误信息并解决问题。")
        return False

def quick_test():
    """快速测试核心功能"""
    print("🚀 快速测试模式")
    print("-" * 40)
    
    # 只测试最关键的功能
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
            print(f"❌ 测试失败: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✅ 核心功能测试通过")
    else:
        print("\n❌ 核心功能测试失败")
    
    return all_passed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DiffLlama 实验环境测试")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--models", action="store_true", help="仅测试模型")
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_test()
    elif args.models:
        success = test_model_availability() and test_model_loading()
    else:
        success = run_comprehensive_test()
    
    sys.exit(0 if success else 1) 
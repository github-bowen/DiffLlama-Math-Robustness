#!/usr/bin/env python3
"""
Google Colab Script for DiffLlama vs Llama Noise Robustness Experiment

This script is specifically designed for Google Colab environment with:
- Google Drive mounting for model storage
- Memory optimization for Colab's GPU limitations
- Progress tracking and checkpointing
- Simplified interface for notebook execution

Usage in Colab:
    !python colab/experiment.py --mode quick
    !python colab/experiment.py --mode full --use-drive
"""

import os
import sys
import time
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def mount_google_drive():
    """Mount Google Drive in Colab environment."""
    try:
        from google.colab import drive
        print("🔗 Mounting Google Drive...")
        drive.mount('/content/drive')
        
        # Check if mount successful
        if os.path.exists('/content/drive/MyDrive'):
            print("✅ Google Drive mounted successfully")
            return True
        else:
            print("❌ Google Drive mount failed")
            return False
    except ImportError:
        print("⚠️  Not running in Google Colab environment")
        return False
    except Exception as e:
        print(f"❌ Error mounting Google Drive: {e}")
        return False

def setup_colab_environment(use_drive=True):
    # Setup directories
    # Use Google Drive for persistent storage
    base_dir = "/content/drive/MyDrive/DiffLlama_Experiment"
    models_dir = f"{base_dir}/models"
    data_dir = f"{base_dir}/data"
    results_dir = f"{base_dir}/results"
    
    # Create symlinks to local workspace
    local_dirs = {
        "cache": models_dir,
        "data": data_dir, 
        "results": results_dir
    }
    
    print(f"📁 Using Google Drive storage: {base_dir}")
        
    # Create directories and symlinks
    for local_name, target_dir in local_dirs.items():
        os.makedirs(target_dir, exist_ok=True)
        
        # Remove existing symlink/directory if it exists
        if os.path.exists(local_name):
            if os.path.islink(local_name):
                os.unlink(local_name)
            elif os.path.isdir(local_name):
                shutil.rmtree(local_name)
        
        # Create symlink
        os.symlink(target_dir, local_name)
        print(f"  ✓ {local_name} -> {target_dir}")
    
    # Additional Colab setup directories
    os.makedirs("src", exist_ok=True)
    os.makedirs("models_finetuned", exist_ok=True)
    
    return True

def install_dependencies():
    """Install required dependencies in Colab."""
    print("📦 Installing dependencies...")
    
    os.system("pip install -r requirements.txt")
    
    print("✅ Dependencies installed")

def check_gpu_memory():
    """Check and display GPU memory status."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_free = memory_total - memory_allocated
            
            print(f"🖥️  GPU: {gpu_name}")
            print(f"💾 Memory: {memory_free:.1f}GB free / {memory_total:.1f}GB total")
            
            if memory_total < 12:  # Less than 12GB
                print("⚠️  Limited GPU memory detected. Using optimized settings.")
                return "limited"
            else:
                return "sufficient"
        else:
            print("❌ No GPU detected")
            return "none"
    except:
        return "unknown"

def copy_source_files():
    """Copy or verify source files are available."""
    print("📋 Setting up source files...")
    
    source_files = [
        "src/utils.py",
        "src/model_loader.py", 
        "src/noise_injection.py",
        "src/evaluation.py",
        "src/fine_tuning.py",
        "src/attention_visualizer.py"
    ]
    
    missing_files = []
    for file_path in source_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing source files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease ensure all source files are uploaded to Colab or run:")
        print("!git clone <your-repo-url> && cd <repo-name>")
        return False
    
    print("✅ All source files found")
    return True

def download_models_colab():
    """Download models with Colab optimizations."""
    print("⬇️ Downloading models...")
    
    # Check if models already exist
    model_paths = {
        "DiffLlama": "./cache/models--reyllama--DiffLlama-375M",
        "Llama": "./cache/models--reyllama--Llama_375M"
    }
    
    models_exist = all(os.path.exists(path) for path in model_paths.values())
    
    if models_exist:
        print("✅ Models already downloaded")
        return True
    
    try:
        # Import and run download script from scripts directory
        sys.path.append('scripts')
        from download_models import download_models
        download_models()
        print("✅ Models downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        print("Please manually download models or check network connection")
        return False

def run_colab_experiment(mode="quick", max_samples=None, skip_sft=True, skip_attention=False):
    """Run the experiment with Colab-specific settings."""
    print(f"\n🚀 Starting {mode} experiment...")
    
    # Set parameters based on mode and GPU memory
    gpu_status = check_gpu_memory()
    
    if mode == "quick" or gpu_status == "limited":
        eval_samples = 20 if max_samples is None else min(max_samples, 50)
        sft_samples = 50
        sft_epochs = 1
        skip_sft = True  # Skip SFT in quick mode or limited memory
        attention_samples = 5
    elif mode == "medium":
        eval_samples = 100 if max_samples is None else min(max_samples, 200)
        sft_samples = 200
        sft_epochs = 2
        attention_samples = 10
    else:  # full mode
        eval_samples = max_samples
        sft_samples = 500
        sft_epochs = 2
        attention_samples = 20
    
    print(f"📊 Experiment settings:")
    print(f"  Mode: {mode}")
    print(f"  Evaluation samples: {eval_samples}")
    print(f"  Skip SFT: {skip_sft}")
    print(f"  Skip attention: {skip_attention}")
    
    # Import experiment modules
    try:
        from src.utils import download_gsm8k
        from src.noise_injection import generate_noisy_datasets
        from src.evaluation import run_comprehensive_evaluation, compare_model_performance
        
        if not skip_attention:
            from src.attention_visualizer import compare_attention_patterns
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    start_time = time.time()
    
    try:
        # Step 1: Data Preparation
        print("\n" + "="*60)
        print("📊 STEP 1: DATA PREPARATION")
        print("="*60)
        
        if not os.path.exists("data/gsm8k_test.jsonl"):
            print("Downloading GSM8K dataset...")
            download_gsm8k()
        
        noisy_files = [
            "data/gsm8k_inf_test.jsonl",
            "data/gsm8k_rcs_test.jsonl", 
            "data/gsm8k_sd_test.jsonl"
        ]
        
        if not all(os.path.exists(f) for f in noisy_files):
            print("Generating noisy datasets...")
            generate_noisy_datasets()
        
        print("✅ Data preparation completed")
        
        # Step 2: Zero-shot Evaluation
        print("\n" + "="*60)
        print("🔍 STEP 2: ZERO-SHOT EVALUATION")
        print("="*60)
        
        results_df, detailed_results = run_comprehensive_evaluation(
            max_samples_per_dataset=eval_samples
        )
        
        compare_model_performance(results_df)
        
        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/colab_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"💾 Results saved to {results_file}")
        
        # Step 3: Attention Analysis (if not skipped)
        if not skip_attention:
            print("\n" + "="*60)
            print("🧠 STEP 3: ATTENTION ANALYSIS")
            print("="*60)
            
            try:
                attention_results = compare_attention_patterns(
                    clean_dataset="data/gsm8k_test.jsonl",
                    noisy_dataset="data/gsm8k_inf_test.jsonl",
                    num_samples=attention_samples
                )
                
                attention_file = f"results/colab_attention_{timestamp}.json"
                with open(attention_file, "w") as f:
                    json.dump(attention_results, f, indent=2)
                print(f"💾 Attention results saved to {attention_file}")
                
            except Exception as e:
                print(f"⚠️  Attention analysis failed: {e}")
        
        # Generate summary report
        end_time = time.time()
        duration = end_time - start_time
        
        summary = {
            "experiment_mode": mode,
            "duration_minutes": duration / 60,
            "samples_evaluated": eval_samples,
            "gpu_status": gpu_status,
            "timestamp": timestamp,
            "results_file": results_file
        }
        
        summary_file = f"results/colab_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("🎉 EXPERIMENT COMPLETED!")
        print("="*60)
        print(f"⏱️  Duration: {duration/60:.1f} minutes")
        print(f"📊 Results: {results_file}")
        print(f"📋 Summary: {summary_file}")
        
        if os.path.exists("/content/drive/MyDrive"):
            print("💾 Results saved to Google Drive")
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def display_colab_instructions():
    """Display instructions for Colab usage."""
    instructions = """
🎯 GOOGLE COLAB USAGE INSTRUCTIONS

1. 📱 Basic Setup (Run once):
   !python colab/experiment.py --setup

2. 🚀 Quick Test (Recommended first run):
   !python colab/experiment.py --mode quick

3. 📊 Medium Experiment:
   !python colab/experiment.py --mode medium --use-drive

4. 🔬 Full Experiment:
   !python colab/experiment.py --mode full --use-drive --max-samples 500

🔧 Options:
   --mode: quick/medium/full
   --use-drive: Save models and results to Google Drive
   --max-samples: Limit number of evaluation samples
   --skip-attention: Skip attention analysis to save time
   --help: Show all options

💡 Tips:
   - Use --use-drive to persist models across sessions
   - Start with quick mode to verify everything works
   - Monitor GPU memory usage in Colab

📁 Results will be saved to:
   - Local: /content/results/
   - Drive: /content/drive/MyDrive/DiffLlama_Experiment/results/
"""
    print(instructions)

def main():
    parser = argparse.ArgumentParser(
        description="DiffLlama vs Llama experiment for Google Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--mode", choices=["quick", "medium", "full"], 
                       default="quick", help="Experiment mode")
    parser.add_argument("--use-drive", action="store_true",
                       help="Use Google Drive for persistent storage")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples for evaluation")
    parser.add_argument("--skip-sft", action="store_true", default=True,
                       help="Skip supervised fine-tuning (default: True)")
    parser.add_argument("--skip-attention", action="store_true",
                       help="Skip attention analysis")
    parser.add_argument("--setup", action="store_true",
                       help="Only run setup (dependencies and environment)")
    parser.add_argument("--instructions", action="store_true",
                       help="Display usage instructions")
    
    args = parser.parse_args()
    
    if args.instructions:
        display_colab_instructions()
        return 0
    
    if args.setup:
        # Environment setup
        install_dependencies()
        drive_mounted = setup_colab_environment(args.use_drive)
        print("✅ Setup completed")
        return 0
    
    print("="*80)
    print("🔬 DIFFLAMA VS LLAMA - GOOGLE COLAB EXPERIMENT")
    print("="*80)
    print(f"🕐 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check source files
    if not copy_source_files():
        print("❌ Please upload all source files first")
        return 1
    
    # Download models
    if not download_models_colab():
        print("❌ Model download failed")
        return 1
    
    # Run experiment
    success = run_colab_experiment(
        mode=args.mode,
        max_samples=args.max_samples,
        skip_sft=args.skip_sft,
        skip_attention=args.skip_attention
    )
    
    if success:
        print("\n🎊 Experiment completed successfully!")
        if drive_mounted:
            print("💾 Results saved to Google Drive for persistence")
        return 0
    else:
        print("\n❌ Experiment failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
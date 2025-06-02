#!/usr/bin/env python3
"""
Google Colab Script for DiffLlama vs Llama Noise Robustness Experiment

This script is specifically designed for Google Colab environment with:
- Google Drive mounting for model storage
- Memory optimization for Colab's GPU limitations
- Progress tracking and checkpointing
- Simplified interface for notebook execution

Usage in Colab:
    !python -m colab.experiment --mode quick
    !python -m colab.experiment --mode full
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

def setup_colab_environment():
    # Setup directories
    os.makedirs("models_finetuned", exist_ok=True)
    
    # Use Google Drive for persistent storage
    base_dir = "/content/drive/MyDrive/DiffLlama_Experiment"
    models_dir = f"{base_dir}/models"
    data_dir = f"{base_dir}/data"
    results_dir = f"{base_dir}/results"
    models_finetuned_dir = f"{base_dir}/models_finetuned"
    
    # Create symlinks to local workspace
    local_dirs = {
        "cache": models_dir,
        "data": data_dir, 
        "results": results_dir,
        "models_finetuned": models_finetuned_dir
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

def run_colab_experiment(mode="quick", max_samples=None, skip_sft=True, skip_attention=False, 
                         skip_zero_shot=False, sft_samples_override=None, sft_epochs_override=None):
    """Run the experiment with Colab-specific settings."""
    print(f"\n🚀 Starting {mode} experiment...")
    
    # Set parameters based on mode and GPU memory
    gpu_status = check_gpu_memory()
    
    if mode == "quick" or gpu_status == "limited":
        eval_samples = 20 if max_samples is None else min(max_samples, 50)
        sft_samples = sft_samples_override or 50
        sft_epochs = sft_epochs_override or 1
        skip_sft = True  # Force skip SFT in quick mode or limited memory
        attention_samples = 5
    elif mode == "medium":
        eval_samples = 100 if max_samples is None else min(max_samples, 200)
        sft_samples = sft_samples_override or 200
        sft_epochs = sft_epochs_override or 2
        attention_samples = 10
    else:  # full mode
        eval_samples = max_samples
        sft_samples = sft_samples_override or 500
        sft_epochs = sft_epochs_override or 2
        attention_samples = 20
    
    # Override skip_sft if we're in quick mode or have limited memory
    if mode == "quick" or gpu_status == "limited":
        if not skip_sft:
            print("⚠️  Forcing SFT skip due to quick mode or limited GPU memory")
            skip_sft = True
    
    print(f"📊 Experiment settings:")
    print(f"  Mode: {mode}")
    print(f"  Evaluation samples: {eval_samples}")
    print(f"  Skip zero-shot: {skip_zero_shot}")
    print(f"  Skip SFT: {skip_sft}")
    if not skip_sft:
        print(f"  SFT samples: {sft_samples}")
        print(f"  SFT epochs: {sft_epochs}")
    print(f"  Skip attention: {skip_attention}")
    print(f"  GPU status: {gpu_status}")
    
    # Import experiment modules
    try:
        from src.utils import download_gsm8k
        from src.noise_injection import generate_noisy_datasets
        
        if not skip_zero_shot:
            from src.evaluation import run_comprehensive_evaluation, compare_model_performance, run_evaluation
        if not skip_sft:
            from src.evaluation import run_evaluation
        
        if not skip_sft:
            from src.fine_tuning import run_full_sft_pipeline
        
        if not skip_attention:
            from src.attention_visualizer import compare_attention_patterns, visualize_sample_attention
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    start_time = time.time()
    
    try:
        # Step 1: Data Preparation
        print("\n" + "="*60)
        print("📊 STEP 1: DATA PREPARATION")
        print("="*60)
        
        # Check if GSM8K files already exist
        gsm8k_files = ["data/gsm8k_test.jsonl", "data/gsm8k_train.jsonl"]
        gsm8k_exists = all(os.path.exists(f) and os.path.getsize(f) > 0 for f in gsm8k_files)
        
        if not gsm8k_exists:
            print("Downloading GSM8K dataset...")
            try:
                download_gsm8k()
            except Exception as e:
                print(f"⚠️  GSM8K download failed: {e}")
                # print("🔧 Trying to create sample data for testing...")
                
                # # Create minimal sample data for testing
                # sample_data = [
                #     {
                #         "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much does she make every day?",
                #         "answer": "The answer is 18"
                #     }
                # ] * 10  # Repeat to have some data
                
                # os.makedirs("data", exist_ok=True)
                # with open("data/gsm8k_test.jsonl", "w") as f:
                #     for item in sample_data:
                #         f.write(json.dumps(item) + "\n")
                # with open("data/gsm8k_train.jsonl", "w") as f:
                #     for item in sample_data:
                #         f.write(json.dumps(item) + "\n")
                
                # print("✅ Created sample data for testing")
                raise Exception("GSM8K download failed")
        else:
            print("✅ GSM8K dataset already exists")
        
        noisy_files = [
            "data/gsm8k_inf_test.jsonl",
            "data/gsm8k_rcs_test.jsonl", 
            "data/gsm8k_sd_test.jsonl"
        ]
        
        if not all(os.path.exists(f) for f in noisy_files):
            print("Generating noisy datasets...")
            try:
                generate_noisy_datasets()
            except Exception as e:
                print(f"⚠️  Noisy dataset generation failed: {e}")
                print("Continuing with clean dataset only...")
        
        print("✅ Data preparation completed")
        
        # Initialize variables for results
        results_df = None
        detailed_results = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 2: Zero-shot Evaluation (optional)
        if not skip_zero_shot:
            print("\n" + "="*60)
            print("🔍 STEP 2: ZERO-SHOT EVALUATION")
            print("="*60)
            
            results_df, detailed_results = run_comprehensive_evaluation(
                max_samples_per_dataset=eval_samples
            )
            
            compare_model_performance(results_df)
            
            # Save results with timestamp
            results_file = f"results/colab_results_{timestamp}.csv"
            results_df.to_csv(results_file, index=False)
            print(f"💾 Results saved to {results_file}")
            
            # Save detailed results
            detailed_file = f"results/colab_detailed_{timestamp}.json"
            with open(detailed_file, "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {}
                for model, model_results in detailed_results.items():
                    serializable_results[model] = {}
                    for dataset, dataset_results in model_results.items():
                        serializable_results[model][dataset] = {
                            'accuracy': dataset_results['accuracy'],
                            'correct': dataset_results['correct'],
                            'total': dataset_results['total'],
                            'elapsed_time': dataset_results['elapsed_time']
                        }
                json.dump(serializable_results, f, indent=2)
            print(f"💾 Detailed results saved to {detailed_file}")
        else:
            print("\n⏭️  Skipping zero-shot evaluation")
            # Create placeholder files for consistency
            results_file = f"results/colab_results_{timestamp}.csv"
            detailed_file = f"results/colab_detailed_{timestamp}.json"
        
        # Step 3: Supervised Fine-tuning (optional)
        sft_model_paths = {}
        if not skip_sft:
            print("\n" + "="*60)
            print("🎯 STEP 3: SUPERVISED FINE-TUNING")
            print("="*60)
            
            # Check if training data exists
            if not os.path.exists("data/gsm8k_train.jsonl"):
                print("⚠️  Training data not found.")
                raise Exception("Training data not found")
            else:
                try:
                    print(f"Running SFT with {sft_samples} samples, {sft_epochs} epochs...")
                    sft_model_paths = run_full_sft_pipeline(
                        max_train_samples=sft_samples,
                        num_epochs=sft_epochs
                    )
                    
                    if sft_model_paths:
                        print("✅ Supervised fine-tuning completed")
                        print(f"📁 Fine-tuned models: {list(sft_model_paths.keys())}")
                    else:
                        print("⚠️  Supervised fine-tuning failed or returned no models")
                        
                except Exception as e:
                    print(f"❌ SFT failed: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("\n⏭️  Skipping supervised fine-tuning")
        
        # Step 4: Post-SFT Evaluation (if models were fine-tuned)
        sft_results = []
        if sft_model_paths:
            print("\n" + "="*60)
            print("📈 STEP 4: POST-SFT EVALUATION")
            print("="*60)
            
            datasets_to_eval = {
                "Clean": "data/gsm8k_test.jsonl",
                "INF": "data/gsm8k_inf_test.jsonl", 
                "RCS": "data/gsm8k_rcs_test.jsonl",
                "SD": "data/gsm8k_sd_test.jsonl"
            }
            
            for model_type, model_path in sft_model_paths.items():
                print(f"\nEvaluating fine-tuned {model_type}...")
                
                for noise_type, dataset_file in datasets_to_eval.items():
                    if not os.path.exists(dataset_file):
                        print(f"  ⚠️  Dataset {noise_type} not found, skipping...")
                        continue
                        
                    try:
                        print(f"  - {noise_type} dataset...")
                        result = run_evaluation(
                            model_type=model_type,
                            dataset_file=dataset_file,
                            max_samples=eval_samples,
                            model_path=model_path
                        )
                        
                        sft_results.append({
                            "model": f"{model_type}_sft",
                            "dataset": noise_type,
                            "accuracy": result['accuracy'],
                            "correct": result['correct'],
                            "total": result['total'],
                            "eval_type": "post_sft"
                        })
                        
                        print(f"    ✅ {noise_type}: {result['accuracy']:.3f} ({result['correct']}/{result['total']})")
                        
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
            
            # Save SFT results
            if sft_results:
                import pandas as pd
                df_sft = pd.DataFrame(sft_results)
                sft_file = f"results/colab_sft_{timestamp}.csv"
                df_sft.to_csv(sft_file, index=False)
                
                print(f"\n💾 SFT results saved to {sft_file}")
                print("\nPost-SFT Results Summary:")
                pivot_sft = df_sft.pivot(index='model', columns='dataset', values='accuracy')
                print(pivot_sft)
                
                print("✅ Post-SFT evaluation completed")
        else:
            print("\n⏭️  Skipping post-SFT evaluation (no fine-tuned models)")
        
        # Step 5: Attention Analysis (if not skipped)
        if not skip_attention:
            print("\n" + "="*60)
            print("🧠 STEP 5: ATTENTION ANALYSIS")
            print("="*60)
            
            # Create output directory
            os.makedirs("results/attention_maps", exist_ok=True)
            
            try:
                # Sample attention visualization
                print("Creating sample attention visualizations...")
                
                sample_questions = [
                    "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. How many eggs does she sell at the farmers' market every day?",
                    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"
                ]
                
                for i, question in enumerate(sample_questions[:2]):  # Limit to 2 for Colab
                    print(f"  Visualizing question {i+1}...")
                    
                    try:
                        # Clean question visualizations
                        visualize_sample_attention(
                            "llama", question, 
                            save_dir=f"results/attention_maps/clean_q{i+1}"
                        )
                        visualize_sample_attention(
                            "diffllama", question,
                            save_dir=f"results/attention_maps/clean_q{i+1}"
                        )
                        
                        # Noisy question visualizations
                        from src.noise_injection import inject_inf_noise
                        noisy_question = inject_inf_noise(question)
                        
                        visualize_sample_attention(
                            "llama", noisy_question,
                            save_dir=f"results/attention_maps/noisy_q{i+1}"
                        )
                        visualize_sample_attention(
                            "diffllama", noisy_question,
                            save_dir=f"results/attention_maps/noisy_q{i+1}"
                        )
                        
                        print(f"    ✅ Question {i+1} completed")
                        
                    except Exception as e:
                        print(f"    ⚠️  Question {i+1} failed: {e}")
                
                # Quantitative attention analysis
                if os.path.exists("data/gsm8k_inf_test.jsonl"):
                    print("Running quantitative attention analysis...")
                    attention_results = compare_attention_patterns(
                        clean_dataset="data/gsm8k_test.jsonl",
                        noisy_dataset="data/gsm8k_inf_test.jsonl",
                        num_samples=attention_samples
                    )
                    
                    attention_file = f"results/colab_attention_{timestamp}.json"
                    with open(attention_file, "w") as f:
                        json.dump(attention_results, f, indent=2)
                    print(f"💾 Attention results saved to {attention_file}")
                    
                    print("✅ Attention analysis completed")
                else:
                    print("⚠️  No noisy dataset found for attention analysis")
                
            except Exception as e:
                print(f"❌ Attention analysis failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n⏭️  Skipping attention analysis")
        
        # Generate comprehensive summary report
        end_time = time.time()
        duration = end_time - start_time
        
        summary = {
            "experiment_info": {
                "mode": mode,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat(),
                "duration_minutes": duration / 60,
                "duration_formatted": f"{duration//3600:.0f}h {(duration%3600)//60:.0f}m {duration%60:.0f}s"
            },
            "settings": {
                "eval_samples": eval_samples,
                "sft_samples": sft_samples,
                "sft_epochs": sft_epochs,
                "skip_zero_shot": skip_zero_shot,
                "skip_sft": skip_sft,
                "skip_attention": skip_attention,
                "gpu_status": gpu_status
            },
            "results": {
                "zero_shot_completed": not skip_zero_shot,
                "sft_completed": bool(sft_model_paths),
                "attention_completed": not skip_attention
            },
            "files_generated": []
        }
        
        # Add zero-shot results to summary if completed
        if not skip_zero_shot and results_df is not None:
            summary["results"]["zero_shot_file"] = results_file
            summary["results"]["detailed_file"] = detailed_file
            summary["files_generated"].extend([results_file, detailed_file])
        
        # Add SFT results to summary if available
        if sft_results:
            sft_file = f"results/colab_sft_{timestamp}.csv"
            summary["results"]["sft_file"] = sft_file
            summary["files_generated"].append(sft_file)
        
        # Add attention results to summary if available
        if not skip_attention and os.path.exists(f"results/colab_attention_{timestamp}.json"):
            attention_file = f"results/colab_attention_{timestamp}.json"
            summary["results"]["attention_file"] = attention_file
            summary["files_generated"].append(attention_file)
        
        summary_file = f"results/colab_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        summary["files_generated"].append(summary_file)
        
        print("\n" + "="*60)
        print("🎉 EXPERIMENT COMPLETED!")
        print("="*60)
        print(f"⏱️  Duration: {summary['experiment_info']['duration_formatted']}")
        
        # Display results based on what was executed
        if not skip_zero_shot and results_df is not None:
            print(f"📊 Zero-shot results: {results_file}")
        elif skip_zero_shot:
            print("⏭️  Zero-shot evaluation was skipped")
        
        if sft_model_paths:
            print(f"🎯 SFT results: {sft_file}")
        elif not skip_sft:
            print("⚠️  SFT was attempted but no models were fine-tuned")
        
        if not skip_attention:
            print(f"🧠 Attention analysis: results/attention_maps/")
        
        print(f"📋 Summary: {summary_file}")
        
        if os.path.exists("/content/drive/MyDrive"):
            print("💾 Results saved to Google Drive")
        
        # Only show files that actually exist
        if summary["files_generated"]:
            print("\nGenerated files:")
            for file_path in summary["files_generated"]:
                if os.path.exists(file_path):
                    print(f"  ✅ {file_path}")
                else:
                    print(f"  ⚠️  {file_path} (not found)")
        else:
            print("\n⚠️  No result files generated (all steps were skipped)")
        
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
   !python -m colab.experiment --setup

2. 🚀 Quick Test (Recommended first run):
   !python -m colab.experiment --mode quick

3. 📊 Medium Experiment:
   !python -m colab.experiment --mode medium

4. 🔬 Full Experiment:
   !python -m colab.experiment --mode full --max-samples 500

5. 🎯 Experiment with Fine-tuning:
   !python -m colab.experiment --mode medium --enable-sft --sft-samples 200

6. 🔄 Skip Zero-shot (only SFT and attention):
   !python -m colab.experiment --mode medium --skip-zero-shot --enable-sft

7. 📈 Only Fine-tuning workflow:
   !python -m colab.experiment --mode medium --skip-zero-shot --enable-sft --skip-attention

🔧 Options:
   --mode: quick/medium/full (experiment scope)
   --max-samples: Limit number of evaluation samples
   --enable-sft: Enable supervised fine-tuning (disabled by default)
   --sft-samples: Number of samples for fine-tuning (default: varies by mode)
   --sft-epochs: Number of epochs for fine-tuning (default: varies by mode)
   --skip-zero-shot: Skip zero-shot evaluation to save time
   --skip-attention: Skip attention analysis to save time
   --help: Show all options

💡 Tips:
   - Use to persist models across sessions
   - Start with quick mode to verify everything works
   - Fine-tuning requires significant GPU memory and time
   - Monitor GPU memory usage in Colab
   - Use medium mode with --enable-sft for balanced experiments
   - Skip zero-shot if you only need SFT results

📁 Results will be saved to:
   - Local: /content/results/
   - Drive: /content/drive/MyDrive/DiffLlama_Experiment/results/

⚠️  Resource Usage:
   - Quick mode: ~15-30 minutes, minimal GPU memory
   - Medium mode: ~1-2 hours, moderate GPU memory
   - Full mode: ~3-6 hours, high GPU memory
   - SFT adds: +30-60 minutes, requires >12GB GPU memory
   - Skipping zero-shot saves: ~30-50% of total time

🎯 Common Workflows:
   - Data validation: --mode quick
   - Zero-shot comparison: --mode medium
   - SFT-only experiment: --mode medium --skip-zero-shot --enable-sft
   - Complete research: --mode full --enable-sft
"""
    print(instructions)

def main():
    parser = argparse.ArgumentParser(
        description="DiffLlama vs Llama experiment for Google Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m colab.experiment --mode quick
    python -m colab.experiment --mode medium --enable-sft
    python -m colab.experiment --mode full --max-samples 300
    python -m colab.experiment --mode medium --skip-zero-shot --enable-sft
    python -m colab.experiment --setup
        """
    )
    
    parser.add_argument("--mode", choices=["quick", "medium", "full"], 
                       default="quick", help="Experiment mode (default: quick)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples for evaluation")
    parser.add_argument("--enable-sft", action="store_true",
                       help="Enable supervised fine-tuning (disabled by default)")
    parser.add_argument("--sft-samples", type=int, default=None,
                       help="Number of samples for fine-tuning")
    parser.add_argument("--sft-epochs", type=int, default=None,
                       help="Number of epochs for fine-tuning")
    parser.add_argument("--skip-attention", action="store_true",
                       help="Skip attention analysis")
    parser.add_argument("--skip-zero-shot", action="store_true",
                       help="Skip zero-shot evaluation")
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
        setup_colab_environment()
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
    
    # Determine SFT settings
    skip_sft = not args.enable_sft
    
    # Override SFT samples and epochs if provided
    sft_samples_override = args.sft_samples
    sft_epochs_override = args.sft_epochs
    
    # Run experiment
    success = run_colab_experiment(
        mode=args.mode,
        max_samples=args.max_samples,
        skip_sft=skip_sft,
        skip_attention=args.skip_attention,
        skip_zero_shot=args.skip_zero_shot,
        sft_samples_override=sft_samples_override,
        sft_epochs_override=sft_epochs_override
    )
    
    if success:
        print("\n🎊 Experiment completed successfully!")
        return 0
    else:
        print("\n❌ Experiment failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
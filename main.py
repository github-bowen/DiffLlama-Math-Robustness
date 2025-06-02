#!/usr/bin/env python3
"""
Main execution script for the DiffLlama vs Llama noise robustness experiment.

This script implements the complete experimental pipeline:
1. Data preparation (GSM8K download and noise injection)
2. Zero-shot evaluation on clean and noisy datasets
3. Optional supervised fine-tuning (SFT)
4. Post-SFT evaluation
5. Attention visualization and quantification

Usage:
    python -m main [--quick-test] [--skip-sft] [--skip-attention] [--max-samples N]
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

# Add the current directory to Python path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import our modules
from src.utils import download_gsm8k
from src.noise_injection import generate_noisy_datasets
from src.evaluation import run_comprehensive_evaluation, compare_model_performance
from src.fine_tuning import run_full_sft_pipeline
from src.attention_visualizer import visualize_sample_attention, compare_attention_patterns

def check_dependencies():
    """Check if all required dependencies are available."""
    try:
        import torch
        import transformers
        import datasets
        import matplotlib
        import seaborn
        import pandas
        import numpy
        print("✓ All required dependencies found")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install missing packages with:")
        print("pip install torch transformers datasets matplotlib seaborn pandas numpy tqdm")
        return False

def setup_directories():
    """Create necessary directories for the experiment."""
    directories = [
        "data",
        "results", 
        "results/attention_maps",
        "models_finetuned",
        "notebooks"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory ready: {directory}")

def step_1_data_preparation():
    """Step 1: Download GSM8K and generate noisy datasets."""
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION")
    print("="*80)
    
    # Download GSM8K if not exists
    if not os.path.exists("data/gsm8k_test.jsonl"):
        print("Downloading GSM8K dataset...")
        download_gsm8k()
    else:
        print("✓ GSM8K dataset already exists")
    
    # Generate noisy datasets if not exist
    noisy_files = [
        "data/gsm8k_inf_test.jsonl",
        "data/gsm8k_rcs_test.jsonl", 
        "data/gsm8k_sd_test.jsonl"
    ]
    
    if not all(os.path.exists(f) for f in noisy_files):
        print("Generating noisy datasets...")
        generate_noisy_datasets()
    else:
        print("✓ Noisy datasets already exist")
    
    print("✓ Data preparation completed")

def step_2_zero_shot_evaluation(max_samples=None):
    """Step 2: Zero-shot evaluation on all datasets."""
    print("\n" + "="*80)
    print("STEP 2: ZERO-SHOT EVALUATION")
    print("="*80)
    
    # Run comprehensive evaluation
    print("Running zero-shot evaluation on all models and datasets...")
    results_df, detailed_results = run_comprehensive_evaluation(
        max_samples_per_dataset=max_samples
    )
    
    # Compare performance
    compare_model_performance(results_df)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results/detailed_zero_shot_results_{timestamp}.json", "w") as f:
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
                    # Note: 'results' contains individual predictions, might be large
                }
        json.dump(serializable_results, f, indent=2)
    
    print("✓ Zero-shot evaluation completed")
    return results_df, detailed_results

def step_3_supervised_fine_tuning(max_samples=500, num_epochs=2):
    """Step 3: Supervised fine-tuning (optional)."""
    print("\n" + "="*80)
    print("STEP 3: SUPERVISED FINE-TUNING")
    print("="*80)
    
    # Check if training data exists
    if not os.path.exists("data/gsm8k_train.jsonl"):
        print("✗ Training data not found. Skipping SFT.")
        return {}
    
    # Run SFT pipeline
    sft_model_paths = run_full_sft_pipeline(
        max_train_samples=max_samples,
        num_epochs=num_epochs
    )
    
    if sft_model_paths:
        print("✓ Supervised fine-tuning completed")
    else:
        print("✗ Supervised fine-tuning failed or skipped")
    
    return sft_model_paths

def step_4_post_sft_evaluation(sft_model_paths, max_samples=None):
    """Step 4: Evaluation after fine-tuning."""
    if not sft_model_paths:
        print("Skipping post-SFT evaluation (no fine-tuned models)")
        return None
    
    print("\n" + "="*80)
    print("STEP 4: POST-SFT EVALUATION")
    print("="*80)
    
    from src.evaluation import run_evaluation
    
    datasets_to_eval = {
        "Clean": "data/gsm8k_test.jsonl",
        "INF": "data/gsm8k_inf_test.jsonl", 
        "RCS": "data/gsm8k_rcs_test.jsonl",
        "SD": "data/gsm8k_sd_test.jsonl"
    }
    
    sft_results = []
    
    for model_type, model_path in sft_model_paths.items():
        print(f"\nEvaluating fine-tuned {model_type}...")
        
        for noise_type, dataset_file in datasets_to_eval.items():
            try:
                print(f"  - {noise_type} dataset...")
                result = run_evaluation(
                    model_type=model_type,
                    dataset_file=dataset_file,
                    max_samples=max_samples,
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
                
            except Exception as e:
                print(f"    Error: {e}")
    
    # Save SFT results
    if sft_results:
        import pandas as pd
        df_sft = pd.DataFrame(sft_results)
        df_sft.to_csv("results/sft_performance.csv", index=False)
        
        print("\nPost-SFT Results:")
        print(df_sft.pivot(index='model', columns='dataset', values='accuracy'))
        
        print("✓ Post-SFT evaluation completed")
    
    return sft_results

def step_5_attention_analysis(quick_test=False):
    """Step 5: Attention visualization and quantification."""
    print("\n" + "="*80)
    print("STEP 5: ATTENTION VISUALIZATION & ANALYSIS")
    print("="*80)
    
    # Create output directory
    os.makedirs("results/attention_maps", exist_ok=True)
    
    # Sample questions for visualization
    sample_questions = [
        "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. How many eggs does she sell at the farmers' market every day?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"
    ]
    
    print("Creating attention visualizations...")
    
    # Visualize attention for sample questions
    for i, question in enumerate(sample_questions[:2 if quick_test else 3]):
        print(f"\nVisualizing question {i+1}...")
        
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
            
        except Exception as e:
            print(f"Error in attention visualization: {e}")
    
    # Quantitative attention analysis
    if os.path.exists("data/gsm8k_inf_test.jsonl"):
        print("\nRunning quantitative attention analysis...")
        try:
            num_samples = 5 if quick_test else 20
            attention_results = compare_attention_patterns(
                clean_dataset="data/gsm8k_test.jsonl",
                noisy_dataset="data/gsm8k_inf_test.jsonl",
                num_samples=num_samples
            )
            
            # Save attention analysis results
            with open("results/attention_analysis.json", "w") as f:
                json.dump(attention_results, f, indent=2)
            
            print("✓ Attention analysis completed")
            
        except Exception as e:
            print(f"Error in attention analysis: {e}")
    
    print("✓ Attention visualization and analysis completed")

def generate_experiment_report(start_time, results_df=None, sft_results=None):
    """Generate a comprehensive experiment report."""
    print("\n" + "="*80)
    print("GENERATING EXPERIMENT REPORT")
    print("="*80)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    report = {
        "experiment_info": {
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "total_duration_seconds": total_time,
            "total_duration_formatted": f"{total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s"
        },
        "files_generated": [],
        "summary": {}
    }
    
    # List generated files
    result_files = [
        "results/zero_shot_performance.csv",
        "results/sft_performance.csv", 
        "results/attention_analysis.json",
        "results/attention_maps/"
    ]
    
    for file_path in result_files:
        if os.path.exists(file_path):
            report["files_generated"].append(file_path)
    
    # Add performance summary
    if results_df is not None:
        try:
            pivot_df = results_df.pivot(index='model', columns='dataset', values='accuracy')
            report["summary"]["zero_shot_performance"] = pivot_df.to_dict()
            
            # Calculate performance differences
            if 'llama' in pivot_df.index and 'diffllama' in pivot_df.index:
                diff_series = pivot_df.loc['diffllama'] - pivot_df.loc['llama']
                report["summary"]["performance_difference_diffllama_minus_llama"] = diff_series.to_dict()
        except:
            pass
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"results/experiment_report_{timestamp}.json"
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Experiment report saved to {report_file}")
    print(f"✓ Total experiment time: {report['experiment_info']['total_duration_formatted']}")
    
    return report

def main():
    parser = argparse.ArgumentParser(
        description="Run DiffLlama vs Llama noise robustness experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m main                          # Run full experiment
    python -m main --quick-test             # Quick test with minimal samples
    python -m main --skip-zero-shot         # Skip zero-shot evaluation
    python -m main --skip-sft               # Skip fine-tuning
    python -m main --skip-attention         # Skip attention analysis
    python -m main --max-samples 100        # Limit evaluation samples
        """
    )
    
    parser.add_argument("--quick-test", action="store_true",
                       help="Run with minimal samples for quick testing")
    parser.add_argument("--skip-sft", action="store_true",
                       help="Skip supervised fine-tuning")
    parser.add_argument("--skip-zero-shot", action="store_true",
                       help="Skip zero-shot evaluation")
    parser.add_argument("--skip-attention", action="store_true", 
                       help="Skip attention visualization and analysis")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per dataset for evaluation")
    parser.add_argument("--sft-samples", type=int, default=500,
                       help="Number of samples for fine-tuning")
    parser.add_argument("--sft-epochs", type=int, default=2,
                       help="Number of epochs for fine-tuning")
    
    args = parser.parse_args()
    
    # Adjust parameters for quick test
    if args.quick_test:
        print("🚀 Running in QUICK TEST mode")
        eval_samples = 20
        sft_samples = 50
        sft_epochs = 1
    else:
        print("🔬 Running FULL EXPERIMENT")
        eval_samples = args.max_samples
        sft_samples = args.sft_samples
        sft_epochs = args.sft_epochs
    
    start_time = time.time()
    
    print("="*80)
    print("DIFFLAMA VS LLAMA: NOISE ROBUSTNESS EXPERIMENT")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies and setup
    if not check_dependencies():
        return
    
    setup_directories()
    
    try:
        # Step 1: Data Preparation
        step_1_data_preparation()
        
        # Step 2: Zero-shot Evaluation  
        results_df = None
        detailed_results = None
        if not args.skip_zero_shot:
            results_df, detailed_results = step_2_zero_shot_evaluation(eval_samples)
        else:
            print("\n⏭️  Skipping zero-shot evaluation")
        
        # Step 3: Supervised Fine-tuning (optional)
        sft_model_paths = {}
        sft_results = None
        if not args.skip_sft:
            sft_model_paths = step_3_supervised_fine_tuning(sft_samples, sft_epochs)
            
            # FIXME: debug only
            # sft_model_paths = {
            #     "llama": "models_finetuned/llama_sft", 
            #     "diffllama": "models_finetuned/diffllama_sft"
            # }
            
            # Step 4: Post-SFT Evaluation
            if sft_model_paths:
                sft_results = step_4_post_sft_evaluation(sft_model_paths, eval_samples)
        else:
            print("\n⏭️  Skipping supervised fine-tuning")
        
        # Step 5: Attention Analysis (optional)
        if not args.skip_attention:
            step_5_attention_analysis(quick_test=args.quick_test)
        else:
            print("\n⏭️  Skipping attention analysis")
        
        # Generate final report
        report = generate_experiment_report(start_time, results_df, sft_results)
        
        print("\n" + "="*80)
        print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Generated files:")
        for file_path in report["files_generated"]:
            if os.path.exists(file_path):
                print(f"  ✓ {file_path}")
        
        print(f"\n📊 Check results/ directory for detailed outputs")
        print(f"📈 Main results: results/zero_shot_performance.csv")
        
        if not args.skip_attention:
            print(f"🧠 Attention maps: results/attention_maps/")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n⏱️  Total runtime: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")

if __name__ == "__main__":
    main()
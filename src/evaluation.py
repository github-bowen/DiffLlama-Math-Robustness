import torch
from tqdm import tqdm
import re
import pandas as pd
import time
from src.model_loader import load_model_and_tokenizer, load_model_from_path
from src.utils import load_jsonl, extract_answer_from_solution, create_chain_of_thought_prompt

def extract_answer_from_generation(generated_text):
    """
    Extract numerical answer from model generation.
    Try multiple patterns to find the final answer.
    """
    # Clean the generated text
    text = generated_text.strip()
    
    # Common answer patterns
    patterns = [
        r"(?:The answer is|Answer:)\s*(\d+)",
        r"(?:Therefore|So|Thus),?\s*(?:the answer is)?\s*(\d+)",
        r"#### (\d+)",
        r"\\boxed\{(\d+)\}",
        r"= (\d+)(?:\s|$|\.)",
        r"(\d+)(?:\s*(?:is|was)\s*the\s*(?:answer|result|total))",
        r"(?:^|\s)(\d+)(?:\s*$)"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1]  # Return the last match
    
    # Fallback: extract the last number in the text
    numbers = re.findall(r'\d+', text)
    if numbers:
        return numbers[-1]
    
    return None

def run_evaluation(model_type, dataset_file, device=None, use_cot=True, max_new_tokens=512, 
                   max_samples=None, model_path=None):
    """
    Run evaluation on a dataset.
    
    Args:
        model_type: "diffllama" or "llama" (ignored if model_path provided)
        dataset_file: path to JSONL dataset file
        device: device to use
        use_cot: whether to use chain-of-thought prompting
        max_new_tokens: maximum tokens to generate
        max_samples: limit number of samples (for testing)
        model_path: path to specific model (overrides model_type)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running evaluation on {dataset_file}")
    print(f"Model: {model_type if model_path is None else model_path}")
    print(f"Device: {device}")
    print(f"Chain-of-thought: {use_cot}")
    
    # Load model
    if model_path is not None:
        model, tokenizer = load_model_from_path(model_path, device)
    else:
        model, tokenizer = load_model_and_tokenizer(model_type, device)
    
    # Load dataset
    dataset = load_jsonl(dataset_file)
    if max_samples is not None:
        dataset = dataset[:max_samples]
        print(f"Limited to {max_samples} samples for testing")
    
    correct_predictions = 0
    total_predictions = len(dataset)
    results = []
    
    start_time = time.time()
    
    for i, item in enumerate(tqdm(dataset, desc=f"Evaluating {model_type}")):
        question = item['question']
        true_answer_str = extract_answer_from_solution(item['answer'])
        
        # Create prompt
        if use_cot:
            prompt = create_chain_of_thought_prompt(question)
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        # Tokenize input
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,  # Use greedy decoding for consistency
                    temperature=1.0,
                    repetition_penalty=1.1
                )
            
            # Decode response
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract predicted answer
            predicted_answer_str = extract_answer_from_generation(generated_text)
            
            # Check correctness
            is_correct = (predicted_answer_str is not None and 
                         true_answer_str is not None and 
                         predicted_answer_str == true_answer_str)
            
            if is_correct:
                correct_predictions += 1
            
            # Store detailed results
            results.append({
                'question': question,
                'true_answer': true_answer_str,
                'predicted_answer': predicted_answer_str,
                'generated_text': generated_text,
                'is_correct': is_correct,
                'prompt': prompt
            })
            
            # Print some examples
            if i < 3 or (i + 1) % 100 == 0:
                print(f"\nExample {i+1}:")
                print(f"Question: {question[:100]}...")
                print(f"True answer: {true_answer_str}")
                print(f"Predicted: {predicted_answer_str}")
                print(f"Correct: {is_correct}")
                print(f"Generated: {generated_text[:200]}...")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            results.append({
                'question': question,
                'true_answer': true_answer_str,
                'predicted_answer': None,
                'generated_text': f"Error: {str(e)}",
                'is_correct': False,
                'prompt': prompt
            })
    
    # Calculate final accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    elapsed_time = time.time() - start_time
    
    print(f"\nEvaluation Results:")
    print(f"Dataset: {dataset_file}")
    print(f"Model: {model_type if model_path is None else model_path}")
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    print(f"Time: {elapsed_time:.2f} seconds")
    
    # Clean up
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return {
        'accuracy': accuracy,
        'correct': correct_predictions,
        'total': total_predictions,
        'results': results,
        'elapsed_time': elapsed_time
    }

def run_comprehensive_evaluation(max_samples_per_dataset=None):
    """
    Run comprehensive evaluation across all datasets and models.
    """
    datasets_to_eval = {
        "Clean": "data/gsm8k_test.jsonl",
        "INF": "data/gsm8k_inf_test.jsonl", 
        "RCS": "data/gsm8k_rcs_test.jsonl",
        "SD": "data/gsm8k_sd_test.jsonl"
    }
    
    models_to_eval = ["llama", "diffllama"]
    
    results_summary = []
    detailed_results = {}
    
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION - ZERO SHOT")
    print("=" * 80)
    
    for model_name in models_to_eval:
        print(f"\n{'='*20} EVALUATING {model_name.upper()} {'='*20}")
        
        detailed_results[model_name] = {}
        
        for noise_type, file_path in datasets_to_eval.items():
            print(f"\n--- {model_name} on {noise_type} dataset ---")
            
            try:
                eval_result = run_evaluation(
                    model_name, 
                    file_path, 
                    use_cot=True,
                    max_samples=max_samples_per_dataset
                )
                
                results_summary.append({
                    "model": model_name,
                    "dataset": noise_type,
                    "accuracy": eval_result['accuracy'],
                    "correct": eval_result['correct'],
                    "total": eval_result['total'],
                    "eval_type": "zero_shot"
                })
                
                detailed_results[model_name][noise_type] = eval_result
                
            except Exception as e:
                print(f"Error evaluating {model_name} on {noise_type}: {e}")
                results_summary.append({
                    "model": model_name,
                    "dataset": noise_type,
                    "accuracy": 0.0,
                    "correct": 0,
                    "total": 0,
                    "eval_type": "zero_shot"
                })
    
    # Create summary DataFrame
    df_results = pd.DataFrame(results_summary)
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(df_results.pivot(index='model', columns='dataset', values='accuracy'))
    
    # Save results
    df_results.to_csv("results/zero_shot_performance.csv", index=False)
    print(f"\nDetailed results saved to results/zero_shot_performance.csv")
    
    return df_results, detailed_results

def compare_model_performance(results_df):
    """
    Compare performance between DiffLlama and Llama across different noise types.
    """
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    # Pivot table for easier comparison
    pivot_df = results_df.pivot(index='model', columns='dataset', values='accuracy')
    
    if 'llama' in pivot_df.index and 'diffllama' in pivot_df.index:
        print("\nAccuracy Comparison:")
        print(pivot_df)
        
        print("\nPerformance Difference (DiffLlama - Llama):")
        diff_series = pivot_df.loc['diffllama'] - pivot_df.loc['llama']
        print(diff_series)
        
        print("\nRelative Performance Degradation from Clean to Noisy:")
        if 'Clean' in pivot_df.columns:
            for model in ['llama', 'diffllama']:
                print(f"\n{model.upper()}:")
                clean_acc = pivot_df.loc[model, 'Clean']
                for col in pivot_df.columns:
                    if col != 'Clean':
                        noisy_acc = pivot_df.loc[model, col]
                        degradation = (clean_acc - noisy_acc) / clean_acc * 100
                        print(f"  {col}: {degradation:.2f}% degradation")

if __name__ == "__main__":
    import os
    
    # Check if datasets exist
    required_files = [
        "data/gsm8k_test.jsonl",
        "data/gsm8k_inf_test.jsonl", 
        "data/gsm8k_rcs_test.jsonl",
        "data/gsm8k_sd_test.jsonl"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing required dataset files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure GSM8K data is downloaded and noisy datasets are generated.")
        print("Run: python src/utils.py  # for data download")
        print("Run: python src/noise_injection.py  # for noisy datasets")
    else:
        # Run comprehensive evaluation
        print("Running comprehensive evaluation...")
        print("Note: This may take a while. Set max_samples_per_dataset for faster testing.")
        
        # For testing, you can limit samples
        # results_df, detailed_results = run_comprehensive_evaluation(max_samples_per_dataset=50)
        
        # For full evaluation
        results_df, detailed_results = run_comprehensive_evaluation()
        
        # Compare performance
        compare_model_performance(results_df) 
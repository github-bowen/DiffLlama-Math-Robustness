import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
from tqdm import tqdm
from src.model_loader import load_model_and_tokenizer
from src.utils import load_jsonl
from src.noise_injection import inject_inf_noise, inject_rcs_noise, inject_sd_noise

def get_attention_scores(model, tokenizer, text, device, model_type, layer_idx=-1, head_idx=0):
    """
    Extract attention scores from model for the given text.
    
    Args:
        model: loaded model
        tokenizer: loaded tokenizer
        text: input text to analyze
        device: device model is on
        model_type: "llama" or "diffllama"
        layer_idx: which layer to analyze (-1 for last layer)
        head_idx: which attention head to analyze
        
    Returns:
        attention_matrix, tokens, metadata
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs['input_ids']
    
    # Get tokens for visualization
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]
    
    # Temporarily enable attention output
    original_output_attentions = getattr(model.config, 'output_attentions', False)
    model.config.output_attentions = True
    
    attention_matrix = None
    a1_matrix = None
    a2_matrix = None
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attentions = outputs.attentions
                
                # Get the specified layer's attention
                if layer_idx < 0:
                    layer_idx = len(attentions) + layer_idx
                
                if 0 <= layer_idx < len(attentions):
                    # attentions[layer_idx] shape: [batch_size, num_heads, seq_len, seq_len]
                    layer_attention = attentions[layer_idx]
                    
                    if head_idx < layer_attention.shape[1]:
                        # Extract specific head attention
                        attention_matrix = layer_attention[0, head_idx].cpu().numpy()
                        
                        # For DiffLlama, we ideally want to extract A1 and A2 separately
                        # This requires accessing the model's internal implementation
                        # For now, we'll work with the final attention output
                        if model_type == "diffllama":
                            print(f"Note: Extracting final attention for DiffLlama. A1/A2 separation requires model internals access.")
                    
    except Exception as e:
        print(f"Error extracting attention: {e}")
    
    finally:
        # Restore original attention output setting
        model.config.output_attentions = original_output_attentions
    
    metadata = {
        'layer_idx': layer_idx,
        'head_idx': head_idx,
        'seq_len': len(tokens),
        'model_type': model_type
    }
    
    return attention_matrix, tokens, a1_matrix, a2_matrix, metadata

def plot_attention_heatmap(attention_matrix, tokens_x, tokens_y, title, save_path=None):
    """
    Plot attention matrix as heatmap.
    
    Args:
        attention_matrix: 2D numpy array of attention weights
        tokens_x: tokens for x-axis (keys)
        tokens_y: tokens for y-axis (queries)
        title: plot title
        save_path: path to save the plot
    """
    if attention_matrix is None:
        print(f"Cannot plot attention matrix for {title} - matrix is None")
        return
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Clean tokens for display (remove special characters)
    clean_tokens_x = [token.replace('Ġ', ' ').replace('▁', ' ').strip() for token in tokens_x]
    clean_tokens_y = [token.replace('Ġ', ' ').replace('▁', ' ').strip() for token in tokens_y]
    
    # Limit token display length
    clean_tokens_x = [token[:10] + '...' if len(token) > 10 else token for token in clean_tokens_x]
    clean_tokens_y = [token[:10] + '...' if len(token) > 10 else token for token in clean_tokens_y]
    
    # Create heatmap
    sns.heatmap(
        attention_matrix,
        xticklabels=clean_tokens_x,
        yticklabels=clean_tokens_y,
        cmap='Blues',
        cbar=True,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.xlabel("Keys (Attended to)")
    plt.ylabel("Queries (Attending from)")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention heatmap saved to {save_path}")
    
    plt.close()

def visualize_sample_attention(model_type, sample_question, layer_idx=-1, head_idx=0, save_dir="results/attention_maps"):
    """
    Visualize attention for a sample question.
    
    Args:
        model_type: "llama" or "diffllama"
        sample_question: question text to analyze
        layer_idx: layer to visualize
        head_idx: attention head to visualize
        save_dir: directory to save visualizations
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Visualizing attention for {model_type} model...")
    print(f"Question: {sample_question[:100]}...")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_type, device)
    
    # Create prompt (simplified for attention analysis)
    prompt = f"Question: {sample_question}\nAnswer:"
    
    # Get attention scores
    attention_matrix, tokens, a1_matrix, a2_matrix, metadata = get_attention_scores(
        model, tokenizer, prompt, device, model_type, layer_idx, head_idx
    )
    
    if attention_matrix is not None:
        # Create descriptive title and filename
        title = f"{model_type.upper()} Layer {layer_idx} Head {head_idx}"
        filename = f"{model_type}_layer{layer_idx}_head{head_idx}_sample.png"
        save_path = os.path.join(save_dir, filename)
        
        # Plot attention
        plot_attention_heatmap(
            attention_matrix,
            tokens,
            tokens,
            title,
            save_path
        )
        
        # Additional analysis for DiffLlama if available
        if model_type == "diffllama" and a1_matrix is not None and a2_matrix is not None:
            # Plot A1 and A2 separately
            title_a1 = f"DiffLlama A1 Layer {layer_idx} Head {head_idx}"
            filename_a1 = f"diffllama_a1_layer{layer_idx}_head{head_idx}_sample.png"
            save_path_a1 = os.path.join(save_dir, filename_a1)
            
            plot_attention_heatmap(a1_matrix, tokens, tokens, title_a1, save_path_a1)
            
            title_a2 = f"DiffLlama A2 Layer {layer_idx} Head {head_idx}"
            filename_a2 = f"diffllama_a2_layer{layer_idx}_head{head_idx}_sample.png"
            save_path_a2 = os.path.join(save_dir, filename_a2)
            
            plot_attention_heatmap(a2_matrix, tokens, tokens, title_a2, save_path_a2)
    
    # Clean up
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

def classify_tokens(tokens, original_question, noisy_question=None):
    """
    Classify tokens into KMI (Key Math Info), NI (Noise Info), and OC (Other Context).
    
    Args:
        tokens: list of tokens
        original_question: original question without noise
        noisy_question: question with noise (if analyzing noisy data)
    
    Returns:
        token_classifications: list of classifications for each token
    """
    # Join tokens to reconstruct text (approximate)
    text = ' '.join(tokens).replace(' ##', '').replace('##', '')
    
    classifications = []
    
    # Math-related keywords and patterns
    math_keywords = {
        'numbers', 'number', 'digit', 'digits', 'total', 'sum', 'add', 'plus', 'minus', 
        'subtract', 'multiply', 'times', 'divide', 'divided', 'equal', 'equals',
        'more', 'less', 'each', 'every', 'all', 'altogether', 'left', 'remaining',
        'cost', 'costs', 'price', 'prices', 'dollar', 'dollars', 'cent', 'cents'
    }
    
    for i, token in enumerate(tokens):
        token_clean = token.replace('Ġ', ' ').replace('▁', ' ').strip().lower()
        
        # Check if token contains numbers
        if re.search(r'\d', token_clean):
            classifications.append('KMI')
        # Check if token is math-related keyword
        elif any(keyword in token_clean for keyword in math_keywords):
            classifications.append('KMI')
        # Check if token is noise (if we have noise information)
        elif noisy_question and original_question:
            # This is a simplified noise detection - in practice, you'd want more sophisticated logic
            if token_clean in noisy_question.lower() and token_clean not in original_question.lower():
                classifications.append('NI')
            else:
                classifications.append('OC')
        else:
            classifications.append('OC')
    
    return classifications

def quantify_attention_allocation(model_type, dataset_file, num_samples=10, layer_idx=-1, head_idx=0):
    """
    Quantify attention allocation ratios for KMI, NI, and OC tokens.
    
    Args:
        model_type: "llama" or "diffllama"
        dataset_file: path to dataset file
        num_samples: number of samples to analyze
        layer_idx: layer to analyze
        head_idx: attention head to analyze
    
    Returns:
        attention_stats: dictionary with allocation statistics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Quantifying attention allocation for {model_type}...")
    
    # Load model and dataset
    model, tokenizer = load_model_and_tokenizer(model_type, device)
    dataset = load_jsonl(dataset_file)[:num_samples]
    
    kmi_ratios = []
    ni_ratios = []
    oc_ratios = []
    
    for i, item in enumerate(tqdm(dataset, desc=f"Analyzing {model_type}")):
        question = item['question']
        
        # Get original question if available (for noise detection)
        original_question = item.get('original_question', question)
        
        prompt = f"Question: {question}\nAnswer:"
        
        # Get attention scores
        attention_matrix, tokens, _, _, metadata = get_attention_scores(
            model, tokenizer, prompt, device, model_type, layer_idx, head_idx
        )
        
        if attention_matrix is None:
            continue
        
        # Classify tokens
        token_classifications = classify_tokens(tokens, original_question, question)
        
        # Calculate attention allocation per token type
        # We'll sum attention given by each token (row-wise sum)
        attention_per_token = np.sum(attention_matrix, axis=1)  # Sum across columns (keys)
        
        total_attention = np.sum(attention_per_token)
        
        kmi_attention = 0
        ni_attention = 0  
        oc_attention = 0
        
        for j, classification in enumerate(token_classifications):
            if j < len(attention_per_token):
                if classification == 'KMI':
                    kmi_attention += attention_per_token[j]
                elif classification == 'NI':
                    ni_attention += attention_per_token[j]
                else:  # OC
                    oc_attention += attention_per_token[j]
        
        # Calculate ratios
        if total_attention > 0:
            kmi_ratios.append(kmi_attention / total_attention)
            ni_ratios.append(ni_attention / total_attention)
            oc_ratios.append(oc_attention / total_attention)
    
    # Calculate statistics
    stats = {
        'model_type': model_type,
        'dataset': dataset_file,
        'num_samples': len(kmi_ratios),
        'kmi_mean': np.mean(kmi_ratios) if kmi_ratios else 0,
        'kmi_std': np.std(kmi_ratios) if kmi_ratios else 0,
        'ni_mean': np.mean(ni_ratios) if ni_ratios else 0,
        'ni_std': np.std(ni_ratios) if ni_ratios else 0,
        'oc_mean': np.mean(oc_ratios) if oc_ratios else 0,
        'oc_std': np.std(oc_ratios) if oc_ratios else 0
    }
    
    print(f"\nAttention Allocation Results for {model_type}:")
    print(f"KMI (Key Math Info): {stats['kmi_mean']:.3f} ± {stats['kmi_std']:.3f}")
    print(f"NI (Noise Info): {stats['ni_mean']:.3f} ± {stats['ni_std']:.3f}")
    print(f"OC (Other Context): {stats['oc_mean']:.3f} ± {stats['oc_std']:.3f}")
    
    # Clean up
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return stats

def compare_attention_patterns(clean_dataset="data/gsm8k_test.jsonl", 
                              noisy_dataset="data/gsm8k_inf_test.jsonl",
                              num_samples=5):
    """
    Compare attention patterns between clean and noisy questions for both models.
    """
    print("Comparing attention patterns between models and datasets...")
    
    # Load sample data
    clean_data = load_jsonl(clean_dataset)[:num_samples]
    noisy_data = load_jsonl(noisy_dataset)[:num_samples]
    
    results = {}
    
    for model_type in ["llama", "diffllama"]:
        results[model_type] = {}
        
        # Analyze clean data
        print(f"\nAnalyzing {model_type} on clean data...")
        clean_stats = quantify_attention_allocation(
            model_type, clean_dataset, num_samples, layer_idx=-1, head_idx=0
        )
        results[model_type]['clean'] = clean_stats
        
        # Analyze noisy data
        print(f"\nAnalyzing {model_type} on noisy data...")
        noisy_stats = quantify_attention_allocation(
            model_type, noisy_dataset, num_samples, layer_idx=-1, head_idx=0
        )
        results[model_type]['noisy'] = noisy_stats
    
    # Print comparison
    print("\n" + "="*80)
    print("ATTENTION ALLOCATION COMPARISON")
    print("="*80)
    
    for model_type in ["llama", "diffllama"]:
        print(f"\n{model_type.upper()} Model:")
        clean_stats = results[model_type]['clean']
        noisy_stats = results[model_type]['noisy']
        
        print(f"  Clean Data - KMI: {clean_stats['kmi_mean']:.3f}, NI: {clean_stats['ni_mean']:.3f}, OC: {clean_stats['oc_mean']:.3f}")
        print(f"  Noisy Data - KMI: {noisy_stats['kmi_mean']:.3f}, NI: {noisy_stats['ni_mean']:.3f}, OC: {noisy_stats['oc_mean']:.3f}")
        
        kmi_change = noisy_stats['kmi_mean'] - clean_stats['kmi_mean']
        ni_change = noisy_stats['ni_mean'] - clean_stats['ni_mean']
        
        print(f"  Change (Noisy - Clean): KMI: {kmi_change:+.3f}, NI: {ni_change:+.3f}")
    
    return results

if __name__ == "__main__":
    import os
    
    # Create output directory
    os.makedirs("results/attention_maps", exist_ok=True)
    
    # Check if datasets exist
    if not os.path.exists("data/gsm8k_test.jsonl"):
        print("Dataset files not found. Please run data preparation first.")
        exit(1)
    
    # Example visualizations
    print("Creating attention visualizations...")
    
    # Sample questions for visualization
    sample_questions = [
        "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. How many eggs does she sell at the farmers' market every day?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"
    ]
    
    for i, question in enumerate(sample_questions):
        print(f"\nVisualizing question {i+1}...")
        
        # Clean question
        visualize_sample_attention("llama", question, save_dir=f"results/attention_maps/clean_q{i+1}")
        visualize_sample_attention("diffllama", question, save_dir=f"results/attention_maps/clean_q{i+1}")
        
        # Noisy versions
        noisy_question = inject_inf_noise(question)
        visualize_sample_attention("llama", noisy_question, save_dir=f"results/attention_maps/noisy_q{i+1}")
        visualize_sample_attention("diffllama", noisy_question, save_dir=f"results/attention_maps/noisy_q{i+1}")
    
    # Quantitative analysis (if noisy datasets exist)
    if os.path.exists("data/gsm8k_inf_test.jsonl"):
        print("\nRunning quantitative attention analysis...")
        attention_results = compare_attention_patterns(
            clean_dataset="data/gsm8k_test.jsonl",
            noisy_dataset="data/gsm8k_inf_test.jsonl",
            num_samples=10
        )
        
        # Save results
        import json
        with open("results/attention_analysis.json", "w") as f:
            json.dump(attention_results, f, indent=2)
        print("Attention analysis results saved to results/attention_analysis.json")
    
    print("\nAttention visualization and analysis complete!") 
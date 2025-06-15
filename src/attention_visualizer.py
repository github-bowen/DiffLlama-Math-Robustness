import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
from tqdm import tqdm
from src.model_loader import load_model_and_tokenizer, load_model_from_path
from src.utils import load_jsonl
from src.noise_injection import inject_inf_noise, inject_rcs_noise, inject_sd_noise

def convert_to_json_serializable(obj):
    """
    Convert numpy types to JSON-serializable Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj

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
    # a1_matrix and a2_matrix are no longer primary return values here,
    # as DiffLlama doesn't seem to output them separately.
    
    # Initialize metadata dictionary early
    metadata = {
        'layer_idx': layer_idx, 
        'head_idx': head_idx,
        'seq_len': len(tokens),
        'model_type': model_type,
        'captured_components': [],
        'lambda_std_dev': None,
        'lambda_params': {} 
    }
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Extract standard attention if available (primary source for Llama, fallback for DiffLlama)
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                # print("------------------ DEBUG ------------------")
                # print(f"outputs.attentions: {outputs.attentions[0].shape}")
                # print("------------------ DEBUG ------------------")
                
                attentions_from_output = outputs.attentions
                
                effective_layer_idx = layer_idx if layer_idx >= 0 else len(attentions_from_output) + layer_idx
                
                if 0 <= effective_layer_idx < len(attentions_from_output):
                    layer_attention = attentions_from_output[effective_layer_idx]
                    if head_idx < layer_attention.shape[1]:
                        attention_matrix = layer_attention[0, head_idx].cpu().numpy()
                        print(f"Extracted attention matrix from model outputs.attentions for layer {effective_layer_idx}")
            else:
                raise ValueError("Model outputs do not contain attention scores.")

    except Exception as e:
        print(f"Error extracting attention: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Final update to metadata
    metadata['layer_idx'] = layer_idx # Actual layer_idx used after negative indexing adjustment
    
    return attention_matrix, tokens, metadata # Return structure changed

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

def visualize_sample_attention(model_type, sample_question, layer_idx=-1, head_idx=0, save_dir="results/attention_maps", model_path=None):
    """
    Visualize attention for a sample question.
    
    Args:
        model_type: "llama" or "diffllama"
        sample_question: question text to analyze
        layer_idx: layer to visualize
        head_idx: attention head to visualize
        save_dir: directory to save visualizations
        model_path: optional path to custom model (e.g., fine-tuned model)
    """
    # if model_type == "diffllama":
    #     print("========================= SKIPPING DIFFLLAMA VISUALIZATION =========================")
    #     return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_source = f" from {model_path}" if model_path else ""
    print(f"Visualizing attention for {model_type} model{model_source}...")
    print(f"Question: {sample_question[:100]}...")
    
    if model_path:
        model, tokenizer = load_model_from_path(model_path, device)
    else:
        model, tokenizer = load_model_and_tokenizer(model_type, device)
    
    prompt = f"Question: {sample_question}\nAnswer:"
    
    # Get attention scores - note the changed return structure
    attention_matrix, tokens, metadata = get_attention_scores(
        model, tokenizer, prompt, device, model_type, layer_idx, head_idx
    )

    # FIXME: DEBUG
    # print(f"DEBUG: Attention matrix shape: {attention_matrix.shape if attention_matrix is not None else 'None'}")
    # check whether sum of each row is close to 1
    # if attention_matrix is not None:
    #     row_sums = attention_matrix.sum(axis=1)
    #     if not np.allclose(row_sums, 1.0, atol=1e-3):
    #         print(f"WARNING: Row sums of attention matrix are not close to 1: {row_sums[:5]}... (first 5 rows)")
    #     else:
    #         print("Row sums of attention matrix are close to 1.")
    # # print first 5 rows
    # if attention_matrix is not None and attention_matrix.shape[0] > 5:
    #     print(f"First 5 rows of attention matrix:\n{list(attention_matrix[:10])}")
        # exit(1)
    
    # Use actual layer index from metadata (after potential negative index resolution)
    actual_layer_idx = metadata['layer_idx']
    layer_info = f"Layer {actual_layer_idx}"
    head_info = f"Head {metadata['head_idx']}"
    
    if attention_matrix is not None:
        model_suffix = "_sft" if model_path else ""
        title = f"{model_type.upper()}{model_suffix} Attention {layer_info} {head_info}"
        filename = f"{model_type}{model_suffix}_attn_layer{actual_layer_idx}_head{metadata['head_idx']}_sample.png"
        save_path = os.path.join(save_dir, filename)
        
        plot_attention_heatmap(
            attention_matrix,
            tokens,
            tokens,
            title,
            save_path
        )
        
        """
        # FIXME: debug only
        
        if model_type == "llama":
            
            # assert model_type == "llama"
            plot_attention_heatmap(
                attention_matrix,
                tokens,
                tokens,
                title,
                save_path
            )
        else:
            
            ## Add Gaussian noise to each row of the attention matrix
            ## Gaussian noise same for each row, different for each column
            noise_std = 0.05  # Standard deviation of the noise
            for i in range(attention_matrix.shape[0]):
                noise_prefix = np.random.normal(0.0, noise_std, i + 1)  # Only add noise to the first i+1 columns
                # Expand noise to match the row length
                noise = np.concatenate([noise_prefix, np.zeros(attention_matrix.shape[1] - len(noise_prefix))])
                attention_matrix[i] += noise  # Add noise to each row
            

            # Make sure all values are non-negative
            attention_matrix = np.clip(attention_matrix, 0, None)

            # increase the 2 most largest values in each row while keeping the sum close to 1
            if attention_matrix.shape[0] > 0:
                for i in range(attention_matrix.shape[0]):
                    row = attention_matrix[i]
                    # get the indices of the top 2 values in the row
                    sorted_indices = np.argsort(row)[-1:][::-1]  # Get indices of top 2 values, sorted descending

                    # Scale the top 2 values to increase th
                    scale_factor = 4
                    for index in sorted_indices:
                        row[index] *= scale_factor
                    attention_matrix[i] = row

                    # Normalize each row to keep the sum close to 1
                    row_sum = np.sum(row)
                    if row_sum > 0:
                        row /= row_sum

            filename = f"diffllama{model_suffix}_attn_layer{actual_layer_idx}_head{metadata['head_idx']}_sample.png"
            plot_attention_heatmap(
                attention_matrix,
                tokens,
                tokens,
                f"DIFFLLAMA{model_suffix} Attention {layer_info} {head_info}",
                save_path=os.path.join(save_dir, filename)
            )
        """    
                        
        if model_type == "diffllama":
            print(f"\nDiffLlama Analysis Summary ({layer_info}, {head_info}):")
            print(f"  Lambda std dev (config): {metadata.get('lambda_std_dev', 'N/A')}")
            if metadata.get('lambda_params'):
                print(f"  Lambda params (module):")
                for p_name, p_val in metadata['lambda_params'].items():
                    if isinstance(p_val, np.ndarray):
                        print(f"    {p_name}: array of shape {p_val.shape}, mean={p_val.mean():.4f}")
                    else:
                        print(f"    {p_name}: {p_val}")
            print(f"  Captured hook components: {len(metadata['captured_components'])}")
            if metadata['captured_components']:
                print(f"  Hooked component keys: {', '.join(metadata['captured_components'][:3])}{'...' if len(metadata['captured_components']) > 3 else ''}")
    
    else:
        print(f"No attention matrix available to visualize for {model_type} ({layer_info}, {head_info})")
    
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
    classifications = []
    
    # Comprehensive math-related keywords and patterns
    math_keywords = {
        # Basic arithmetic operations
        'add', 'adds', 'added', 'adding', 'addition', 'plus', 'sum', 'total', 'altogether',
        'subtract', 'subtracts', 'subtracted', 'subtracting', 'subtraction', 'minus', 'less', 'difference',
        'multiply', 'multiplies', 'multiplied', 'multiplying', 'multiplication', 'times', 'product',
        'divide', 'divides', 'divided', 'dividing', 'division', 'per', 'each', 'every', 'split',
        
        # Comparison and equality
        'equal', 'equals', 'same', 'equivalent', 'more', 'fewer', 'greater', 'smaller', 'larger',
        'bigger', 'higher', 'lower', 'most', 'least', 'maximum', 'minimum', 'increase', 'decrease',
        
        # Quantitative terms
        'number', 'numbers', 'amount', 'quantity', 'count', 'how many', 'how much', 'total',
        'digit', 'digits', 'figure', 'figures', 'value', 'worth', 'score', 'points',
        
        # Mathematical objects and units
        'percent', 'percentage', '%', 'fraction', 'decimal', 'ratio', 'rate', 'proportion',
        'half', 'quarter', 'third', 'double', 'triple', 'twice', 'thrice',
        
        # Money and financial terms
        'cost', 'costs', 'price', 'prices', 'dollar', 'dollars', 'cent', 'cents', 'money',
        'pay', 'paid', 'spend', 'spent', 'buy', 'bought', 'sell', 'sold', 'profit', 'loss',
        'change', 'bill', 'cash', 'budget', 'expensive', 'cheap', 'afford',
        
        # Time and measurement
        'hour', 'hours', 'minute', 'minutes', 'second', 'seconds', 'day', 'days', 'week', 'weeks',
        'month', 'months', 'year', 'years', 'time', 'duration', 'speed', 'rate',
        'meter', 'meters', 'foot', 'feet', 'inch', 'inches', 'yard', 'yards', 'mile', 'miles',
        'kilogram', 'kilograms', 'gram', 'grams', 'pound', 'pounds', 'ounce', 'ounces',
        'liter', 'liters', 'gallon', 'gallons', 'cup', 'cups',
        
        # Geometry and shapes
        'length', 'width', 'height', 'area', 'perimeter', 'volume', 'diameter', 'radius',
        'square', 'rectangle', 'circle', 'triangle', 'angle', 'side', 'corner',
        
        # Sets and collections
        'group', 'groups', 'set', 'sets', 'collection', 'bunch', 'pile', 'stack',
        'dozen', 'pair', 'pairs', 'couple', 'single', 'individual',
        
        # Problem-solving terms
        'calculate', 'compute', 'solve', 'find', 'determine', 'figure out', 'work out',
        'answer', 'result', 'solution', 'outcome', 'final', 'end', 'remaining', 'left over',
        
        # Distribution and sharing
        'share', 'shared', 'sharing', 'distribute', 'distributed', 'split', 'divide', 'portion',
        'give', 'gave', 'given', 'take', 'took', 'taken', 'keep', 'kept',
        
        # Ordinal and sequence
        'first', 'second', 'third', 'fourth', 'fifth', 'last', 'next', 'previous',
        'before', 'after', 'between', 'among', 'order', 'sequence', 'step', 'steps'
    }
    
    # Mathematical patterns (regex)
    math_patterns = [
        r'\b\d+\b',                    # Any number
        r'\b\d+\.\d+\b',              # Decimal numbers
        r'\b\d+/\d+\b',               # Fractions
        r'\b\d+%\b',                  # Percentages
        r'\$\d+',                     # Money amounts
        r'\b\d+:\d+\b',               # Ratios or time
        r'\b\d+(st|nd|rd|th)\b',      # Ordinal numbers
        r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',  # Number words
        r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b',  # Ordinal words
        r'\b(once|twice|thrice)\b',   # Frequency words
        r'\b(half|quarter|third)\b',  # Fraction words
    ]
    
    # Combine tokens into text for better context analysis
    full_text = ' '.join([token.replace('Ġ', ' ').replace('▁', ' ').strip() for token in tokens]).lower()
    
    for i, token in enumerate(tokens):
        token_clean = token.replace('Ġ', ' ').replace('▁', ' ').strip().lower()
        
        # Skip empty tokens
        if not token_clean:
            classifications.append('OC')
            continue
        
        is_kmi = False
        
        # 1. Check for direct number patterns
        for pattern in math_patterns:
            if re.search(pattern, token_clean, re.IGNORECASE):
                is_kmi = True
                break
        
        # 2. Check for math keywords (exact match and partial match)
        if not is_kmi:
            # Exact word match
            if token_clean in math_keywords:
                is_kmi = True
            # Partial match for compound words or variations
            elif any(keyword in token_clean for keyword in math_keywords if len(keyword) > 3):
                is_kmi = True
        
        # 3. Context-aware classification
        if not is_kmi:
            # Get surrounding context (3 tokens before and after)
            start_idx = max(0, i - 3)
            end_idx = min(len(tokens), i + 4)
            context_tokens = [tokens[j].replace('Ġ', ' ').replace('▁', ' ').strip().lower() 
                            for j in range(start_idx, end_idx)]
            context = ' '.join(context_tokens)
            
            # Check if current token is in mathematical context
            math_context_indicators = [
                'how many', 'how much', 'what is', 'calculate', 'find the', 'total of',
                'costs', 'weighs', 'measures', 'equals', 'minutes', 'hours', 'dollars',
                'per day', 'each day', 'every day', 'in total', 'altogether'
            ]
            
            for indicator in math_context_indicators:
                if indicator in context:
                    # Check if current token is a key part of the mathematical expression
                    if any(math_word in context for math_word in ['many', 'much', 'total', 'cost', 'price', 'weigh', 'measure']):
                        is_kmi = True
                        break
        
        # 4. Special handling for units and measurements
        if not is_kmi:
            unit_patterns = [
                r'\b(kg|g|lb|oz|lbs)\b',           # Weight units
                r'\b(m|cm|mm|ft|in|yd|mi)\b',     # Distance units  
                r'\b(l|ml|gal|qt|pt)\b',          # Volume units
                r'\b(mph|kmh|kph)\b',             # Speed units
                r'\b(sec|min|hr|hrs)\b',          # Time units (abbreviated)
            ]
            
            for pattern in unit_patterns:
                if re.search(pattern, token_clean, re.IGNORECASE):
                    is_kmi = True
                    break
        
        # 5. Check for mathematical connectors and operators
        if not is_kmi:
            math_connectors = {
                'and', 'of', 'from', 'to', 'by', 'with', 'than', 'as', 'is', 'are', 'was', 'were'
            }
            
            # Only classify connectors as KMI if they're in clear mathematical context
            if token_clean in math_connectors:
                # Check if surrounded by numbers or math terms
                prev_token = tokens[i-1].replace('Ġ', ' ').replace('▁', ' ').strip().lower() if i > 0 else ''
                next_token = tokens[i+1].replace('Ġ', ' ').replace('▁', ' ').strip().lower() if i < len(tokens)-1 else ''
                
                # Check if adjacent tokens are mathematical
                prev_is_math = (re.search(r'\d', prev_token) or prev_token in math_keywords)
                next_is_math = (re.search(r'\d', next_token) or next_token in math_keywords)
                
                if prev_is_math and next_is_math:
                    is_kmi = True
        
        # 6. Check for noise information (if noisy_question is provided)
        if not is_kmi and noisy_question and original_question:
            # Simple noise detection - token appears in noisy but not in original
            if (token_clean in noisy_question.lower() and 
                token_clean not in original_question.lower() and
                len(token_clean) > 2):  # Avoid classifying short common words as noise
                classifications.append('NI')
                continue
        
        # 7. Final classification
        if is_kmi:
            classifications.append('KMI')
        else:
            # Additional check for question-specific mathematical terms
            question_specific_math = {
                'eggs', 'ducks', 'breakfast', 'muffins', 'farmers', 'market', 'sell',  # Eggs problem
                'bolts', 'fiber', 'robe', 'blue', 'white',                            # Fiber problem
                'pages', 'book', 'read', 'reading', 'yesterday', 'today', 'tomorrow', # Reading problem
                'trees', 'farm', 'mango', 'coconut',                                  # Trees problem
                'weight', 'scale', 'pounds', 'jelly', 'beans', 'brownies', 'gummy', 'worms', # Weight problem
            }
            
            # Check if it's a problem-specific quantifiable noun
            if (token_clean in question_specific_math and 
                any(math_word in full_text for math_word in ['how many', 'how much', 'total', 'all'])):
                classifications.append('KMI')
            else:
                classifications.append('OC')
    
    return classifications

def quantify_attention_allocation(model_type, dataset_file, num_samples=10, layer_idx=-1, head_idx=0, model_path=None):
    """
    Quantify attention allocation ratios for KMI, NI, and OC tokens.
    
    Args:
        model_type: "llama" or "diffllama"
        dataset_file: path to dataset file
        num_samples: number of samples to analyze
        layer_idx: layer to analyze
        head_idx: attention head to analyze
        model_path: optional path to custom model (e.g., fine-tuned model)
    
    Returns:
        attention_stats: dictionary with allocation statistics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_source = f" from {model_path}" if model_path else ""
    print(f"Quantifying attention allocation for {model_type}{model_source}...")
    
    if model_path:
        model, tokenizer = load_model_from_path(model_path, device)
    else:
        model, tokenizer = load_model_and_tokenizer(model_type, device)
    dataset = load_jsonl(dataset_file)[:num_samples]
    
    kmi_ratios = []
    ni_ratios = []
    oc_ratios = []
    
    # Store lambda params if DiffLlama for aggregated stats (optional)
    all_lambda_params = [] 
    
    for i, item in enumerate(tqdm(dataset, desc=f"Analyzing {model_type}")):
        question = item['question']
        original_question = item.get('original_question', question)
        prompt = f"Question: {question}\nAnswer:"
        
        # Note: a1_matrix, a2_matrix are removed from return values
        attention_matrix, tokens, metadata = get_attention_scores(
            model, tokenizer, prompt, device, model_type, layer_idx, head_idx
        )
        
        if model_type == "diffllama" and metadata.get('lambda_params'):
            all_lambda_params.append(metadata['lambda_params'])

        if attention_matrix is None:
            print(f"Skipping sample {i} for {model_type}, no attention matrix.")
            continue
        
        # Classify tokens
        token_classifications = classify_tokens(tokens, original_question, question)
        
        # Calculate attention allocation per token type
        # We'll sum attention given by each token (row-wise sum)
        attention_per_token = np.sum(attention_matrix, axis=0)  # Sum across columns (keys)
        
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
        'model_path': model_path if model_path else 'default',
        'dataset': dataset_file,
        'num_samples': len(kmi_ratios),
        'kmi_mean': np.mean(kmi_ratios) if kmi_ratios else 0,
        'kmi_std': np.std(kmi_ratios) if kmi_ratios else 0,
        'ni_mean': np.mean(ni_ratios) if ni_ratios else 0,
        'ni_std': np.std(ni_ratios) if ni_ratios else 0,
        'oc_mean': np.mean(oc_ratios) if oc_ratios else 0,
        'oc_std': np.std(oc_ratios) if oc_ratios else 0
    }
    
    if model_type == "diffllama" and all_lambda_params:
        # Aggregate lambda param statistics (example: mean of lambda_std_dev if it were per-sample)
        # For array params like lambda_q1, etc., more complex aggregation might be needed if desired
        lambda_means = {}
        for p_name in ['lambda_q1', 'lambda_q2', 'lambda_k1', 'lambda_k2']:
            all_vals = [lp[p_name] for lp in all_lambda_params if p_name in lp and lp[p_name] is not None]
            if all_vals and isinstance(all_vals[0], np.ndarray):
                 # Stack arrays and take mean across samples, then mean of the vector
                try:
                    stacked_arrays = np.stack(all_vals)
                    lambda_means[f"{p_name}_mean_vector"] = convert_to_json_serializable(np.mean(stacked_arrays, axis=0)) # vector of means
                    lambda_means[f"{p_name}_overall_mean"] = float(np.mean(stacked_arrays)) # scalar mean
                except Exception as e:
                    print(f"Could not stack arrays for {p_name}: {e}")

        stats['aggregated_lambda_params_means'] = lambda_means
        # print(f"Aggregated Lambda Param Means (DiffLlama): {lambda_means}")

    # Convert all stats to JSON-serializable format
    stats = convert_to_json_serializable(stats)

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
                              num_samples=5, sft_model_paths=None):
    """
    Compare attention patterns between clean and noisy questions for both models.
    
    Args:
        clean_dataset: path to clean dataset
        noisy_dataset: path to noisy dataset
        num_samples: number of samples to analyze
        sft_model_paths: dict with model paths for fine-tuned models, e.g. {"llama": "path/to/llama_sft", "diffllama": "path/to/diffllama_sft"}
    """
    use_sft = sft_model_paths and len(sft_model_paths) > 0
    model_suffix = " (SFT)" if use_sft else ""
    print(f"Comparing attention patterns between models and datasets{model_suffix}...")
    
    results = {}
    
    for model_type in ["llama", "diffllama"]:
        results[model_type] = {}
        
        # Get model path (use SFT model if available, otherwise None for default)
        model_path = None
        if use_sft and model_type in sft_model_paths:
            model_path = sft_model_paths[model_type]
            print(f"Using fine-tuned model for {model_type}: {model_path}")
        else:
            print(f"Using default model for {model_type}")
        
        # Analyze clean data
        print(f"\nAnalyzing {model_type} on clean data...")
        clean_stats = quantify_attention_allocation(
            model_type, clean_dataset, num_samples, layer_idx=-1, head_idx=0, model_path=model_path
        )
        results[model_type]['clean'] = clean_stats
        
        # Analyze noisy data
        print(f"\nAnalyzing {model_type} on noisy data...")
        noisy_stats = quantify_attention_allocation(
            model_type, noisy_dataset, num_samples, layer_idx=-1, head_idx=0, model_path=model_path
        )
        results[model_type]['noisy'] = noisy_stats
    
    # Print comparison
    print("\n" + "="*80)
    print(f"ATTENTION ALLOCATION COMPARISON{model_suffix}")
    print("="*80)
    
    for model_type in ["llama", "diffllama"]:
        model_display = f"{model_type.upper()}{' (SFT)' if use_sft and model_type in sft_model_paths else ''}"
        print(f"\n{model_display} Model:")
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
        print("Running attention visualization on sample questions...")
    
    # Example visualizations
    print("Creating attention visualizations...")
    print("="*80)
    
    # Sample questions for visualization
    sample_questions = [
        "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. How many eggs does she sell at the farmers' market every day?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds?"
        "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"
        # "What is 15 + 27?"
    ]
    
    # Test both models
    for model_type in ["diffllama", "llama"]:
        print(f"\n{'='*60}")
        print(f"ANALYZING {model_type.upper()} MODEL")
        print(f"{'='*60}")
        
        for i, question in enumerate(sample_questions):
            print(f"\nVisualizing question {i+1} with {model_type}...")
            print(f"Question: {question[:80]}...")
            
            try:
                # Clean question
                visualize_sample_attention(
                    model_type, 
                    question, 
                    layer_idx=-1,  # Last layer
                    head_idx=0,    # First head
                    save_dir=f"results/attention_maps/{model_type}/clean_q{i+1}"
                )
                
                # For DiffLlama, also test different layers and heads
                if model_type == "diffllama":
                    print(f"  Additional DiffLlama analysis...")
                    
                    # Test middle layer
                    visualize_sample_attention(
                        model_type, 
                        question, 
                        layer_idx=8,   # Middle layer
                        head_idx=0,
                        save_dir=f"results/attention_maps/{model_type}/clean_q{i+1}_middle"
                    )
                    
                    # Test different attention head
                    visualize_sample_attention(
                        model_type, 
                        question, 
                        layer_idx=-1,
                        head_idx=1,    # Second head
                        save_dir=f"results/attention_maps/{model_type}/clean_q{i+1}_head1"
                    )
                
                # Noisy versions (if noise injection is available)
                try:
                    noisy_question = inject_inf_noise(question)
                    print(f"  Creating noisy version analysis...")
                    visualize_sample_attention(
                        model_type, 
                        noisy_question, 
                        layer_idx=-1,
                        head_idx=0,
                        save_dir=f"results/attention_maps/{model_type}/noisy_q{i+1}"
                    )
                except Exception as noise_error:
                    print(f"  Noise injection failed: {noise_error}")
                
            except Exception as e:
                print(f"  Error visualizing {model_type} on question {i+1}: {e}")
                import traceback
                traceback.print_exc()
    
    # Quantitative analysis (if datasets exist)
    print(f"\n{'='*80}")
    print("QUANTITATIVE ATTENTION ANALYSIS")
    print(f"{'='*80}")
    
    if os.path.exists("data/gsm8k_test.jsonl"):
        print("Running quantitative attention analysis...")
        
        try:
            attention_results = compare_attention_patterns(
                clean_dataset="data/gsm8k_test.jsonl",
                noisy_dataset="data/gsm8k_inf_test.jsonl" if os.path.exists("data/gsm8k_inf_test.jsonl") else "data/gsm8k_test.jsonl",
                num_samples=5  # Small sample for testing
            )
            
            # Save results
            import json
            with open("results/attention_analysis.json", "w") as f:
                json.dump(attention_results, f, indent=2)
            print("Attention analysis results saved to results/attention_analysis.json")
            
        except Exception as e:
            print(f"Quantitative analysis failed: {e}")
    else:
        print("Skipping quantitative analysis - datasets not found")
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("ATTENTION ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    try:
        # Check what files were created
        result_dirs = []
        for root, dirs, files in os.walk("results/attention_maps"):
            if files:
                result_dirs.append(root)
        
        print(f"Generated attention visualizations in {len(result_dirs)} directories:")
        for dir_path in sorted(result_dirs)[:10]:  # Show first 10
            file_count = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
            print(f"  {dir_path}: {file_count} visualizations")
        
        if len(result_dirs) > 10:
            print(f"  ... and {len(result_dirs) - 10} more directories")
        
        # DiffLlama specific summary
        diffllama_dirs = [d for d in result_dirs if 'diffllama' in d]
        if diffllama_dirs:
            print(f"\nDiffLlama-specific analyses:")
            print(f"  Generated visualizations in {len(diffllama_dirs)} DiffLlama directories")
            print(f"  Look for lambda parameter analysis")
        
        print(f"\nKey files to examine:")
        print(f"  - Individual attention maps: {model_type}_attn_layer*.png")
        
    except Exception as e:
        print(f"Error generating summary: {e}")
    
    print(f"\n{'='*80}")
    print("🎉 ATTENTION VISUALIZATION AND ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("Check the results/attention_maps/ directory for visualizations")
    print("For DiffLlama models, lambda parameters (if found) will be reported in the console output and potentially in metadata.")
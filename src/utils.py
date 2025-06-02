import json
import re
from datasets import load_dataset

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def download_gsm8k():
    """Download GSM8K dataset and save to data folder."""
    print("Downloading GSM8K dataset...")
    
    # Ensure data directory exists
    import os
    os.makedirs("data", exist_ok=True)
    
    try:
        # Try the standard approach first
        dataset = load_dataset("gsm8k", "main")
        dataset['train'].to_json("data/gsm8k_train.jsonl")
        dataset['test'].to_json("data/gsm8k_test.jsonl")
        print("GSM8K dataset downloaded and saved to data/ folder.")
    except ValueError as e:
        if "Invalid pattern" in str(e):
            print("Standard download failed due to pattern issue. Trying alternative approach...")
            try:
                # Alternative approach with explicit configuration
                dataset = load_dataset("gsm8k", "main", trust_remote_code=True)
                dataset['train'].to_json("data/gsm8k_train.jsonl")
                dataset['test'].to_json("data/gsm8k_test.jsonl")
                print("GSM8K dataset downloaded and saved to data/ folder.")
            except Exception as e2:
                print(f"Alternative download also failed: {e2}")
                # Try downloading with different parameters
                try:
                    from datasets import load_dataset_builder
                    builder = load_dataset_builder("gsm8k", "main")
                    dataset = builder.as_dataset()
                    dataset['train'].to_json("data/gsm8k_train.jsonl") 
                    dataset['test'].to_json("data/gsm8k_test.jsonl")
                    print("GSM8K dataset downloaded using builder approach.")
                except Exception as e3:
                    print(f"All download methods failed. Last error: {e3}")
                    # Create empty files to prevent further download attempts
                    with open("data/gsm8k_train.jsonl", "w") as f:
                        f.write("")
                    with open("data/gsm8k_test.jsonl", "w") as f:
                        f.write("")
                    raise Exception("Failed to download GSM8K dataset. Please download manually.")
        else:
            raise e

def extract_answer_from_solution(solution_str):
    """Extract the final numerical answer from a GSM8K solution string."""
    # GSM8K answers usually in format "The answer is X" or similar
    # Try to find the final answer pattern
    patterns = [
        r"The answer is (\d+)",
        r"The final answer is (\d+)", 
        r"#### (\d+)",
        r"\*\*(\d+)\*\*",
        r"The answer: (\d+)",
        r"Answer: (\d+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, solution_str)
        if match:
            return match.group(1)
    
    # As fallback, try to extract the last number in the solution
    numbers = re.findall(r"\d+", solution_str)
    return numbers[-1] if numbers else None

def create_chain_of_thought_prompt(question):
    """Create a chain-of-thought prompt for math problems."""
    return f"""Question: {question}

Let's solve this step by step:

Answer:"""

def create_answer_extraction_prompt(question):
    """Create a simpler answer extraction prompt."""
    return f"""Question: {question}

Answer:"""
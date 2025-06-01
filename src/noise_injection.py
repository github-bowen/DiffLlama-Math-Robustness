import json
import random
import re
from src.utils import load_jsonl, save_jsonl

def inject_inf_noise(question):
    """
    Inject Irrelevant Numbers/Facts (INF) noise into the question.
    Adds completely unrelated information with numbers or facts.
    """
    noises = [
        "Her cat is 3 years old and weighs 5 kilograms.",
        "The local library has 10,000 books on its shelves.",
        "He bought 2 apples and 3 oranges yesterday.",
        "The temperature outside was 25 degrees Celsius.",
        "There are 7 days in a week and 12 months in a year.",
        "The parking lot has 150 spaces available.",
        "She read 4 chapters of her favorite book last night.",
        "The store opens at 9 AM and closes at 8 PM.",
        "His bicycle has 2 wheels and weighs 15 pounds.",
        "The movie lasted for 120 minutes and had 5 main characters."
    ]
    
    noise = random.choice(noises)
    
    # Find a good insertion point
    sentences = question.split('.')
    if len(sentences) > 1:
        # Insert noise in the middle of the question
        insert_idx = random.randint(0, len(sentences) - 2)
        sentences.insert(insert_idx + 1, " " + noise.strip())
        return '.'.join(sentences)
    else:
        # If no sentences, just append
        return question + " " + noise

def inject_rcs_noise(question):
    """
    Inject Redundant Calculation Steps (RCS) noise into the question.
    Adds calculations or steps that seem relevant but don't affect the answer.
    """
    noises = [
        "First, let's note that there are 100 cents in a dollar.",
        "Remember that there are 60 minutes in an hour and 24 hours in a day.",
        "We should keep in mind that multiplication is commutative.",
        "It's worth noting that 10 times 10 equals 100.",
        "Consider that the sum of any number and zero is the number itself.",
        "We can observe that 5 plus 5 equals 10, which might be useful.",
        "Note that dividing by 1 doesn't change the value.",
        "Keep in mind that there are 4 quarters in a whole.",
        "Remember that half of 20 is 10, which is a basic fact.",
        "We should consider that 3 times 3 equals 9."
    ]
    
    noise = random.choice(noises)
    
    sentences = question.split('.')
    if len(sentences) > 1:
        insert_idx = random.randint(0, len(sentences) - 2)
        sentences.insert(insert_idx + 1, " " + noise.strip())
        return '.'.join(sentences)
    else:
        return question + " " + noise

def inject_sd_noise(question):
    """
    Inject Semantic Distraction (SD) noise into the question.
    Adds information related to the problem context but irrelevant to the solution.
    """
    # Extract context clues from the question to make relevant distractions
    context_mapping = {
        'shop|store|buy|sell|price|cost|dollar': [
            "The shop also sells various other items at different prices.",
            "The store manager mentioned they might have a sale next week.",
            "Other customers were buying different items in the same store.",
            "The cashier was friendly and provided excellent customer service."
        ],
        'school|student|class|teacher|homework': [
            "The teacher also assigned reading homework for tomorrow.",
            "Other students in the class were working on different subjects.",
            "The school cafeteria serves lunch at noon every day.",
            "The classroom has 30 desks and large windows facing south."
        ],
        'cook|bake|recipe|kitchen|food': [
            "The kitchen also has other cooking utensils and appliances.",
            "She learned this recipe from her grandmother years ago.",
            "The oven temperature needs to be preheated before baking.",
            "Other family members enjoy different types of food."
        ],
        'car|drive|travel|road|mile': [
            "The car also has a GPS system and air conditioning.",
            "Traffic conditions can vary significantly throughout the day.",
            "The road has several scenic viewpoints along the way.",
            "Other drivers were traveling at different speeds on the highway."
        ],
        'book|read|page|chapter|story': [
            "The book also contains illustrations and a detailed index.",
            "She enjoys reading different genres of literature.",
            "The library has thousands of books on various topics.",
            "Other readers have given positive reviews about this book."
        ]
    }
    
    # Find matching context
    selected_noise = "The weather was sunny and pleasant that day."  # default
    for pattern, noise_list in context_mapping.items():
        if re.search(pattern, question.lower()):
            selected_noise = random.choice(noise_list)
            break
    
    sentences = question.split('.')
    if len(sentences) > 1:
        insert_idx = random.randint(0, len(sentences) - 2)
        sentences.insert(insert_idx + 1, " " + selected_noise.strip())
        return '.'.join(sentences)
    else:
        return question + " " + selected_noise

def generate_noisy_datasets(original_test_file="data/gsm8k_test.jsonl"):
    """
    Generate all three types of noisy datasets from the original test file.
    """
    print(f"Loading original dataset from {original_test_file}...")
    original_data = load_jsonl(original_test_file)
    
    print(f"Generating noisy datasets from {len(original_data)} samples...")
    
    noisy_data_inf = []
    noisy_data_rcs = []
    noisy_data_sd = []
    
    for i, item in enumerate(original_data):
        question = item['question']
        answer = item['answer']  # Keep answer unchanged
        
        # Generate INF noise
        noisy_q_inf = inject_inf_noise(question)
        noisy_data_inf.append({
            'question': noisy_q_inf, 
            'answer': answer,
            'original_question': question,
            'noise_type': 'INF'
        })
        
        # Generate RCS noise
        noisy_q_rcs = inject_rcs_noise(question)
        noisy_data_rcs.append({
            'question': noisy_q_rcs, 
            'answer': answer,
            'original_question': question,
            'noise_type': 'RCS'
        })
        
        # Generate SD noise
        noisy_q_sd = inject_sd_noise(question)
        noisy_data_sd.append({
            'question': noisy_q_sd, 
            'answer': answer,
            'original_question': question,
            'noise_type': 'SD'
        })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(original_data)} samples...")
    
    # Save noisy datasets
    save_jsonl(noisy_data_inf, "data/gsm8k_inf_test.jsonl")
    save_jsonl(noisy_data_rcs, "data/gsm8k_rcs_test.jsonl")
    save_jsonl(noisy_data_sd, "data/gsm8k_sd_test.jsonl")
    
    print("Noisy datasets generated successfully:")
    print(f"  - INF: data/gsm8k_inf_test.jsonl ({len(noisy_data_inf)} samples)")
    print(f"  - RCS: data/gsm8k_rcs_test.jsonl ({len(noisy_data_rcs)} samples)")
    print(f"  - SD: data/gsm8k_sd_test.jsonl ({len(noisy_data_sd)} samples)")

def create_sample_comparison(original_test_file="data/gsm8k_test.jsonl", num_samples=5):
    """
    Create a sample comparison showing original vs noisy questions.
    """
    original_data = load_jsonl(original_test_file)[:num_samples]
    
    print("Sample Noise Injection Examples:")
    print("=" * 80)
    
    for i, item in enumerate(original_data):
        question = item['question']
        print(f"\nExample {i+1}:")
        print(f"Original: {question}")
        print(f"INF:      {inject_inf_noise(question)}")
        print(f"RCS:      {inject_rcs_noise(question)}")
        print(f"SD:       {inject_sd_noise(question)}")
        print("-" * 80)

if __name__ == "__main__":
    import os
    
    # Check if original test file exists
    if not os.path.exists("data/gsm8k_test.jsonl"):
        print("GSM8K test file not found. Please run data download first.")
        print("You can run: python -c 'from src.utils import download_gsm8k; download_gsm8k()'")
    else:
        # Generate sample comparisons first
        print("Creating sample noise injection examples...")
        create_sample_comparison()
        
        # Generate full noisy datasets
        print("\nGenerating full noisy datasets...")
        generate_noisy_datasets() 
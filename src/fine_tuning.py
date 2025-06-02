import torch
import os
from transformers import (
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer
)
from datasets import Dataset
import json
from src.model_loader import load_model_and_tokenizer, get_model_paths
from src.utils import load_jsonl

def prepare_training_data(train_file="data/gsm8k_train.jsonl", max_samples=1000):
    """
    Prepare training data in the format needed for language model fine-tuning.
    
    Args:
        train_file: path to training JSONL file
        max_samples: maximum number of samples to use for training
    
    Returns:
        list of formatted training examples
    """
    print(f"Loading training data from {train_file}...")
    
    if not os.path.exists(train_file):
        print(f"Training file {train_file} not found!")
        return []
    
    train_data = load_jsonl(train_file)
    
    if max_samples and len(train_data) > max_samples:
        train_data = train_data[:max_samples]
        print(f"Limited training data to {max_samples} samples")
    
    formatted_examples = []
    
    for item in train_data:
        question = item['question']
        answer = item['answer']
        
        # Format as instruction-following format
        # The model should learn to generate the complete solution
        text = f"Question: {question}\n\nAnswer: {answer}<|endoftext|>"
        
        formatted_examples.append({"text": text})
    
    print(f"Prepared {len(formatted_examples)} training examples")
    return formatted_examples

def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokenize the training examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

def fine_tune_model(model_type, train_file="data/gsm8k_train.jsonl", 
                   output_dir_base="./models_finetuned", 
                   max_samples=1000,
                   num_epochs=3,
                   batch_size=8,
                   learning_rate=2e-5):
    """
    Fine-tune a model on GSM8K training data.
    
    Args:
        model_type: "diffllama" or "llama"
        train_file: path to training data
        output_dir_base: base directory for saving fine-tuned models
        max_samples: maximum number of training samples
        num_epochs: number of training epochs
        batch_size: training batch size
        learning_rate: learning rate for training
        
    Returns:
        path to fine-tuned model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Fine-tuning {model_type} model...")
    print(f"Device: {device}")
    print(f"Training samples: {max_samples}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    # Create output directory
    output_dir = os.path.join(output_dir_base, f"{model_type}_sft")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_type, device, for_training=True)
    
    # Prepare training data
    training_examples = prepare_training_data(train_file, max_samples)
    if not training_examples:
        print("No training data available. Skipping fine-tuning.")
        return None
    
    # Create dataset
    train_dataset = Dataset.from_list(training_examples)
    
    # Tokenize dataset
    print("Tokenizing training data...")
    tokenized_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,  # Effective batch size = batch_size * 2
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        fp16=False,  # Disable FP16
        max_grad_norm=1.0,  # Add gradient clipping
        dataloader_pin_memory=False,
        report_to="none",  # Disable wandb/tensorboard
        optim="adamw_torch",  # Explicitly set optimizer
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        # eval_dataset=tokenized_val_dataset,  # If you have validation data
    )
    
    print(f"Starting fine-tuning for {model_type}...")
    
    try:
        # Start training
        trainer.train()
        
        print(f"Fine-tuning completed for {model_type}")
        
        # Save the final model
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"Fine-tuned model saved to {output_dir}")
        
        # Save training info
        training_info = {
            "model_type": model_type,
            "train_file": train_file,
            "max_samples": max_samples,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "output_dir": output_dir
        }
        
        with open(os.path.join(output_dir, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        output_dir = None
    
    finally:
        # Clean up GPU memory
        del model, trainer
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return output_dir

def create_training_subset(original_file="data/gsm8k_train.jsonl", 
                          output_file="data/gsm8k_train_sft.jsonl",
                          num_samples=500):
    """
    Create a subset of the training data for SFT experiments.
    
    Args:
        original_file: path to original training file
        output_file: path to save subset
        num_samples: number of samples to include
    """
    if not os.path.exists(original_file):
        print(f"Original training file {original_file} not found!")
        return False
    
    print(f"Creating training subset with {num_samples} samples...")
    
    train_data = load_jsonl(original_file)
    
    if len(train_data) < num_samples:
        print(f"Warning: Original file has only {len(train_data)} samples, using all")
        subset_data = train_data
    else:
        # Take first num_samples for reproducibility
        subset_data = train_data[:num_samples]
    
    # Save subset
    with open(output_file, "w") as f:
        for item in subset_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Training subset saved to {output_file} with {len(subset_data)} samples")
    return True

def run_full_sft_pipeline(max_train_samples=500, num_epochs=2):
    """
    Run the complete SFT pipeline for both models.
    
    Args:
        max_train_samples: maximum number of training samples to use
        num_epochs: number of training epochs
        
    Returns:
        dict with paths to fine-tuned models
    """
    print("=" * 80)
    print("STARTING SUPERVISED FINE-TUNING PIPELINE")
    print("=" * 80)
    
    # Check if training data exists
    if not os.path.exists("data/gsm8k_train.jsonl"):
        print("Training data not found. Please download GSM8K dataset first.")
        return {}
    
    # Create training subset (Overwrite if exists)
    sft_train_file = "data/gsm8k_train_sft.jsonl"
    create_training_subset(
        original_file="data/gsm8k_train.jsonl",
        output_file=sft_train_file,
        num_samples=max_train_samples
    )
        
    
    sft_model_paths = {}
    
    # Fine-tune both models
    for model_type in ["llama", "diffllama"]:
        print(f"\n{'='*20} FINE-TUNING {model_type.upper()} {'='*20}")
        
        try:
            output_path = fine_tune_model(
                model_type=model_type,
                train_file=sft_train_file,
                max_samples=max_train_samples,
                num_epochs=num_epochs,
                batch_size=8,  # Small batch size for memory efficiency
                learning_rate=5e-5
            )
            
            if output_path:
                sft_model_paths[model_type] = output_path
                print(f"✓ {model_type} fine-tuning completed: {output_path}")
            else:
                print(f"✗ {model_type} fine-tuning failed")
                
        except Exception as e:
            print(f"✗ Error fine-tuning {model_type}: {e}")
    
    print("\n" + "=" * 80)
    print("SFT PIPELINE RESULTS")
    print("=" * 80)
    
    for model_type, path in sft_model_paths.items():
        print(f"{model_type}: {path}")
    
    return sft_model_paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune models on GSM8K")
    parser.add_argument("--model", choices=["llama", "diffllama", "both"], 
                       default="both", help="Which model(s) to fine-tune")
    parser.add_argument("--samples", type=int, default=500, 
                       help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=2, 
                       help="Number of training epochs")
    parser.add_argument("--test", action="store_true", 
                       help="Run with minimal samples for testing")
    
    args = parser.parse_args()
    
    if args.test:
        print("Running in test mode with minimal samples...")
        max_samples = 10
        epochs = 1
    else:
        max_samples = args.samples
        epochs = args.epochs
    
    if args.model == "both":
        # Run full pipeline
        sft_paths = run_full_sft_pipeline(max_samples, epochs)
    else:
        # Fine-tune single model
        train_file = "data/gsm8k_train_sft.jsonl"
        if not os.path.exists(train_file):
            create_training_subset(num_samples=max_samples)
        
        output_path = fine_tune_model(
            model_type=args.model,
            train_file=train_file,
            max_samples=max_samples,
            num_epochs=epochs
        )
        
        if output_path:
            print(f"Fine-tuned model saved to: {output_path}")
        else:
            print("Fine-tuning failed") 
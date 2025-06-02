# DiffLlama vs Llama Experiment Usage Guide

This guide provides detailed instructions on how to use the DiffLlama vs Llama noise robustness experiment framework, including the complete workflow for setup, execution, and result analysis.

## 🎯 Quick Start

### 1. Environment Check

First, verify that the environment is properly configured:

```bash
# Run environment test
python -m scripts.test_setup

# Quick test mode
python -m scripts.test_setup --quick
```

### 2. Download Models (if needed)

```bash
# Download models required for experiments
python -m scripts.download_models
```

### 3. Inspect Model Structure and Attention Patterns (Optional)

Models used in this step are zero-shot models, which can be downloaded in the previous step.

```bash
python -m scripts.inspect_model_structure --model llama
python -m scripts.inspect_model_structure --model diffllama
```

```bash
python -m scripts.test_llama_attention
python -m scripts.test_diffllama_attention
```

### 4. Run Quick Experiment

```bash
# Run quick test with 20 samples
python -m main --quick-test
```

## 🔧 Detailed Usage Instructions

### Command Line Arguments

#### Main Experiment Script (`main.py`)

```bash
python -m main [OPTIONS]
```

**Main Options:**

- `--quick-test`: Run quick test (20 samples)
- `--max-samples N`: Limit maximum number of samples per dataset
- `--skip-data-gen`: Skip data generation (use existing data)
- `--skip-zero-shot`: Skip zero-shot evaluation
- `--skip-sft`: Skip supervised fine-tuning
- `--skip-attention`: Skip attention analysis
- `--models MODEL1,MODEL2`: Specify models to test (default: diffllama,llama)
- `--datasets DATASET1,DATASET2`: Specify datasets to test (default: clean,inf,rcs,sd)

**Example Usage:**

```bash
# Full experiment
python -m main

# Custom sample count
python -m main --max-samples 100

# Skip zero-shot evaluation step
python -m main --skip-zero-shot

# Skip time-consuming fine-tuning step
python -m main --skip-sft

# Test specific datasets only
python -m main --datasets clean,inf

# Test specific models only
python -m main --models diffllama
```

For a specific setup (50 samples for evaluation, 100 samples for SFT, 1 epoch for SFT)

```bash
## Evaluation only
python -m main --max-samples 50 --sft-samples 100 --sft-epochs 1 --skip-sft --skip-attention

## SFT only
python -m main --max-samples 50 --sft-samples 100 --sft-epochs 1 --skip-zero-shot --skip-attention

## Attention Analysis only
python -m main --max-samples 50 --sft-samples 100 --sft-epochs 1 --skip-sft --skip-zero-shot

## Evaluation + SFT
python -m main --max-samples 50 --sft-samples 100 --sft-epochs 1 --skip-attention

## SFT + Attention Analysis
python -m main --max-samples 50 --sft-samples 100 --sft-epochs 1 --skip-zero-shot

## All steps: Evaluation + SFT + Attention Analysis
python -m main --max-samples 50 --sft-samples 100 --sft-epochs 1
```

SFT on all samples:
```bash
## SFT + Attention Analysis
python -m main --max-samples 50 --sft-samples 7473 --sft-epochs 1 --skip-zero-shot
```

#### Colab Experiment Script (`colab/experiment.py`)

```bash
python -m colab.experiment [OPTIONS]
```

**Colab-Specific Options:**

- `--mode {quick,medium,full}`: Experiment mode
- `--max-samples N`: Limit number of evaluation samples
- `--enable-sft`: Enable supervised fine-tuning (disabled by default)
- `--sft-samples N`: Number of samples for fine-tuning
- `--sft-epochs N`: Number of epochs for fine-tuning
- `--skip-zero-shot`: Skip zero-shot evaluation to save time
- `--skip-attention`: Skip attention analysis to save time
- `--setup`: Only perform environment setup
- `--instructions`: Show usage instructions

**Example Usage:**

```bash
# Different experiment modes
!python -m colab.experiment --mode quick    # Quick (20 samples)
!python -m colab.experiment --mode medium  # Medium (100 samples)  
!python -m colab.experiment --mode full    # Complete (custom)

# Environment setup only
!python -m colab.experiment --setup

# Skip specific steps
!python -m colab.experiment --mode medium --skip-zero-shot --enable-sft
!python -m colab.experiment --mode medium --skip-attention
```

For `full` mode:

```bash
# Full experiment (run only when you have enough time)
## Evaluation only
!python -m colab.experiment --mode full --skip-attention

## SFT only
!python -m colab.experiment --mode full --skip-zero-shot --enable-sft --skip-attention

## Attention Analysis only
!python -m colab.experiment --mode full --skip-zero-shot

## Evaluation + SFT
!python -m colab.experiment --mode full --enable-sft --skip-attention

## SFT + Attention Analysis
!python -m colab.experiment --mode full --skip-zero-shot --enable-sft

## All steps: Evaluation + SFT + Attention Analysis
!python -m colab.experiment --mode full --enable-sft
```

### Running Modules Independently

#### Data Generation

```python
from src.utils import download_gsm8k
from src.noise_injection import generate_noisy_datasets

# Download original data
download_gsm8k()

# Generate noisy datasets
generate_noisy_datasets()
```

#### Model Evaluation

```python
from src.evaluation import run_comprehensive_evaluation

# Run evaluation
results_df, detailed_results = run_comprehensive_evaluation(
    models=['diffllama', 'llama'],
    datasets=['clean', 'inf'],
    max_samples_per_dataset=50
)
```

#### Attention Analysis

```python
from src.attention_visualizer import compare_attention_patterns

# Compare attention patterns
results = compare_attention_patterns(
    clean_dataset="data/gsm8k_test.jsonl",
    noisy_dataset="data/gsm8k_inf_test.jsonl",
    num_samples=10
)
```

### Adding New Noise Type

Implement new noise injection function:

```python
# src/noise_injection.py
def inject_custom_noise(question):
    """Custom noise injection function"""
    # Implement your noise injection logic
    return modified_question

# In generation script
def generate_custom_noisy_dataset():
    """Generate custom noisy dataset"""
    # Use your noise function to generate dataset
    pass
```

## Llama and DiffLlama Attention Visualizer

### Running the Main Attention Visualizer

The primary script for visualizing attention is `src/attention_visualizer.py`. Run it from the project root directory:

```bash
python -m src.attention_visualizer
```

-   This command will generate visualizations for predefined sample questions within the script.
-   If `data/gsm8k_test.jsonl` is present (and the script is configured to use it), it may also process questions from this dataset.
-   Output maps are saved in `results/attention_maps/`.
-   For DiffLlama, observe the console output for reported lambda parameters and other metadata.

### Running Attention Visualizer Tests

Test scripts are provided in the `scripts/` directory to verify the functionality. Run these from the project root directory.

1.  **Test Llama Attention Visualization:**
    ```bash
    python -m scripts.test_llama_attention
    ```
    This script loads a standard Llama model, extracts attention scores, and generates sample visualizations in `test_results/llama_test/`.

2.  **Test DiffLlama Attention Visualization and Comparison:**
    ```bash
    python -m scripts.test_diffllama_attention
    ```
    This script tests DiffLlama specific attention extraction (including lambda parameters), compares its attention with a standard Llama model, and generates sample visualizations in `test_results/diffllama_test/`.

Always check the console output for test results (PASSED/FAILED) and any error messages.
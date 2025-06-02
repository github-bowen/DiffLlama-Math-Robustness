# DiffLlama vs Llama Experiment Usage Guide

This guide provides detailed instructions on how to use the DiffLlama vs Llama noise robustness experiment framework, including the complete workflow for setup, execution, and result analysis.

## 🎯 Quick Start

### 1. Environment Check

First, verify that the environment is properly configured:

```bash
# Run environment test
python scripts/test_setup.py

# Quick test mode
python scripts/test_setup.py --quick
```

### 2. Download Models (if needed)

```bash
# Download models required for experiments
python scripts/download_models.py
```

### 3. Run Quick Experiment

```bash
# Run quick test with 20 samples
python main.py --quick-test
```

## 🔧 Detailed Usage Instructions

### Command Line Arguments

#### Main Experiment Script (`main.py`)

```bash
python main.py [OPTIONS]
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
python main.py

# Custom sample count
python main.py --max-samples 100

# Skip zero-shot evaluation step
python main.py --skip-zero-shot

# Skip time-consuming fine-tuning step
python main.py --skip-sft

# Test specific datasets only
python main.py --datasets clean,inf

# Test specific models only
python main.py --models diffllama
```

#### Colab Experiment Script (`colab/experiment.py`)

```bash
python colab/experiment.py [OPTIONS]
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
# Colab quick test
python colab/experiment.py --mode quick

# Colab medium experiment
python colab/experiment.py --mode medium --max-samples 100

# Colab experiment with SFT
python colab/experiment.py --mode medium --enable-sft --sft-samples 200

# Skip zero-shot evaluation
python colab/experiment.py --mode medium --skip-zero-shot --enable-sft

# Environment setup only
python colab/experiment.py --setup
```

### Experiment Modes

#### 1. Quick Test Mode

- **Purpose**: Verify environment configuration and code correctness
- **Sample Count**: 20 per dataset
- **Runtime**: 30-60 minutes
- **Command**: `python main.py --quick-test`

#### 2. Standard Mode

- **Purpose**: Balanced experimental results
- **Sample Count**: 200 per dataset (customizable)
- **Runtime**: 2-4 hours
- **Command**: `python main.py`

#### 3. Full Mode

- **Purpose**: Complete research results
- **Sample Count**: All data (~1300 samples)
- **Runtime**: 6-12 hours
- **Command**: `python main.py --max-samples -1`

## 📊 Dataset Description

### Original Dataset

- **Clean**: GSM8K original test set
- **Size**: 1,319 math problems

### Noisy Datasets

- **INF** (Irrelevant Numbers/Facts): Add irrelevant numbers and facts
- **RCS** (Redundant Calculation Steps): Add redundant calculation steps
- **SD** (Semantic Distraction): Add semantic distraction information

Each noisy dataset is generated based on the original Clean dataset.

## 🤖 Model Description

### DiffLlama-375M

- **Type**: Llama variant based on differential attention mechanism
- **Parameters**: 375M
- **Features**: Specialized differential attention mechanism

### Llama-375M

- **Type**: Standard Llama architecture
- **Parameters**: 375M
- **Features**: Traditional attention mechanism

## 📈 Result Analysis

### Output Files

After experiment completion, results are saved in the `results/` directory:

```
results/
├── experiment_results_[timestamp].csv     # 主要性能结果
├── detailed_results_[timestamp].json      # 详细结果
├── attention_analysis_[timestamp].json    # 注意力分析
├── model_comparison_[timestamp].png       # 性能对比图
└── attention_maps/                        # 注意力热力图
    ├── clean_samples/
    ├── inf_samples/
    ├── rcs_samples/
    └── sd_samples/
```

### Performance Metrics

#### Pass@1 Accuracy

- The proportion of models that give the correct answer on their first attempt
- Main evaluation metric

#### Attention Analysis Metrics

- **KMI Ratio**: Proportion of attention focused on key mathematical information
- **NI Ratio**: Proportion of attention focused on noise information
- **OC Ratio**: Proportion of attention focused on other content

### Interpreting Results

#### Example Performance Comparison Table

```
           Clean    INF      RCS      SD
llama      0.145    0.098    0.110    0.105
diffllama  0.162    0.123    0.135    0.128
```

**Interpretation:**

- DiffLlama outperforms Llama on all datasets
- Noise significantly reduces the performance of both models
- DiffLlama's performance drop is smaller on noisy data

#### Example Attention Analysis

```json
{
  "llama": {
    "clean": {"kmi_ratio": 0.45, "ni_ratio": 0.0, "oc_ratio": 0.55},
    "inf": {"kmi_ratio": 0.32, "ni_ratio": 0.18, "oc_ratio": 0.50}
  },
  "diffllama": {
    "clean": {"kmi_ratio": 0.50, "ni_ratio": 0.0, "oc_ratio": 0.50},
    "inf": {"kmi_ratio": 0.43, "ni_ratio": 0.12, "oc_ratio": 0.45}
  }
}
```

**Interpretation:**

- DiffLlama focuses better on key mathematical information (KMI)
- When facing noise, DiffLlama's attention dispersion is smaller
- Proves the effectiveness of the differential attention mechanism

## 🛠 Advanced Usage

### Custom Configuration

Edit configuration file to customize experiment parameters:

```python
# src/config.py (if exists)
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.1,
    "top_p": 0.95,
    "do_sample": False
}

EVALUATION_CONFIG = {
    "batch_size": 8,
    "max_samples_per_dataset": 200
}
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

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Memory Not Enough

```bash
# Solution: Reduce batch size or sample count
python main.py --quick-test --max-samples 10
```

#### 2. Model Download Failure

```bash
# Manually download model
python scripts/download_models.py

# Check network connection and Hugging Face access
```

#### 3. Import Error

```bash
# Check environment configuration
python scripts/test_setup.py

# Install missing dependencies
pip install -r requirements.txt
```

#### 4. Data Generation Failure

```bash
# Check network connection and data directory permissions
ls -la data/

# Manually download GSM8K
python -c "from src.utils import download_gsm8k; download_gsm8k()"
```

### Performance Optimization

#### GPU Optimization

```python
# Enable mixed precision training (if supported)
import torch
torch.backends.cudnn.benchmark = True
```

#### Memory Optimization

```python
# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Use gradient checkpointing (if applicable)
```

## 📚 Extended Experiments

### Adding New Model

1. Add model loading logic in `src/model_loader.py`
2. Update configuration to include new model
3. Run experiment: `python main.py --models your_model,diffllama,llama`

### Adding New Dataset

1. Add data download function in `src/utils.py`
2. Add corresponding noise generation in `src/noise_injection.py`
3. Update evaluation process to support new dataset

### Custom Evaluation Metrics

1. Add new evaluation function in `src/evaluation.py`
2. Modify result aggregation logic
3. Update result visualization

## 🎓 Academic Usage

### Citation Format

If using this framework in academic work, please consider citing related papers and datasets.

### Experiment Reproducibility

To ensure experiment reproducibility:

1. Record the model version used
2. Save random seed settings
3. Record hardware configuration information
4. Save complete result files

---

**More Help:**

- Check `README.md` for project overview
- Check `colab/README.md` for Colab usage
- Run `python main.py --help` to view all options

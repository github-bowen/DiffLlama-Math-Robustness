# 🔬 Google Colab Experiment Guide

This guide provides detailed instructions for running experiments in Google Colab.

## 🚀 Quick Start

### First Run (No Cache)
```python
# 1. Mount Google Drive (recommended)
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone repository
!git clone https://github.com/github-bowen/DiffLlama-Math-Robustness.git
!cd DiffLlama-Math-Robustness

# 3. Quick test
!python -m colab.experiment --mode quick

# 4. Medium scale experiment (if quick test succeeds)
!python -m colab.experiment --mode medium

# 5. View results
import pandas as pd
df = pd.read_csv('results/colab_results_[latest_timestamp].csv')
print(df.pivot(index='model', columns='dataset', values='accuracy'))
```

### Subsequent Runs (With Cache)
```python
# Run experiment directly (models already in Drive)
!python -m colab.experiment --mode medium --skip-attention
```

## 🎯 Expected Results

### Performance Comparison Table
```
           Clean    INF    RCS     SD
llama      0.123   0.098  0.110  0.105
diffllama  0.145   0.123  0.135  0.128
```

### Attention Analysis
```json
{
  "llama": {
    "clean": {"kmi_mean": 0.45, "ni_mean": 0.0, "oc_mean": 0.55},
    "noisy": {"kmi_mean": 0.35, "ni_mean": 0.15, "oc_mean": 0.50}
  },
  "diffllama": {
    "clean": {"kmi_mean": 0.50, "ni_mean": 0.0, "oc_mean": 0.50},
    "noisy": {"kmi_mean": 0.45, "ni_mean": 0.10, "oc_mean": 0.45}
  }
}
```

## 📚 Advanced Usage

### Custom Parameters
```bash
# Limit evaluation samples
!python -m colab.experiment --mode medium --max-samples 50

# Skip zero-shot evaluation
!python -m colab.experiment --mode medium --skip-zero-shot --enable-sft

# Skip attention analysis
!python -m colab.experiment --mode full --skip-attention

# Enable supervised fine-tuning
!python -m colab.experiment --mode medium --enable-sft --sft-samples 200

# Run setup only
!python -m colab.experiment --setup
```

### Manual Module Running
```python
# Run data generation separately
from src.utils import download_gsm8k
from src.noise_injection import generate_noisy_datasets

download_gsm8k()
generate_noisy_datasets()

# Run evaluation separately
from src.evaluation import run_comprehensive_evaluation
results = run_comprehensive_evaluation(max_samples_per_dataset=20)
```

## 💡 Best Practices

1. **Run quick test first**: Verify all components work normally
2. **Use Google Drive**: Ensure data and results persistence
3. **Monitor runtime**: Colab has time limits, save progress regularly
4. **Check memory regularly**: Avoid crashes due to memory overflow
5. **Download important results**: Save key results locally before session ends
6. **Use skip options**: Use `--skip-zero-shot` or `--skip-attention` to save time
7. **Start with small samples**: Use `--max-samples` to test with smaller datasets first

## 🔧 Troubleshooting

### Common Issues

#### GPU Memory Insufficient
```bash
# Use smaller sample sizes
!python -m colab.experiment --mode quick --max-samples 10
```

#### Model Download Failed
```bash
# Retry setup
!python -m colab.experiment --setup
```

#### Session Timeout
```bash
# Use skip options to reduce runtime
!python -m colab.experiment --mode medium --skip-attention --skip-zero-shot --enable-sft
```

## 🎓 Educational Use

This experiment framework is especially suitable for:
- **Course assignments**: Compare different model architectures' performance
- **Research projects**: Explore the role of attention mechanisms
- **Technical demonstrations**: Show the impact of noise on NLP models
- **Learning practice**: Understand modern Transformer models

## ⏱️ Runtime Estimates

- **Quick mode**: ~15-30 minutes
- **Medium mode**: ~1-2 hours  
- **Full mode**: ~3-6 hours
- **With SFT**: +30-60 minutes
- **Skip zero-shot**: Saves ~30-50% time

---

**Need help?** 

- 📖 Check main README.md for experiment background
- 🔧 Run `!python -m colab.experiment --help` to see all options
- 💬 Check detailed notes and instructions in the Notebook

**Good luck with your experiment!** 🎉 
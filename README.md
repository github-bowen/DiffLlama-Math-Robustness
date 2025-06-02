# DiffLlama vs Llama Noise Robustness Experiment

A research project on mathematical reasoning noise robustness based on differential attention mechanism, comparing the performance of [DiffLlama-375M](https://huggingface.co/reyllama/DiffLlama-375M) and [Llama-375M](https://huggingface.co/reyllama/Llama_375M) on noisy mathematical problems.

## 📋 Project Overview

This project implements a complete experimental framework for studying and comparing the noise robustness of different attention mechanisms in mathematical reasoning tasks. Through systematic experimental design, it deeply explores the advantages of DiffLlama's differential attention mechanism compared to traditional attention mechanisms.

### 🎯 Research Objectives

- **Performance Comparison**: Evaluate DiffLlama and Llama's performance on clean and noisy mathematical problems
- **Noise Robustness**: Analyze the impact of different types of noise on model performance
- **Attention Mechanism**: Visualize and quantify models' attention allocation patterns
- **In-depth Analysis**: Explore the working principles and advantages of differential attention mechanism

## 🏗️ Project Structure

```bash
DiffLlama_Experiment/
├── src/                                     # Core source code
│   ├── utils.py                             # Data download and processing tools
│   ├── model_loader.py                      # Model loading and configuration
│   ├── noise_injection.py                   # Three noise injection strategies
│   ├── evaluation.py                        # Evaluation and performance analysis
│   ├── fine_tuning.py                       # Supervised fine-tuning (optional)
│   └── attention_visualizer.py              # Attention visualization and analysis
├── colab/                                   # Google Colab specific
│   ├── experiment.py                        # Colab main experiment script
│   ├── config.py                            # Colab environment configuration
│   ├── README.md                            # Colab detailed guide
│   └── DiffLlama_Colab_Experiment.ipynb     # Main experiment notebook
├── scripts/                                 # Helper scripts
│   ├── download_models.py                   # Model download script
│   ├── test_setup.py                        # Environment configuration test
│   └── interactive_inference.py             # Interactive inference tool
├── docs/                                    # Documentation
│   ├── USAGE_GUIDE.md                       # Detailed usage instructions
│   └── PROJECT_SUMMARY.md                   # Project file organization
├── results/                                 # Experiment results
│   └── attention_maps/                      # Attention visualization outputs
├── models_finetuned/                        # Fine-tuned model storage
├── main.py                                  # Main experiment entry point
├── requirements.txt                         # Python dependencies
├── README.md                                # Project documentation
└── LICENSE                                  # License file
```

### Example of Noise Injection

```bash
Original: "A pizza is cut into 8 slices. If 3 slices are eaten, how many remain?"
Noisy: "A delicious pizza with cheese and pepperoni is cut into 8 equal slices. The pizza smells great. If 3 slices are eaten quickly, how many slices remain on the plate?"
```

### Evaluation Metrics

- **Pass@1 Accuracy**: Model's accuracy on first attempt
- **Attention Allocation Ratio**: Attention proportion for KMI/NI/OC tokens
- **Robustness Degradation**: Performance change from clean to noisy data

## 📊 Experiment Modes

### 🏃 Quick Test Mode

- **Purpose**: Verify environment and code correctness
- **Samples**: 20 per dataset
- **Time**: 30-60 minutes
- **Command**: `python main.py --quick-test`

### 📈 Standard Mode  

- **Purpose**: Balanced experimental results
- **Samples**: 200 per dataset
- **Time**: 2-4 hours
- **Command**: `python main.py`

### 🔬 Complete Mode

- **Purpose**: Complete research data
- **Samples**: All data (~1300)
- **Time**: 6-12 hours  
- **Command**: `python main.py --max-samples -1`

## 🎛️ Advanced Configuration

### Custom Experiment Parameters

```bash
# Specify sample count
python main.py --max-samples 100

# Skip time-consuming fine-tuning step
python main.py --skip-sft

# Test specific model only
python main.py --models diffllama

# Test specific datasets only
python main.py --datasets clean,inf

# Skip attention analysis to save time
python main.py --skip-attention
```

### Google Colab Specific Options

```bash
# Different experiment modes
!python colab/experiment.py --mode quick    # Quick (20 samples)
!python colab/experiment.py --mode medium  # Medium (100 samples)  
!python colab/experiment.py --mode full    # Complete (custom)

# Google Drive integration
!python colab/experiment.py --use-drive    # Persistent storage

# Environment setup only
!python colab/experiment.py --setup
```

## 📈 Result Analysis

### Output Files

```bash
results/
├── experiment_results_[timestamp].csv      # Main performance data
├── detailed_results_[timestamp].json       # Detailed results
├── attention_analysis_[timestamp].json     # Attention analysis
├── model_comparison.png                    # Performance comparison charts
└── attention_maps/                         # Attention heatmaps
    ├── clean_samples/
    ├── inf_samples/  
    ├── rcs_samples/
    └── sd_samples/
```

### Example Expected Results

#### Performance Comparison

```
           Clean    INF      RCS      SD       Average
llama      0.145    0.098    0.110    0.105    0.115
diffllama  0.162    0.123    0.135    0.128    0.137
improvement +0.017   +0.025   +0.025   +0.023   +0.022
```

#### Attention Analysis

```json
{
  "attention_summary": {
    "llama": {
      "clean": {"kmi_ratio": 0.45, "noise_ratio": 0.0},
      "noisy": {"kmi_ratio": 0.32, "noise_ratio": 0.18}
    },
    "diffllama": {
      "clean": {"kmi_ratio": 0.50, "noise_ratio": 0.0}, 
      "noisy": {"kmi_ratio": 0.43, "noise_ratio": 0.12}
    }
  }
}
```

## 🛠 Development and Extension

### Adding New Noise Types

1. Implement noise function in `src/noise_injection.py`
2. Integrate new function in data generation pipeline
3. Update evaluation pipeline to support new dataset

### Adding New Models

1. Add loading logic in `src/model_loader.py`
2. Configure model paths and parameters
3. Update evaluation pipeline

### Custom Evaluation Metrics

1. Implement new metrics in `src/evaluation.py`
2. Modify result aggregation and visualization logic
3. Update result output format

## 📚 Documentation Resources

- **[Detailed Usage Guide](docs/USAGE_GUIDE.md)**: Complete usage instructions and advanced configuration
- **[Colab Usage Guide](colab/README.md)**: Google Colab specific instructions
- **[Project File Organization](docs/PROJECT_SUMMARY.md)**: File structure and organization explanation
- **Command Line Help**: `python main.py --help`

## 🔧 Troubleshooting

### Common Issues

#### GPU Memory Insufficient

```bash
# Use smaller batch size and sample count
python main.py --quick-test --max-samples 10
```

#### Model Download Failed  

```bash
# Check network connection, retry manually
python scripts/download_models.py
```

#### Import Errors

```bash
# Verify environment configuration
python scripts/test_setup.py --quick
```

### Performance Optimization

- **GPU**: Use CUDA acceleration, 12GB+ VRAM recommended
- **Memory**: 16GB+ system memory recommended
- **Storage**: At least 15GB available space for model cache
- **Network**: First run requires model and data downloads

## 🎓 Academic Usage

### Citation Information

If you use this framework in your research, please consider citing the relevant DiffLlama paper and GSM8K dataset.

### Experiment Reproduction

- Set random seeds for reproducibility
- Record hardware configuration and model versions
- Save complete experiment configuration and results

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🤝 Contributing

Contributions in code, issue reports, or improvement suggestions are welcome. Please participate through GitHub Issues or Pull Requests.

---

**🚀 Start your noise robustness research journey!**

Choose your usage mode:

- 🏃 **Quick Experience**: `python main.py --quick-test`  
- 🔬 **Complete Research**: `python main.py`
- 📱 **Cloud Experiment**: Use Google Colab

**Need help?** Check the [Usage Guide](docs/USAGE_GUIDE.md) or run `python scripts/test_setup.py` to check environment configuration.

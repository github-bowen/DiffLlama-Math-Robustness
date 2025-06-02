# DiffLlama vs Llama Noise Robustness Experiment

A research project on mathematical reasoning noise robustness based on differential attention mechanism, comparing the performance of [DiffLlama-375M](https://huggingface.co/reyllama/DiffLlama-375M) and [Llama-375M](https://huggingface.co/reyllama/Llama_375M) on noisy mathematical problems.

> [!NOTE]
> For usage guide, please refer to [USAGE_GUIDE.md](USAGE_GUIDE.md).

## 📋 Project Overview

This project implements a complete experimental framework for studying and comparing the noise robustness of different attention mechanisms in mathematical reasoning tasks. Through systematic experimental design, it deeply explores the advantages of DiffLlama's differential attention mechanism compared to traditional attention mechanisms.

### 🎯 Research Objectives

- **Performance Comparison**: Evaluate DiffLlama and Llama's performance on clean and noisy mathematical problems
- **Noise Robustness**: Analyze the impact of different types of noise on model performance
- **Attention Mechanism**: Visualize and quantify models' attention allocation patterns
- **In-depth Analysis**: Explore the working principles and advantages of differential attention mechanism

## 🏗️ Project Structure

```bash
.
├── colab
│   ├── config.py
│   ├── DiffLlama_Colab_Experiment.ipynb
│   ├── experiment.py
│   ├── __init__.py
│   └── README.md
├── data
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
├── scripts
│   ├── download_models.py
│   ├── __init__.py
│   ├── interactive_inference.py
│   ├── test_diffllama_attention.py
│   ├── test_llama_attention.py
│   └── test_setup.py
├── src
│   ├── attention_visualizer.py
│   ├── evaluation.py
│   ├── fine_tuning.py
│   ├── __init__.py
│   ├── model_loader.py
│   ├── noise_injection.py
│   └── utils.py
└── USAGE_GUIDE.md
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

### Experiment Modes

#### 1. Quick Test Mode

- **Purpose**: Verify environment configuration and code correctness
- **Sample Count**: 20 per dataset
- **Runtime**: 30-60 minutes
- **Command**: `python -m main --quick-test`

#### 2. Standard Mode

- **Purpose**: Balanced experimental results
- **Sample Count**: 200 per dataset (customizable)
- **Runtime**: 2-4 hours
- **Command**: `python -m main`

#### 3. Full Mode

- **Purpose**: Complete research results
- **Sample Count**: All data (~1300 samples)
- **Runtime**: 6-12 hours
- **Command**: `python -m main --max-samples -1`

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

```bash
results/
├── experiment_results_[timestamp].csv     # Main performance results
├── detailed_results_[timestamp].json      # Detailed results
├── attention_analysis_[timestamp].json    # Attention analysis
├── model_comparison_[timestamp].png       # Performance comparison
└── attention_maps/                        # Attention heatmaps
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
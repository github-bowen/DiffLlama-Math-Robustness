# DiffLlama vs Llama Noise Robustness Experiment

A research project on mathematical reasoning noise robustness based on differential attention mechanism, comparing the performance of [DiffLlama-375M](https://huggingface.co/reyllama/DiffLlama-375M) and [Llama-375M](https://huggingface.co/reyllama/Llama_375M) on noisy mathematical problems.

This project is an extension of the work presented in [Differential Transformer](https://arxiv.org/abs/2410.05258), adapted to use small-scale models for mathematical reasoning robustness analysis.

> [!NOTE]
> For usage guide, please refer to [USAGE_GUIDE.md](USAGE_GUIDE.md).

## 📚 Table of Contents

- [DiffLlama vs Llama Noise Robustness Experiment](#diffllama-vs-llama-noise-robustness-experiment)
  - [📚 Table of Contents](#-table-of-contents)
  - [📋 Project Overview](#-project-overview)
    - [🎯 Research Objectives](#-research-objectives)
  - [🏗️ Project Structure](#️-project-structure)
    - [Example of Noise Injection](#example-of-noise-injection)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Experiment Modes](#experiment-modes)
      - [1. Quick Test Mode](#1-quick-test-mode)
      - [2. Standard Mode](#2-standard-mode)
      - [3. Full Mode](#3-full-mode)
  - [📊 Dataset Description](#-dataset-description)
    - [Original Dataset](#original-dataset)
    - [Noisy Datasets](#noisy-datasets)
  - [🤖 Model Description](#-model-description)
    - [DiffLlama-375M](#diffllama-375m)
    - [Llama-375M](#llama-375m)
  - [📈 Result Analysis](#-result-analysis)
    - [Output Files](#output-files)
    - [Performance Metrics](#performance-metrics)
      - [Pass@1 Accuracy](#pass1-accuracy)
      - [Attention Analysis Metrics](#attention-analysis-metrics)
  - [📄 License](#-license)

## 📋 Project Overview

This project implements a complete experimental framework for studying and comparing the noise robustness of different attention mechanisms in mathematical reasoning tasks. Through systematic experimental design, it deeply explores the advantages of DiffLlama's differential attention mechanism compared to traditional attention mechanisms.

**Paper Reference**: This work extends the differential attention mechanism introduced in *Differential Transformer* ([arXiv:2410.05258](https://arxiv.org/abs/2410.05258)) by applying it to mathematical reasoning tasks using smaller-scale models (375M parameters) to investigate noise robustness properties.

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
├── scripts
│   ├── download_models.py
│   ├── __init__.py
│   ├── inspect_model_structure.py
│   ├── inspect_attention_allocation_ratio_results.py
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
├── data
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
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
├── experiment_report_[timestamp].csv      # Main performance results
├── detailed_results_[timestamp].json      # Detailed results
├── attention_analysis_sft.json            # Attention analysis
├── sft_performance.csv                    # Performance comparison
└── attention_maps/                        # Attention heatmaps
    ├── clean_q1_sft/
    ├── ......
    ├── INF_noise_q1_sft/
    ├── ......
    ├── RCS_noise_q1_sft/
    ├── ......
    ├── SD_noise_q1_sft/
    └── ......
```

### Performance Metrics

#### Pass@1 Accuracy

- The proportion of models that give the correct answer on their first attempt
- Main evaluation metric

#### Attention Analysis Metrics

- **KMI Ratio**: Proportion of attention focused on key mathematical information
- **NI Ratio**: Proportion of attention focused on noise information
- **OC Ratio**: Proportion of attention focused on other content

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

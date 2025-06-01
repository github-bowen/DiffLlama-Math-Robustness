# 📖 DiffLlama vs Llama 实验使用指南

本指南详细说明了如何使用 DiffLlama vs Llama 噪声鲁棒性实验框架，包括设置、运行和分析结果的完整流程。

## 🎯 快速开始

### 1. 环境检查
首先验证环境是否正确配置：

```bash
# 运行环境测试
python scripts/test_setup.py

# 快速测试模式
python scripts/test_setup.py --quick
```

### 2. 下载模型（如果需要）
```bash
# 下载实验所需的模型
python scripts/download_models.py
```

### 3. 运行快速实验
```bash
# 运行包含20个样本的快速测试
python main.py --quick-test
```

## 🔧 详细使用说明

### 命令行参数

#### 主实验脚本 (`main.py`)

```bash
python main.py [OPTIONS]
```

**主要选项:**
- `--quick-test`: 运行快速测试（20个样本）
- `--max-samples N`: 限制每个数据集的最大样本数
- `--skip-data-gen`: 跳过数据生成（使用现有数据）
- `--skip-evaluation`: 跳过零样本评估
- `--skip-sft`: 跳过监督微调
- `--skip-attention`: 跳过注意力分析
- `--models MODEL1,MODEL2`: 指定要测试的模型（默认: diffllama,llama）
- `--datasets DATASET1,DATASET2`: 指定要测试的数据集（默认: clean,inf,rcs,sd）

**示例用法:**
```bash
# 完整实验
python main.py

# 自定义样本数量
python main.py --max-samples 100

# 跳过耗时的微调步骤
python main.py --skip-sft

# 只测试特定数据集
python main.py --datasets clean,inf

# 只测试特定模型
python main.py --models diffllama
```

#### Colab 实验脚本 (`colab/experiment.py`)

```bash
python colab/experiment.py [OPTIONS]
```

**Colab 特定选项:**
- `--mode {quick,medium,full}`: 实验模式
- `--use-drive`: 使用 Google Drive 持久存储
- `--setup`: 仅执行环境设置
- `--instructions`: 显示使用说明

**示例用法:**
```bash
# Colab 快速测试
python colab/experiment.py --mode quick --use-drive

# Colab 完整实验
python colab/experiment.py --mode full --use-drive --max-samples 500
```

### 实验模式

#### 1. 快速测试模式
- **用途**: 验证环境配置和代码正确性
- **样本数**: 20个/数据集
- **运行时间**: 30-60分钟
- **命令**: `python main.py --quick-test`

#### 2. 标准模式
- **用途**: 平衡的实验结果
- **样本数**: 200个/数据集（可自定义）
- **运行时间**: 2-4小时
- **命令**: `python main.py`

#### 3. 完整模式
- **用途**: 完整的研究结果
- **样本数**: 全部数据（~1300个）
- **运行时间**: 6-12小时
- **命令**: `python main.py --max-samples -1`

## 📊 数据集说明

### 原始数据集
- **Clean**: GSM8K 原始测试集
- **大小**: 1,319个数学问题

### 噪声数据集
- **INF** (Irrelevant Numbers/Facts): 添加无关数字和事实
- **RCS** (Redundant Calculation Steps): 添加冗余计算步骤  
- **SD** (Semantic Distraction): 添加语义干扰信息

每个噪声数据集都基于原始 Clean 数据集生成。

## 🤖 模型说明

### DiffLlama-375M
- **类型**: 基于差分注意力机制的 Llama 变体
- **参数量**: 375M
- **特点**: 具有专门的差分注意力机制

### Llama-375M  
- **类型**: 标准 Llama 架构
- **参数量**: 375M
- **特点**: 传统的注意力机制

## 📈 结果分析

### 输出文件

实验完成后，结果保存在 `results/` 目录：

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

### 性能指标

#### Pass@1 准确率
- 模型第一次尝试给出正确答案的比例
- 主要评估指标

#### 注意力分析指标
- **KMI比例**: 关键数学信息的注意力占比
- **NI比例**: 噪声信息的注意力占比  
- **OC比例**: 其他内容的注意力占比

### 解读结果

#### 性能对比表格示例
```
           Clean    INF      RCS      SD
llama      0.145    0.098    0.110    0.105
diffllama  0.162    0.123    0.135    0.128
```

**解读:**
- DiffLlama 在所有数据集上都优于 Llama
- 噪声显著降低了两个模型的性能
- DiffLlama 在噪声数据上的性能下降较小

#### 注意力分析示例
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

**解读:**
- DiffLlama 能更好地关注关键数学信息（KMI）
- 面对噪声时，DiffLlama 的注意力分散程度较小
- 证明了差分注意力机制的有效性

## 🛠 高级使用

### 自定义配置

编辑配置文件以自定义实验参数：

```python
# src/config.py (如果存在)
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

### 单独运行模块

#### 数据生成
```python
from src.utils import download_gsm8k
from src.noise_injection import generate_noisy_datasets

# 下载原始数据
download_gsm8k()

# 生成噪声数据集
generate_noisy_datasets()
```

#### 模型评估
```python
from src.evaluation import run_comprehensive_evaluation

# 运行评估
results_df, detailed_results = run_comprehensive_evaluation(
    models=['diffllama', 'llama'],
    datasets=['clean', 'inf'],
    max_samples_per_dataset=50
)
```

#### 注意力分析
```python
from src.attention_visualizer import compare_attention_patterns

# 比较注意力模式
results = compare_attention_patterns(
    clean_dataset="data/gsm8k_test.jsonl",
    noisy_dataset="data/gsm8k_inf_test.jsonl",
    num_samples=10
)
```

### 添加新的噪声类型

实现新的噪声注入函数：

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

## 🐛 故障排除

### 常见问题

#### 1. CUDA 内存不足
```bash
# 解决方案：减少批量大小或样本数
python main.py --quick-test --max-samples 10
```

#### 2. 模型下载失败
```bash
# 手动下载模型
python scripts/download_models.py

# 检查网络连接和 Hugging Face 访问
```

#### 3. 导入错误
```bash
# 检查环境配置
python scripts/test_setup.py

# 安装缺失的依赖
pip install -r requirements.txt
```

#### 4. 数据生成失败
```bash
# 检查网络连接和数据目录权限
ls -la data/

# 手动下载 GSM8K
python -c "from src.utils import download_gsm8k; download_gsm8k()"
```

### 性能优化

#### GPU 优化
```python
# 启用混合精度训练（如果支持）
import torch
torch.backends.cudnn.benchmark = True
```

#### 内存优化
```python
# 清理 GPU 缓存
import torch
torch.cuda.empty_cache()

# 使用梯度检查点（如适用）
```

## 📚 扩展实验

### 添加新模型

1. 在 `src/model_loader.py` 中添加模型加载逻辑
2. 更新配置以包含新模型
3. 运行实验：`python main.py --models your_model,diffllama,llama`

### 添加新数据集

1. 在 `src/utils.py` 中添加数据下载函数
2. 在 `src/noise_injection.py` 中添加对应的噪声生成
3. 更新评估流程以支持新数据集

### 自定义评估指标

1. 在 `src/evaluation.py` 中添加新的评估函数
2. 修改结果聚合逻辑
3. 更新结果可视化

## 🎓 学术使用

### 引用格式
如果在学术工作中使用此框架，请考虑引用相关论文和数据集。

### 实验复现
为确保实验可复现：
1. 记录所使用的模型版本
2. 保存随机种子设置
3. 记录硬件配置信息
4. 保存完整的结果文件

---

**更多帮助:**
- 查看 `README.md` 了解项目概览
- 查看 `colab/README.md` 了解 Colab 使用
- 运行 `python main.py --help` 查看所有选项 
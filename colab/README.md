# 🔬 Google Colab 专用实验指南

本指南专门为 Google Colab 用户设计，提供在 Colab 环境中运行 DiffLlama vs Llama 噪声鲁棒性实验的完整流程。

## 📋 Colab 环境优势

- ✅ **免费 GPU**: 使用 Google 提供的免费 GPU 资源
- ✅ **Google Drive 集成**: 模型和结果持久化存储
- ✅ **内存优化**: 针对 Colab 的内存限制进行优化
- ✅ **简化设置**: 自动化环境配置和依赖安装

## 🚀 快速开始

### 方法一：使用预制 Notebook（推荐）

1. **打开 Colab Notebook**
   - 上传 `DiffLlama_Colab_Experiment.ipynb` 到 Google Colab
   - 或者直接在 Colab 中创建新的 notebook

2. **设置 GPU 运行时**
   ```
   Runtime → Change runtime type → Hardware accelerator → GPU
   ```

3. **按照 Notebook 中的步骤执行**
   - 所有步骤都有详细说明
   - 包含完整的代码示例

### 方法二：使用命令行脚本

1. **上传项目文件**
   ```python
   # 在 Colab 中上传以下文件：
   # - colab/ 目录下的所有文件
   # - src/ 目录下的所有文件
   # - scripts/ 目录下的模型下载脚本
   ```

2. **运行快速测试**
   ```bash
   !python colab/experiment.py --mode quick --use-drive
   ```

## 📁 文件结构

上传到 Colab 的必需文件：

```
Colab Environment/
├── colab/                        # Colab 专用脚本
│   ├── experiment.py             # 主要的 Colab 实验脚本
│   ├── config.py                 # Colab 专用配置
│   ├── quick_run.py              # 快速运行示例
│   └── README.md                 # 本文档
├── src/                          # 核心实验代码
│   ├── utils.py
│   ├── model_loader.py
│   ├── noise_injection.py
│   ├── evaluation.py
│   ├── fine_tuning.py
│   └── attention_visualizer.py
├── scripts/                      # 辅助脚本
│   └── download_models.py        # 模型下载脚本
└── DiffLlama_Colab_Experiment.ipynb  # Notebook 版本
```

## 🔧 实验模式

### 🏃 快速测试模式
```bash
!python colab/experiment.py --mode quick --use-drive
```
- **样本数量**: 20个
- **运行时间**: 30-60分钟
- **内存需求**: 最低
- **推荐用途**: 验证环境设置

### 📊 中等规模模式
```bash
!python colab/experiment.py --mode medium --use-drive
```
- **样本数量**: 100个
- **运行时间**: 1-2小时
- **内存需求**: 中等
- **推荐用途**: 平衡时间与结果质量

### 🔬 完整实验模式
```bash
!python colab/experiment.py --mode full --use-drive --max-samples 500
```
- **样本数量**: 自定义
- **运行时间**: 数小时
- **内存需求**: 较高
- **推荐用途**: 完整研究结果

## 💾 Google Drive 集成

### 自动挂载和配置
```python
# 脚本会自动执行以下操作：
# 1. 挂载 Google Drive
# 2. 在 Drive 中创建实验目录
# 3. 设置符号链接以实现透明访问
```

### Drive 目录结构
```
Google Drive/
└── MyDrive/
    └── DiffLlama_Experiment/
        ├── models/           # 模型文件 (持久化)
        ├── data/            # 数据集 (持久化)
        ├── results/         # 实验结果 (持久化)
        └── models_finetuned/ # 微调模型 (可选)
```

### 手动挂载（如需要）
```python
from google.colab import drive
drive.mount('/content/drive')
```

## ⚙️ 自动化内存优化

脚本会根据可用的 GPU 自动调整设置：

### T4 GPU (15GB)
```python
settings = {
    "max_samples": 50,
    "batch_size": 1,
    "attention_samples": 5,
    "skip_sft": True  # 跳过微调以节省内存
}
```

### V100/A100 GPU (16GB+)
```python
settings = {
    "max_samples": 200,
    "batch_size": 2, 
    "attention_samples": 10,
    "skip_sft": False  # 可以运行微调
}
```

## 📊 结果管理

### 自动保存位置
- **本地**: `/content/results/` (临时)
- **Google Drive**: `/content/drive/MyDrive/DiffLlama_Experiment/results/` (持久)

### 结果文件类型
```
results/
├── colab_results_[timestamp].csv      # 主要性能结果
├── colab_attention_[timestamp].json   # 注意力分析
├── colab_summary_[timestamp].json     # 实验摘要
└── attention_maps/                    # 注意力热力图
    ├── clean_q1/
    └── noisy_q1/
```

### 下载结果
```python
# 在 Notebook 中自动压缩结果
import zipfile
with zipfile.ZipFile('experiment_results.zip', 'w') as zipf:
    # ... 压缩代码 ...
    
# 或者直接从 Google Drive 访问
```

## 🛠 故障排除

### 常见问题及解决方案

#### 问题 1: GPU 内存不足
```python
# 解决方案 1: 清理缓存
import torch
torch.cuda.empty_cache()

# 解决方案 2: 使用更小的样本
!python colab/experiment.py --mode quick --max-samples 20
```

#### 问题 2: 运行时断开连接
```python
# 解决方案：使用 Google Drive 持久化
!python colab/experiment.py --use-drive
# 重新连接后可以继续使用保存的模型和数据
```

#### 问题 3: 文件上传失败
```python
# 解决方案：逐个上传核心文件
# 优先级顺序：
# 1. colab/experiment.py
# 2. src/ 目录
# 3. 其他支持文件
```

#### 问题 4: 模型下载慢
```python
# 解决方案：使用 Google Drive 缓存
# 第一次下载后，模型会保存在 Drive 中
# 后续运行直接从 Drive 加载
```

### 性能优化建议

1. **使用 GPU 运行时**: 必须选择 GPU 以获得合理的运行速度
2. **启用 Drive 集成**: 避免重复下载和处理
3. **分批运行**: 对于大规模实验，考虑分多次运行
4. **监控资源**: 定期检查内存和 GPU 使用情况

## 📈 实验流程示例

### 完整流程（首次运行）
```python
# 1. 环境检查
!python colab/experiment.py --instructions

# 2. 初始设置
!python colab/experiment.py --setup --use-drive

# 3. 快速测试
!python colab/experiment.py --mode quick --use-drive

# 4. 中等规模实验（如果快速测试成功）
!python colab/experiment.py --mode medium --use-drive

# 5. 查看结果
import pandas as pd
df = pd.read_csv('results/colab_results_[最新时间戳].csv')
print(df.pivot(index='model', columns='dataset', values='accuracy'))
```

### 后续运行（已有缓存）
```python
# 直接运行实验（模型已在 Drive 中）
!python colab/experiment.py --mode medium --use-drive --skip-attention
```

## 🎯 预期结果

### 性能对比表格
```
           Clean    INF    RCS     SD
llama      0.123   0.098  0.110  0.105
diffllama  0.145   0.123  0.135  0.128
```

### 注意力分析
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

## 📚 进阶使用

### 自定义参数
```bash
# 限制评估样本数
!python colab/experiment.py --mode medium --max-samples 50

# 跳过注意力分析
!python colab/experiment.py --mode full --skip-attention

# 只运行特定部分
!python colab/experiment.py --setup  # 仅设置
```

### 手动运行模块
```python
# 单独运行数据生成
from src.utils import download_gsm8k
from src.noise_injection import generate_noisy_datasets

download_gsm8k()
generate_noisy_datasets()

# 单独运行评估
from src.evaluation import run_comprehensive_evaluation
results = run_comprehensive_evaluation(max_samples_per_dataset=20)
```

## 💡 最佳实践

1. **先运行快速测试**: 验证所有组件正常工作
2. **使用 Google Drive**: 确保数据和结果持久化
3. **监控运行时**: Colab 有时间限制，注意保存进度
4. **定期检查内存**: 避免内存溢出导致的崩溃
5. **备份重要结果**: 将关键结果下载到本地

## 🎓 教学使用

此实验框架特别适合：
- **课程作业**: 比较不同模型架构的性能
- **研究项目**: 探索注意力机制的作用
- **技术演示**: 展示噪声对 NLP 模型的影响
- **学习实践**: 理解现代 Transformer 模型

---

**需要帮助？** 

- 📖 查看主 README.md 了解实验背景
- 🔧 运行 `!python colab/experiment.py --help` 查看所有选项
- 💬 检查 Notebook 中的详细注释和说明

**祝您实验顺利！** 🎉 
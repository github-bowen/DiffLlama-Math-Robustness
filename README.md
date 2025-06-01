# 🔬 DiffLlama vs Llama 噪声鲁棒性实验

基于差分注意力机制的数学推理噪声鲁棒性研究项目，比较 DiffLlama-375M 与 Llama-375M 在带噪声数学问题上的性能表现。

## 📋 项目概述

本项目实现了一个完整的实验框架，用于研究和比较不同注意力机制在数学推理任务中的噪声鲁棒性。通过系统化的实验设计，深入探索DiffLlama的差分注意力机制相比传统注意力机制的优势。

### 🎯 研究目标

- **性能对比**: 评估 DiffLlama 与 Llama 在 clean 和 noisy 数学问题上的表现
- **噪声鲁棒性**: 分析不同类型噪声对模型性能的影响
- **注意力机制**: 可视化和量化模型的注意力分配模式
- **深度分析**: 探索差分注意力机制的工作原理和优势

## 🏗️ 项目结构

```
DiffLlama_Experiment/
├── 🧠 src/                          # 核心源代码
│   ├── utils.py                     # 数据下载和处理工具
│   ├── model_loader.py              # 模型加载和配置
│   ├── noise_injection.py           # 三种噪声注入策略
│   ├── evaluation.py                # 评估和性能分析
│   ├── fine_tuning.py               # 监督微调(可选)
│   └── attention_visualizer.py      # 注意力可视化和分析
├── 🔬 colab/                        # Google Colab 专用
│   ├── experiment.py                # Colab 主实验脚本
│   ├── config.py                    # Colab 环境配置
│   ├── quick_run.py                 # 快速运行示例
│   └── README.md                    # Colab 详细指南
├── 🛠 scripts/                      # 辅助脚本
│   ├── download_models.py           # 模型下载脚本
│   └── test_setup.py                # 环境配置测试
├── 📚 docs/                         # 文档资料
│   ├── USAGE_GUIDE.md               # 详细使用指南
│   └── PROJECT_SUMMARY.md          # 项目文件组织总结
├── 🚀 main.py                       # 主实验脚本（本地）
├── 📦 requirements.txt              # Python 依赖包
├── 📊 data/                         # 数据集目录
├── 📈 results/                      # 实验结果目录
└── 💾 cache/                        # 模型缓存目录
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- transformers 4.20+
- CUDA 11.0+ (推荐，用于GPU加速)
- 约15GB磁盘空间（模型缓存）

### 安装和设置

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd DiffLlama_Experiment
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **环境验证**
   ```bash
   python scripts/test_setup.py
   ```

4. **下载模型**
   ```bash
   python scripts/download_models.py
   ```

### 快速运行

#### 本地环境
```bash
# 快速测试 (20个样本，30-60分钟)
python main.py --quick-test

# 完整实验 (全部数据，数小时)
python main.py
```

#### Google Colab
```bash
# 上传项目文件到 Colab，然后运行：
!python colab/experiment.py --mode quick --use-drive
```

## 🔬 实验设计

### 噪声类型

#### 1. INF (Irrelevant Numbers/Facts)
添加与问题无关的数字和事实信息
```
原始: "Tom has 5 apples. He gives 2 to his friend. How many apples does he have left?"
噪声: "Tom has 5 apples and 3 oranges. He gives 2 apples to his friend who has 7 books. How many apples does Tom have left?"
```

#### 2. RCS (Redundant Calculation Steps)  
添加冗余的计算步骤和无用信息
```
原始: "Calculate 8 + 7"
噪声: "First, note that 8 = 4 + 4. Also, 7 = 3 + 4. Now calculate 8 + 7. Remember that 8 > 7."
```

#### 3. SD (Semantic Distraction)
添加语义相关但逻辑无关的干扰信息
```
原始: "A pizza is cut into 8 slices. If 3 slices are eaten, how many remain?"
噪声: "A delicious pizza with cheese and pepperoni is cut into 8 equal slices. The pizza smells great. If 3 slices are eaten quickly, how many slices remain on the plate?"
```

### 评估指标

- **Pass@1 准确率**: 模型第一次尝试的正确率
- **注意力分配比例**: KMI/NI/OC token的注意力占比
- **鲁棒性下降**: 从clean到noisy数据的性能变化

## 📊 实验模式

### 🏃 快速测试模式
- **目的**: 验证环境和代码正确性
- **样本数**: 20个/数据集
- **时间**: 30-60分钟
- **命令**: `python main.py --quick-test`

### 📈 标准模式  
- **目的**: 平衡的实验结果
- **样本数**: 200个/数据集
- **时间**: 2-4小时
- **命令**: `python main.py`

### 🔬 完整模式
- **目的**: 完整研究数据
- **样本数**: 全部数据(~1300个)
- **时间**: 6-12小时  
- **命令**: `python main.py --max-samples -1`

## 🎛️ 高级配置

### 自定义实验参数

```bash
# 指定样本数量
python main.py --max-samples 100

# 跳过耗时的微调步骤
python main.py --skip-sft

# 只测试特定模型
python main.py --models diffllama

# 只测试特定数据集
python main.py --datasets clean,inf

# 跳过注意力分析以节省时间
python main.py --skip-attention
```

### Google Colab 专用选项

```bash
# 不同实验模式
!python colab/experiment.py --mode quick    # 快速(20样本)
!python colab/experiment.py --mode medium  # 中等(100样本)  
!python colab/experiment.py --mode full    # 完整(自定义)

# Google Drive 集成
!python colab/experiment.py --use-drive    # 持久化存储

# 仅环境设置
!python colab/experiment.py --setup
```

## 📈 结果分析

### 输出文件

```
results/
├── experiment_results_[timestamp].csv      # 主要性能数据
├── detailed_results_[timestamp].json       # 详细结果
├── attention_analysis_[timestamp].json     # 注意力分析
├── model_comparison.png                    # 性能对比图表
└── attention_maps/                         # 注意力热力图
    ├── clean_samples/
    ├── inf_samples/  
    ├── rcs_samples/
    └── sd_samples/
```

### 预期结果示例

#### 性能对比
```
           Clean    INF      RCS      SD       Average
llama      0.145    0.098    0.110    0.105    0.115
diffllama  0.162    0.123    0.135    0.128    0.137
improvement +0.017   +0.025   +0.025   +0.023   +0.022
```

#### 注意力分析
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

## 🛠 开发和扩展

### 添加新的噪声类型

1. 在 `src/noise_injection.py` 中实现噪声函数
2. 在数据生成流程中集成新函数
3. 更新评估流程以支持新数据集

### 添加新的模型

1. 在 `src/model_loader.py` 中添加加载逻辑
2. 配置模型路径和参数
3. 更新评估流程

### 自定义评估指标

1. 在 `src/evaluation.py` 中实现新指标
2. 修改结果聚合和可视化逻辑
3. 更新结果输出格式

## 📚 文档资源

- **[详细使用指南](docs/USAGE_GUIDE.md)**: 完整的使用说明和高级配置
- **[Colab 使用指南](colab/README.md)**: Google Colab 专用说明
- **[项目文件组织](docs/PROJECT_SUMMARY.md)**: 文件结构和组织说明
- **命令行帮助**: `python main.py --help`

## 🔧 故障排除

### 常见问题

#### GPU 内存不足
```bash
# 使用更小的批量大小和样本数
python main.py --quick-test --max-samples 10
```

#### 模型下载失败  
```bash
# 检查网络连接，手动重试
python scripts/download_models.py
```

#### 导入错误
```bash
# 验证环境配置
python scripts/test_setup.py --quick
```

### 性能优化

- **GPU**: 使用CUDA加速，推荐12GB+显存
- **内存**: 推荐16GB+系统内存 
- **存储**: 至少15GB可用空间用于模型缓存
- **网络**: 首次运行需要下载模型和数据

## 🎓 学术使用

### 引用信息
如果您在研究中使用此框架，请考虑引用相关的DiffLlama论文和GSM8K数据集。

### 实验复现
- 设置随机种子确保可复现性
- 记录硬件配置和模型版本
- 保存完整的实验配置和结果

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 🤝 贡献

欢迎贡献代码、报告问题或建议改进。请通过 GitHub Issues 或 Pull Requests 参与项目。

---

**🚀 开始您的噪声鲁棒性研究之旅！** 

选择适合您的使用模式：
- 🏃 **快速体验**: `python main.py --quick-test`  
- 🔬 **完整研究**: `python main.py`
- 📱 **云端实验**: 使用 Google Colab

**需要帮助？** 查看 [使用指南](docs/USAGE_GUIDE.md) 或运行 `python scripts/test_setup.py` 检查环境配置。 
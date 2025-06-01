# DiffLlama vs Llama: 数学推理中的噪声鲁棒性与注意力机制研究

本项目实现了对比 DiffLlama-375M 与 Llama-375M 在数学推理任务中噪声鲁棒性和注意力机制的全面实验框架。

## 📋 实验概述

### 研究目标
深入探究 DiffLlama-375M 在处理含噪数学问题时的表现及其注意力机制的独特性，作为对原论文的扩展研究。

### 核心研究问题
1. **噪声鲁棒性**：DiffLlama 是否在含噪声的数学推理任务中表现更好？
2. **注意力机制**：差分注意力是否能更有效地聚焦关键数学信息并抑制噪声？
3. **微调效果**：监督微调后两个模型的性能差异如何变化？

## 🔬 实验设计

### 噪声类型
- **INF (Irrelevant Numbers/Facts)**: 不相关数字/事实
- **RCS (Redundant Calculation Steps)**: 冗余计算步骤  
- **SD (Semantic Distraction)**: 语义干扰信息

### 评估指标
- **Pass@1 准确率**：零样本和微调后的准确率
- **注意力分配比例**：KMI、NI、OC token的注意力分配
- **性能下降幅度**：从干净数据到含噪数据的性能变化

## 🚀 快速开始

### 环境配置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 模型下载
确保已下载模型（如果尚未下载）：
```bash
python pre_download_models.py
```

### 运行实验

#### 快速测试（推荐首次运行）
```bash
python main.py --quick-test
```

#### 完整实验
```bash
python main.py
```

#### 自定义运行
```bash
# 跳过微调，只运行评估和注意力分析
python main.py --skip-sft --max-samples 200

# 跳过注意力分析，只运行评估
python main.py --skip-attention

# 自定义微调参数
python main.py --sft-samples 1000 --sft-epochs 3
```

## 📁 项目结构

```
project_root/
├── cache/                          # Hugging Face 模型缓存
│   ├── models--reyllama--DiffLlama-375M/
│   └── models--reyllama--Llama_375M/
├── data/                           # 数据集
│   ├── gsm8k_train.jsonl          # GSM8K 训练集
│   ├── gsm8k_test.jsonl           # GSM8K 测试集
│   ├── gsm8k_inf_test.jsonl       # INF 噪声测试集
│   ├── gsm8k_rcs_test.jsonl       # RCS 噪声测试集
│   └── gsm8k_sd_test.jsonl        # SD 噪声测试集
├── src/                            # 源代码
│   ├── utils.py                   # 工具函数
│   ├── model_loader.py            # 模型加载
│   ├── noise_injection.py         # 噪声注入
│   ├── evaluation.py              # 模型评估
│   ├── fine_tuning.py             # 监督微调
│   └── attention_visualizer.py    # 注意力可视化
├── results/                        # 实验结果
│   ├── zero_shot_performance.csv  # 零样本性能
│   ├── sft_performance.csv        # 微调后性能
│   ├── attention_analysis.json    # 注意力分析
│   └── attention_maps/            # 注意力热力图
├── models_finetuned/              # 微调后的模型
├── main.py                        # 主执行脚本
├── requirements.txt               # 依赖列表
└── README.md                      # 项目说明
```

## 📊 实验步骤

### 1. 数据准备
- 自动下载 GSM8K 数据集
- 生成三种类型的噪声数据集
- 数据格式验证和预处理

### 2. 零样本评估
- 在干净和含噪数据集上评估两个模型
- 使用思维链 (Chain-of-Thought) 提示
- 计算 Pass@1 准确率

### 3. 监督微调（可选）
- 在 GSM8K 训练子集上微调两个模型
- 使用指令遵循格式训练
- 支持自定义训练参数

### 4. 微调后评估
- 评估微调后模型在所有数据集上的性能
- 对比微调前后的性能变化

### 5. 注意力分析
- **可视化**：生成注意力热力图
- **量化**：计算 KMI、NI、OC 的注意力分配比例
- **对比**：分析两个模型的注意力模式差异

## 📈 结果分析

### 性能指标
- **准确率对比**：各数据集上的模型性能
- **鲁棒性评估**：性能下降幅度分析
- **相对提升**：DiffLlama 相对于 Llama 的改进

### 注意力分析
- **注意力分配**：不同类型 token 的注意力权重
- **噪声抑制**：模型对噪声信息的注意力分配
- **关键信息聚焦**：对数学关键信息的注意力集中度

## 🔧 高级用法

### 单独运行模块

#### 数据准备
```bash
python -c "from src.utils import download_gsm8k; download_gsm8k()"
python src/noise_injection.py
```

#### 模型评估
```bash
python src/evaluation.py
```

#### 注意力可视化
```bash
python src/attention_visualizer.py
```

#### 监督微调
```bash
python src/fine_tuning.py --model diffllama --samples 500 --epochs 2
```

### 自定义实验

#### 修改噪声策略
在 `src/noise_injection.py` 中自定义噪声注入函数：
```python
def inject_custom_noise(question):
    # 实现自定义噪声注入逻辑
    return noisy_question
```

#### 调整评估参数
在 `src/evaluation.py` 中修改生成参数：
```python
max_new_tokens=512,  # 最大生成长度
temperature=0.7,     # 生成温度
```

## 📋 实验检查清单

- [ ] 环境配置完成
- [ ] 模型下载成功
- [ ] GSM8K 数据集准备就绪
- [ ] 噪声数据集生成完成
- [ ] 零样本评估运行成功
- [ ] 注意力可视化生成
- [ ] 结果文件产生并可分析

## ⚠️ 注意事项

### 计算资源
- **GPU 内存**：推荐至少 8GB VRAM
- **存储空间**：至少 10GB 可用空间
- **运行时间**：完整实验可能需要数小时

### 模型兼容性
- 项目基于 Hugging Face Transformers
- DiffLlama 的差分注意力提取可能需要特殊处理
- 注意力可视化依赖模型的 `output_attentions` 参数

### 结果可重现性
- 使用固定随机种子
- 贪婪解码确保一致性
- 详细记录实验参数

## 🔍 故障排除

### 常见问题

**Q: 模型加载失败**
A: 检查模型路径和网络连接，确保模型已正确下载

**Q: CUDA 内存不足**
A: 减少批处理大小或使用 `--max-samples` 限制样本数量

**Q: 注意力提取失败**
A: 确保模型支持 `output_attentions=True` 参数

**Q: 数据集下载慢**
A: 配置 Hugging Face 缓存目录或使用镜像源

### 调试模式
```bash
# 使用最小样本快速测试
python main.py --quick-test

# 跳过耗时步骤
python main.py --skip-sft --skip-attention
```

## 📚 参考文献

1. DiffLlama 原论文及其注意力机制实现
2. GSM8K 数学推理数据集
3. Transformer 注意力机制相关研究

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目！

## 📄 许可证

本项目遵循 MIT 许可证。

---

**注**：这是一个研究性项目，用于探索差分注意力机制在数学推理任务中的效果。实际结果可能因模型版本、硬件配置等因素而有所不同。 
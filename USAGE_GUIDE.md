# 🚀 快速使用指南

本指南将带您逐步运行 DiffLlama vs Llama 噪声鲁棒性实验。

## 📋 准备工作检查清单

### 1. 环境准备
```bash
# 检查 Python 版本 (需要 Python 3.8+)
python --version

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型验证
```bash
# 验证模型是否已下载
python test_setup.py
```

如果模型未下载，运行：
```bash
python pre_download_models.py
```

## 🎯 运行方式

### 方式一：快速测试 (推荐首次使用)
```bash
python main.py --quick-test
```
- 使用少量样本 (20个)
- 跳过耗时的微调步骤
- 运行时间：约 30-60 分钟
- 适合验证实验流程

### 方式二：完整实验
```bash
python main.py
```
- 使用完整数据集
- 包含微调步骤
- 运行时间：数小时
- 产生完整研究结果

### 方式三：自定义实验
```bash
# 跳过微调，只评估零样本性能
python main.py --skip-sft --max-samples 500

# 跳过注意力分析，只运行评估
python main.py --skip-attention --max-samples 200

# 自定义微调参数
python main.py --sft-samples 1000 --sft-epochs 3
```

## 📊 结果文件说明

实验完成后，`results/` 目录将包含：

### 核心结果文件
- `zero_shot_performance.csv`: 零样本评估结果对比表
- `sft_performance.csv`: 微调后评估结果 (如果运行了微调)
- `experiment_report_[timestamp].json`: 完整实验报告

### 注意力分析文件
- `attention_analysis.json`: 注意力分配量化结果
- `attention_maps/`: 注意力热力图可视化

### 详细数据文件
- `detailed_zero_shot_results_[timestamp].json`: 每个样本的详细预测结果

## 🔍 结果解读

### 1. 零样本性能对比
查看 `zero_shot_performance.csv`:
```csv
model,dataset,accuracy,correct,total,eval_type
llama,Clean,0.1234,123,1000,zero_shot
llama,INF,0.0987,98,1000,zero_shot
diffllama,Clean,0.1456,145,1000,zero_shot
diffllama,INF,0.1234,123,1000,zero_shot
```

### 2. 性能下降分析
- **鲁棒性指标**: 比较 Clean vs 噪声数据集的准确率下降
- **相对改进**: DiffLlama 相对 Llama 的性能提升
- **噪声类型影响**: 不同噪声类型对性能的影响程度

### 3. 注意力分析结果
查看 `attention_analysis.json`:
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

关键指标：
- **KMI (Key Math Info)**: 数学关键信息的注意力比例
- **NI (Noise Info)**: 噪声信息的注意力比例  
- **OC (Other Context)**: 其他上下文的注意力比例

## 🛠 故障排除

### 常见问题及解决方案

**问题 1: CUDA 内存不足**
```bash
# 解决方案：减少样本数量
python main.py --quick-test --max-samples 50
```

**问题 2: 模型加载失败**
```bash
# 检查模型路径
python test_setup.py

# 重新下载模型
python pre_download_models.py
```

**问题 3: 数据集下载慢**
```bash
# 设置 Hugging Face 镜像 (中国用户)
export HF_ENDPOINT=https://hf-mirror.com
python main.py
```

**问题 4: 依赖包冲突**
```bash
# 创建新的虚拟环境
python -m venv fresh_env
source fresh_env/bin/activate
pip install -r requirements.txt
```

## 📈 高级分析建议

### 1. 性能分析
```python
# 在 Python 中进一步分析结果
import pandas as pd

# 加载结果
df = pd.read_csv('results/zero_shot_performance.csv')

# 计算性能下降
pivot_df = df.pivot(index='model', columns='dataset', values='accuracy')
degradation = (pivot_df['Clean'] - pivot_df['INF']) / pivot_df['Clean'] * 100

print("Performance degradation (%):")
print(degradation)
```

### 2. 注意力模式分析
查看 `results/attention_maps/` 中的热力图，关注：
- 数学关键词的高亮程度
- 噪声部分的注意力分布
- 两个模型的差异模式

### 3. 错误案例分析
查看 `detailed_zero_shot_results_[timestamp].json` 中的预测结果：
- DiffLlama 正确但 Llama 错误的案例
- 两个模型都失败的困难案例
- 噪声对预测的具体影响

## 🔄 重复实验

为确保结果可靠性，建议：

1. **多次运行**: 由于随机性，运行多次实验
2. **不同噪声强度**: 修改噪声注入强度
3. **不同数据集规模**: 使用不同的 `--max-samples` 参数

```bash
# 运行多次快速测试
for i in {1..3}; do
    echo "Run $i"
    python main.py --quick-test --skip-sft
done
```

## 📝 实验记录建议

建议记录：
- 实验参数设置
- 硬件配置信息
- 运行时间和资源使用
- 观察到的现象和结论

## 🎯 预期结果

根据 DiffLlama 的设计理念，预期观察到：

1. **更好的噪声鲁棒性**: DiffLlama 在含噪数据上的性能下降更小
2. **更精准的注意力**: DiffLlama 对 KMI 的注意力分配更高，对 NI 更低
3. **一致的改进**: 在不同类型噪声上都有改进效果

## 📞 获取帮助

如果遇到问题：
1. 首先运行 `python test_setup.py` 进行诊断
2. 查看终端输出的错误信息
3. 检查 `results/` 目录中的日志文件
4. 尝试使用 `--quick-test` 模式排查问题

---

**祝您实验顺利！🎉** 
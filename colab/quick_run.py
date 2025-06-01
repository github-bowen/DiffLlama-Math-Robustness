#!/usr/bin/env python3
"""
Quick Colab Run Script - 在 Google Colab 中快速运行实验的示例脚本

这个脚本展示了在 Colab 中运行实验的关键步骤和命令。
可以直接复制粘贴到 Colab 的代码单元格中运行。
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# 步骤 1: 环境检查和初始设置
# ============================================================================

def step_1_environment_check():
    """步骤 1: 检查 Colab 环境并进行初始设置"""
    print("🔍 步骤 1: 环境检查")
    
    # 检查 GPU
    import torch
    print(f"GPU 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 检查是否在 Colab 环境
    try:
        import google.colab
        print("✅ 正在 Google Colab 中运行")
        return True
    except ImportError:
        print("⚠️  不在 Google Colab 环境中")
        return False

# ============================================================================
# 步骤 2: 挂载 Google Drive（可选但推荐）
# ============================================================================

def step_2_mount_drive():
    """步骤 2: 挂载 Google Drive"""
    print("\n💾 步骤 2: 挂载 Google Drive")
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive 挂载成功")
        return True
    except Exception as e:
        print(f"❌ Google Drive 挂载失败: {e}")
        return False

# ============================================================================
# 步骤 3: 安装依赖和设置环境
# ============================================================================

def step_3_install_dependencies():
    """步骤 3: 安装必要的依赖包"""
    print("\n📦 步骤 3: 安装依赖包")
    
    packages = [
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "accelerate>=0.20.0",
        "seaborn>=0.11.0"
    ]
    
    for package in packages:
        print(f"安装 {package}...")
        os.system(f"pip install -q {package}")
    
    print("✅ 依赖包安装完成")

# ============================================================================
# 步骤 4: 文件上传检查
# ============================================================================

def step_4_check_files():
    """步骤 4: 检查必要文件是否已上传"""
    print("\n📁 步骤 4: 检查项目文件")
    
    required_files = [
        "colab/experiment.py",
        "src/utils.py",
        "src/model_loader.py",
        "src/noise_injection.py",
        "src/evaluation.py",
        "src/attention_visualizer.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  缺少 {len(missing_files)} 个文件，请先上传：")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ 所有必需文件已找到")
    return True

# ============================================================================
# 步骤 5: 运行实验
# ============================================================================

def step_5_run_experiment(mode="quick"):
    """步骤 5: 运行实验"""
    print(f"\n🚀 步骤 5: 运行 {mode} 模式实验")
    
    # 构建命令
    cmd = f"python colab/experiment.py --mode {mode} --use-drive"
    
    print(f"执行命令: {cmd}")
    print("这可能需要一些时间...")
    
    # 执行命令
    result = os.system(cmd)
    
    if result == 0:
        print("✅ 实验完成")
        return True
    else:
        print("❌ 实验失败")
        return False

# ============================================================================
# 步骤 6: 查看结果
# ============================================================================

def step_6_view_results():
    """步骤 6: 查看实验结果"""
    print("\n📊 步骤 6: 查看实验结果")
    
    import glob
    import pandas as pd
    
    # 查找结果文件
    result_files = glob.glob("results/colab_results_*.csv")
    
    if result_files:
        latest_file = max(result_files)
        print(f"📈 最新结果文件: {latest_file}")
        
        # 读取和显示结果
        df = pd.read_csv(latest_file)
        print("\n📊 性能对比:")
        pivot_df = df.pivot(index='model', columns='dataset', values='accuracy')
        print(pivot_df)
        
        # 计算改进
        if 'llama' in pivot_df.index and 'diffllama' in pivot_df.index:
            print("\n🔍 DiffLlama 相对 Llama 的改进:")
            improvement = pivot_df.loc['diffllama'] - pivot_df.loc['llama']
            print(improvement)
        
        return True
    else:
        print("❌ 未找到结果文件")
        return False

# ============================================================================
# 主函数 - 运行完整流程
# ============================================================================

def run_complete_experiment():
    """运行完整的实验流程"""
    print("="*80)
    print("🔬 GOOGLE COLAB 实验 - 快速运行")
    print("="*80)
    
    # 执行所有步骤
    if not step_1_environment_check():
        print("❌ 环境检查失败")
        return
    
    step_2_mount_drive()  # 可选，但推荐
    step_3_install_dependencies()
    
    if not step_4_check_files():
        print("❌ 请先上传必要的项目文件")
        return
    
    if step_5_run_experiment("quick"):
        step_6_view_results()
    
    print("\n🎉 实验流程完成！")

# ============================================================================
# Colab 代码单元格示例
# ============================================================================

def show_colab_cells():
    """显示可以直接在 Colab 中使用的代码单元格"""
    
    cells = [
        {
            "title": "单元格 1: 环境检查",
            "code": """# 检查 GPU 和环境
import torch
print(f"GPU 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
try:
    import google.colab
    print("✅ 在 Colab 环境中")
except ImportError:
    print("❌ 不在 Colab 环境中")"""
        },
        
        {
            "title": "单元格 2: 挂载 Google Drive",
            "code": """# 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')"""
        },
        
        {
            "title": "单元格 3: 安装依赖",
            "code": """# 安装必要包
!pip install -q transformers>=4.20.0 datasets>=2.0.0 accelerate>=0.20.0 seaborn>=0.11.0"""
        },
        
        {
            "title": "单元格 4: 运行快速实验",
            "code": """# 运行快速实验（确保已上传所有项目文件）
!python colab/experiment.py --mode quick --use-drive"""
        },
        
        {
            "title": "单元格 5: 查看结果",
            "code": """# 查看实验结果
import pandas as pd
import glob

# 找到最新结果
result_files = glob.glob('results/colab_results_*.csv')
if result_files:
    latest_file = max(result_files)
    df = pd.read_csv(latest_file)
    
    print("📊 性能对比:")
    pivot_df = df.pivot(index='model', columns='dataset', values='accuracy')
    print(pivot_df)
    
    # 计算改进
    if 'llama' in pivot_df.index and 'diffllama' in pivot_df.index:
        print("\\n🔍 改进幅度:")
        print(pivot_df.loc['diffllama'] - pivot_df.loc['llama'])
else:
    print("未找到结果文件")"""
        },
        
        {
            "title": "单元格 6: 下载结果（可选）",
            "code": """# 打包并下载结果
import zipfile
import os
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
zip_filename = f'experiment_results_{timestamp}.zip'

with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for root, dirs, files in os.walk('results'):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path)

print(f"📦 结果已打包: {zip_filename}")
print("可以从 Colab 文件面板下载此文件")"""
        }
    ]
    
    print("\n" + "="*80)
    print("📱 COLAB 代码单元格示例")
    print("="*80)
    print("将以下代码分别复制到 Colab 的不同单元格中运行：")
    
    for i, cell in enumerate(cells, 1):
        print(f"\n--- {cell['title']} ---")
        print(cell['code'])
        print()

# ============================================================================
# 如果直接运行此脚本
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--show-cells":
            show_colab_cells()
        elif sys.argv[1] == "--run":
            run_complete_experiment()
        else:
            print("用法:")
            print("  python colab/quick_run.py --show-cells  # 显示 Colab 代码单元格")
            print("  python colab/quick_run.py --run         # 运行完整实验")
    else:
        print("🔬 Google Colab 快速实验脚本")
        print("此脚本提供了在 Colab 中运行实验的示例代码。")
        print()
        print("选项:")
        print("  --show-cells: 显示可直接在 Colab 中使用的代码单元格")
        print("  --run: 运行完整的实验流程")
        print()
        print("或者直接导入并使用其中的函数:")
        print("  from colab.quick_run import run_complete_experiment")
        print("  run_complete_experiment()") 
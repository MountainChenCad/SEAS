# SEAS: 基于散射感知的元学习少样本HRRP分类框架

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **SEAS** (Scattering-aware Episodic Learning，散射感知元学习) 是一个用于跨类别少样本飞机目标识别的元学习框架，基于高分辨距离像(HRRP)雷达信号。

---

## 项目概述

### 核心研究问题
**通过LoRA微调，大语言模型能否将学到的分类能力泛化到完全未见过的新飞机类别？**

### 主要创新点
- **3-way 1-shot 元学习**: 以任务(episode)为单位进行训练，而非单个样本
- **比较式思维链(CoT)**: 通过将查询样本与所有支持类别对比进行推理
- **跨类别泛化**: 前6类训练，后6类测试，验证模型的迁移能力

### 数据集划分
| 数据划分 | 飞机类别 | 用途 |
|---------|---------|------|
| **训练集** | EA-18G, EP-3E, F15, F16, F18, F2 | LoRA微调 |
| **测试集** | F22, F35, GlobalHawk, IDF, Mirage2000, Predator | 少样本评估 |

### 实验结果
| 模型 | 准确率 | 提升 |
|-----|--------|-----|
| 基线模型 (Qwen3-8B) | 62.00% | - |
| **SEAS (LoRA微调后)** | **67.33%** | **+5.33pp** |

---

## 项目结构

```
SEAS/
├── scripts/                      # 核心脚本 (6个)
│   ├── 01_prepare_train_data.py  # 生成训练数据（反向CoT）
│   ├── 02_prepare_eval_data.py   # 生成评估任务
│   ├── 03_train_lora.py          # LoRA微调训练
│   ├── 04_inference_local.py     # 本地推理（4-bit量化）
│   ├── 05_inference_jsonl.py     # JSONL批量推理
│   └── 06_evaluate.py            # API模型评估
│
├── src/                          # 核心模块 (5个)
│   ├── config.py                 # 全局配置
│   ├── feature_extractor.py      # 散射中心提取
│   ├── encoder.py                # 文本编码
│   ├── llm_utils.py              # LLM输出解析
│   └── prompt_builder.py         # Prompt构建
│
├── data/                         # 数据目录
│   ├── hrrp_episodes_train_3way.jsonl   # 训练数据 (3,263 episodes)
│   └── hrrp_episodes_eval_3way.jsonl    # 评估数据 (150 episodes)
│
├── README.md                     # 英文文档
├── README_CN.md                  # 本文档（中文）
├── LICENSE                       # MIT许可证
└── requirements.txt              # 依赖配置
```

---

## 快速开始

### 环境配置

```bash
# Python >= 3.10
# CUDA >= 12.0 (用于GPU训练)
# 显存 >= 16GB (推荐 32GB+)

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

```bash
# 步骤1: 生成训练数据（需要API密钥）
export SILICONFLOW_API_KEY="your_api_key"
python scripts/01_prepare_train_data.py

# 步骤2: 生成评估数据
python scripts/02_prepare_eval_data.py
```

### 模型训练

```bash
# LoRA微调
python scripts/03_train_lora.py \
    --model-path /path/to/Qwen3-8B \
    --train-data data/hrrp_episodes_train_3way.jsonl \
    --output-dir output/seas-3way \
    --epochs 2 \
    --lr 3e-4 \
    --rank 2
```

### 模型评估

```bash
# 本地4-bit量化推理（省内存）
python scripts/04_inference_local.py \
    --model-path output/seas-3way/final \
    --base-model /path/to/Qwen3-8B \
    --eval-tasks data/hrrp_episodes_eval_3way.jsonl

# JSONL批量推理
python scripts/05_inference_jsonl.py \
    --model-path output/seas-3way/final \
    --eval-data data/hrrp_episodes_eval_3way.jsonl

# API评估（云端微调模型）
export SILICONFLOW_API_KEY="your_api_key"
python scripts/06_evaluate.py
```

---

## 技术细节

### 3-way 1-shot Episode结构

```
任务(Episode):
├── 支持集 (3个样本，每类1个)
│   ├── 类别A: 散射中心特征
│   ├── 类别B: 散射中心特征
│   └── 类别C: 散射中心特征
├── 查询样本 (1个待分类样本)
│   └── 未知飞机的散射中心特征
└── 目标: 从{A, B, C}中预测查询样本类别
```

### 反向思维链(CoT)生成

与传统CoT（先推理后给出答案）不同，我们使用**反向CoT**：
1. 先告诉API正确答案
2. 要求生成合理的推理过程，对比查询样本与所有支持类别的异同
3. 移除答案提示，仅保留推理过程作为训练数据

这确保了训练数据具有一致且高质量的推理过程。

### HRRP数据处理流程

```
原始HRRP (.mat文件)
    ↓
复数向量 → 实数幅度 → 归一化到[0,1]
    ↓
峰值检测 (scipy.signal.find_peaks)
    ├── 显著性阈值: 0.15
    ├── 最小间距: 5个采样点
    └── 保留幅度最高的10个峰值
    ↓
文本编码
    格式: "Position: 459:Amplitude: 1.000; Position: 568:Amplitude: 0.922; ..."
    ↓
少样本Prompt构建
    ↓
大模型推理 (Qwen3-8B + LoRA)
    ↓
分类结果
```

### LoRA配置

```python
r = 2               # LoRA秩（低秩快速适应）
alpha = 32          # 缩放因子
target_modules = ["q_proj", "v_proj"]  # 注意力投影层
epochs = 2
learning_rate = 3e-4
```

### 4-bit量化（Jetson部署）

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 启用4-bit
    bnb_4bit_compute_dtype=torch.float16,  # 计算使用FP16
    bnb_4bit_use_double_quant=True,       # 嵌套量化
    bnb_4bit_quant_type="nf4",            # NF4量化类型
)
```

---

## 命令速查

### 数据生成
```bash
# 训练数据（需要API）
python scripts/01_prepare_train_data.py --output data/hrrp_episodes_train_3way.jsonl

# 评估数据
python scripts/02_prepare_eval_data.py --output data/hrrp_episodes_eval_3way.jsonl
```

### 模型训练
```bash
# 完整训练
python scripts/03_train_lora.py \
    --model-path /path/to/Qwen3-8B \
    --train-data data/hrrp_episodes_train_3way.jsonl \
    --output-dir output/seas-3way \
    --epochs 2 --lr 3e-4 --rank 2 --batch-size 4

# 监控GPU
watch -n 5 nvidia-smi
```

### 推理评估
```bash
# 本地4-bit推理
python scripts/04_inference_local.py \
    --model-path output/seas-3way/final \
    --eval-tasks data/hrrp_episodes_eval_3way.jsonl \
    --limit-samples 10  # 快速测试

# JSONL批量推理
python scripts/05_inference_jsonl.py \
    --model-path output/seas-3way/final \
    --eval-data data/hrrp_episodes_eval_3way.jsonl

# API评估
python scripts/06_evaluate.py \
    --model-id ft:LoRA/Qwen/Qwen3-8B:your-model-id \
    --eval-tasks data/hrrp_episodes_eval_3way.jsonl
```

---

## 模型部署

### Jetson Orin Nano (8GB)

提供支持4-bit量化的部署包：

```bash
# 在Jetson上下载基础模型
python scripts/download_model.py --model Qwen/Qwen3-8B

# 运行推理
python scripts/inference.py --adapter-path model/ --max-samples 10
```

详细部署说明请参见 `docs/JETSON_DEPLOYMENT.md`。

---

## 引用

如果本代码对您的研究有帮助，请引用：

```bibtex
@misc{seas2025,
  title={SEAS: Scattering-aware Episodic Learning for Few-Shot HRRP Classification},
  author={SEAS Project Contributors},
  year={2025},
  url={https://github.com/MountainChenCad/SEAS}
}
```

---

## 许可证

MIT许可证 - 详情请参见 [LICENSE](LICENSE)。

---

**最后更新**: 2026-03-31
**项目状态**: 活跃开发中

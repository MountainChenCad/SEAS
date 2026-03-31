# Scattering Aware Episodic Adaptation for Few-Shot HRRP ATR Using Large Language Models, Under Review

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![English](https://img.shields.io/badge/Language-English-red)](README.md)

> **SEAS** 是一个用于跨类别少样本飞机目标识别的小样本学习框架，基于高分辨距离像(HRRP)雷达信号。

---

## 项目概述

### 数据集划分
| 数据划分 | 飞机类别 | 用途 |
|---------|---------|------|
| **训练集** | EA-18G, EP-3E, F15, F16, F18, F2 | LoRA微调 |
| **测试集** | F22, F35, GlobalHawk, IDF, Mirage2000, Predator | 少样本评估 |

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

---

## 命令速查

### 数据生成
```bash
# 训练数据（需要API）
python scripts/01_prepare_train_data.py --output data/hrrp_episodes_train_3way.jsonl

# 评估数据
python scripts/02_prepare_eval_data.py --output data/hrrp_episodes_eval_3way.jsonl
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

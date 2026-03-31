# TableLlama-HRRP: LoRA微调的飞机识别系统

> 基于Qwen3-8B的LoRA微调，用于高分辨率雷达信号(HRRP)的飞机目标识别任务

[![License](https://img.shields.io/badge/License-Private-red)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green)]()

## 项目概览

### 核心研究问题
**能否通过LoRA微调提升LLM在未见过的新飞机类别上的Few-Shot分类准确率？**

### 数据集划分
- **训练集**: 前6类飞机 `(EA-18G, EP-3E, F15, F16, F18, F2)`
- **测试集**: 后6类飞机 `(F22, F35, GlobalHawk, IDF, Mirage2000, Predator)`
- **Few-Shot设置**: 6-way 1-shot (150个评估任务)

### 核心方法论
使用**LoRA监督微调(SFT)**在前6类飞机上训练，验证模型能否将学到的分类能力泛化到未见过的后6类飞机。

---

## 项目结构

```
TableLlama-HRRP/
├── scripts/                      # 9个核心脚本
│   ├── 01_generate_baseline_results.py     # 生成基线推理结果
│   ├── 02_extract_sft_data.py              # 从基线提取SFT训练数据
│   ├── 03_convert_to_sharegpt.py           # 转换为ShareGPT格式
│   ├── 04_train_lora.py                    # LoRA SFT 训练
│   ├── 05_generate_eval_tasks.py           # 生成评估任务
│   ├── 06_run_inference.py                 # 统一推理脚本
│   ├── 07_compare_results.py               # 结果对比分析
│   ├── 08_eval_api_models.py               # API模型批量评估
│   └── 09_eval_72b_models.py               # 72B模型评估
│
├── src/                          # 核心模块
│   ├── config.py                 # 全局配置
│   ├── feature_extractor.py      # HRRP散射中心提取
│   ├── scattering_center_encoder.py  # 文本编码
│   ├── prompt_constructor_sc.py  # Prompt构建
│   ├── llm_utils.py              # LLM工具函数
│   ├── api_caller_siliconflow.py # SiliconFlow API调用器
│   └── __init__.py
│
├── data/                         # 核心数据文件
│   ├── hrrp_sft_from_baseline.json        # SFT源数据 (1800 samples)
│   ├── hrrp_sft_train_sharegpt.json       # ShareGPT格式训练数据
│   ├── hrrp_sft_stats.json                # 数据统计
│   ├── eval_tasks_new_6classes.json       # 150个评估任务
│   └── qwen_baseline_6way_1shot_results_final.json  # 基线结果
│
├── output/                       # 训练输出
│   └── qwen3-hrrp-lora-sft-curve-0to10/   # LoRA训练结果
│
├── eval_results/                 # 评估结果
├── archive/                      # 归档内容
│   └── dpo/                      # DPO相关文件(已归档)
│
├── CLAUDE.md                     # 项目维护指南
└── README.md                     # 本文档
```

---

## 快速开始

### 环境要求
```bash
# 系统环境
Python >= 3.10.8
CUDA >= 12.0
GPU内存 >= 16GB (推荐 32GB+)

# 核心库
transformers >= 4.50.0
peft >= 0.15.0
datasets >= 4.0.0
torch >= 2.0.0
```

### 安装依赖
```bash
pip install transformers peft datasets torch scipy numpy

# API调用 (可选)
pip install openai aiohttp
```

### 本地模型准备
```bash
# 使用本地Qwen3-8B模型
export QWEN_MODEL_PATH=/root/autodl-tmp/Qwen3-8B
```

---

## 核心实验

### 训练LoRA模型

```bash
python scripts/04_train_lora.py \
  --model-path /root/autodl-tmp/Qwen3-8B \
  --data-path data/hrrp_sft_train_sharegpt.json \
  --output-dir output/qwen3-hrrp-lora-sft \
  --num-epochs 10 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --lora-rank 16
```

### 评估模型

```bash
python scripts/06_run_inference.py \
  --model-id output/qwen3-hrrp-lora-sft \
  --eval-tasks data/eval_tasks_new_6classes.json \
  --output-dir eval_results/lora_baseline
```

---

## 技术细节

### HRRP数据处理流程

```
HRRP数据 (.mat文件)
    ↓
读取复数向量 [200维]
    ↓
提取实部和幅度 → 归一化到[0,1]
    ↓
散射中心提取 (peak detection)
    ├─ 峰值显著性阈值: 0.15
    ├─ 最小间距: 5个采样点
    └─ 保留Top-10最高峰值
    ↓
文本编码 (scattering center encoding)
    ↓
Few-Shot Prompt构建
    ↓
LLM推理 (Qwen3-8B)
    ↓
分类结果
```

### Few-Shot Prompt 结构

```
System: You are an expert in aircraft classification...

User:
Known aircraft examples (1 per class, 6 classes):

Class 'F22':
Scattering Centers: Position: 459:Amplitude: 1.000; Position: 568:Amplitude: 0.922; ...

Class 'F35':
Scattering Centers: Position: 478:Amplitude: 0.918; Position: 595:Amplitude: 0.850; ...

[... 共6个类别的support样本]

Unknown aircraft to classify:
Scattering Centers: Position: 487:Amplitude: 1.000; Position: 602:Amplitude: 0.896; ...

Candidate classes: F22, F35, GlobalHawk, IDF, Mirage2000, Predator
Respond with ONLY the class name.
```

---

## 命令速查表

### 数据处理
```bash
# 生成评估任务
python scripts/05_generate_eval_tasks.py

# 查看数据统计
python -c "import json; d=json.load(open('data/hrrp_sft_stats.json')); print(json.dumps(d, indent=2))"
```

### 模型训练
```bash
# LoRA训练
python scripts/04_train_lora.py --num-epochs 10 --lora-rank 16

# 查看GPU使用
nvidia-smi
watch -n 5 nvidia-smi
```

### 模型评估
```bash
# 评估本地模型
python scripts/06_run_inference.py \
  --model-id output/qwen3-hrrp-lora-sft \
  --eval-tasks data/eval_tasks_new_6classes.json

# 批量评估API模型
python scripts/08_eval_api_models.py --eval-tasks data/eval_tasks_new_6classes.json

# 对比多个模型
python scripts/07_compare_results.py
```

---

## 参考资源

### 开源库
- **PEFT**: https://huggingface.co/docs/peft/ (LoRA适配器)
- **Transformers**: https://huggingface.co/docs/transformers/ (模型加载)

### 相关项目
- **Qwen官方**: https://huggingface.co/Qwen

---

## 项目信息

**项目名称**: TableLlama-HRRP
**研究方向**: LoRA微调用于Few-Shot飞机识别
**核心方法**: LoRA + SFT
**数据来源**: HRRP飞机识别任务
**模型基础**: Qwen3-8B

---

## 许可证

Private Repository - TableLlama-HRRP Research Project

---

**最后更新**: 2026-03-27
**项目状态**: 活跃开发中


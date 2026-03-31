# Scattering Aware Episodic Adaptation for Few-Shot HRRP ATR Using Large Language Models, Under Review

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![中文](https://img.shields.io/badge/语言-简体中文-red)](README_CN.md)

> **SEAS** (Scattering-aware Episodic Adaptation via Semantics) is a meta-learning framework for cross-category few-shot aircraft classification using High-Resolution Range Profile (HRRP) radar signals.

---

## Overview

### Dataset Split
| Split | Aircraft Classes | Purpose |
|-------|------------------|---------|
| **Train** | EA-18G, EP-3E, F15, F16, F18, F2 | LoRA fine-tuning |
| **Test** | F22, F35, GlobalHawk, IDF, Mirage2000, Predator | Few-shot evaluation |

---

## Project Structure

```
SEAS/
├── scripts/                      # Core scripts (6 files)
│   ├── 01_prepare_train_data.py  # Generate training data with reverse CoT
│   ├── 02_prepare_eval_data.py   # Generate evaluation episodes
│   ├── 03_train_lora.py          # LoRA fine-tuning
│   ├── 04_inference_local.py     # Local inference (4-bit quantization)
│   ├── 05_inference_jsonl.py     # Batch JSONL inference
│   └── 06_evaluate.py            # API model evaluation
│
├── src/                          # Core modules (5 files)
│   ├── config.py                 # Global configuration
│   ├── feature_extractor.py      # Scattering center extraction
│   ├── encoder.py                # Text encoding
│   ├── llm_utils.py              # LLM output parsing
│   └── prompt_builder.py         # Prompt construction
│
├── data/                         # Data directory
│   ├── hrrp_episodes_train_3way.jsonl   # Training data (3,263 episodes)
│   └── hrrp_episodes_eval_3way.jsonl    # Evaluation data (150 episodes)
│
├── README.md                     # This file (English)
├── README_CN.md                  # Chinese version
├── LICENSE                       # MIT License
└── requirements.txt              # Dependencies
```

---

## Quick Start

### Environment Setup

```bash
# Python >= 3.10
# CUDA >= 12.0 (for GPU training)
# GPU Memory >= 16GB (recommended 32GB+)

# Install dependencies
pip install -r requirements.txt
```

### Reverse CoT Generation

Unlike traditional CoT where the model reasons then answers, we use **reverse CoT**:
1. Tell the API the correct answer
2. Ask it to generate plausible reasoning comparing query against all support classes
3. Remove the answer hint, keep only the reasoning for training

This ensures consistent, high-quality reasoning for training data.

### HRRP Processing Pipeline

```
Raw HRRP (.mat file)
    ↓
Complex vector → Real amplitude → Normalization [0,1]
    ↓
Peak Detection (scipy.signal.find_peaks)
    ├── Prominence threshold: 0.15
    ├── Min distance: 5 samples
    └── Keep top-10 peaks
    ↓
Text Encoding
    Format: "Position: 459:Amplitude: 1.000; Position: 568:Amplitude: 0.922; ..."
    ↓
Few-Shot Prompt Construction
    ↓
LLM Inference (Qwen3-8B + LoRA)
    ↓
Classification Result
```

---

## Command Reference

### Data Generation
```bash
# Training data (requires API)
python scripts/01_prepare_train_data.py --output data/hrrp_episodes_train_3way.jsonl

# Evaluation data
python scripts/02_prepare_eval_data.py --output data/hrrp_episodes_eval_3way.jsonl
```

### Inference
```bash
# Local 4-bit inference
python scripts/04_inference_local.py \
    --model-path output/seas-3way/final \
    --eval-tasks data/hrrp_episodes_eval_3way.jsonl \
    --limit-samples 10  # Quick test

# JSONL batch inference
python scripts/05_inference_jsonl.py \
    --model-path output/seas-3way/final \
    --eval-data data/hrrp_episodes_eval_3way.jsonl

# API evaluation
python scripts/06_evaluate.py \
    --model-id ft:LoRA/Qwen/Qwen3-8B:your-model-id \
    --eval-tasks data/hrrp_episodes_eval_3way.jsonl
```

---

## Deployment

### Jetson Orin Nano (8GB)

A deployment package is available with 4-bit quantization support:

```bash
# Download base model on Jetson
python scripts/download_model.py --model Qwen/Qwen3-8B

# Run inference
python scripts/inference.py --adapter-path model/ --max-samples 10
```

See `docs/JETSON_DEPLOYMENT.md` for detailed setup instructions.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{seas2026,
  title={Scattering Aware Episodic Adaptation for Few-Shot HRRP ATR Using Large Language Models},
  author={Lingfeng Chen, Panhe Hu, Shuanghui Zhang, Xiangfeng Qiu, Zhen Liu},
  year={2026},
  url={https://github.com/MountainChenCad/SEAS}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Last Updated**: 2026-03-31
**Status**: Active Development

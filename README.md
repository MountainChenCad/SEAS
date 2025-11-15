# LLMs_LORA_HRRP: TableLlama-HRRP 项目

基于Qwen3-8B的LoRA微调，用于高分辨率雷达信号(HRRP)的飞机目标识别任务。

## 项目概览

**研究问题**: 能否通过LoRA微调提升LLM在未见过的新飞机类别上的Few-Shot分类准确率？

**数据集划分**:
- **训练集**: 前6类飞机 (EA-18G, EP-3E, F15, F16, F18, F2)
- **测试集**: 后6类飞机 (F22, F35, GlobalHawk, IDF, Mirage2000, Predator)

## 项目结构

```
TableLlama-HRRP/
├── scripts/              # 7个核心脚本
│   ├── 01_generate_baseline_results.py
│   ├── 02_extract_sft_data.py
│   ├── 03_convert_to_sharegpt.py
│   ├── 04_train_lora.py              # 核心训练脚本
│   ├── 05_generate_eval_tasks.py
│   ├── 06_run_inference.py
│   └── 07_compare_results.py
├── src/                 # 核心模块
│   ├── config.py
│   ├── feature_extractor.py          # HRRP散射中心提取
│   ├── scattering_center_encoder.py  # 文本编码
│   ├── prompt_constructor_sc.py      # Prompt构建
│   ├── llm_utils.py
│   └── api_callers/                  # LLM API调用器
├── configs/             # 训练配置
│   ├── qwen3_lora_sft.yaml           # 基础配置 (7 epoch)
│   └── qwen3_lora_sft_curve_0to10.yaml  # 核心实验 (10 epoch, 30 checkpoints)
├── data/                # 数据
│   ├── hrrp_sft_train_sharegpt.json  # ShareGPT格式训练数据
│   ├── eval_tasks_new_6classes.json  # 评估任务
│   └── qwen_baseline_6way_1shot_results_final.json  # 基线结果
├── output/              # 训练输出 (空目录，用于新训练)
├── eval_results/        # 评估结果 (空目录，用于新评估)
└── CLAUDE.md            # 项目维护指南
```

## 核心实验

### PHASE 1: 基线 vs LoRA初步验证
- **基线方法**: Training-free, 仅使用prompt engineering
- **LoRA微调**: 7 epoch, 8-way 1-shot evaluation
- **结果**: 
  - 基线准确率: 52.00%
  - LoRA微调: 57.33%
  - **改进**: +5.33%

### PHASE 2: 核心实验曲线 (当前进行中)
- **目标**: 生成完整的学习曲线 (0-10 epoch)
- **采样策略**: 30个均匀分布的checkpoints (每188步保存一次)
- **配置**:
  - 总步数: 1130步 (1800样本 × 10 epoch / 有效批次16)
  - Batch Size: 1 (内存优化)
  - 梯度累积: 16步
  - Sequence长度: 5000 tokens
  - 量化: 4-bit (BitsAndBytes)
- **预计耗时**: ~21小时

## 运行方式

### 启动训练

```bash
python scripts/04_train_lora.py \
  --model-path /root/autodl-tmp/Qwen3-8B \
  --data-path data/hrrp_sft_train_sharegpt.json \
  --output-dir output/qwen3-hrrp-lora-sft-curve-0to10 \
  --num-epochs 10 \
  --save-steps 188 \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --learning-rate 5e-5 \
  --max-length 5000 \
  --quantization 4bit
```

### 运行推理评估

```bash
python scripts/06_run_inference.py \
  --model-path /root/autodl-tmp/Qwen3-8B \
  --adapter-path output/qwen3-hrrp-lora-sft-curve-0to10/checkpoint-188 \
  --eval-tasks data/eval_tasks_new_6classes.json \
  --output-file eval_results/results.json
```

## 技术细节

### HRRP数据处理流程

```
HRRP (.mat) → 复数向量 → 实数向量 → 归一化 → 散射中心提取 → 文本编码
```

**散射中心提取** (feature_extractor.py):
```python
scipy.signal.find_peaks(
    normalized_hrrp,
    prominence=0.15,    # 峰值显著性阈值
    distance=5          # 峰值最小间距
)
```

**文本格式**:
```
Position: 459:Amplitude: 1.000; Position: 568:Amplitude: 0.922; ...
```

### Few-Shot Prompt结构

```
System: You are an expert in aircraft classification...

User:
Known aircraft examples (1 per class):
Class 'F22': Scattering Centers: Position: 459:Amplitude: 1.000; ...
Class 'F35': Scattering Centers: Position: 478:Amplitude: 0.922; ...
[... 6个类]

Unknown aircraft to classify:
Scattering Centers: Position: 487:Amplitude: 1.000; ...

Candidate classes: F22, F35, GlobalHawk, IDF, Mirage2000, Predator
Respond with ONLY the class name.
```

## 关键指标

| 指标 | 值 |
|------|-----|
| 模型 | Qwen3-8B |
| 量化 | 4-bit (BitsAndBytes) |
| 微调方法 | LoRA (rank=16, alpha=32) |
| 可训练参数 | 7.67M (0.09% of 8.2B) |
| Few-Shot设置 | 6-way, 1-shot |
| 评估任务数 | 150 (30 checkpoints × 5 samples) |

## 依赖

- transformers >= 4.36
- peft >= 0.7.1
- torch >= 2.0
- bitsandbytes >= 0.41.0
- scipy
- numpy
- jsonlines

## 作者

Mountain Chen (SHA256:I1eETP5j8vEVhexAJUR1cMeHpfpt/MPrswakpdOwMu8)

## 许可证

Private Repository - TableLlama-HRRP Research Project

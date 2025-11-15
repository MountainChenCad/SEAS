# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude's Code of Conduct (Claude Code 八荣八耻)

This is the guiding philosophy for all development. Adhere to it strictly.

- 以瞎猜接口为耻，以认真查询为荣。(Shame on guessing interfaces; glory in diligent verification.)
- 以模糊执行为耻，以寻求确认为荣。(Shame on ambiguous execution; glory in seeking confirmation.)
- 以臆想业务为耻，以人类确认为荣。(Shame on assuming business logic; glory in confirming with humans.)
- 以创造接口为耻，以复用现有为荣。(Shame on creating new interfaces; glory in reusing existing ones.)
- 以跳过验证为耻，以主动测试为荣。(Shame on skipping validation; glory in proactive testing.)
- 以破坏架构为耻，以遵循规范为荣。(Shame on breaking the architecture; glory in following specifications.)
- 以假装理解为耻，以诚实无知为荣。(Shame on pretending to understand; glory in admitting ignorance.)
- 以盲目修改为耻，以谨慎重构为荣。(Shame on blind modification; glory in cautious refactoring.)

## 项目维护原则

### 代码整洁原则
1. **删除优先于保留** - 发现冗余文件/代码时，立即删除而非标记为"待删除"
2. **整合优先于重复** - 发现重复实现时，提取公共逻辑，删除重复代码
3. **实际使用优先于理论完整** - 只保留实际workflow中使用的代码，删除"可能有用"的模块
4. **规范命名优先于历史命名** - 使用清晰的、有序的命名（如01_、02_前缀），而非含糊的版本号

### 文档管理原则
1. **单一文档真相** - 仅维护一个CLAUDE.md，删除所有临时报告/总结文档
2. **代码即文档** - 代码清晰性优先于注释丰富性
3. **结构化优于详细化** - 文档重点展示架构和数据流，而非每个函数的细节

---

## 项目概览

**HRRPLLM (TableLlama-HRRP)** 是一个研究项目，验证**LoRA微调**能否提升LLM在**未见过的新飞机类别**上的Few-Shot分类准确率。

### 核心研究问题
模型在前6类飞机上微调后，能否将学到的"分类推理能力"泛化到完全未见过的后6类？

### 数据集划分
- **前6类 (训练)**: EA-18G, EP-3E, F15, F16, F18, F2
- **后6类 (测试)**: F22, F35, GlobalHawk, IDF, Mirage2000, Predator

### 关键特性
- **Training-Free原则**: 基线方法不更新模型参数，仅用prompt engineering
- **LoRA微调增强**: 可选的参数高效微调来提升性能
- **Few-Shot Learning**: 使用6-way 1-shot设置评估泛化能力

---

## 项目结构 (2025-11-11 已整合)

```
TableLlama-HRRP/
├── data/                                # 核心数据 (仅5个文件)
│   ├── hrrp_sft_from_baseline.json      # SFT源数据 (8.1MB, 1800样本)
│   ├── hrrp_sft_train_sharegpt.json     # ShareGPT格式训练数据 (7.2MB)
│   ├── hrrp_sft_stats.json              # 数据统计
│   ├── eval_tasks_new_6classes.json     # 评估任务 (150个6-way 1-shot)
│   └── qwen_baseline_6way_1shot_results_final.json  # 基线结果 (52MB)
│
├── scripts/                             # 7个核心脚本 (按执行顺序)
│   ├── 01_generate_baseline_results.py # 生成基线推理结果
│   ├── 02_extract_sft_data.py          # 从基线结果提取SFT数据
│   ├── 03_convert_to_sharegpt.py       # 转换为ShareGPT格式
│   ├── 04_train_lora.py                # LoRA微调训练
│   ├── 05_generate_eval_tasks.py       # 生成新类评估任务
│   ├── 06_run_inference.py             # 统一推理(baseline|finetuned)
│   └── 07_compare_results.py           # 对比分析
│
├── src/                                 # 6个核心模块
│   ├── config.py                        # 全局配置
│   ├── feature_extractor.py            # 散射中心提取
│   ├── scattering_center_encoder.py    # 文本编码
│   ├── prompt_constructor_sc.py        # Prompt构建
│   ├── llm_utils.py                    # LLM工具函数
│   └── __init__.py
│
├── output/                              # 训练输出 (当前训练)
│   └── qwen3-hrrp-lora-sft-baseline/
│
├── eval_results/                        # 评估结果
│
├── CLAUDE.md                            # 本文档
├── PROJECT_STRUCTURE.md                 # 详细结构说明
└── monitor_training.sh                  # 训练监控
```

### 整合历史 (2025-11-11)
**删除内容**:
- 旧checkpoints: 7.5GB (13个过期训练输出)
- 冗余数据: 180MB (11个中间/重复JSON)
- 冗余脚本: 17个 (从25个减至7个)
- 冗余文档: 30个MD文件 (仅保留本文档)
- 未使用模块: 5个src文件 (从11个减至6个)

**代码整合**:
1. 提取`llm_utils.py` - 从main_experiment.py抽取parse_llm_output_for_label
2. 统一`06_run_inference.py` - 合并baseline和finetuned推理脚本
3. 规范化命名 - 脚本按执行顺序编号(01-07)

**空间节省**: 8.5GB → 168MB (98%减少)

---

## 当前工作流程 (清理后)

### 完整数据流
```
前6类HRRP (.mat)
  ↓ 01_generate_baseline_results.py
基线推理结果 (qwen_baseline_6way_1shot_results_final.json, 52MB)
  ↓ 02_extract_sft_data.py
SFT源数据 (hrrp_sft_from_baseline.json, 8.1MB, 1800样本)
  ↓ 03_convert_to_sharegpt.py
ShareGPT训练数据 (hrrp_sft_train_sharegpt.json, 7.2MB)
  ↓ 04_train_lora.py
LoRA Adapter (output/qwen3-hrrp-lora-sft-baseline/)
  ↓ 05_generate_eval_tasks.py
评估任务 (eval_tasks_new_6classes.json, 150 tasks)
  ↓ 06_run_inference.py (baseline | finetuned)
推理结果
  ↓ 07_compare_results.py
对比分析报告
```

### 当前训练状态
- **进度**: 训练中 (~Step 150/1800, 8.3%)
- **身份**: PID 475771, 正常运行
- **预期完成**: 23:16 (约2.4小时后)
- **第一批checkpoint**: Step 500 (~21:33)

---

## 核心技术细节

### 1. HRRP数据预处理

```python
# 完整pipeline
.mat文件 → 复数向量 → 实数向量 → 归一化[0,1] → 散射中心提取 → 文本编码
```

**散射中心提取** (feature_extractor.py):
```python
scipy.signal.find_peaks(
    normalized_hrrp,
    prominence=0.15,    # 峰值显著性阈值
    distance=5          # 峰值最小间距
)
# 保留幅度最大的10个峰值
```

**文本编码** (scattering_center_encoder.py):
```python
# 格式: "Position: 459:Amplitude: 1.000; Position: 568:Amplitude: 0.922; ..."
# 位置: 0-1000 (HRRP长度)
# 幅度: [0, 1] 归一化, 3位小数
```

### 2. Few-Shot Prompt结构

```
System: You are an expert in aircraft classification...

User:
Known aircraft examples (1 per class):

Class 'F22':
Scattering Centers: Position: 459:Amplitude: 1.000; ...

Class 'F35':
Scattering Centers: Position: 478:Amplitude: 0.922; ...

[... 6个类的support样本]

Unknown aircraft to classify:
Scattering Centers: Position: 487:Amplitude: 1.000; ...

Candidate classes: F22, F35, GlobalHawk, IDF, Mirage2000, Predator
Respond with ONLY the class name.
```

---

## 工作原则总结

**最终整合原则**:
1. **只保留实际使用的内容** - 删除所有历史遗留文件，只保留当前训练和评估逻辑所需的文件
2. **消除重复实现** - 整合baseline和finetuned推理为统一脚本，提取公共函数到独立模块
3. **规范化命名** - 使用按执行顺序的01-07前缀，提高可读性和可维护性
4. **代码即文档** - 通过清晰的结构化代码展示工作流程，减少文档负担
5. **训练在后台稳定进行** - 当前训练正常进行，预计21:33达到第一个checkpoint，23:16完成训练

**下一步工作**:
- 继续监控训练直到21:33，验证损失函数值
- 训练完成后自动触发评估管线
- 最终对比分析baseline vs finetuned在新类上的性能提升

---

**项目状态**: 清理整合完成，训练正常进行中，等待首个checkpoint验证。整个项目从混乱的8.5GB精简为规范化的~170MB，代码清晰度和可维护性显著提升。

---

## 📦 项目存档说明 (2025-11-15)

### 存档结构
项目历史文件已迁移到 `archive/` 目录：
- **archive/checkpoints/** - 中间checkpoint历史 (1.2GB)
- **archive/logs/** - 训练/评估日志 (430KB)
- **archive/experiments_deprecated/** - 过期实验脚本

### 存档索引
详细说明见 `archive/archive_index.md`

### 当前项目状态
- **已完成**: Baseline评估 (52.00%) + LoRA初步验证 (最优57.33%)
- **已生成**: 完整对比报告 `EVALUATION_REPORT.md`
- **下一步**: 核心实验曲线扩展 (0-10 Epoch, 30个checkpoint精细采样)
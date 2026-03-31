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

**HRRPLLM (TableLlama-HRRP)** 是一个研究项目，验证**LoRA微调(SFT)**能否提升LLM在**未见过的新飞机类别**上的Few-Shot分类准确率。

本项目采用**SiliconFlow API框架**，通过OpenAI兼容接口调用云端微调模型，无需本地GPU即可进行推理评估。

### 核心研究问题
模型在前6类飞机上微调后，能否将学到的"分类推理能力"泛化到完全未见过的后6类？

### 数据集划分
- **前6类 (训练)**: EA-18G, EP-3E, F15, F16, F18, F2
- **后6类 (测试)**: F22, F35, GlobalHawk, IDF, Mirage2000, Predator

### 关键特性
- **API驱动**: 通过SiliconFlow API调用云端Qwen2.5-7B微调模型
- **三个Checkpoint评估**: initial_commit、ckpt_step_406、ckpt_step_203
- **Few-Shot Learning**: 使用6-way 1-shot设置评估泛化能力
- **自动化评估**: 支持批量测试多个模型版本并生成对比报告

---

## 项目结构 (2025-11-11 已整合)

```
TableLlama-HRRP/
├── data/                                # 核心数据 (5个文件)
│   ├── hrrp_sft_from_baseline.json      # SFT源数据 (8.1MB, 1800样本)
│   ├── hrrp_sft_train_sharegpt.json     # ShareGPT格式训练数据 (7.2MB)
│   ├── hrrp_sft_stats.json              # 数据统计
│   ├── eval_tasks_new_6classes.json     # 评估任务 (150个6-way 1-shot)
│   └── qwen_baseline_6way_1shot_results_final.json  # 基线结果 (52MB)
│
├── scripts/                             # 9个核心脚本 (按执行顺序)
│   ├── 01_generate_baseline_results.py # 生成基线推理结果
│   ├── 02_extract_sft_data.py          # 从基线结果提取SFT数据
│   ├── 03_convert_to_sharegpt.py       # 转换为ShareGPT格式
│   ├── 04_train_lora.py                # LoRA微调训练
│   ├── 05_generate_eval_tasks.py       # 生成新类评估任务
│   ├── 06_run_inference.py             # 统一推理(baseline|finetuned)
│   ├── 07_compare_results.py           # 对比分析
│   ├── 08_eval_api_models.py           # API模型批量评估
│   └── 09_eval_72b_models.py           # 72B模型评估
│
├── src/                                 # 7个核心模块
│   ├── config.py                        # 全局配置
│   ├── feature_extractor.py            # 散射中心提取
│   ├── scattering_center_encoder.py    # 文本编码
│   ├── prompt_constructor_sc.py        # Prompt构建
│   ├── llm_utils.py                    # LLM工具函数
│   ├── api_caller_siliconflow.py       # SiliconFlow API调用器
│   └── __init__.py
│
├── output/                              # 训练输出
│   └── qwen3-hrrp-lora-sft-curve-0to10/ # LoRA训练结果
│
├── eval_results/                        # 评估结果
│
├── archive/                             # 归档内容
│   └── dpo/                             # DPO相关文件(已归档)
│
├── CLAUDE.md                            # 本文档
└── README.md                            # 项目说明
```

### 整合历史 (2026-03-27)
**删除内容**:
- DPO相关文件: 已归档到 archive/dpo/
  - 文档: 8个 (DPO_SETUP.md等)
  - 脚本: 3个 (10-12)
  - 数据: 2个 (hrrp_dpo_train.json等)
  - 模型: qwen3-hrrp-dpo-baseline/
  - 日志: 6个

**当前结构**:
- 9个脚本 (01-09)
- 7个src模块
- 5个数据文件

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

### 项目命令

**监控训练进度**:
```bash
bash monitor_training_core.sh
# 或查看实时日志
tail -f training_core_curve.log
```

**检查GPU状态**:
```bash
nvidia-smi
# 或持续监控
watch -n 5 nvidia-smi
```

**验证数据准备**:
```bash
# 检查训练数据
ls -lh data/hrrp_sft_train_sharegpt.json
# 检查评估任务
ls -lh data/eval_tasks_new_6classes.json
```

---

## 常用开发命令

### 前置设置

**设置API密钥**:
```bash
# 从 https://cloud.siliconflow.cn/account/ak 获取API密钥
export SILICONFLOW_API_KEY='your_api_key_here'
```

### API推理评估

**测试单个微调模型**:
```bash
# 测试 initial_commit 版本
python scripts/06_run_inference.py \
  --model-id ft:LoRA/Qwen/Qwen2.5-7B-Instruct:rpl47v9x40:initial_commit:uyulemtufwhthcnywhcj \
  --eval-tasks data/eval_tasks_new_6classes.json \
  --output-dir eval_results/api_initial_commit

# 测试 step_406 checkpoint
python scripts/06_run_inference.py \
  --model-id ft:LoRA/Qwen/Qwen2.5-7B-Instruct:rpl47v9x40:initial_commit:uyulemtufwhthcnywhcj-ckpt_step_406 \
  --output-dir eval_results/api_ckpt_406

# 测试 step_203 checkpoint
python scripts/06_run_inference.py \
  --model-id ft:LoRA/Qwen/Qwen2.5-7B-Instruct:rpl47v9x40:initial_commit:uyulemtufwhthcnywhcj-ckpt_step_203 \
  --output-dir eval_results/api_ckpt_203
```

**批量评估全部3个微调模型**:
```bash
python scripts/08_eval_api_models.py \
  --eval-tasks data/eval_tasks_new_6classes.json \
  --output-dir eval_results/api_models
```

**快速测试 (仅评估10个样本)**:
```bash
python scripts/08_eval_api_models.py \
  --eval-tasks data/eval_tasks_new_6classes.json \
  --limit-samples 10
```

**跳过推理，仅生成对比报告**:
```bash
python scripts/08_eval_api_models.py \
  --eval-tasks data/eval_tasks_new_6classes.json \
  --skip-inference
```

### 推理参数调整

**调整温度参数** (影响输出的随机性):
```bash
# 低温度 (更确定)
python scripts/06_run_inference.py \
  --model-id <model_id> \
  --temperature 0.1

# 高温度 (更多样)
python scripts/06_run_inference.py \
  --model-id <model_id> \
  --temperature 0.7
```

**调整最大生成长度**:
```bash
python scripts/06_run_inference.py \
  --model-id <model_id> \
  --max-tokens 5000
```

### 结果查看

**查看评估结果**:
```bash
# 查看单个模型结果
cat eval_results/api_initial_commit/results.json | python -m json.tool

# 查看对比报告
cat eval_results/api_models/comparison_report.md
```

**查看实时日志**:
```bash
tail -f inference_api.log
tail -f eval_api_models.log
```

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

---

## 架构和模块设计

### API框架架构

```
SiliconFlow云端微调模型 (3个版本)
  ├─ initial_commit
  ├─ ckpt_step_406
  └─ ckpt_step_203
         ↑
         | OpenAI兼容API
         |
    src/api_caller_siliconflow.py
    (SiliconFlowAPICaller)
         ↑
         |
scripts/06_run_inference.py
    └─> 数据处理流程
        ├─ src/feature_extractor.py (HRRP峰值提取)
        ├─ src/scattering_center_encoder.py (文本编码)
        ├─ src/prompt_constructor_sc.py (Prompt构建)
        ├─ src/llm_utils.py (输出解析)
        └─> 评估结果
            └─> eval_results/
```

### 关键模块说明

**src/api_caller_siliconflow.py**:
- 使用 OpenAI Python SDK
- 实现SiliconFlowAPICaller类
- 支持自动重试（3次，指数退避）
- 处理超时、限流等异常
- 环境变量自动读取API密钥

**scripts/06_run_inference.py**:
- 核心推理脚本（API版本）
- 加载150个评估任务
- 对每个任务: 加载HRRP → 特征提取 → Prompt构建 → API调用 → 解析结果
- 计算准确率等统计指标
- 保存详细结果JSON

**scripts/08_eval_api_models.py**:
- 批量评估脚本
- 自动调用06_run_inference.py处理3个模型
- 生成Markdown对比报告
- 支持跳过推理，仅生成报告

### 数据流向

**推理流程**:
```
评估任务 (eval_tasks_new_6classes.json, 150个任务)
  ↓
加载query和support HRRP文件
  ↓
特征提取 & 文本编码
  ├─ extract_scattering_centers_peak_detection()
  └─ encode_single_sc_set_to_text()
  ↓
构建Few-Shot Prompt
  ├─ PromptConstructorSC 构建context header
  └─ 插入support样本 + query样本
  ↓
SiliconFlow API调用
  ├─ OpenAI兼容接口
  └─ 模型: ft:LoRA/Qwen/...
  ↓
解析LLM输出
  └─ parse_llm_output_for_label() 提取类别
  ↓
结果保存 (results.json)
  └─ 准确率、逐题详情等
```

---

## 关键实现细节

### API调用机制 (src/api_caller_siliconflow.py)

**SiliconFlowAPICaller类**:
```python
caller = SiliconFlowAPICaller(
    api_key="your_key",
    base_url="https://api.siliconflow.cn/v1",
    max_retries=3,
    timeout=30
)

# 调用API
success, response = caller.call(
    model="ft:LoRA/Qwen/...",
    messages=[{"role": "user", "content": "prompt"}],
    temperature=0.1,
    max_tokens=3000
)
```

**重试机制**:
- 速率限制 (RateLimitError): 自动重试，指数退避
- 连接错误 (APIConnectionError): 自动重试
- 最大重试次数: 3次，基础延迟1秒

### HRRP特征提取 (feature_extractor.py)

```python
# 使用scipy.signal.find_peaks检测显著的散射中心
peaks, properties = scipy.signal.find_peaks(
    normalized_hrrp,
    prominence=0.15,      # 峰值相对高度阈值
    distance=5            # 峰值间最小距离
)
# 保留幅度最大的10个峰值
```

参数说明:
- `prominence=0.15`: 只保留相对高度超过0.15的峰值
- `distance=5`: 峰值间距不小于5个采样点
- 最终保留Top-10最高峰值

### 文本编码格式 (scattering_center_encoder.py)

```
Position: 459:Amplitude: 1.000; Position: 568:Amplitude: 0.922; ...
```

规范:
- 位置: 整数, 范围[0-1000]
- 幅度: 3位小数, 范围[0.0-1.0]
- 分隔符: `; ` (分号+空格)

### Few-Shot Prompt结构 (prompt_constructor_sc.py)

标准Prompt包含5个部分:
1. **系统提示**: 任务定义和背景知识
2. **Support样本**: 每个类别各1个已知样本
3. **Query样本**: 待分类的未知样本
4. **候选类别**: F22, F35, GlobalHawk, IDF, Mirage2000, Predator
5. **输出格式指导**: "Predicted Target Class: [class_name]"

---

## SiliconFlow API配置 (src/config.py)

**SiliconFlowConfig数据类**:
```python
@dataclass
class SiliconFlowConfig:
    api_key: str = ""  # 从环境变量读取
    base_url: str = "https://api.siliconflow.cn/v1"

    # 三个微调模型标识符
    model_initial: str = "ft:LoRA/Qwen/..."
    model_ckpt_406: str = "ft:LoRA/Qwen/...-ckpt_step_406"
    model_ckpt_203: str = "ft:LoRA/Qwen/...-ckpt_step_203"

    # 推理参数
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: int = 3000

    # 重试策略
    max_retries: int = 3
    timeout: int = 30
```

**环境变量优先级**:
1. 命令行参数 `--api-key`
2. 环境变量 `SILICONFLOW_API_KEY`
3. src/config.py中的默认值（留空）

---

## 项目现状

**框架演进**:
- ✅ **本地模型推理**: scripts/06_run_inference.py (旧版，已替换)
- ✅ **API框架**: scripts/06_run_inference.py (新版，OpenAI兼容)
- ✅ **批量评估**: scripts/08_eval_api_models.py (3个模型自动化测试)

**代码结构**:
- 9个脚本 (01-09)
- 7个源模块 (config, feature_extractor, scattering_center_encoder, prompt_constructor_sc, llm_utils, api_caller_siliconflow + __init__)
- 单一CLAUDE.md (项目指南)

**评估数据**:
- 150个6-way 1-shot任务 (eval_tasks_new_6classes.json)
- 6个新类别 (F22, F35, GlobalHawk, IDF, Mirage2000, Predator)

**下一步**:
1. 设置SILICONFLOW_API_KEY环境变量
2. 运行 `python scripts/08_eval_api_models.py` 批量评估模型
3. 查看 `eval_results/api_models/comparison_report.md` 对比结果
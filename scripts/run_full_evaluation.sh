#!/bin/bash
# scripts/run_full_evaluation.sh
# 运行微调模型的完整评估 (0/1/3/5-shot)

set -e

MODEL_PATH="models/qwen3-hrrp-finetuned"
MODEL_NAME="Qwen3-8B-HRRP-Finetuned"
DATASET="simulated"
TAG="finetuned_full_eval"
NUM_TASKS=180

echo "═══════════════════════════════════════════════════════════"
echo "🚀 开始完整评估: $MODEL_NAME"
echo "═══════════════════════════════════════════════════════════"
echo "数据集: $DATASET"
echo "任务数: $NUM_TASKS"
echo "配置: 0/1/3/5-shot"
echo "═══════════════════════════════════════════════════════════"

# 0-shot
echo ""
echo "📊 [1/4] 运行 0-shot 评估..."
python src/main_experiment.py \
  --model_name "$MODEL_NAME" \
  --api_key "local" \
  --api_provider "qwen_local" \
  --model_path "$MODEL_PATH" \
  --dataset_key "$DATASET" \
  --experiment_tag "${TAG}_0shot" \
  --n_way 3 \
  --k_shot_support 0 \
  --q_shot_query 1 \
  --num_fsl_tasks $NUM_TASKS \
  --temperature 0.7 \
  --max_tokens_completion 400

echo "✅ 0-shot 完成"

# 1-shot
echo ""
echo "📊 [2/4] 运行 1-shot 评估..."
python src/main_experiment.py \
  --model_name "$MODEL_NAME" \
  --api_key "local" \
  --api_provider "qwen_local" \
  --model_path "$MODEL_PATH" \
  --dataset_key "$DATASET" \
  --experiment_tag "${TAG}_1shot" \
  --n_way 3 \
  --k_shot_support 1 \
  --q_shot_query 1 \
  --num_fsl_tasks $NUM_TASKS \
  --temperature 0.7 \
  --max_tokens_completion 400

echo "✅ 1-shot 完成"

# 3-shot
echo ""
echo "📊 [3/4] 运行 3-shot 评估..."
python src/main_experiment.py \
  --model_name "$MODEL_NAME" \
  --api_key "local" \
  --api_provider "qwen_local" \
  --model_path "$MODEL_PATH" \
  --dataset_key "$DATASET" \
  --experiment_tag "${TAG}_3shot" \
  --n_way 3 \
  --k_shot_support 3 \
  --q_shot_query 1 \
  --num_fsl_tasks $NUM_TASKS \
  --temperature 0.7 \
  --max_tokens_completion 400

echo "✅ 3-shot 完成"

# 5-shot
echo ""
echo "📊 [4/4] 运行 5-shot 评估..."
python src/main_experiment.py \
  --model_name "$MODEL_NAME" \
  --api_key "local" \
  --api_provider "qwen_local" \
  --model_path "$MODEL_PATH" \
  --dataset_key "$DATASET" \
  --experiment_tag "${TAG}_5shot" \
  --n_way 3 \
  --k_shot_support 5 \
  --q_shot_query 1 \
  --num_fsl_tasks $NUM_TASKS \
  --temperature 0.7 \
  --max_tokens_completion 400

echo "✅ 5-shot 完成"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "🎉 所有评估完成！"
echo "═══════════════════════════════════════════════════════════"
echo "结果保存在: results/llm_experiments_log.csv"
echo ""
echo "查看结果:"
echo "  tail -4 results/llm_experiments_log.csv"

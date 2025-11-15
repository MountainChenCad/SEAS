#!/bin/bash
# scripts/run_full_pipeline.sh
# Qwen3-8B HRRP 目标识别完整微调流程

set -e  # 遇到错误立即退出

echo "======================================"
echo "🚀 Qwen3-8B HRRP 微调完整流程"
echo "======================================"

# 配置参数
BASE_MODEL_PATH="/root/autodl-tmp/Qwen3-8B"
DATASET_DIR="datasets/simulated_hrrp"
OUTPUT_DATA_FILE="data/hrrp_sft_train.json"
ADAPTER_OUTPUT_DIR="output/qwen3-hrrp-lora"
MERGED_MODEL_DIR="models/qwen3-hrrp-finetuned"
NUM_SAMPLES=1000
K_SHOT=3
SEED=42

# ====================================
# Step 1: 生成训练数据
# ====================================
echo ""
echo "📊 Step 1/4: 生成指令微调数据集"
echo "======================================"
echo "  输入: $DATASET_DIR"
echo "  输出: $OUTPUT_DATA_FILE"
echo "  样本数: $NUM_SAMPLES"
echo "  K-shot: $K_SHOT"
echo ""

if [ -f "$OUTPUT_DATA_FILE" ]; then
    read -p "⚠️  数据文件已存在，是否重新生成? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$OUTPUT_DATA_FILE"
    else
        echo "跳过数据生成步骤..."
    fi
fi

if [ ! -f "$OUTPUT_DATA_FILE" ]; then
    python scripts/prepare_sft_data.py \
        --input_dir "$DATASET_DIR" \
        --output_file "$OUTPUT_DATA_FILE" \
        --num_samples $NUM_SAMPLES \
        --k_shot $K_SHOT \
        --seed $SEED

    echo "✅ 数据生成完成！"
else
    echo "✅ 使用现有数据文件"
fi

# ====================================
# Step 2: 检查 LLaMA-Factory
# ====================================
echo ""
echo "🔍 Step 2/4: 检查 LLaMA-Factory 环境"
echo "======================================"

if ! command -v llamafactory-cli &> /dev/null; then
    echo "❌ 错误: 未找到 llamafactory-cli"
    echo "   请先安装 LLaMA-Factory:"
    echo "   git clone https://github.com/hiyouga/LLaMA-Factory.git"
    echo "   cd LLaMA-Factory && pip install -e ."
    exit 1
fi

echo "✅ LLaMA-Factory 已安装"

# ====================================
# Step 3: 启动 LoRA 微调
# ====================================
echo ""
echo "🔥 Step 3/4: 启动 LoRA 微调"
echo "======================================"
echo "  基础模型: $BASE_MODEL_PATH"
echo "  输出目录: $ADAPTER_OUTPUT_DIR"
echo "  配置文件: configs/qwen3_lora_sft.yaml"
echo ""

read -p "⚠️  微调将耗时 2-3 小时，是否继续? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 用户取消微调"
    exit 0
fi

# 启动微调
llamafactory-cli train configs/qwen3_lora_sft.yaml

echo "✅ 微调完成！"

# ====================================
# Step 4: 合并 LoRA 权重
# ====================================
echo ""
echo "🔧 Step 4/4: 合并 LoRA 适配器"
echo "======================================"
echo "  基础模型: $BASE_MODEL_PATH"
echo "  适配器: $ADAPTER_OUTPUT_DIR"
echo "  输出目录: $MERGED_MODEL_DIR"
echo ""

python scripts/merge_lora_adapter.py \
    --base_model "$BASE_MODEL_PATH" \
    --adapter_path "$ADAPTER_OUTPUT_DIR" \
    --output_dir "$MERGED_MODEL_DIR"

echo "✅ 模型合并完成！"

# ====================================
# Step 5: 生成测试脚本
# ====================================
echo ""
echo "📝 生成测试脚本..."

cat > test_finetuned_model.sh << EOF
#!/bin/bash
# 自动生成的测试脚本

echo "🧪 测试微调后的模型"

python src/main_experiment.py \\
    --model_name "Qwen3-8B-HRRP-Finetuned" \\
    --api_key "local" \\
    --api_provider "qwen_local" \\
    --model_path "$MERGED_MODEL_DIR" \\
    --dataset_key "simulated" \\
    --experiment_tag "finetuned_eval" \\
    --n_way 3 \\
    --k_shot_support 0 1 3 5 \\
    --q_shot_query 1 \\
    --num_fsl_tasks 180 \\
    --temperature 0.7 \\
    --max_tokens_completion 400

echo "✅ 测试完成！"
echo "📊 查看结果: results/finetuned_eval/"
EOF

chmod +x test_finetuned_model.sh

# ====================================
# 完成总结
# ====================================
echo ""
echo "======================================"
echo "🎉 完整流程执行成功！"
echo "======================================"
echo ""
echo "📦 输出文件:"
echo "  ✓ 训练数据: $OUTPUT_DATA_FILE"
echo "  ✓ LoRA 适配器: $ADAPTER_OUTPUT_DIR"
echo "  ✓ 完整模型: $MERGED_MODEL_DIR"
echo ""
echo "🚀 下一步操作:"
echo "  1. 运行测试: ./test_finetuned_model.sh"
echo "  2. 查看日志: tensorboard --logdir $ADAPTER_OUTPUT_DIR/logs"
echo "  3. 对比结果: 与 v3 基线对比 (results/)"
echo ""
echo "📊 预期性能提升:"
echo "  • 0-shot: 35.56% → 45-50% (+10-15%)"
echo "  • 5-shot: 66.67% → 75-80% (+8-13%)"
echo ""
echo "======================================"

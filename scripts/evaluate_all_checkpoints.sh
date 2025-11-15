#!/bin/bash
# scripts/evaluate_all_checkpoints.sh
# Automatically merge and evaluate all early stopping checkpoints

set -e

BASE_MODEL="/root/autodl-tmp/Qwen3-8B"
ADAPTER_DIR="output/qwen3-hrrp-lora-earlystop"
DATASET="simulated"
NUM_TASKS=30

echo "═══════════════════════════════════════════════════════════════"
echo "🚀 Early Stopping Experiment - Checkpoint Evaluation Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Define checkpoint-epoch mapping
declare -A CHECKPOINTS
CHECKPOINTS[437]="0.5e"
CHECKPOINTS[874]="1.0e"
CHECKPOINTS[1311]="1.5e"
CHECKPOINTS[1748]="2.0e"
CHECKPOINTS[2185]="2.5e"
CHECKPOINTS[2622]="3.0e"

# Count available checkpoints
echo "📂 Scanning for checkpoints..."
available_checkpoints=()
for step in "${!CHECKPOINTS[@]}"; do
    if [ -d "$ADAPTER_DIR/checkpoint-$step" ]; then
        available_checkpoints+=($step)
        echo "   ✓ Found checkpoint-$step (${CHECKPOINTS[$step]})"
    else
        echo "   ✗ Missing checkpoint-$step (${CHECKPOINTS[$step]})"
    fi
done

if [ ${#available_checkpoints[@]} -eq 0 ]; then
    echo ""
    echo "❌ No checkpoints found in $ADAPTER_DIR"
    echo "   Make sure training has completed and created checkpoints"
    exit 1
fi

echo ""
echo "Found ${#available_checkpoints[@]} checkpoint(s) to process"
echo ""

# Sort checkpoints by step number
IFS=$'\n' sorted_checkpoints=($(sort -n <<<"${available_checkpoints[*]}"))
unset IFS

# Process each checkpoint
for step in "${sorted_checkpoints[@]}"; do
    epoch=${CHECKPOINTS[$step]}

    echo "═══════════════════════════════════════════════════════════════"
    echo "Processing checkpoint-$step ($epoch)"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    # Step 1: Merge LoRA adapter
    echo "📦 [1/2] Merging LoRA adapter..."
    merged_model_dir="models/qwen3-hrrp-earlystop-$epoch"

    if [ -d "$merged_model_dir" ]; then
        echo "   ⚠️ Merged model already exists at $merged_model_dir"
        echo "   Skipping merge step..."
    else
        python scripts/merge_lora_adapter.py \
          --base_model "$BASE_MODEL" \
          --adapter_path "$ADAPTER_DIR/checkpoint-$step" \
          --output_dir "$merged_model_dir"

        echo "   ✅ Merge complete: $merged_model_dir"
    fi

    echo ""

    # Step 2: Evaluate on 30-task benchmark
    echo "🧪 [2/2] Evaluating on $NUM_TASKS-task benchmark..."

    python src/main_experiment.py \
      --model_name "Qwen3-8B-HRRP-EarlyStop-$epoch" \
      --api_key "local" \
      --api_provider "qwen_local" \
      --model_path "$merged_model_dir" \
      --dataset_key "$DATASET" \
      --experiment_tag "earlystop_${epoch}_1shot" \
      --n_way 3 \
      --k_shot_support 1 \
      --q_shot_query 1 \
      --num_fsl_tasks $NUM_TASKS \
      --temperature 0.7 \
      --max_tokens_completion 400

    echo "   ✅ Evaluation complete"
    echo ""

    # Show quick result
    echo "📊 Quick Result:"
    result=$(tail -1 results/llm_experiments_log.csv | cut -d',' -f2,19,20,21)
    model_name=$(echo "$result" | cut -d',' -f1)
    accuracy=$(echo "$result" | cut -d',' -f2)
    f1=$(echo "$result" | cut -d',' -f3)
    valid_preds=$(echo "$result" | cut -d',' -f4)

    echo "   Model: $model_name"
    echo "   Accuracy: $accuracy"
    echo "   F1 Score: $f1"
    echo "   Valid Predictions: $valid_preds"
    echo ""
done

echo "═══════════════════════════════════════════════════════════════"
echo "🎉 All Checkpoints Processed!"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Generate comparison report
echo "📊 Generating Comparison Report..."
echo ""

python -c "
import pandas as pd
import sys

# Read results
df = pd.read_csv('results/llm_experiments_log.csv')

# Filter early stopping results
earlystop_df = df[df['model_name'].str.contains('EarlyStop', na=False)].copy()

if len(earlystop_df) == 0:
    print('❌ No early stopping results found in CSV')
    sys.exit(1)

# Extract epoch from experiment_tag
earlystop_df['epoch'] = earlystop_df['experiment_tag'].str.extract(r'earlystop_([\d.]+)e')[0].astype(float)

# Sort by epoch
earlystop_df = earlystop_df.sort_values('epoch')

# Get baseline for comparison
baseline_df = df[(df['model_name'] == 'Qwen3-8B-Local') & (df['k_shot_support'] == 1)]
baseline_acc = baseline_df['accuracy'].values[0] if len(baseline_df) > 0 else 0.65

print('┌─────────────────────────────────────────────────────────────┐')
print('│         Early Stopping Results Comparison                  │')
print('├─────────────────────────────────────────────────────────────┤')
print(f'│ Baseline (Qwen3-8B, 1-shot): {baseline_acc:.2%}' + ' ' * 24 + '│')
print('├────────┬──────────┬─────────┬─────────┬────────────────────┤')
print('│ Epoch  │ Accuracy │ F1 Score│ Valid   │ vs Baseline        │')
print('├────────┼──────────┼─────────┼─────────┼────────────────────┤')

best_acc = 0
best_epoch = 0

for _, row in earlystop_df.iterrows():
    epoch = row['epoch']
    acc = row['accuracy']
    f1 = row['f1_macro']
    valid = f'{int(row[\"valid_preds_count\"])}/{int(row[\"total_queries_eval\"])}'
    diff = acc - baseline_acc
    diff_str = f'{diff:+.2%}' if diff < 0 else f'+{diff:.2%}'

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch

    marker = ' ★' if acc == best_acc else '  '
    print(f'│ {epoch:4.1f}e │ {acc:7.2%} │ {f1:7.4f} │ {valid:7} │ {diff_str:18} {marker}│')

print('└────────┴──────────┴─────────┴─────────┴────────────────────┘')
print()
print(f'🏆 Best Performance: {best_acc:.2%} at {best_epoch:.1f} epochs')

if best_acc >= baseline_acc:
    print(f'✅ SUCCESS! Exceeded baseline by {(best_acc - baseline_acc):.2%}')
elif best_acc >= 0.60:
    print(f'✅ GOOD! Within 5% of baseline ({abs(best_acc - baseline_acc):.2%} difference)')
elif best_acc >= 0.50:
    print(f'⚠️ PARTIAL: Minimum target met, but below baseline by {(baseline_acc - best_acc):.2%}')
else:
    print(f'❌ FAILED: Below minimum target (50%), {(baseline_acc - best_acc):.2%} below baseline')

print()
print('📝 Full results saved in: results/llm_experiments_log.csv')
"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Evaluation Pipeline Complete!"
echo "═══════════════════════════════════════════════════════════════"

#!/bin/bash
# scripts/monitor_training.sh
# Monitor early stopping experiment training progress

OUTPUT_DIR="output/qwen3-hrrp-lora-earlystop"
LOG_FILE="logs/training_earlystop.log"

echo "════════════════════════════════════════════════════════"
echo "📊 Early Stopping Experiment - Training Monitor"
echo "════════════════════════════════════════════════════════"
echo ""

# Check if training process is running
echo "🔍 Training Process Status:"
if pgrep -f "train_lora_direct.py.*earlystop" > /dev/null; then
    echo "   ✅ Training is RUNNING"
    ps aux | grep "train_lora_direct.py.*earlystop" | grep -v grep | awk '{print "   PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
else
    echo "   ⚠️ Training process NOT found"
fi
echo ""

# Show recent log entries
echo "📝 Recent Training Log (last 20 lines):"
echo "────────────────────────────────────────────────────────"
if [ -f "$LOG_FILE" ]; then
    tail -20 "$LOG_FILE"
else
    echo "   ⚠️ Log file not found: $LOG_FILE"
fi
echo ""

# List checkpoints
echo "💾 Saved Checkpoints:"
echo "────────────────────────────────────────────────────────"
if [ -d "$OUTPUT_DIR" ]; then
    for checkpoint in $(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | sort -V); do
        step=$(basename "$checkpoint" | sed 's/checkpoint-//')
        epoch=$(echo "scale=1; $step / 875" | bc)
        size=$(du -sh "$checkpoint" 2>/dev/null | awk '{print $1}')
        echo "   ✓ $(basename $checkpoint) (~${epoch} epochs) - Size: $size"
    done

    checkpoint_count=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | wc -l)
    if [ "$checkpoint_count" -eq 0 ]; then
        echo "   ⏳ No checkpoints yet (training just started)"
    else
        echo ""
        echo "   Total checkpoints: $checkpoint_count"
    fi
else
    echo "   ⚠️ Output directory not found: $OUTPUT_DIR"
fi
echo ""

echo "════════════════════════════════════════════════════════"
echo "ℹ️ Expected Checkpoints:"
echo "   - checkpoint-437  (~0.5 epochs)"
echo "   - checkpoint-874  (~1.0 epochs)"
echo "   - checkpoint-1311 (~1.5 epochs)"
echo "   - checkpoint-1748 (~2.0 epochs)"
echo "   - checkpoint-2185 (~2.5 epochs)"
echo "   - checkpoint-2622 (~3.0 epochs)"
echo ""
echo "⏱️ Estimated total time: 8-9 hours"
echo "════════════════════════════════════════════════════════"

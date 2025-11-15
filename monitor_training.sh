#!/bin/bash
# 监控后台训练进度脚本

echo "════════════════════════════════════════════════════════════════════════════════"
echo "                    🚀 连训带测 后台训练监控"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# 检查进程是否存在
if pgrep -f "train_lora.py" > /dev/null; then
    echo "✅ 训练进程运行中..."
    PID=$(pgrep -f "train_lora.py")
    echo "   PID: $PID"
else
    echo "❌ 未找到训练进程"
    exit 1
fi

echo ""
echo "📊 实时监控信息:"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# 监控循环
while true; do
    clear
    
    echo "⏰ 时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "PID: $PID"
    echo ""
    
    # 检查进程是否仍在运行
    if ! pgrep -p $PID > /dev/null; then
        echo "❌ 训练进程已结束"
        echo "查看最终日志:"
        echo "  tail -100 training.log"
        break
    fi
    
    # GPU状态
    echo "🖥️  GPU使用状态:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader | sed 's/^/   /'
    echo ""
    
    # 日志进度
    echo "📝 日志进度 (最近20行):"
    echo "────────────────────────────────────────────────────────────────────────────────"
    tail -20 training.log | sed 's/^/   /'
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo ""
    
    # 输出目录
    if [ -d "output/exp_20251022_v1_full" ]; then
        CHECKPOINT_COUNT=$(find output/exp_20251022_v1_full -name "checkpoint-*" -type d 2>/dev/null | wc -l)
        echo "💾 Checkpoint数量: $CHECKPOINT_COUNT"
    fi
    
    echo ""
    echo "按 Ctrl+C 停止监控 (训练继续在后台运行)"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    
    sleep 30
done

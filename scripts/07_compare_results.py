#!/usr/bin/env python3
"""
综合评估分析脚本 - 四层对比分析框架

功能:
1. Layer 1: 改进幅度对比 (improvement magnitude)
2. Layer 2: 按类泛化模式 (per-class generalization)
3. Layer 3: 混淆矩阵演化 (confusion matrix evolution)
4. Layer 4: 跨类泛化差距 (cross-class generalization gap)

输入:
  - eval_results/baseline_new_classes/baseline_eval_new_classes.json
  - eval_results/finetuned_new_classes/finetuned_eval_new_classes.json

输出:
  - eval_results/comprehensive_analysis.json (详细分析)
  - eval_results/comparative_report.txt (文本报告)
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from datetime import datetime

EVAL_CLASSES = ["F22", "F35", "GlobalHawk", "IDF", "Mirage2000", "Predator"]
OUTPUT_DIR = "eval_results"


def load_results(filepath: str) -> dict:
    """加载评估结果"""
    with open(filepath, 'r') as f:
        return json.load(f)


def build_confusion_matrix(predictions: list, classes: list) -> np.ndarray:
    """构建混淆矩阵"""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for pred in predictions:
        true_label = pred['query_label']
        pred_label = pred['predicted_label']

        if true_label in class_to_idx and pred_label in class_to_idx:
            true_idx = class_to_idx[true_label]
            pred_idx = class_to_idx[pred_label]
            cm[true_idx, pred_idx] += 1

    return cm


def compute_per_class_metrics(predictions: list, classes: list) -> dict:
    """计算每类的指标"""
    metrics = {}

    for cls in classes:
        class_preds = [p for p in predictions if p['query_label'] == cls]
        if not class_preds:
            metrics[cls] = {'accuracy': 0, 'count': 0, 'correct': 0}
            continue

        correct = sum(1 for p in class_preds if p['is_correct'])
        total = len(class_preds)
        accuracy = correct / total if total > 0 else 0

        metrics[cls] = {
            'accuracy': accuracy,
            'count': total,
            'correct': correct
        }

    return metrics


def compute_layer1_improvement(baseline_metrics: dict, finetuned_metrics: dict) -> dict:
    """Layer 1: 改进幅度分析"""
    layer1 = {
        'baseline_overall': 0,
        'finetuned_overall': 0,
        'absolute_improvement': 0,
        'relative_improvement_percent': 0,
        'per_class': {}
    }

    # 计算总体准确率
    baseline_acc = np.mean([m['accuracy'] for m in baseline_metrics.values()])
    finetuned_acc = np.mean([m['accuracy'] for m in finetuned_metrics.values()])

    layer1['baseline_overall'] = float(baseline_acc)
    layer1['finetuned_overall'] = float(finetuned_acc)
    layer1['absolute_improvement'] = float(finetuned_acc - baseline_acc)

    if baseline_acc > 0:
        layer1['relative_improvement_percent'] = float((finetuned_acc - baseline_acc) / baseline_acc * 100)

    # 每类改进
    for cls in EVAL_CLASSES:
        baseline_cls_acc = baseline_metrics[cls]['accuracy']
        finetuned_cls_acc = finetuned_metrics[cls]['accuracy']
        abs_imp = finetuned_cls_acc - baseline_cls_acc
        rel_imp = (abs_imp / baseline_cls_acc * 100) if baseline_cls_acc > 0 else 0

        layer1['per_class'][cls] = {
            'baseline_accuracy': float(baseline_cls_acc),
            'finetuned_accuracy': float(finetuned_cls_acc),
            'absolute_improvement': float(abs_imp),
            'relative_improvement_percent': float(rel_imp)
        }

    return layer1


def compute_layer2_generalization(baseline_metrics: dict, finetuned_metrics: dict) -> dict:
    """Layer 2: 泛化模式分析"""
    layer2 = {
        'baseline_per_class': {},
        'finetuned_per_class': {},
        'generalization_success_rate': 0,
        'difficult_classes': [],
        'easy_classes': []
    }

    # 记录每类性能
    for cls in EVAL_CLASSES:
        layer2['baseline_per_class'][cls] = {
            'accuracy': float(baseline_metrics[cls]['accuracy']),
            'samples': baseline_metrics[cls]['count']
        }
        layer2['finetuned_per_class'][cls] = {
            'accuracy': float(finetuned_metrics[cls]['accuracy']),
            'samples': finetuned_metrics[cls]['count']
        }

    # 计算泛化成功率 (多少类得到改进)
    improved_classes = sum(
        1 for cls in EVAL_CLASSES
        if finetuned_metrics[cls]['accuracy'] > baseline_metrics[cls]['accuracy']
    )
    layer2['generalization_success_rate'] = float(improved_classes / len(EVAL_CLASSES))

    # 识别困难类和简单类
    finetuned_accs = [finetuned_metrics[cls]['accuracy'] for cls in EVAL_CLASSES]
    median_acc = np.median(finetuned_accs)

    for cls in EVAL_CLASSES:
        if finetuned_metrics[cls]['accuracy'] < median_acc * 0.7:
            layer2['difficult_classes'].append(cls)
        elif finetuned_metrics[cls]['accuracy'] > median_acc * 1.3:
            layer2['easy_classes'].append(cls)

    return layer2


def compute_layer3_confusion(baseline_cm: np.ndarray, finetuned_cm: np.ndarray) -> dict:
    """Layer 3: 混淆矩阵演化分析"""
    layer3 = {
        'baseline_confusion_matrix': baseline_cm.tolist(),
        'finetuned_confusion_matrix': finetuned_cm.tolist(),
        'confusion_reduction': {},
        'persistent_confusions': []
    }

    # 计算混淆减少
    for i, cls_i in enumerate(EVAL_CLASSES):
        for j, cls_j in enumerate(EVAL_CLASSES):
            if i != j:
                baseline_conf = baseline_cm[i, j]
                finetuned_conf = finetuned_cm[i, j]
                reduction = baseline_conf - finetuned_conf

                if baseline_conf > 0:
                    layer3['confusion_reduction'][f'{cls_i}_to_{cls_j}'] = {
                        'baseline': int(baseline_conf),
                        'finetuned': int(finetuned_conf),
                        'absolute_reduction': int(reduction),
                        'reduction_percent': float(reduction / baseline_conf * 100) if baseline_conf > 0 else 0
                    }

                    # 持续混淆 (微调后仍然存在的混淆)
                    if finetuned_conf > 0 and baseline_conf > 0:
                        layer3['persistent_confusions'].append({
                            'from_class': cls_i,
                            'to_class': cls_j,
                            'count': int(finetuned_conf)
                        })

    return layer3


def compute_layer4_generalization_gap(baseline_metrics: dict, finetuned_metrics: dict) -> dict:
    """Layer 4: 跨类泛化差距分析"""
    layer4 = {
        'training_to_evaluation_gap': 0,
        'baseline_std': 0,
        'finetuned_std': 0,
        'consistency_improvement': 0,
        'class_wise_gaps': {}
    }

    baseline_accs = np.array([baseline_metrics[cls]['accuracy'] for cls in EVAL_CLASSES])
    finetuned_accs = np.array([finetuned_metrics[cls]['accuracy'] for cls in EVAL_CLASSES])

    # 标准差 (衡量类间不一致)
    baseline_std = float(np.std(baseline_accs))
    finetuned_std = float(np.std(finetuned_accs))

    layer4['baseline_std'] = baseline_std
    layer4['finetuned_std'] = finetuned_std
    layer4['consistency_improvement'] = float(baseline_std - finetuned_std)

    # 类间准确率差距
    baseline_gap = float(np.max(baseline_accs) - np.min(baseline_accs))
    finetuned_gap = float(np.max(finetuned_accs) - np.min(finetuned_accs))
    layer4['training_to_evaluation_gap'] = finetuned_gap

    # 每类与平均值的偏差
    baseline_mean = np.mean(baseline_accs)
    finetuned_mean = np.mean(finetuned_accs)

    for cls in EVAL_CLASSES:
        baseline_dev = baseline_metrics[cls]['accuracy'] - baseline_mean
        finetuned_dev = finetuned_metrics[cls]['accuracy'] - finetuned_mean

        layer4['class_wise_gaps'][cls] = {
            'baseline_deviation': float(baseline_dev),
            'finetuned_deviation': float(finetuned_dev),
            'consistency_change': float(baseline_dev - finetuned_dev)
        }

    return layer4


def main():
    """主函数"""
    print("\\n" + "="*70)
    print("综合评估分析")
    print("="*70 + "\\n")

    # 加载结果
    baseline_file = Path(OUTPUT_DIR) / "baseline_new_classes" / "baseline_eval_new_classes.json"
    finetuned_file = Path(OUTPUT_DIR) / "finetuned_new_classes" / "finetuned_eval_new_classes.json"

    if not baseline_file.exists():
        print(f"✗ 基线结果不存在: {baseline_file}")
        return

    if not finetuned_file.exists():
        print(f"⚠ 微调结果不存在: {finetuned_file}")
        print("  将使用基线结果作为微调结果")
        finetuned_results = load_results(str(baseline_file))
    else:
        finetuned_results = load_results(str(finetuned_file))

    baseline_results = load_results(str(baseline_file))

    print(f"✓ 加载基线结果: {baseline_file}")
    print(f"✓ 加载微调结果: {finetuned_file}\\n")

    # 计算指标
    print("计算每类指标...")
    baseline_metrics = compute_per_class_metrics(baseline_results['predictions'], EVAL_CLASSES)
    finetuned_metrics = compute_per_class_metrics(finetuned_results['predictions'], EVAL_CLASSES)
    print("✓ 完成\\n")

    print("构建混淆矩阵...")
    baseline_cm = build_confusion_matrix(baseline_results['predictions'], EVAL_CLASSES)
    finetuned_cm = build_confusion_matrix(finetuned_results['predictions'], EVAL_CLASSES)
    print("✓ 完成\\n")

    # 四层分析
    print("执行四层对比分析...")
    layer1 = compute_layer1_improvement(baseline_metrics, finetuned_metrics)
    print("  ✓ Layer 1: 改进幅度分析")

    layer2 = compute_layer2_generalization(baseline_metrics, finetuned_metrics)
    print("  ✓ Layer 2: 泛化模式分析")

    layer3 = compute_layer3_confusion(baseline_cm, finetuned_cm)
    print("  ✓ Layer 3: 混淆矩阵演化分析")

    layer4 = compute_layer4_generalization_gap(baseline_metrics, finetuned_metrics)
    print("  ✓ Layer 4: 跨类泛化差距分析\\n")

    # 保存综合分析结果
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'eval_classes': EVAL_CLASSES,
        'layer1_improvement': layer1,
        'layer2_generalization': layer2,
        'layer3_confusion': layer3,
        'layer4_cross_class_gap': layer4
    }

    analysis_file = Path(OUTPUT_DIR) / "comprehensive_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"✓ 分析结果保存到: {analysis_file}\\n")

    # 生成文本报告
    print("生成文本报告...")
    report = generate_report(analysis)

    report_file = Path(OUTPUT_DIR) / "comparative_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"✓ 文本报告保存到: {report_file}\\n")

    # 打印摘要
    print(report)


def generate_report(analysis: dict) -> str:
    """生成文本报告"""
    layer1 = analysis['layer1_improvement']
    layer2 = analysis['layer2_generalization']
    layer3 = analysis['layer3_confusion']
    layer4 = analysis['layer4_cross_class_gap']

    report = f"""
{'='*70}
综合评估分析报告
{'='*70}

生成时间: {analysis['timestamp']}
评估类别: {', '.join(analysis['eval_classes'])}

{'='*70}
Layer 1: 改进幅度分析
{'='*70}

总体准确率:
  基线:    {layer1['baseline_overall']:.2%}
  微调后:  {layer1['finetuned_overall']:.2%}

改进情况:
  绝对改进: {layer1['absolute_improvement']:+.4f}
  相对改进: {layer1['relative_improvement_percent']:+.2f}%

每类改进:
"""

    for cls, metrics in sorted(layer1['per_class'].items()):
        report += f"  {cls:15s}: {metrics['baseline_accuracy']:.2%} → {metrics['finetuned_accuracy']:.2%} "
        report += f"({metrics['absolute_improvement']:+.4f}, {metrics['relative_improvement_percent']:+.2f}%)\\n"

    report += f"""
{'='*70}
Layer 2: 泛化模式分析
{'='*70}

泛化成功率: {layer2['generalization_success_rate']:.1%} (改进的类数 / 总类数)

困难类 (准确率较低):
"""

    if layer2['difficult_classes']:
        for cls in layer2['difficult_classes']:
            acc = layer2['finetuned_per_class'][cls]['accuracy']
            report += f"  {cls:15s}: {acc:.2%}\\n"
    else:
        report += "  无\\n"

    report += f"""
简单类 (准确率较高):
"""

    if layer2['easy_classes']:
        for cls in layer2['easy_classes']:
            acc = layer2['finetuned_per_class'][cls]['accuracy']
            report += f"  {cls:15s}: {acc:.2%}\\n"
    else:
        report += "  无\\n"

    report += f"""
{'='*70}
Layer 3: 混淆矩阵演化分析
{'='*70}

混淆减少的前5个类对:
"""

    # 排序混淆减少
    sorted_confusions = sorted(
        [(k, v) for k, v in layer3['confusion_reduction'].items() if v['reduction_percent'] > 0],
        key=lambda x: x[1]['reduction_percent'],
        reverse=True
    )[:5]

    for pair, metrics in sorted_confusions:
        if metrics['baseline'] > 0:
            report += f"  {pair:30s}: {metrics['baseline']} → {metrics['finetuned']} "
            report += f"({metrics['reduction_percent']:.1f}% 减少)\\n"

    report += f"""
{'='*70}
Layer 4: 跨类泛化差距分析
{'='*70}

准确率一致性:
  基线标准差:   {layer4['baseline_std']:.4f}
  微调后标准差: {layer4['finetuned_std']:.4f}
  一致性改进:   {layer4['consistency_improvement']:+.4f}

类间准确率差距: {layer4['training_to_evaluation_gap']:.4f}

{'='*70}
总体结论
{'='*70}

微调模型在新类上的表现:
  ✓ 总体准确率改进: {layer1['absolute_improvement']:+.4f} ({layer1['relative_improvement_percent']:+.2f}%)
  ✓ 泛化成功率: {layer2['generalization_success_rate']:.1%}
  ✓ 类间一致性改进: {layer4['consistency_improvement']:+.4f}

建议:
  - 重点关注困难类: {', '.join(layer2['difficult_classes']) if layer2['difficult_classes'] else '无'}
  - 模型已较好地学习的类: {', '.join(layer2['easy_classes']) if layer2['easy_classes'] else '所有'}

{'='*70}
"""

    return report


if __name__ == '__main__':
    main()

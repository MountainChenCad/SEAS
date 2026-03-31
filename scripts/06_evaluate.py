#!/usr/bin/env python3
"""
批量评估SiliconFlow上的3个微调模型
对比initial_commit, ckpt_step_406, ckpt_step_203三个版本的性能

Usage:
    export SILICONFLOW_API_KEY='your_api_key'
    python scripts/08_eval_api_models.py --eval-tasks data/eval_tasks_new_6classes.json
"""

import json
import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("eval_api_models.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# 模型配置
MODELS = {
    "initial_commit": {
        "id": "ft:LoRA/Qwen/Qwen2.5-7B-Instruct:rpl47v9x40:initial_commit:uyulemtufwhthcnywhcj",
        "desc": "Initial Commit (Epoch 1)",
        "output_dir": "eval_results/api_models/initial_commit",
    },
    "ckpt_step_406": {
        "id": "ft:LoRA/Qwen/Qwen2.5-7B-Instruct:rpl47v9x40:initial_commit:uyulemtufwhthcnywhcj-ckpt_step_406",
        "desc": "Checkpoint at Step 406 (Epoch 2)",
        "output_dir": "eval_results/api_models/ckpt_step_406",
    },
    "ckpt_step_203": {
        "id": "ft:LoRA/Qwen/Qwen2.5-7B-Instruct:rpl47v9x40:initial_commit:uyulemtufwhthcnywhcj-ckpt_step_203",
        "desc": "Checkpoint at Step 203 (Epoch 1)",
        "output_dir": "eval_results/api_models/ckpt_step_203",
    },
}

EVAL_TASKS_FILE = "data/eval_tasks_new_6classes.json"
OUTPUT_BASE_DIR = "eval_results/api_models"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批量评估SiliconFlow微调模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--eval-tasks",
        type=str,
        default=EVAL_TASKS_FILE,
        help="评估任务JSON文件",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_BASE_DIR,
        help="输出基目录",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="SiliconFlow API密钥",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="要评估的模型 (默认全部)",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="限制每个模型的样本数（用于快速测试）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="温度参数",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="跳过推理，仅生成对比报告",
    )

    return parser.parse_args()


def run_inference_for_model(
    model_key: str,
    model_id: str,
    eval_tasks_file: str,
    output_dir: str,
    api_key: str = None,
    limit_samples: int = None,
    temperature: float = 0.1,
) -> Tuple[bool, Dict]:
    """
    为单个模型运行推理

    Returns:
        (success, results_dict)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"正在评估模型: {model_key}")
    logger.info(f"{'='*80}")

    # 构建命令
    cmd = [
        "python",
        "scripts/06_run_inference.py",
        "--model-id",
        model_id,
        "--eval-tasks",
        eval_tasks_file,
        "--output-dir",
        output_dir,
        "--temperature",
        str(temperature),
    ]

    if api_key:
        cmd.extend(["--api-key", api_key])

    if limit_samples:
        cmd.extend(["--limit-samples", str(limit_samples)])

    logger.info(f"执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=False, check=False)
        if result.returncode == 0:
            logger.info(f"✓ 模型{model_key}推理成功")
            # 加载结果
            result_file = Path(output_dir) / "results.json"
            if result_file.exists():
                with open(result_file, "r") as f:
                    results = json.load(f)
                return True, results
            else:
                logger.error(f"✗ 找不到结果文件: {result_file}")
                return False, {}
        else:
            logger.error(f"✗ 模型{model_key}推理失败 (return code: {result.returncode})")
            return False, {}

    except Exception as e:
        logger.error(f"✗ 模型{model_key}推理过程出错: {e}")
        return False, {}


def generate_comparison_report(
    all_results: Dict[str, Dict],
    output_file: str,
) -> bool:
    """
    生成对比报告

    Args:
        all_results: 所有模型的结果字典
        output_file: 输出文件路径

    Returns:
        是否成功生成报告
    """
    logger.info(f"\n生成对比报告: {output_file}")

    try:
        # 准备对比数据
        comparison_rows = []

        for model_key in ["initial_commit", "ckpt_step_406", "ckpt_step_203"]:
            if model_key not in all_results:
                continue

            results = all_results[model_key]
            stats = results.get("statistics", {})

            comparison_rows.append(
                {
                    "模型": MODELS[model_key]["desc"],
                    "总任务": stats.get("total_tasks", "N/A"),
                    "成功任务": stats.get("successful_tasks", "N/A"),
                    "正确预测": stats.get("correct_count", "N/A"),
                    "准确率": f"{stats.get('accuracy', 0):.2f}%",
                }
            )

        # 生成Markdown报告
        report = []
        report.append("# SiliconFlow 微调模型评估报告\n")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**评估集**: 6个新类 (F22, F35, GlobalHawk, IDF, Mirage2000, Predator)\n")
        report.append(f"**评估任务数**: 150 (6-way 1-shot)\n")
        report.append("\n## 性能对比\n")
        report.append("| 模型 | 总任务 | 成功任务 | 正确预测 | 准确率 |\n")
        report.append("|------|--------|--------|--------|--------|\n")

        for row in comparison_rows:
            report.append(
                f"| {row['模型']} | {row['总任务']} | {row['成功任务']} | {row['正确预测']} | {row['准确率']} |\n"
            )

        report.append("\n## 详细分析\n")

        for model_key in ["initial_commit", "ckpt_step_406", "ckpt_step_203"]:
            if model_key not in all_results:
                continue

            results = all_results[model_key]
            stats = results.get("statistics", {})

            report.append(f"\n### {MODELS[model_key]['desc']}\n")
            report.append(f"- **准确率**: {stats.get('accuracy', 0):.2f}%\n")
            report.append(f"- **成功任务**: {stats.get('successful_tasks', 'N/A')} / {stats.get('total_tasks', 'N/A')}\n")
            report.append(f"- **正确预测**: {stats.get('correct_count', 'N/A')}\n")

            # 分析错误情况
            detailed_results = results.get("results", [])
            error_count = sum(1 for r in detailed_results if not r.get("success", False))
            parse_errors = sum(
                1
                for r in detailed_results
                if r.get("success", False) and r.get("predicted_label") == "PARSE_ERROR"
            )

            if error_count > 0:
                report.append(f"- **错误任务**: {error_count}\n")
            if parse_errors > 0:
                report.append(f"- **解析错误**: {parse_errors}\n")

        report.append("\n## 结论\n")

        # 找出最佳模型
        best_model_key = max(
            all_results.keys(),
            key=lambda k: all_results[k].get("statistics", {}).get("accuracy", 0),
        )
        best_accuracy = all_results[best_model_key].get("statistics", {}).get("accuracy", 0)

        report.append(f"\n**最佳模型**: {MODELS[best_model_key]['desc']} (准确率: {best_accuracy:.2f}%)\n")

        # 保存报告
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("".join(report))

        logger.info(f"✓ 对比报告已保存: {output_file}")
        return True

    except Exception as e:
        logger.error(f"✗ 生成报告失败: {e}")
        return False


def print_summary(all_results: Dict[str, Dict]):
    """打印汇总信息"""
    logger.info("\n" + "=" * 80)
    logger.info("评估汇总")
    logger.info("=" * 80)

    for model_key in ["initial_commit", "ckpt_step_406", "ckpt_step_203"]:
        if model_key not in all_results:
            logger.info(f"\n{MODELS[model_key]['desc']}: 跳过")
            continue

        results = all_results[model_key]
        stats = results.get("statistics", {})

        logger.info(f"\n{MODELS[model_key]['desc']}:")
        logger.info(f"  准确率: {stats.get('accuracy', 0):.2f}%")
        logger.info(f"  成功/总任务: {stats.get('successful_tasks', 'N/A')} / {stats.get('total_tasks', 'N/A')}")
        logger.info(f"  正确预测: {stats.get('correct_count', 'N/A')}")

    logger.info("\n" + "=" * 80)


def main():
    """主函数"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("SiliconFlow微调模型批量评估")
    logger.info("=" * 80)
    logger.info(f"评估任务文件: {args.eval_tasks}")
    logger.info(f"输出基目录: {args.output_dir}")
    logger.info(f"要评估的模型: {', '.join(args.models)}")

    # 验证评估任务文件
    if not Path(args.eval_tasks).exists():
        logger.error(f"✗ 评估任务文件不存在: {args.eval_tasks}")
        return 1

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 收集所有模型的结果
    all_results = {}

    # 为每个模型运行推理
    if not args.skip_inference:
        for model_key in args.models:
            if model_key not in MODELS:
                logger.warning(f"⚠ 未知模型: {model_key}")
                continue

            model_config = MODELS[model_key]
            success, results = run_inference_for_model(
                model_key=model_key,
                model_id=model_config["id"],
                eval_tasks_file=args.eval_tasks,
                output_dir=model_config["output_dir"],
                api_key=args.api_key,
                limit_samples=args.limit_samples,
                temperature=args.temperature,
            )

            if success:
                all_results[model_key] = results
            else:
                logger.warning(f"⚠ 模型{model_key}推理失败，跳过")
    else:
        # 跳过推理，加载现有结果
        logger.info("\n跳过推理，加载现有结果...")
        for model_key in args.models:
            model_config = MODELS[model_key]
            result_file = Path(model_config["output_dir"]) / "results.json"
            if result_file.exists():
                with open(result_file, "r") as f:
                    all_results[model_key] = json.load(f)
                logger.info(f"✓ 已加载模型{model_key}的结果")
            else:
                logger.warning(f"⚠ 找不到模型{model_key}的结果: {result_file}")

    # 打印汇总
    print_summary(all_results)

    # 生成对比报告
    if all_results:
        report_file = Path(args.output_dir) / "comparison_report.md"
        if generate_comparison_report(all_results, str(report_file)):
            logger.info(f"✓ 对比报告已生成: {report_file}")
        else:
            logger.warning("⚠ 对比报告生成失败")
    else:
        logger.warning("⚠ 没有可用的评估结果")
        return 1

    logger.info("\n✓ 批量评估完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())

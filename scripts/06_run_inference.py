#!/usr/bin/env python3
"""
统一推理脚本:在后6个新类上进行推理评估
支持基线模型和微调模型

Usage:
    python inference_new_classes.py --model_type baseline
    python inference_new_classes.py --model_type finetuned --adapter_path output/qwen3-hrrp-lora-sft-baseline
"""

import json
import os
import random
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.io import loadmat
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference_new_classes.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入项目代码
sys.path.insert(0, '/root/autodl-tmp/projects/TableLlama-HRRP')
sys.path.insert(0, '/root/autodl-tmp/projects/TableLlama-HRRP/src')

from src.config import SCATTERING_CENTER_EXTRACTION, SCATTERING_CENTER_ENCODING
from src.feature_extractor import extract_scattering_centers_peak_detection
from src.scattering_center_encoder import encode_single_sc_set_to_text
from src.llm_utils import parse_llm_output_for_label

# 配置
EVAL_CLASSES = ["F22", "F35", "GlobalHawk", "IDF", "Mirage2000", "Predator"]
MODEL_PATH = "/root/autodl-tmp/Qwen3-8B"
EVAL_TASKS_FILE = "data/eval_tasks_new_6classes.json"
OUTPUT_DIR = "eval_results"

# 推理参数
MAX_NEW_TOKENS = 3000
TEMPERATURE = 0.1
TOP_P = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description='在新类上运行推理评估')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['baseline', 'finetuned'],
                        help='模型类型: baseline(原始) 或 finetuned(微调)')
    parser.add_argument('--adapter_path', type=str, default=None,
                        help='LoRA adapter路径 (仅finetuned需要)')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='基础模型路径')
    parser.add_argument('--eval_tasks', type=str, default=EVAL_TASKS_FILE,
                        help='评估任务JSON文件')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='输出目录')
    return parser.parse_args()


def load_hrrp_file(filepath: str) -> Optional[np.ndarray]:
    """加载HRRP数据"""
    try:
        mat_data = loadmat(filepath)
        hrrp = mat_data.get('hrrp', mat_data.get('data', None))
        if hrrp is None:
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if keys:
                hrrp = mat_data[keys[0]]

        if isinstance(hrrp, np.ndarray):
            if hrrp.ndim > 1:
                hrrp = hrrp.flatten()
            if np.iscomplexobj(hrrp):
                hrrp = np.abs(hrrp)
            return hrrp.astype(np.float32)
    except Exception as e:
        logger.warning(f"加载失败 {filepath}: {e}")
    return None


def extract_and_encode_sc(hrrp: np.ndarray) -> str:
    """提取和编码散射中心"""
    centers = extract_scattering_centers_peak_detection(
        hrrp,
        prominence=SCATTERING_CENTER_EXTRACTION['prominence'],
        min_distance=SCATTERING_CENTER_EXTRACTION['min_distance'],
        max_centers_to_keep=SCATTERING_CENTER_EXTRACTION['max_centers_to_keep'],
        normalize_hrrp_before_extraction=SCATTERING_CENTER_EXTRACTION['normalize_hrrp_before_extraction']
    )
    sc_text = encode_single_sc_set_to_text(centers, SCATTERING_CENTER_ENCODING)
    return sc_text


def construct_prompt(task: Dict) -> Tuple[Optional[str], Optional[str]]:
    """构建推理提示"""
    system_prompt = (
        "You are an expert in aircraft classification based on radar HRRP (High-Resolution Range Profile) signals. "
        "Your task is to classify an unknown aircraft based on its scattering center features compared with known examples. "
        "Respond with only the class name."
    )

    support_text = "Known aircraft examples (1 per class):\n\n"
    for class_name in EVAL_CLASSES:
        if class_name in task['support_examples']:
            support_file = task['support_examples'][class_name]
            support_hrrp = load_hrrp_file(support_file)
            if support_hrrp is not None:
                support_sc = extract_and_encode_sc(support_hrrp)
                support_text += f"Class '{class_name}':\nScattering Centers: {support_sc}\n\n"

    query_hrrp = load_hrrp_file(task['query_file'])
    if query_hrrp is None:
        return None, None

    query_sc = extract_and_encode_sc(query_hrrp)

    query_text = (
        f"Unknown aircraft to classify:\nScattering Centers: {query_sc}\n\n"
        f"Candidate classes: {', '.join(EVAL_CLASSES)}\n"
        f"Respond with ONLY the class name (one of: {', '.join(EVAL_CLASSES)})"
    )

    user_prompt = support_text + query_text

    return system_prompt, user_prompt


def run_inference(model, tokenizer, system_prompt: str, user_prompt: str) -> Optional[str]:
    """运行推理"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=False
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response.strip()
    except Exception as e:
        logger.error(f"推理错误: {e}")
        return None


def load_model(args):
    """加载模型 (基线或微调)"""
    logger.info(f"加载模型: {args.model_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.model_type == 'baseline':
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        logger.info("✓ 基线模型加载完成")
    else:  # finetuned
        if not args.adapter_path:
            raise ValueError("微调模式需要提供 --adapter_path")

        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )

        # 加载LoRA adapter
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        logger.info(f"✓ 微调模型加载完成 (adapter: {args.adapter_path})")

    model.eval()
    return model, tokenizer


def main():
    args = parse_args()

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 打印配置
    logger.info("\n" + "="*70)
    logger.info(f"{'基线' if args.model_type == 'baseline' else '微调'}模型推理 (后6类新类)")
    logger.info("="*70)
    logger.info(f"类别: {', '.join(EVAL_CLASSES)}")
    logger.info(f"模型: Qwen3-8B ({args.model_type})")
    if args.model_type == 'finetuned':
        logger.info(f"Adapter: {args.adapter_path}")
    logger.info("="*70 + "\n")

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_subdir = Path(args.output_dir) / f"{args.model_type}_new_classes"
    output_subdir.mkdir(parents=True, exist_ok=True)

    # 加载评估任务
    logger.info(f"加载评估任务: {args.eval_tasks}")
    with open(args.eval_tasks, 'r') as f:
        tasks = json.load(f)
    logger.info(f"✓ 加载了 {len(tasks)} 个任务\n")

    # 加载模型
    model, tokenizer = load_model(args)

    # 推理
    logger.info("开始推理...")
    logger.info("="*70 + "\n")

    start_time = datetime.now()
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': f'Qwen3-8B ({args.model_type})',
        'adapter_path': args.adapter_path if args.model_type == 'finetuned' else None,
        'eval_classes': EVAL_CLASSES,
        'config': {
            'n_way': 6,
            'k_shot': 1,
            'q_shot': 1,
            'classes': EVAL_CLASSES,
            'max_tokens': MAX_NEW_TOKENS,
            'temperature': TEMPERATURE
        },
        'predictions': [],
        'statistics': {}
    }

    total_tasks = 0
    correct_count = 0
    class_results = {c: {'correct': 0, 'total': 0} for c in EVAL_CLASSES}

    for task_idx, task in enumerate(tasks, 1):
        target_class = task['query_label']

        # 构建提示
        system_prompt, user_prompt = construct_prompt(task)
        if system_prompt is None:
            continue

        # 运行推理
        prediction = run_inference(model, tokenizer, system_prompt, user_prompt)
        if prediction is None:
            predicted_label = "UNKNOWN"
        else:
            predicted_label = parse_llm_output_for_label(prediction, EVAL_CLASSES)
            if predicted_label is None:
                predicted_label = "UNKNOWN"

        # 检查正确性
        is_correct = predicted_label == target_class
        if is_correct:
            correct_count += 1
        class_results[target_class]['correct'] += (1 if is_correct else 0)
        class_results[target_class]['total'] += 1
        total_tasks += 1

        # 保存结果
        result = {
            'task_idx': task_idx,
            'query_file': task['query_file'],
            'query_label': target_class,
            'predicted_label': predicted_label,
            'is_correct': is_correct,
            'raw_response': prediction if prediction else "N/A"
        }
        results['predictions'].append(result)

        # 进度日志
        if task_idx % 10 == 0 or is_correct:
            status = "✓" if is_correct else "✗"
            logger.info(f"[{task_idx}/{len(tasks)}] {status} {target_class} -> {predicted_label}")

    # 计算统计
    overall_accuracy = correct_count / total_tasks if total_tasks > 0 else 0
    class_accuracies = {
        c: class_results[c]['correct'] / class_results[c]['total']
        if class_results[c]['total'] > 0 else 0
        for c in EVAL_CLASSES
    }

    results['statistics'] = {
        'total_tasks': total_tasks,
        'correct_count': correct_count,
        'overall_accuracy': overall_accuracy,
        'class_accuracies': class_accuracies,
        'class_results': class_results,
        'runtime_seconds': (datetime.now() - start_time).total_seconds()
    }

    # 保存结果 (为微调模型添加checkpoint标识以避免覆盖)
    if args.model_type == 'finetuned' and args.adapter_path:
        # 从adapter_path提取checkpoint信息: output/qwen3-hrrp-lora-sft-baseline/checkpoint-1000 -> checkpoint-1000
        checkpoint_name = Path(args.adapter_path).name
        output_file = output_subdir / f"{args.model_type}_{checkpoint_name}_eval_new_classes.json"
    else:
        output_file = output_subdir / f"{args.model_type}_eval_new_classes.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印总结
    logger.info("\n" + "="*70)
    logger.info("评估完成")
    logger.info("="*70)
    logger.info(f"总体准确率: {overall_accuracy:.2%} ({correct_count}/{total_tasks})")
    logger.info("\n各类别准确率:")
    for c in EVAL_CLASSES:
        acc = class_accuracies[c]
        logger.info(f"  {c:15s}: {acc:.2%} ({class_results[c]['correct']}/{class_results[c]['total']})")
    logger.info(f"\n结果已保存到: {output_file}")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()

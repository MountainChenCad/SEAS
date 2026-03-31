#!/usr/bin/env python3
"""
本地模型推理脚本 - 用于边训练边评估

支持从本地加载LoRA微调模型进行推理评估。

Usage:
    # 评估本地checkpoint
    python scripts/06b_run_inference_local.py \
        --model-path output/qwen3-hrrp-seas-cot/checkpoint-500 \
        --eval-tasks data/eval_tasks_new_6classes.json \
        --output-dir eval_results/local_ckpt_500

    # 评估最终模型
    python scripts/06b_run_inference_local.py \
        --model-path output/qwen3-hrrp-seas-cot \
        --eval-tasks data/eval_tasks_new_6classes.json \
        --output-dir eval_results/local_final
"""

import json
import os
import sys
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from scipy.io import loadmat
from datetime import datetime
from tqdm import tqdm

# 导入transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import SCATTERING_CENTER_EXTRACTION, SCATTERING_CENTER_ENCODING
from src.feature_extractor import extract_scattering_centers_peak_detection
from src.scattering_center_encoder import encode_single_sc_set_to_text
from src.llm_utils import parse_llm_output_for_label
from src.prompt_constructor_seas import SEASPromptConstructor, construct_seas_prompt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# 常量
EVAL_CLASSES = ["F22", "F35", "GlobalHawk", "IDF", "Mirage2000", "Predator"]
EVAL_TASKS_FILE = "data/eval_tasks_new_6classes.json"
OUTPUT_DIR = "eval_results"

# 推理超参数
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 3000


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="本地模型推理评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    # 评估本地checkpoint
    python scripts/06b_run_inference_local.py \\
        --model-path output/qwen3-hrrp-seas-cot/checkpoint-500 \\
        --eval-tasks data/eval_tasks_new_6classes.json

    # 快速测试（仅评估10个样本）
    python scripts/06b_run_inference_local.py \\
        --model-path output/qwen3-hrrp-seas-cot \\
        --limit-samples 10
        """,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="本地模型路径（LoRA adapter目录或完整模型路径）",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="/root/autodl-tmp/Qwen3-8B",
        help="基础模型路径（用于加载LoRA adapter）",
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
        default=OUTPUT_DIR,
        help="输出结果目录",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="温度参数（0-2）",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="top_p采样参数",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="最大生成token数",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="仅处理前N个任务（用于快速测试）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批处理大小（本地推理建议设为1）",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="使用bf16精度",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="使用fp16精度",
    )

    return parser.parse_args()


def load_hrrp_file(filepath: str) -> Optional[np.ndarray]:
    """加载HRRP数据文件"""
    try:
        mat_data = loadmat(filepath)
        hrrp = mat_data.get("hrrp", mat_data.get("data", None))

        if hrrp is None:
            keys = [k for k in mat_data.keys() if not k.startswith("__")]
            if keys:
                hrrp = mat_data[keys[0]]

        if isinstance(hrrp, np.ndarray):
            if hrrp.ndim > 1:
                hrrp = hrrp.flatten()
            if np.iscomplexobj(hrrp):
                hrrp = np.abs(hrrp)
            return hrrp.astype(np.float32)

    except Exception as e:
        logger.warning(f"加载HRRP文件失败 {filepath}: {e}")

    return None


def extract_and_encode_sc(hrrp: np.ndarray) -> str:
    """提取和编码散射中心"""
    centers = extract_scattering_centers_peak_detection(
        hrrp,
        prominence=SCATTERING_CENTER_EXTRACTION["prominence"],
        min_distance=SCATTERING_CENTER_EXTRACTION["min_distance"],
        max_centers_to_keep=SCATTERING_CENTER_EXTRACTION["max_centers_to_keep"],
        normalize_hrrp_before_extraction=SCATTERING_CENTER_EXTRACTION[
            "normalize_hrrp_before_extraction"
        ],
    )
    sc_text = encode_single_sc_set_to_text(centers, SCATTERING_CENTER_ENCODING)
    return sc_text


def construct_few_shot_prompt(
    query_sc_text: str,
    query_label: str,
    support_examples: Dict[str, Tuple[str, str]],
) -> str:
    """构建Few-Shot Prompt（使用SEAS简化版格式，与训练数据一致）"""
    # 准备支持样本列表
    support_list = []
    for class_name in EVAL_CLASSES:
        if class_name in support_examples:
            sc_text, _ = support_examples[class_name]
            support_list.append((sc_text, class_name))

    # 打乱顺序以消除Position Bias (Primacy Effect)
    random.seed(42)  # 固定种子便于复现
    random.shuffle(support_list)

    # 使用SEAS简化版构造器
    prompt = construct_seas_prompt(
        support_examples=support_list,
        query_sc_text=query_sc_text,
        class_names=EVAL_CLASSES,
        use_answer_tag=True,
    )

    # 添加response前缀引导模型开始reasoning
    prompt += "\n\n### Response:\n<reasoning>\n"

    return prompt


def load_model_and_tokenizer(model_path: str, base_model: str, bf16: bool = False, fp16: bool = False):
    """加载模型和tokenizer"""
    logger.info(f"加载模型: {model_path}")

    # 确定数据类型
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 检查是否是LoRA adapter
    adapter_config_path = Path(model_path) / "adapter_config.json"
    is_lora = adapter_config_path.exists()

    if is_lora:
        logger.info(f"检测到LoRA adapter，基础模型: {base_model}")
        # 加载基础模型
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        # 加载LoRA adapter
        model = PeftModel.from_pretrained(base, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        # 加载完整模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    logger.info("✓ 模型加载完成")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 0.1,
    top_p: float = 1.0,
    max_new_tokens: int = 3000,
) -> str:
    """生成模型响应"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码输出，只取新生成的部分
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


def run_inference_batch(
    model,
    tokenizer,
    eval_tasks: List[Dict],
    temperature: float,
    top_p: float,
    max_tokens: int,
    limit_samples: Optional[int] = None,
) -> Tuple[List[Dict], int]:
    """批量运行推理"""
    results = []
    correct_count = 0

    total_tasks = min(len(eval_tasks), limit_samples) if limit_samples else len(eval_tasks)
    logger.info(f"开始推理，总任务数: {total_tasks}")

    for task_idx, task in enumerate(tqdm(eval_tasks[:total_tasks], desc="推理进度")):
        task_id = task.get("task_id", f"task_{task_idx}")

        try:
            # 加载和编码查询样本
            query_file = task.get("query_file")
            query_label = task.get("query_label", "Unknown")

            if not query_file:
                logger.warning(f"[{task_id}] 缺少query_file")
                continue

            query_hrrp = load_hrrp_file(query_file)
            if query_hrrp is None:
                logger.warning(f"[{task_id}] 无法加载query文件: {query_file}")
                continue

            query_sc_text = extract_and_encode_sc(query_hrrp)

            # 加载和编码支持样本
            support_examples = {}
            support_data = task.get("support_examples", {})

            for class_name, file_path in support_data.items():
                support_hrrp = load_hrrp_file(file_path)
                if support_hrrp is not None:
                    sc_text = extract_and_encode_sc(support_hrrp)
                    support_examples[class_name] = (sc_text, file_path)

            if not support_examples:
                logger.warning(f"[{task_id}] 无法加载支持样本")
                continue

            # 构建prompt
            prompt = construct_few_shot_prompt(query_sc_text, query_label, support_examples)

            # 生成响应
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
            )

            # 解析模型输出 (使用SEAS格式的<answer>标签解析)
            predicted_label = parse_llm_output_for_label(response, EVAL_CLASSES, prefer_answer_tag=True)

            if predicted_label is None:
                logger.warning(f"[{task_id}] 无法解析预测结果，原始输出: {response[:100]}...")
                predicted_label = "PARSE_ERROR"

            is_correct = predicted_label == query_label
            if is_correct:
                correct_count += 1

            result = {
                "task_id": task_id,
                "query_label": query_label,
                "predicted_label": predicted_label,
                "is_correct": is_correct,
                "success": True,
                "response_sample": response[:200],
            }

            status_symbol = "✓" if is_correct else "✗"
            logger.info(
                f"[{task_id}] {status_symbol} {query_label} → {predicted_label} "
                f"({task_idx + 1}/{total_tasks})"
            )

            results.append(result)

        except Exception as e:
            logger.error(f"[{task_id}] 处理失败: {e}", exc_info=True)
            results.append(
                {
                    "task_id": task_id,
                    "error": str(e),
                    "success": False,
                }
            )

    return results, correct_count


def main():
    """主函数"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("本地模型推理开始")
    logger.info("=" * 80)
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"基础模型: {args.base_model}")
    logger.info(f"评估任务: {args.eval_tasks}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"超参数: temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载评估任务
    try:
        with open(args.eval_tasks, "r") as f:
            eval_tasks = json.load(f)
        logger.info(f"✓ 已加载 {len(eval_tasks)} 个评估任务")
    except Exception as e:
        logger.error(f"加载评估任务失败: {e}")
        return 1

    # 加载模型
    try:
        model, tokenizer = load_model_and_tokenizer(
            model_path=args.model_path,
            base_model=args.base_model,
            bf16=args.bf16,
            fp16=args.fp16,
        )
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return 1

    # 运行推理
    try:
        results, correct_count = run_inference_batch(
            model=model,
            tokenizer=tokenizer,
            eval_tasks=eval_tasks,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            limit_samples=args.limit_samples,
        )
    except Exception as e:
        logger.error(f"推理过程出错: {e}", exc_info=True)
        return 1

    # 计算统计信息
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r.get("success", False))
    accuracy = (correct_count / successful_tasks * 100) if successful_tasks > 0 else 0

    # 输出统计信息
    logger.info("=" * 80)
    logger.info("推理完成！统计信息：")
    logger.info("=" * 80)
    logger.info(f"总任务数: {total_tasks}")
    logger.info(f"成功任务: {successful_tasks}")
    logger.info(f"正确预测: {correct_count}")
    logger.info(f"准确率: {accuracy:.2f}%")
    logger.info("=" * 80)

    # 保存结果
    result_file = Path(args.output_dir) / "results.json"
    try:
        with open(result_file, "w") as f:
            json.dump(
                {
                    "model_path": args.model_path,
                    "timestamp": datetime.now().isoformat(),
                    "config": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": args.max_tokens,
                    },
                    "statistics": {
                        "total_tasks": total_tasks,
                        "successful_tasks": successful_tasks,
                        "correct_count": correct_count,
                        "accuracy": accuracy,
                    },
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"✓ 结果已保存到: {result_file}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

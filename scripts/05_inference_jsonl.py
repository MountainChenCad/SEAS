#!/usr/bin/env python3
"""
Zero-Shot Inference Script for JSONL format evaluation data.

Usage:
    # Quick test (10 samples)
    python scripts/06c_run_inference_jsonl.py \
        --model-path output/qwen3-hrrp-zeroshot/checkpoint-450 \
        --eval-data data/hrrp_sft_eval.jsonl \
        --base-model /root/autodl-tmp/Qwen3-8B \
        --output-dir eval_results/zeroshot_ckpt450 \
        --limit-samples 10

    # Full evaluation
    python scripts/06c_run_inference_jsonl.py \
        --model-path output/qwen3-hrrp-zeroshot/final \
        --eval-data data/hrrp_sft_eval.jsonl \
        --base-model /root/autodl-tmp/Qwen3-8B \
        --output-dir eval_results/zeroshot_final
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.llm_utils import parse_zeroshot_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Class names for new 6 classes (0-shot target)
CLASS_NAMES = ["F22", "F35", "GlobalHawk", "IDF", "Mirage2000", "Predator"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-Shot Inference for JSONL format data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model (LoRA adapter or full model)",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="data/hrrp_sft_eval.jsonl",
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="/root/autodl-tmp/Qwen3-8B",
        help="Base model path (for loading LoRA adapter)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max new tokens to generate (class names are short)",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Limit to N samples for quick test",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 precision",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 precision",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, base_model: str, bf16: bool = False, fp16: bool = False):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_path}")

    # Determine dtype
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Check if LoRA adapter
    adapter_config_path = Path(model_path) / "adapter_config.json"
    is_lora = adapter_config_path.exists()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model if is_lora else model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_lora:
        logger.info(f"Loading LoRA adapter with base model: {base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        logger.info("Loading full model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    logger.info("Model loaded successfully")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    temperature: float = 0.1,
    max_new_tokens: int = 50,
) -> str:
    """Generate model response."""
    # Apply chat template if not already applied
    if not prompt.startswith("<|im_start|>"):
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


def extract_query_label(messages: List[dict]) -> str:
    """Extract true label from messages (assistant content)."""
    for msg in messages:
        if msg["role"] == "assistant":
            return msg["content"].strip()
    return "Unknown"


def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Zero-Shot Inference Started")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Eval data: {args.eval_data}")
    logger.info(f"Output dir: {args.output_dir}")

    # Check eval data exists
    if not Path(args.eval_data).exists():
        logger.error(f"Eval data not found: {args.eval_data}")
        sys.exit(1)

    # Load eval data
    logger.info("Loading evaluation data...")
    eval_samples = []
    with open(args.eval_data, "r") as f:
        for line in f:
            eval_samples.append(json.loads(line))
    logger.info(f"Loaded {len(eval_samples)} evaluation samples")

    # Limit samples if specified
    if args.limit_samples:
        eval_samples = eval_samples[: args.limit_samples]
        logger.info(f"Limited to {len(eval_samples)} samples for quick test")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.base_model,
        bf16=args.bf16,
        fp16=args.fp16,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    results = []
    correct_count = 0
    total = len(eval_samples)

    logger.info(f"Starting inference on {total} samples...")

    for idx, sample in enumerate(tqdm(eval_samples, desc="Inferencing")):
        try:
            messages = sample["messages"]

            # Extract user prompt and true label
            user_prompt = None
            query_label = None
            for msg in messages:
                if msg["role"] == "user":
                    user_prompt = msg["content"]
                elif msg["role"] == "assistant":
                    query_label = msg["content"].strip()

            if not user_prompt or not query_label:
                logger.warning(f"[{idx}] Missing user_prompt or query_label")
                continue

            # Print first sample for verification
            if idx == 0:
                logger.info("=" * 80)
                logger.info("First sample prompt (for verification):")
                logger.info(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)
                logger.info("=" * 80)

            # Generate response
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=user_prompt,
                temperature=args.temperature,
                max_new_tokens=args.max_tokens,
            )

            # Parse prediction
            predicted_label = parse_zeroshot_output(response, CLASS_NAMES)

            if predicted_label is None:
                logger.warning(f"[{idx}] Parse error, raw response: {response[:100]}")
                predicted_label = "PARSE_ERROR"

            # Check correctness
            is_correct = predicted_label == query_label
            if is_correct:
                correct_count += 1

            status = "✓" if is_correct else "✗"
            logger.info(f"[{idx}] {status} {query_label} → {predicted_label}")

            results.append({
                "index": idx,
                "query_label": query_label,
                "predicted_label": predicted_label,
                "is_correct": is_correct,
                "response": response,
            })

        except Exception as e:
            logger.error(f"[{idx}] Error: {e}", exc_info=True)
            results.append({
                "index": idx,
                "error": str(e),
            })

    # Calculate statistics
    successful = sum(1 for r in results if "error" not in r)
    accuracy = (correct_count / successful * 100) if successful > 0 else 0

    logger.info("=" * 80)
    logger.info("Inference Complete!")
    logger.info(f"Total: {total}, Successful: {successful}, Correct: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info("=" * 80)

    # Save results
    result_file = output_dir / "results.json"
    with open(result_file, "w") as f:
        json.dump({
            "model_path": args.model_path,
            "eval_data": args.eval_data,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            },
            "statistics": {
                "total": total,
                "successful": successful,
                "correct": correct_count,
                "accuracy": accuracy,
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {result_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

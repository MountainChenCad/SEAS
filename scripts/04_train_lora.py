#!/usr/bin/env python3
"""
Unified LoRA Fine-tuning Script for Qwen Models

Consolidates train_lora_direct.py and train_simple_single_gpu.py
Supports multiple quantization strategies and hardware configurations

Usage:
    # Basic training with 4-bit quantization
    python scripts/train_lora.py \\
        --model-path /root/autodl-tmp/Qwen3-8B \\
        --data-path data/hrrp_sft_train.json \\
        --quantization 4bit

    # Full custom configuration
    python scripts/train_lora.py \\
        --model-path /root/autodl-tmp/Qwen3-8B \\
        --data-path data/hrrp_sft_train.json \\
        --output-dir output/qwen3-hrrp-lora \\
        --quantization 4bit \\
        --gpu 0 \\
        --num-epochs 3 \\
        --batch-size 4 \\
        --grad-accum-steps 4 \\
        --lora-rank 16 \\
        --lora-alpha 32 \\
        --learning-rate 5e-5 \\
        --max-length 5000
"""

import os
import sys
import json
import torch
import argparse
import gc
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified LoRA Fine-tuning Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model and data paths
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/Qwen3-8B",
        help="Path to base model",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/hrrp_sft_train.json",
        help="Path to training data (JSON)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/qwen3-hrrp-lora",
        help="Output directory for LoRA adapter",
    )

    # Hardware configuration
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "8bit", "fp16"],
        default="4bit",
        help="Quantization strategy (default: 4bit)",
    )

    # Training parameters
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=5000,
        help="Max sequence length (default: 5000)",
    )

    # LoRA parameters
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
    )

    # Optional flags
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluate every N steps (default: 100)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps (default: 100)",
    )

    return parser.parse_args()


def setup_device(gpu_id: int, quantization: str):
    """Setup GPU and environment variables"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if quantization == "4bit":
        # Optimize memory for 4-bit training
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    gc.collect()


def load_training_data(data_path: str) -> list:
    """Load training data from JSON file"""
    print(f"\n📥 Loading training data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   ✓ Loaded {len(data)} samples")
    return data


def format_instruction(sample: dict) -> str:
    """Format instruction sample - supports ShareGPT and standard formats"""

    if "conversations" in sample:
        # ShareGPT format: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        messages = sample["conversations"]
        text = ""
        for msg in messages:
            role = "User:" if msg["from"] == "human" else "Assistant:"
            text += f"{role}\n{msg['value']}\n\n"
        return text.strip()

    # Standard format (fallback)
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")

    # Build complete training text
    text = f"{instruction}\n\n{input_text}\n\n{output}"
    return text


def prepare_dataset(data: list, tokenizer, max_length: int) -> Dataset:
    """Prepare dataset for training"""
    print(f"\n🔧 Preparing dataset...")

    def tokenize_function(example):
        # Format the sample
        text = format_instruction(example)

        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        # Set labels with padding masking: -100 tells CrossEntropyLoss to ignore these positions
        # This prevents padding tokens from contributing to loss and gradients
        labels = tokenized["input_ids"].copy()
        labels = [
            (-100 if mask == 0 else token)
            for mask, token in zip(tokenized["attention_mask"], labels)
        ]
        tokenized["labels"] = labels
        return tokenized

    # Create dataset
    dataset = Dataset.from_list(data)

    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        desc="Tokenizing training data",
    )

    print(f"   ✓ Prepared {len(tokenized_dataset)} samples for training")
    return tokenized_dataset


def create_quantization_config(quantization: str) -> Optional[BitsAndBytesConfig]:
    """Create quantization configuration"""
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8,
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["lm_head"],
        )
    else:  # fp16
        return None


def load_model(model_path: str, quantization: str):
    """Load and configure model"""
    print(f"\n🤖 Loading model from {model_path}...")

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✓ Tokenizer loaded")

    # Create quantization config
    quantization_config = create_quantization_config(quantization)

    # Load model
    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if quantization in ["4bit", "8bit"]:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    # Load model - skip lm_head for quantization to avoid memory issues
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )

    # For 4-bit quantization, move lm_head to float32 to avoid OOM
    if quantization == "4bit" and hasattr(model, "lm_head"):
        model.lm_head = model.lm_head.float()

    print(f"   ✓ Model loaded ({quantization} quantization)")
    return model, tokenizer


def setup_lora(model, lora_rank: int, lora_alpha: int, lora_dropout: float):
    """Setup LoRA configuration and apply to model"""
    print(f"\n⚙️ Setting up LoRA...")

    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],  # For Qwen models
    )

    # Prepare model for k-bit training if using quantization
    if hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    print(f"   ✓ LoRA applied (rank={lora_rank}, alpha={lora_alpha})")
    model.print_trainable_parameters()

    return model


def main():
    args = parse_args()

    # Print configuration
    print("\n" + "=" * 70)
    print("🚀 LoRA Fine-tuning: Unified Training Script")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Model: {args.model_path}")
    print(f"   Data: {args.data_path}")
    print(f"   Output: {args.output_dir}")
    print(f"   GPU: {args.gpu}")
    print(f"   Quantization: {args.quantization}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Grad Accum Steps: {args.grad_accum_steps}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   LoRA Rank: {args.lora_rank}, Alpha: {args.lora_alpha}")
    print("=" * 70)

    try:
        # Setup device
        setup_device(args.gpu, args.quantization)

        # Load data
        data = load_training_data(args.data_path)

        # Load model and tokenizer
        model, tokenizer = load_model(args.model_path, args.quantization)

        # Prepare dataset
        train_dataset = prepare_dataset(data, tokenizer, args.max_length)

        # Setup LoRA
        model = setup_lora(
            model,
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
        )

        # Setup training arguments
        print(f"\n⚙️ Setting up training...")
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=args.warmup_steps,
            save_steps=args.save_steps,
            save_total_limit=None,  # Keep all checkpoints
            logging_steps=10,
            logging_dir=f"{args.output_dir}/logs",
            seed=42,
            ddp_find_unused_parameters=False,
            optim="paged_adamw_32bit",
            bf16=False,
            fp16=True if args.quantization == "fp16" else False,
            report_to=[],  # Disable wandb and other reporting
            remove_unused_columns=False,
            gradient_checkpointing=False,  # Disable gradient checkpointing to save memory
        )
        print(f"   ✓ Training configured")

        # Create trainer with improved data collator
        # DataCollatorForSeq2Seq handles padding masks correctly for causal LM
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            return_tensors="pt",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Train
        print(f"\n🎯 Starting training...")
        print("=" * 70)
        trainer.train()

        # Save final model
        print("\n💾 Saving final model...")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"   ✓ Model saved to {args.output_dir}")

        print("\n" + "=" * 70)
        print("✅ Training completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

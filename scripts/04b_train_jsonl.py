#!/usr/bin/env python3
"""
Zero-Shot LoRA Training Script for JSONL format data.

Usage:
    python scripts/04b_train_jsonl.py \
        --model-path /root/autodl-tmp/Qwen3-8B \
        --train-data data/hrrp_sft_train.jsonl \
        --output-dir output/qwen3-hrrp-zeroshot
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-Shot LoRA Training for JSONL format data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/Qwen3-8B",
        help="Path to base model",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/hrrp_sft_train.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/qwen3-hrrp-zeroshot",
        help="Output directory for checkpoints",
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


def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Zero-Shot LoRA Training Started")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Train data: {args.train_data}")
    logger.info(f"Output dir: {args.output_dir}")

    # Check train data exists
    if not Path(args.train_data).exists():
        logger.error(f"Training data not found: {args.train_data}")
        sys.exit(1)

    # Determine dtype
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Load dataset from JSONL
    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files=args.train_data, split="train")
    logger.info(f"Loaded {len(dataset)} training samples")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA configuration (strict: r=2, alpha=32, q_proj/v_proj only)
    logger.info("Configuring LoRA (r=2, alpha=32, target_modules=[q_proj, v_proj])...")
    lora_config = LoraConfig(
        r=2,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Preprocessing function
    max_seq_length = 1024  # 3-way shorter than 6-way, 1024 is sufficient

    def preprocess_function(examples):
        """Process messages format into model inputs."""
        # Extract messages
        messages_list = examples["messages"]

        # Apply chat template to convert messages to text
        texts = []
        for messages in messages_list:
            # Qwen3 uses apply_chat_template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        # Tokenize
        model_inputs = tokenizer(
            texts,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        )

        # For causal LM, labels are same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    logger.info("Preprocessing dataset...")
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    logger.info(f"Preprocessed {len(processed_dataset)} samples")

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate steps
    num_epochs = 2  # 3263 samples > 900, 2 epochs sufficient
    batch_size = 2  # Reduced from 4 to avoid OOM
    grad_accum = 8  # Increased to maintain effective batch size
    effective_batch = batch_size * grad_accum
    steps_per_epoch = len(processed_dataset) // effective_batch
    total_steps = steps_per_epoch * num_epochs
    save_steps = 400  # ~8 checkpoints (3263*2/16≈408 steps/epoch)

    logger.info(f"Training config: {num_epochs} epochs, {steps_per_epoch} steps/epoch, {total_steps} total")
    logger.info(f"Save checkpoint every {save_steps} steps")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=3e-4,  # 3-way task simpler, can be more aggressive
        warmup_ratio=0.05,  # 2 epochs, ~20 steps warmup
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=max_seq_length,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Final model saved to {final_dir}")

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

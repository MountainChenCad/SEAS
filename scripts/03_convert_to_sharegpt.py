#!/usr/bin/env python3
"""
Convert SFT data to standard LLaMA training format.

Supports two output formats:
1. ShareGPT format (messages list) - for LLaMA Factory, Ollama, etc.
2. Simple text format (prompt + completion) - for basic fine-tuning

Outputs:
  - hrrp_sft_train_sharegpt.json (messages format)
  - hrrp_sft_train_text.json (text format)
  - hrrp_sft_stats.json (statistics)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

INPUT_FILE = "/root/autodl-tmp/projects/TableLlama-HRRP/data/hrrp_sft_from_baseline.json"
OUTPUT_SHAREGPT = "/root/autodl-tmp/projects/TableLlama-HRRP/data/hrrp_sft_train_sharegpt.json"
OUTPUT_TEXT = "/root/autodl-tmp/projects/TableLlama-HRRP/data/hrrp_sft_train_text.json"
OUTPUT_STATS = "/root/autodl-tmp/projects/TableLlama-HRRP/data/hrrp_sft_stats.json"


def convert_to_sharegpt_format(examples: List[Dict]) -> List[Dict]:
    """
    Convert examples to ShareGPT format.

    ShareGPT format (used by LLaMA Factory):
    {
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    """
    sharegpt_data = []

    for ex in examples:
        conversations = []

        # Add system message as user message (some frameworks ignore system separately)
        # Skip system and just use user + assistant
        for msg in ex['messages']:
            if msg['role'] == 'user':
                conversations.append({
                    "from": "human",
                    "value": msg['content']
                })
            elif msg['role'] == 'assistant':
                conversations.append({
                    "from": "gpt",
                    "value": msg['content']
                })

        sharegpt_data.append({
            "conversations": conversations
        })

    return sharegpt_data


def convert_to_text_format(examples: List[Dict]) -> List[Dict]:
    """
    Convert examples to simple text format.

    Text format:
    {
        "prompt": "...",
        "completion": "..."
    }
    """
    text_data = []

    for ex in examples:
        # Combine system + user as prompt
        prompt_parts = []
        completion = ""

        for msg in ex['messages']:
            if msg['role'] == 'system':
                prompt_parts.append(msg['content'])
            elif msg['role'] == 'user':
                prompt_parts.append(msg['content'])
            elif msg['role'] == 'assistant':
                completion = msg['content']

        prompt = "\n\n".join(prompt_parts)

        text_data.append({
            "prompt": prompt,
            "completion": completion
        })

    return text_data


def compute_statistics(examples: List[Dict], sharegpt_data: List[Dict]) -> Dict:
    """Compute dataset statistics."""
    stats = {
        "total_examples": len(examples),
        "correct_examples": sum(1 for ex in examples if ex['metadata']['was_correct']),
        "corrected_examples": sum(1 for ex in examples if not ex['metadata']['was_correct']),
        "by_class": defaultdict(lambda: {"total": 0, "correct": 0, "corrected": 0}),
        "by_correctness": {
            "correct": 0,
            "corrected": 0
        }
    }

    for ex in examples:
        true_label = ex['metadata']['true_label']
        was_correct = ex['metadata']['was_correct']

        stats['by_class'][true_label]['total'] += 1
        if was_correct:
            stats['by_class'][true_label]['correct'] += 1
            stats['by_correctness']['correct'] += 1
        else:
            stats['by_class'][true_label]['corrected'] += 1
            stats['by_correctness']['corrected'] += 1

    # Convert defaultdict to regular dict
    stats['by_class'] = dict(stats['by_class'])

    # Compute average message lengths
    avg_prompt_len = sum(len(ex['messages'][1]['content']) for ex in examples) / len(examples)
    avg_completion_len = sum(len(ex['messages'][-1]['content']) for ex in examples) / len(examples)

    stats['avg_prompt_tokens_estimate'] = int(avg_prompt_len / 4)  # Rough estimate
    stats['avg_completion_tokens_estimate'] = int(avg_completion_len / 4)

    return stats


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("Converting SFT Data to Standard Training Formats")
    print("="*70 + "\n")

    # Load input
    print(f"Loading: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        examples = json.load(f)
    print(f"✓ Loaded {len(examples)} examples\n")

    # Convert to ShareGPT format
    print("Converting to ShareGPT format (for LLaMA Factory)...")
    sharegpt_data = convert_to_sharegpt_format(examples)
    print(f"✓ Converted {len(sharegpt_data)} examples\n")

    # Convert to text format
    print("Converting to text format (simple prompt+completion)...")
    text_data = convert_to_text_format(examples)
    print(f"✓ Converted {len(text_data)} examples\n")

    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(examples, sharegpt_data)

    # Save outputs
    print("Saving outputs...")

    with open(OUTPUT_SHAREGPT, 'w') as f:
        json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)
    print(f"✓ ShareGPT format: {OUTPUT_SHAREGPT}")

    with open(OUTPUT_TEXT, 'w') as f:
        json.dump(text_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Text format: {OUTPUT_TEXT}")

    with open(OUTPUT_STATS, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ Statistics: {OUTPUT_STATS}\n")

    # Print statistics
    print("="*70)
    print("Dataset Statistics")
    print("="*70)
    print(f"Total examples: {stats['total_examples']}")
    print(f"  - Correct predictions (with reasoning): {stats['by_correctness']['correct']} ({stats['by_correctness']['correct']/stats['total_examples']:.1%})")
    print(f"  - Corrected errors (clean answers): {stats['by_correctness']['corrected']} ({stats['by_correctness']['corrected']/stats['total_examples']:.1%})")
    print()
    print(f"Estimated token usage:")
    print(f"  - Average prompt: ~{stats['avg_prompt_tokens_estimate']} tokens")
    print(f"  - Average completion: ~{stats['avg_completion_tokens_estimate']} tokens")
    print()
    print("By class:")
    for cls in sorted(stats['by_class'].keys()):
        class_stats = stats['by_class'][cls]
        print(f"  {cls:15s}: {class_stats['total']:3d} total ({class_stats['correct']:3d} correct + {class_stats['corrected']:3d} corrected)")

    print()
    print("="*70)
    print("✓ Format conversion complete!")
    print("="*70 + "\n")

    print("Next steps:")
    print("1. For LLaMA Factory training:")
    print(f"   python scripts/train_lora.py --data-path {OUTPUT_SHAREGPT}")
    print()
    print("2. For other frameworks:")
    print(f"   Use {OUTPUT_TEXT} or {OUTPUT_SHAREGPT} depending on framework requirements")
    print()


if __name__ == '__main__':
    main()

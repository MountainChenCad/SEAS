#!/usr/bin/env python3
"""
Generate SFT training data from baseline inference results.

Strategy:
1. Correct predictions → positive training examples (demonstrate correct reasoning)
2. Incorrect predictions → corrected examples (contrastive learning)
3. Balance across all 6 classes
4. Use same prompt format as inference but add correct answer as completion
"""

import json
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from scipy.io import loadmat
from collections import defaultdict

# Add project to path
sys.path.insert(0, '/root/autodl-tmp/projects/TableLlama-HRRP')
sys.path.insert(0, '/root/autodl-tmp/projects/TableLlama-HRRP/src')

from src.config import SCATTERING_CENTER_EXTRACTION, SCATTERING_CENTER_ENCODING
from src.feature_extractor import extract_scattering_centers_peak_detection
from src.scattering_center_encoder import encode_single_sc_set_to_text

# Configuration
TRAIN_CLASSES = ["EA-18G", "EP-3E", "F15", "F16", "F18", "F2"]
HRRP_DATA_DIR = "/root/autodl-tmp/projects/hrrplib/data/simulated_hrrp"
BASELINE_RESULTS = "/root/autodl-tmp/projects/TableLlama-HRRP/data/qwen_baseline_6way_1shot_results_final.json"
OUTPUT_FILE = "/root/autodl-tmp/projects/TableLlama-HRRP/data/hrrp_sft_from_baseline.json"

# Sampling strategy
MAX_SAMPLES_PER_CLASS = 300  # Balance across classes
CORRECT_RATIO = 0.5  # 50% correct, 50% corrected errors


def load_hrrp_file(filepath: str) -> np.ndarray:
    """Load HRRP data from .mat file."""
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
        print(f"Warning: Error loading {filepath}: {e}")
    return None


def extract_and_encode_sc(hrrp: np.ndarray) -> str:
    """Extract scattering centers and encode to text."""
    centers = extract_scattering_centers_peak_detection(
        hrrp,
        prominence=SCATTERING_CENTER_EXTRACTION['prominence'],
        min_distance=SCATTERING_CENTER_EXTRACTION['min_distance'],
        max_centers_to_keep=SCATTERING_CENTER_EXTRACTION['max_centers_to_keep'],
        normalize_hrrp_before_extraction=SCATTERING_CENTER_EXTRACTION['normalize_hrrp_before_extraction']
    )
    sc_text = encode_single_sc_set_to_text(centers, SCATTERING_CENTER_ENCODING)
    return sc_text


def reconstruct_prompt_from_prediction(pred: Dict) -> Tuple[str, str, str]:
    """
    Reconstruct the original prompt from a prediction record.

    Returns:
        (system_prompt, user_prompt, correct_answer)
    """
    system_prompt = (
        "You are an expert in aircraft classification based on radar HRRP (High-Resolution Range Profile) signals. "
        "Your task is to classify an unknown aircraft based on its scattering center features compared with known examples. "
        "Respond with only the class name."
    )

    # We don't have the support examples stored, so we'll need to note this limitation
    # For now, we'll create a simplified version focusing on the query
    query_file = pred['query_file']
    true_label = pred['query_label']

    # Load query HRRP
    query_hrrp = load_hrrp_file(query_file)
    if query_hrrp is None:
        return None, None, None

    query_sc = extract_and_encode_sc(query_hrrp)

    # Note: We can't reconstruct exact support examples without task metadata
    # But we can create training data with query + correct answer
    user_prompt = (
        f"Known aircraft classes: {', '.join(TRAIN_CLASSES)}\n\n"
        f"Unknown aircraft to classify:\nScattering Centers: {query_sc}\n\n"
        f"Candidate classes: {', '.join(TRAIN_CLASSES)}\n"
        f"Respond with ONLY the class name."
    )

    correct_answer = true_label

    return system_prompt, user_prompt, correct_answer


def create_sft_example(pred: Dict, include_reasoning: bool = False) -> Dict:
    """
    Create an SFT training example from a prediction.

    Args:
        pred: Prediction record from baseline results
        include_reasoning: If True, include model's reasoning (for correct predictions)

    Returns:
        SFT example in conversational format
    """
    system_prompt, user_prompt, correct_answer = reconstruct_prompt_from_prediction(pred)

    if system_prompt is None:
        return None

    # Format for conversational fine-tuning
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # For correct predictions, optionally include the original reasoning
    if include_reasoning and pred['is_correct'] and pred.get('raw_response'):
        assistant_response = pred['raw_response']
    else:
        # For incorrect predictions, provide clean correct answer
        assistant_response = correct_answer

    messages.append({"role": "assistant", "content": assistant_response})

    return {
        "messages": messages,
        "metadata": {
            "query_file": pred['query_file'],
            "true_label": pred['query_label'],
            "was_correct": pred['is_correct'],
            "original_prediction": pred.get('predicted_label', 'UNKNOWN')
        }
    }


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("Generating SFT Data from Baseline Inference Results")
    print("="*70 + "\n")

    # Load baseline results
    print(f"Loading baseline results from: {BASELINE_RESULTS}")
    with open(BASELINE_RESULTS, 'r') as f:
        baseline_data = json.load(f)

    predictions = baseline_data['predictions']
    print(f"✓ Loaded {len(predictions)} predictions")
    print(f"  Overall accuracy: {baseline_data['statistics']['overall_accuracy']:.2%}")
    print()

    # Separate by correctness and class
    correct_by_class = defaultdict(list)
    incorrect_by_class = defaultdict(list)

    for pred in predictions:
        true_label = pred['query_label']
        if pred['is_correct']:
            correct_by_class[true_label].append(pred)
        else:
            incorrect_by_class[true_label].append(pred)

    # Print distribution
    print("Distribution by class:")
    for cls in sorted(TRAIN_CLASSES):
        n_correct = len(correct_by_class[cls])
        n_incorrect = len(incorrect_by_class[cls])
        print(f"  {cls:15s}: {n_correct:4d} correct, {n_incorrect:4d} incorrect")
    print()

    # Sample balanced dataset
    print(f"Sampling up to {MAX_SAMPLES_PER_CLASS} examples per class...")
    print(f"  Target ratio: {CORRECT_RATIO:.0%} correct, {1-CORRECT_RATIO:.0%} corrected errors")
    print()

    sft_examples = []
    random.seed(42)

    for cls in TRAIN_CLASSES:
        n_correct_target = int(MAX_SAMPLES_PER_CLASS * CORRECT_RATIO)
        n_incorrect_target = MAX_SAMPLES_PER_CLASS - n_correct_target

        # Sample correct predictions (keep reasoning)
        correct_samples = correct_by_class[cls]
        if len(correct_samples) > n_correct_target:
            correct_samples = random.sample(correct_samples, n_correct_target)

        for pred in correct_samples:
            example = create_sft_example(pred, include_reasoning=True)
            if example:
                sft_examples.append(example)

        # Sample incorrect predictions (provide correction)
        incorrect_samples = incorrect_by_class[cls]
        if len(incorrect_samples) > n_incorrect_target:
            incorrect_samples = random.sample(incorrect_samples, n_incorrect_target)

        for pred in incorrect_samples:
            example = create_sft_example(pred, include_reasoning=False)
            if example:
                sft_examples.append(example)

        print(f"  {cls:15s}: {len(correct_samples):3d} correct + {len(incorrect_samples):3d} corrected = {len(correct_samples) + len(incorrect_samples):3d} total")

    print()
    print(f"✓ Generated {len(sft_examples)} SFT training examples")

    # Shuffle
    random.shuffle(sft_examples)

    # Save
    print(f"\nSaving to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(sft_examples, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(sft_examples)} examples")
    print()

    # Statistics
    n_correct_kept = sum(1 for ex in sft_examples if ex['metadata']['was_correct'])
    n_corrected = len(sft_examples) - n_correct_kept

    print("Final statistics:")
    print(f"  Total examples: {len(sft_examples)}")
    print(f"  Correct predictions (with reasoning): {n_correct_kept} ({n_correct_kept/len(sft_examples):.1%})")
    print(f"  Corrected errors (clean answers): {n_corrected} ({n_corrected/len(sft_examples):.1%})")
    print()

    print("="*70)
    print("✓ SFT data generation complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

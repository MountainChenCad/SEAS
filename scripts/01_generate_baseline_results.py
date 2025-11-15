#!/usr/bin/env python3
"""
Improved inference script with checkpointing and timeout handling.

This version includes:
- Periodic log flushing
- Timeout mechanism per sample
- Progress checkpoints every N samples
- Better error handling
- Resumable from checkpoints
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from scipy.io import loadmat
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from datetime import datetime
import signal

# Setup logging with flush
class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        FlushingFileHandler('inference_baseline_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import from project
sys.path.insert(0, '/root/autodl-tmp/projects/TableLlama-HRRP')
sys.path.insert(0, '/root/autodl-tmp/projects/TableLlama-HRRP/src')

from src.config import SCATTERING_CENTER_EXTRACTION, SCATTERING_CENTER_ENCODING
from src.feature_extractor import extract_scattering_centers_peak_detection
from src.scattering_center_encoder import encode_single_sc_set_to_text
from src.llm_utils import parse_llm_output_for_label

# Configuration
TRAIN_CLASSES = ["EA-18G", "EP-3E", "F15", "F16", "F18", "F2"]
MODEL_PATH = "/root/autodl-tmp/Qwen3-8B"
HRRP_DATA_DIR = "/root/autodl-tmp/projects/hrrplib/data/simulated_hrrp"
OUTPUT_DIR = "/root/autodl-tmp/projects/TableLlama-HRRP/data"

# Inference config
INFERENCE_BATCH_SIZE = 2
MAX_NEW_TOKENS = 3000
TEMPERATURE = 0.1
TOP_P = 1.0

# Checkpointing config
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N samples
TIMEOUT_PER_SAMPLE = 300  # 5 minutes timeout per sample
CHECKPOINT_DIR = OUTPUT_DIR


def timeout_handler(signum, frame):
    raise TimeoutError("Sample inference timeout")


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
        logger.warning(f"Error loading {filepath}: {e}")
    return None


def get_class_samples(class_name: str) -> List[str]:
    """Get all HRRP file paths for a given class."""
    pattern = f"{class_name}_*.mat"
    files = list(Path(HRRP_DATA_DIR).glob(pattern))
    return sorted([str(f) for f in files])


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


def build_6way_1shot_task(
    query_file: str,
    query_label: str,
    all_class_files: Dict[str, List[str]]
) -> Dict:
    """Build a 6-way 1-shot task."""
    task = {
        'n_way': 6,
        'k_shot': 1,
        'q_shot': 1,
        'target_classes': TRAIN_CLASSES,
        'query_label': query_label,
        'query_file': query_file,
        'support_examples': {}
    }

    for class_name in TRAIN_CLASSES:
        available_files = list(all_class_files.get(class_name, []))

        if class_name == query_label and query_file in available_files:
            try:
                available_files.remove(query_file)
            except ValueError:
                pass

        if len(available_files) > 0:
            support_file = random.choice(available_files)
            task['support_examples'][class_name] = support_file
        else:
            logger.warning(f"No support samples available for {class_name}")
            return None

    return task


def construct_6way_1shot_prompt(task: Dict) -> Tuple[str, str]:
    """Construct prompt for 6-way 1-shot task."""
    system_prompt = (
        "You are an expert in aircraft classification based on radar HRRP (High-Resolution Range Profile) signals. "
        "Your task is to classify an unknown aircraft based on its scattering center features compared with known examples. "
        "Respond with only the class name."
    )

    support_text = "Known aircraft examples (1 per class):\n\n"
    for class_name in TRAIN_CLASSES:
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
        f"Candidate classes: {', '.join(TRAIN_CLASSES)}\n"
        f"Respond with ONLY the class name (one of: {', '.join(TRAIN_CLASSES)})"
    )

    user_prompt = support_text + query_text

    return system_prompt, user_prompt


def run_inference(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str
) -> str:
    """Run inference with timeout handling."""
    try:
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_PER_SAMPLE)

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

        # Cancel timeout
        signal.alarm(0)

        return response.strip()
    except TimeoutError:
        logger.warning("Sample inference timeout - skipping")
        signal.alarm(0)
        return None
    except Exception as e:
        logger.error(f"Inference error: {e}")
        signal.alarm(0)
        return None


def save_checkpoint(results: Dict, checkpoint_file: str):
    """Save checkpoint with current progress."""
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"✓ Checkpoint saved: {checkpoint_file}")


def load_checkpoint(checkpoint_file: str) -> Dict:
    """Load checkpoint to resume from."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None


def main():
    """Main entry point."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    logger.info("\n" + "="*70)
    logger.info("6-Way 1-Shot Baseline Inference v2 (Improved, Resumable)")
    logger.info("="*70)
    logger.info(f"Classes: {', '.join(TRAIN_CLASSES)}")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Checkpoint interval: Every {CHECKPOINT_INTERVAL} samples")
    logger.info(f"Timeout per sample: {TIMEOUT_PER_SAMPLE}s")
    logger.info("="*70 + "\n")

    # Check for existing checkpoint
    checkpoint_file = os.path.join(OUTPUT_DIR, 'inference_checkpoint_latest.json')
    resume_from = None
    if os.path.exists(checkpoint_file):
        logger.info(f"Found checkpoint file: {checkpoint_file}")
        resume_from = load_checkpoint(checkpoint_file)
        if resume_from:
            logger.info(f"Resuming from: {len(resume_from['predictions'])} previous predictions")
            logger.info("")

    # Load model and tokenizer
    logger.info("Loading Qwen3-8B model (no LoRA)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    logger.info("✓ Model loaded\n")

    # Load all class files
    logger.info("Loading class file lists...")
    all_class_files = {}
    for class_name in TRAIN_CLASSES:
        files = get_class_samples(class_name)
        all_class_files[class_name] = files
        logger.info(f"  ✓ {class_name}: {len(files)} files")
    logger.info("")

    # Initialize results
    if resume_from:
        results = resume_from
    else:
        results = {
            'timestamp': datetime.now().isoformat(),
            'model': 'Qwen3-8B (baseline, no LoRA)',
            'config': {
                'n_way': 6,
                'k_shot': 1,
                'q_shot': 1,
                'classes': TRAIN_CLASSES,
                'max_tokens': MAX_NEW_TOKENS,
                'temperature': TEMPERATURE
            },
            'predictions': [],
            'statistics': {}
        }

    total_tasks = 0
    correct_count = 0
    class_results = {c: {'correct': 0, 'total': 0} for c in TRAIN_CLASSES}

    # Restore statistics from checkpoint
    if resume_from and 'statistics' in resume_from:
        for class_name in TRAIN_CLASSES:
            if class_name in resume_from['statistics'].get('per_class', {}):
                class_results[class_name] = resume_from['statistics']['per_class'][class_name]
                total_tasks += class_results[class_name]['total']
                correct_count += class_results[class_name]['correct']

    logger.info("Starting inference...")
    logger.info("="*70 + "\n")

    start_time = datetime.now()

    for target_class in TRAIN_CLASSES:
        available_files = all_class_files[target_class]
        query_files = random.sample(available_files, len(available_files))

        logger.info(f"Inferring on class '{target_class}' ({len(query_files)} queries)...")

        for idx, query_file in enumerate(query_files, 1):
            # Build 6-way 1-shot task
            task = build_6way_1shot_task(query_file, target_class, all_class_files)
            if task is None:
                continue

            # Construct prompt
            system_prompt, user_prompt = construct_6way_1shot_prompt(task)
            if system_prompt is None:
                continue

            # Run inference
            prediction = run_inference(model, tokenizer, system_prompt, user_prompt)
            if prediction is None:
                predicted_label = "UNKNOWN"
            else:
                predicted_label = parse_llm_output_for_label(prediction, TRAIN_CLASSES)
                if predicted_label is None:
                    predicted_label = "UNKNOWN"

            # Check correctness
            is_correct = predicted_label == target_class
            if is_correct:
                correct_count += 1
            class_results[target_class]['correct'] += (1 if is_correct else 0)
            class_results[target_class]['total'] += 1

            # Store result
            result = {
                'query_file': query_file,
                'query_label': target_class,
                'predicted_label': predicted_label,
                'is_correct': is_correct,
                'raw_response': prediction if prediction else "TIMEOUT/ERROR"
            }
            results['predictions'].append(result)

            total_tasks += 1

            # Periodic logging and checkpointing
            if idx % 50 == 0:
                elapsed = datetime.now() - start_time
                logger.info(f"  Processed {idx}/{len(query_files)} queries... (elapsed: {elapsed})")

            if total_tasks % CHECKPOINT_INTERVAL == 0:
                # Update statistics
                results['statistics'] = {
                    'total_tasks': total_tasks,
                    'correct_predictions': correct_count,
                    'overall_accuracy': correct_count / total_tasks if total_tasks > 0 else 0,
                    'per_class': class_results,
                    'elapsed_time': str(datetime.now() - start_time)
                }
                save_checkpoint(results, checkpoint_file)

        # Log class completion
        class_acc = 0
        if class_results[target_class]['total'] > 0:
            class_acc = class_results[target_class]['correct'] / class_results[target_class]['total']
        logger.info(
            f"  ✓ {target_class}: {class_results[target_class]['correct']}/{class_results[target_class]['total']} "
            f"({class_acc:.1%})\n"
        )

    # Final results
    logger.info("\n" + "="*70)
    logger.info("INFERENCE COMPLETED")
    logger.info("="*70)

    overall_accuracy = correct_count / total_tasks if total_tasks > 0 else 0
    elapsed_total = datetime.now() - start_time

    logger.info(f"\nTotal time: {elapsed_total}")
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f} ({correct_count}/{total_tasks})")

    logger.info(f"\nPer-Class Accuracy:")
    for class_name in sorted(TRAIN_CLASSES):
        class_acc = 0
        if class_results[class_name]['total'] > 0:
            class_acc = class_results[class_name]['correct'] / class_results[class_name]['total']
        logger.info(
            f"  {class_name:15s}: {class_results[class_name]['correct']:4d}/{class_results[class_name]['total']:4d} "
            f"({class_acc:.1%})"
        )

    # Final statistics
    results['statistics'] = {
        'total_tasks': total_tasks,
        'correct_predictions': correct_count,
        'overall_accuracy': overall_accuracy,
        'per_class': class_results,
        'total_time': str(elapsed_total)
    }

    logger.info("\n" + "="*70 + "\n")

    # Save final results
    results_file = os.path.join(OUTPUT_DIR, 'qwen_baseline_6way_1shot_results_v2.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"✓ Final results saved to: {results_file}\n")

    return results


if __name__ == '__main__':
    results = main()

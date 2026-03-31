#!/usr/bin/env python3
"""
生成3-way 1-shot评估数据（新6类）- 无需API调用

评估数据只需要episode结构，不需要CoT推理
格式与训练数据一致，但assistant字段用于评估时对比
"""

import argparse
import json
import logging
import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 6个新类别（评估类别）
EVAL_CLASSES = ["F22", "F35", "GlobalHawk", "IDF", "Mirage2000", "Predator"]
BASE_DATA_DIR = "/root/autodl-tmp/projects/hrrplib/data/simulated_hrrp"


def get_mat_files(cls: str) -> List[str]:
    """获取某类的所有MAT文件路径"""
    pattern = f"{BASE_DATA_DIR}/{cls}_hrrp_theta_90_phi_*.mat"
    return sorted(glob.glob(pattern))


def extract_scattering_centers(mat_path: str) -> List[Dict]:
    """从MAT文件提取散射中心"""
    try:
        mat_data = loadmat(mat_path)
        hrrp = mat_data.get("hrrp_data", None)
        if hrrp is None:
            keys = [k for k in mat_data.keys() if not k.startswith("__")]
            if keys:
                hrrp = mat_data[keys[0]]

        if isinstance(hrrp, np.ndarray):
            if hrrp.ndim > 1:
                hrrp = hrrp.flatten()
            if np.iscomplexobj(hrrp):
                hrrp = np.abs(hrrp)

            hrrp_normalized = hrrp / np.max(hrrp)
            peaks, properties = find_peaks(hrrp_normalized, prominence=0.15, distance=5)
            amplitudes = hrrp_normalized[peaks]
            top_indices = np.argsort(amplitudes)[-10:][::-1]

            return [{"range index": int(peaks[idx]), "normalized amplitude": round(float(amplitudes[idx]), 3)} for idx in top_indices]
    except Exception as e:
        logger.warning(f"Error: {e}")
    return []


def format_sc_text(sc_list: List[Dict]) -> str:
    """格式化散射中心为文本"""
    lines = [f"  {{'range index': {sc['range index']}, 'normalized amplitude': {sc['normalized amplitude']:.3f}}}" for sc in sc_list]
    return "[\n" + ",\n".join(lines) + "\n]"


def build_eval_sample(episode: Dict) -> Dict:
    """构建评估样本（无需API，只需要结构）"""
    support_texts = []
    for s in episode["support_set"]:
        sc_text = format_sc_text(s["sc"])
        support_texts.append(f"Class '{s['class']}':\nScattering Centers: {sc_text}")

    support_text_joined = '\n\n'.join(support_texts)
    query_sc_text = format_sc_text(episode["query_sc"])

    user_prompt = f"""You are an expert in aircraft classification based on radar HRRP signals.

Known aircraft examples (1 per class):

{support_text_joined}

Unknown aircraft to classify:
Scattering Centers: {query_sc_text}

Candidate classes: {', '.join(episode['candidate_classes'])}

Analyze by comparing the query with each support class. Provide class name followed by detailed reasoning."""

    # 评估数据：assistant content 只需要正确答案（用于评估对比）
    # 实际推理时模型会生成自己的推理
    assistant_content = episode["query_class"]

    return {
        "messages": [
            {"role": "system", "content": "You are a radar target recognition expert."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content}
        ],
        "metadata": {
            "query_class": episode["query_class"],
            "query_file": episode["query_file"],
            "support_files": [s["file"] for s in episode["support_set"]],
            "3way_combo": episode["candidate_classes"]
        }
    }


def generate_3way_eval_episodes(num_samples: int, seed: int = 42) -> List[Dict]:
    """生成3-way评估episodes"""
    random.seed(seed)

    class_files = {cls: get_mat_files(cls) for cls in EVAL_CLASSES}
    logger.info("每类可用样本数:")
    for cls, files in class_files.items():
        logger.info(f"  {cls}: {len(files)}")

    all_combinations = list(combinations(EVAL_CLASSES, 3))
    logger.info(f"\n3类组合总数: {len(all_combinations)}")

    samples_per_combo = num_samples // len(all_combinations)
    remainder = num_samples % len(all_combinations)
    logger.info(f"每组合生成: {samples_per_combo} 条，余数: {remainder}")

    episodes = []
    for combo_idx, combo in enumerate(all_combinations):
        # 余数分配：前remainder个组合多1条
        n_samples = samples_per_combo + (1 if combo_idx < remainder else 0)
        logger.info(f"生成组合 {combo}: {n_samples} 条")

        for _ in range(n_samples):
            query_class = random.choice(combo)
            support_set = []
            for cls in combo:
                mat_file = random.choice(class_files[cls])
                sc = extract_scattering_centers(mat_file)
                support_set.append({"class": cls, "sc": sc, "file": mat_file})

            query_file = random.choice(class_files[query_class])
            query_sc = extract_scattering_centers(query_file)

            episodes.append({
                "query_class": query_class,
                "query_file": query_file,
                "query_sc": query_sc,
                "support_set": support_set,
                "candidate_classes": list(combo)
            })

    random.shuffle(episodes)
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Generate 3-way 1-shot evaluation data (no API needed)")
    parser.add_argument("--output", type=str, default="data/hrrp_episodes_eval_3way.jsonl")
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("3-Way 1-Shot Evaluation Data Generation (No API)")
    logger.info("=" * 70)
    logger.info(f"Total samples: {args.num_samples}")
    logger.info(f"Output: {args.output}")

    episodes = generate_3way_eval_episodes(args.num_samples, seed=args.seed)
    logger.info(f"\nGenerated {len(episodes)} episodes")

    logger.info("Building eval samples...")
    results = []
    for episode in tqdm(episodes, desc="Building samples"):
        results.append(build_eval_sample(episode))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("=" * 70)
    logger.info("Complete!")
    logger.info(f"Total: {len(results)} samples")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
生成3-way 1-shot训练数据（使用反向CoT生成）

核心设计：
1. 从旧6类中随机选3类构建episode
2. 通过API反向提示（告诉正确答案）生成CoT推理
3. 训练数据中不保留答案提示信息
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import glob
import numpy as np
import requests
from scipy.io import loadmat
from scipy.signal import find_peaks
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 6个训练类别
TRAIN_CLASSES = ["EA-18G", "EP-3E", "F15", "F16", "F18", "F2"]
BASE_DATA_DIR = "/root/autodl-tmp/projects/hrrplib/data/simulated_hrrp"


class CoTGenerator:
    """使用反向提示生成CoT"""

    def __init__(self, api_key: str, model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def generate_cot(
        self,
        episode: Dict,
        max_retries: int = 3,
        delay: float = 0.5,
    ) -> Optional[str]:
        """使用反向提示生成CoT"""
        prompt = self._build_reverse_prompt(episode)

        for attempt in range(max_retries):
            try:
                time.sleep(delay)

                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert in radar HRRP aircraft classification. Your task is to provide detailed comparative reasoning."
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 600,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # 验证格式
                if self._validate_format(content, episode["query_class"]):
                    return content
                else:
                    logger.warning(f"Invalid format, retrying... (attempt {attempt + 1})")

            except Exception as e:
                logger.warning(f"API call failed: {e}, retrying... (attempt {attempt + 1})")
                time.sleep(delay * (2 ** attempt))

        return None

    def _build_reverse_prompt(self, episode: Dict) -> str:
        """构建反向提示（告诉API正确答案）"""
        correct_class = episode["query_class"]
        support_set = episode["support_set"]

        # 构建support文本
        support_texts = []
        for s in support_set:
            sc_lines = []
            for sc in s["sc"]:
                sc_lines.append(f"  {{'range index': {sc['range index']}, 'normalized amplitude': {sc['normalized amplitude']:.3f}}}")
            sc_text = "[\n" + ",\n".join(sc_lines) + "\n]"
            support_texts.append(f"Class '{s['class']}':\nScattering Centers: {sc_text}")

        # 构建query文本
        query_sc_lines = []
        for sc in episode["query_sc"]:
            query_sc_lines.append(f"  {{'range index': {sc['range index']}, 'normalized amplitude': {sc['normalized amplitude']:.3f}}}")
        query_sc_text = "[\n" + ",\n".join(query_sc_lines) + "\n]"

        support_text_joined = '\n\n'.join(support_texts)
        candidate_classes = episode['candidate_classes']

        prompt = f"""You are an expert in aircraft classification based on radar HRRP signals.

Below are 3 known aircraft (support set) and 1 unknown aircraft (query).

SUPPORT SET (1 sample per class):

{support_text_joined}

QUERY AIRCRAFT (to classify):
Scattering Centers: {query_sc_text}

CANDIDATE CLASSES: {', '.join(episode['candidate_classes'])}

FOR TRAINING PURPOSES: The correct answer is '{correct_class}'.

Your task: Generate CONCISE comparative reasoning to justify why '{correct_class}' is correct.

CRITICAL FORMAT REQUIREMENTS:
1. First line: ONLY the class name '{correct_class}'
2. Second line: EMPTY (blank line)
3. Third line+: Start with "Physical analysis:"
4. For EACH of the 3 support classes, write EXACTLY ONE sentence comparing with query
5. End with ONE conclusion sentence starting with "Conclusion:"
6. TOTAL LENGTH MUST BE UNDER 600 CHARACTERS

EXAMPLE OUTPUT FORMAT:
{correct_class}

Physical analysis:
Comparing with Class '{candidate_classes[0]}': [One sentence comparing positions and amplitudes].
Comparing with Class '{candidate_classes[1]}': [One sentence analysis].
Comparing with Class '{candidate_classes[2]}': [One sentence analysis].
Conclusion: The query matches {correct_class} best because [one sentence summary]."""

        return prompt

    def _validate_format(self, content: str, expected_class: str) -> bool:
        """验证输出格式"""
        content = content.strip()
        lines = content.split('\n')

        if len(lines) < 5:
            logger.warning(f"Too few lines: {len(lines)}")
            return False

        # 检查第一行是类别名
        first_line = lines[0].strip()
        if first_line != expected_class:
            logger.warning(f"First line '{first_line}' != expected '{expected_class}'")
            return False

        # 检查有Physical analysis
        if 'Physical analysis' not in content:
            logger.warning("Missing 'Physical analysis'")
            return False

        # 检查有Conclusion
        if 'Conclusion:' not in content:
            logger.warning("Missing 'Conclusion:'")
            return False

        return True


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

            # 归一化
            hrrp_normalized = hrrp / np.max(hrrp)

            # 提取散射中心
            peaks, properties = find_peaks(
                hrrp_normalized,
                prominence=0.15,
                distance=5
            )

            # 获取幅度并排序
            amplitudes = hrrp_normalized[peaks]
            top_indices = np.argsort(amplitudes)[-10:][::-1]

            sc_entries = []
            for idx in top_indices:
                peak_idx = peaks[idx]
                amp = amplitudes[idx]
                sc_entries.append({
                    "range index": int(peak_idx),
                    "normalized amplitude": round(float(amp), 3)
                })

            return sc_entries
    except Exception as e:
        logger.warning(f"Error extracting {mat_path}: {e}")

    return []


def build_training_sample(episode: Dict, cot_content: str) -> Dict:
    """构建最终训练样本（ShareGPT格式，不包含答案提示）"""
    # 提取推理部分（去掉第一行类别名）
    lines = cot_content.split('\n')
    if len(lines) >= 3:
        # 第一行是类别名，第二行是空行，从第三行开始是推理
        reasoning = '\n'.join(lines[2:]).strip()
    else:
        reasoning = cot_content.strip()

    # 构建support文本（用于user prompt）
    support_texts = []
    for s in episode["support_set"]:
        sc_lines = []
        for sc in s["sc"]:
            sc_lines.append(f"  {{'range index': {sc['range index']}, 'normalized amplitude': {sc['normalized amplitude']:.3f}}}")
        sc_text = "[\n" + ",\n".join(sc_lines) + "\n]"
        support_texts.append(f"Class '{s['class']}':\nScattering Centers: {sc_text}")

    # 构建query文本
    query_sc_lines = []
    for sc in episode["query_sc"]:
        query_sc_lines.append(f"  {{'range index': {sc['range index']}, 'normalized amplitude': {sc['normalized amplitude']:.3f}}}")
    query_sc_text = "[\n" + ",\n".join(query_sc_lines) + "\n]"

    # 构建user prompt（不包含答案提示！）
    support_text_joined = '\n\n'.join(support_texts)
    user_prompt = f"""You are an expert in aircraft classification based on radar HRRP signals.

Known aircraft examples (1 per class):

{support_text_joined}

Unknown aircraft to classify:
Scattering Centers: {query_sc_text}

Candidate classes: {', '.join(episode['candidate_classes'])}

Analyze by comparing the query with each support class. Provide class name followed by detailed reasoning."""

    # 构建assistant content（第一行类别名 + 推理）
    assistant_content = f"""{episode['query_class']}

{reasoning}"""

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a radar target recognition expert."
            },
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ],
        "metadata": {
            "query_class": episode["query_class"],
            "query_file": episode["query_file"],
            "support_files": [s["file"] for s in episode["support_set"]],
            "3way_combo": episode["candidate_classes"]
        }
    }


def generate_3way_episodes(num_samples: int, seed: int = 42) -> List[Dict]:
    """生成3-way episodes"""
    random.seed(seed)

    # 获取每类的所有MAT文件
    class_files = {cls: get_mat_files(cls) for cls in TRAIN_CLASSES}

    logger.info("每类可用样本数:")
    for cls, files in class_files.items():
        logger.info(f"  {cls}: {len(files)}")

    # 生成所有3类组合
    all_combinations = list(combinations(TRAIN_CLASSES, 3))
    logger.info(f"\n3类组合总数: {len(all_combinations)}")

    # 每组合生成样本数
    samples_per_combo = num_samples // len(all_combinations)
    logger.info(f"每组合生成: {samples_per_combo} 条")

    episodes = []

    for combo in all_combinations:
        logger.info(f"\n生成组合 {combo} 的 {samples_per_combo} 条episodes...")

        for _ in range(samples_per_combo):
            # 1. 随机选择query类别（从combo的3类中）
            query_class = random.choice(combo)

            # 2. 为每个support类别选择1个样本
            support_set = []
            for cls in combo:
                mat_file = random.choice(class_files[cls])
                sc = extract_scattering_centers(mat_file)
                support_set.append({
                    "class": cls,
                    "sc": sc,
                    "file": mat_file
                })

            # 3. 选择query样本
            query_file = random.choice(class_files[query_class])
            query_sc = extract_scattering_centers(query_file)

            # 4. 构建episode
            episode = {
                "query_class": query_class,
                "query_file": query_file,
                "query_sc": query_sc,
                "support_set": support_set,
                "candidate_classes": list(combo)
            }
            episodes.append(episode)

    # 打乱顺序
    random.shuffle(episodes)

    return episodes


def main():
    parser = argparse.ArgumentParser(description="Generate 3-way 1-shot training data with CoT")
    parser.add_argument("--output", type=str, default="data/hrrp_episodes_train_3way.jsonl")
    parser.add_argument("--num-samples", type=int, default=3600)
    parser.add_argument("--api-key", type=str, default=os.environ.get("SILICONFLOW_API_KEY"))
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.api_key:
        logger.error("Please provide API key via --api-key or SILICONFLOW_API_KEY env var")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("3-Way 1-Shot Training Data Generation")
    logger.info("=" * 70)
    logger.info(f"Total samples: {args.num_samples}")
    logger.info(f"Output: {args.output}")

    # 生成episodes
    logger.info("\nGenerating 3-way episodes...")
    episodes = generate_3way_episodes(args.num_samples, seed=args.seed)
    logger.info(f"Generated {len(episodes)} episodes")

    # 初始化CoT生成器
    generator = CoTGenerator(args.api_key, args.model)

    # 生成CoT
    results = []
    success_count = 0

    for i, episode in enumerate(tqdm(episodes, desc="Generating CoT")):
        try:
            cot_content = generator.generate_cot(episode)

            if cot_content is None:
                logger.warning(f"Failed to generate CoT for episode {i}")
                continue

            # 构建最终训练样本
            train_sample = build_training_sample(episode, cot_content)
            results.append(train_sample)
            success_count += 1

            # 保存checkpoint
            if (i + 1) % args.checkpoint_interval == 0:
                checkpoint_path = Path(args.output).with_suffix(f".checkpoint_{i+1}.jsonl")
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                logger.info(f"Checkpoint saved: {checkpoint_path} ({len(results)} samples)")

        except Exception as e:
            logger.error(f"Error processing episode {i}: {e}")
            continue

    # 保存最终结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info("=" * 70)
    logger.info("Generation Complete!")
    logger.info(f"Total episodes: {len(episodes)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Success rate: {success_count / len(episodes) * 100:.1f}%")
    logger.info(f"Output saved to: {args.output}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

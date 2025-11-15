#!/usr/bin/env python3
"""
生成后6个类别(新类)的6-way 1-shot FSL评估任务

目的: 用完全不同的飞机类别评估微调模型的泛化能力
类别: F22, F35, GlobalHawk, IDF, Mirage2000, Predator (均未用于训练)
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from scipy.io import loadmat

# 配置
HRRP_DATA_DIR = "/root/autodl-tmp/projects/hrrplib/data/simulated_hrrp"
OUTPUT_DIR = "/root/autodl-tmp/projects/TableLlama-HRRP/data"
EVAL_CLASSES = ["F22", "F35", "GlobalHawk", "IDF", "Mirage2000", "Predator"]
NUM_EVAL_TASKS = 150  # 总任务数
RANDOM_SEED = 42


def load_hrrp_file(filepath: str) -> np.ndarray:
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
        print(f"警告: 加载失败 {filepath}: {e}")
    return None


def get_class_samples(class_name: str) -> List[str]:
    """获取某个类别的所有HRRP文件路径"""
    pattern = f"{class_name}_*.mat"
    files = list(Path(HRRP_DATA_DIR).glob(pattern))
    return sorted([str(f) for f in files])


def build_6way_1shot_task(support_files: Dict[str, str], query_file: str, query_label: str) -> Dict:
    """
    构建一个6-way 1-shot评估任务

    Args:
        support_files: {class_name: file_path} 每个类一个支持样本
        query_file: 查询样本的文件路径
        query_label: 查询样本的真实标签
    """
    task = {
        'n_way': 6,
        'k_shot': 1,
        'q_shot': 1,
        'target_classes': EVAL_CLASSES,
        'query_label': query_label,
        'query_file': query_file,
        'support_examples': support_files
    }
    return task


def main():
    """主函数"""
    print("\n" + "="*70)
    print("生成后6类(新类)的6-way 1-shot FSL评估任务")
    print("="*70)
    print(f"类别: {', '.join(EVAL_CLASSES)}")
    print(f"任务数: {NUM_EVAL_TASKS}")
    print(f"每任务结构: 6-way 1-shot (6个类, 每类1个支持, 1个查询)")
    print("="*70 + "\n")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 加载所有类别的文件列表
    print("加载类别文件...")
    all_class_files = {}
    for class_name in EVAL_CLASSES:
        files = get_class_samples(class_name)
        all_class_files[class_name] = files
        print(f"  ✓ {class_name:15s}: {len(files):4d} 个样本")

    total_samples = sum(len(files) for files in all_class_files.values())
    print(f"\n总计: {total_samples} 个样本\n")

    # 生成任务
    print(f"生成 {NUM_EVAL_TASKS} 个评估任务...")
    tasks = []

    # 策略: 循环遍历查询样本，确保均匀覆盖
    # 每个类约 NUM_EVAL_TASKS / 6 个任务
    samples_per_class = NUM_EVAL_TASKS // len(EVAL_CLASSES)

    for target_class_idx, target_class in enumerate(EVAL_CLASSES):
        available_query_files = list(all_class_files[target_class])

        # 采样该类的查询文件
        if len(available_query_files) >= samples_per_class:
            query_files = random.sample(available_query_files, samples_per_class)
        else:
            # 如果样本不足，使用全部
            query_files = available_query_files

        print(f"  {target_class:15s}: 生成 {len(query_files)} 个任务")

        for query_file in query_files:
            # 为每个类选择一个支持样本
            support_examples = {}

            for class_name in EVAL_CLASSES:
                available_files = list(all_class_files[class_name])

                # 如果是同类，排除查询文件
                if class_name == target_class and query_file in available_files:
                    try:
                        available_files.remove(query_file)
                    except ValueError:
                        pass

                if len(available_files) > 0:
                    support_file = random.choice(available_files)
                    support_examples[class_name] = support_file
                else:
                    print(f"    警告: {class_name} 没有可用的支持样本")
                    continue

            # 构建任务
            task = build_6way_1shot_task(support_examples, query_file, target_class)
            tasks.append(task)

    print(f"\n✓ 生成了 {len(tasks)} 个任务\n")

    # 打乱任务顺序
    random.shuffle(tasks)

    # 验证任务
    print("验证任务有效性...")

    valid_count = 0
    for idx, task in enumerate(tasks):
        # 检查支持集是否完整
        if len(task['support_examples']) != len(EVAL_CLASSES):
            print(f"  警告: 任务 {idx} 支持集不完整")
            continue

        # 检查查询文件是否存在
        if not Path(task['query_file']).exists():
            print(f"  警告: 任务 {idx} 查询文件不存在: {task['query_file']}")
            continue

        # 检查支持文件是否存在
        support_valid = True
        for class_name, file_path in task['support_examples'].items():
            if not Path(file_path).exists():
                print(f"  警告: 任务 {idx} 支持文件不存在: {file_path}")
                support_valid = False
                break

        if support_valid:
            valid_count += 1

    print(f"✓ 有效任务: {valid_count} / {len(tasks)} ({valid_count/len(tasks)*100:.1f}%)\n")

    # 保存任务
    output_file = Path(OUTPUT_DIR) / "eval_tasks_new_6classes.json"
    print(f"保存任务到: {output_file}")

    with open(output_file, 'w') as f:
        json.dump(tasks, f, indent=2, default=str)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✓ 保存成功 ({file_size_mb:.1f} MB)\n")

    # 统计信息
    print("="*70)
    print("任务统计信息")
    print("="*70)
    print(f"总任务数: {len(tasks)}")
    print(f"目标类别: 6 (F22, F35, GlobalHawk, IDF, Mirage2000, Predator)")
    print(f"任务配置: 6-way 1-shot")
    print(f"总推理次数: {len(tasks)} 任务 × 6 查询 = {len(tasks) * 6} 次推理")
    print(f"随机种子: {RANDOM_SEED} (可复现)")
    print("="*70 + "\n")

    return tasks


if __name__ == '__main__':
    tasks = main()

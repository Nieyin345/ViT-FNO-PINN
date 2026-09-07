"""
预处理 v2 数据：把体数据在指定 d/λ 处插值，缓存为小文件。
运行一次后训练速度大幅提升。

用法：python data/preprocess_v2.py
"""

import os
import numpy as np
from tqdm import tqdm
from interpolate_field import extract_plane_at_normalized_distance, FREQ_LIST

DATA_DIR = "data/data"
TARGET_D_OVER_LAMBDA = 0.1
CACHE_SUFFIX = f"_plane_d{TARGET_D_OVER_LAMBDA:.2f}.npy"  # e.g. _plane_d0.10.npy


def preprocess_all(data_dir=DATA_DIR, target_d=TARGET_D_OVER_LAMBDA):
    categories = [d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d))]

    total = 0
    skipped = 0

    for cat in sorted(categories):
        cat_dir = os.path.join(data_dir, cat)
        samples = sorted([d for d in os.listdir(cat_dir)
                          if os.path.isdir(os.path.join(cat_dir, d))])

        for sample in tqdm(samples, desc=cat):
            sample_dir = os.path.join(cat_dir, sample)
            volume_file = os.path.join(sample_dir, "field_volume_merged.npy")
            cache_file  = os.path.join(sample_dir, f"field_plane_d{target_d:.2f}.npy")

            if not os.path.isfile(volume_file):
                continue

            if os.path.isfile(cache_file):
                skipped += 1
                continue

            # 加载体数据并插值
            field_volume = np.load(volume_file)  # (9, 22, 64, 64, 12)
            planes = extract_plane_at_normalized_distance(
                field_volume, target_d, kind="cubic"
            )  # (9, 64, 64, 12)

            np.save(cache_file, planes)
            total += 1

    print(f"\n完成：新生成 {total} 个，跳过已有 {skipped} 个")
    print(f"缓存文件名格式：field_plane_d{target_d:.2f}.npy")


if __name__ == "__main__":
    preprocess_all()

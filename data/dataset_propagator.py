"""
场传播数据集

源场：field_plane_d0.10.npy（已有缓存，0.1λ 处的散射场）
目标场：field_volume_merged.npy 的固定 z 层（直接读取，不插值）

每条记录：(样本, 频率, 目标z层) 三元组
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

FREQ_LIST = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
FREQ_MIN, FREQ_MAX = FREQ_LIST[0], FREQ_LIST[-1]
C_MM = 3e11  # 光速 mm/s

# 体数据的 z 层（mm），步长 1mm，共 22 层
Z_MM = np.arange(2, 24, 1, dtype=np.float32)  # [2, 3, ..., 23]

# 源场在 0.1λ，目标层从第 1 层开始（z=3mm）到第 21 层（z=23mm）
TARGET_Z_INDICES = list(range(1, 22))  # 21 个目标层


def wavelength_mm(freq_ghz):
    return C_MM / (freq_ghz * 1e9)


class PropagatorDataset(Dataset):
    """
    场传播数据集。
    源场：0.1λ 处的散射场（来自缓存文件 field_plane_d0.10.npy）
    目标场：体数据中固定 z 层的散射场
    """
    def __init__(
        self,
        data_dir: str,
        target_z_indices=None,
        train_categories=None,
        train_freqs=None,
        normalize: bool = False,
        output_norm_mean=None,
        output_norm_std=None,
        bg_params_path: str = "data/blank/background_params.npz",
    ):
        self.target_z_indices = target_z_indices or TARGET_Z_INDICES
        self.normalize        = normalize
        self.bg_params_path   = bg_params_path

        self.output_norm_mean = None
        self.output_norm_std  = None
        if output_norm_mean is not None and output_norm_std is not None:
            self.set_output_normalization(output_norm_mean, output_norm_std)

        freq_set = set(train_freqs) if train_freqs else set(FREQ_LIST)

        def _iter_dirs(path):
            try:
                return [e.path for e in os.scandir(path) if e.is_dir()]
            except:
                return []

        self.samples = []
        for cat_dir in sorted(_iter_dirs(data_dir)):
            cat_name = os.path.basename(cat_dir)
            if train_categories and not any(
                cat_name == c or cat_name.startswith(c + "/")
                for c in train_categories
            ):
                continue
            for sample_dir in sorted(_iter_dirs(cat_dir)):
                # 源场缓存文件（0.1λ，已有）
                cache_file  = os.path.join(sample_dir, "field_plane_d0.10.npy")
                # 体数据文件（用于取目标层）
                volume_file = os.path.join(sample_dir, "field_volume_merged.npy")
                if not os.path.isfile(cache_file) or not os.path.isfile(volume_file):
                    continue
                for fi, f in enumerate(FREQ_LIST):
                    if f not in freq_set:
                        continue
                    lam = wavelength_mm(f)
                    source_z = 0.1 * lam
                    # 计算该频率下有效的目标层（z > 0.1λ）
                    valid_tgt = [i for i in self.target_z_indices
                                 if float(Z_MM[i]) > source_z]
                    if not valid_tgt:
                        continue
                    self.samples.append({
                        "cache":      cache_file,
                        "volume":     volume_file,
                        "category":   cat_name,
                        "name":       os.path.basename(sample_dir),
                        "freq_ghz":   f,
                        "freq_idx":   fi,
                        "valid_tgt":  valid_tgt,   # 该频率下有效的目标层索引列表
                    })

        if not self.samples:
            raise ValueError(f"未找到任何样本: {data_dir}")

        cat_counts = Counter(s["category"] for s in self.samples)
        print(f"[PropagatorDataset] {len(self.samples)} 条记录")
        print(f"  类别: {dict(sorted(cat_counts.items()))}")
        print(f"  目标层数: {len(self.target_z_indices)}")

    def set_output_normalization(self, mean, std):
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        std  = np.asarray(std,  dtype=np.float32).reshape(-1)
        std  = np.maximum(std, 1e-8)
        self.output_norm_mean = mean
        self.output_norm_std  = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        fi      = s["freq_idx"]
        freq    = s["freq_ghz"]
        lam     = wavelength_mm(freq)

        # 随机选一个有效目标层
        tgt_idx = int(np.random.choice(s["valid_tgt"]))

        # ── 源场：从缓存读取 0.1λ 处的散射场 ──────────────────
        # cache shape: (9, 64, 64, 12)，已经是散射场（之前预处理时减过背景场）
        cache = np.load(s["cache"], mmap_mode="r")          # (9, 64, 64, 12)
        src_plane = cache[fi]                # (64, 64, 12)

        # ── 目标场：从体数据取对应 z 层，减去背景场 ────────────
        volume = np.load(s["volume"], mmap_mode="r")        # (9, 22, 64, 64, 12)
        tgt_plane = volume[fi, tgt_idx]      # (64, 64, 12)

        from data.fit_background import get_background_field
        z_tgt  = float(Z_MM[tgt_idx])
        bg_tgt = get_background_field(freq, z_tgt, self.bg_params_path)  # (12,)
        tgt_scatter = tgt_plane - bg_tgt[None, None, :]  # (64, 64, 12)

        # 转成 (12, H, W)
        src_t = torch.from_numpy(src_plane.transpose(2, 0, 1).copy())
        tgt_t = torch.from_numpy(tgt_scatter.transpose(2, 0, 1).copy())

        if self.normalize:
            if self.output_norm_mean is None:
                raise RuntimeError("normalize=True 但未设置归一化统计量")
            mean = torch.from_numpy(self.output_norm_mean).view(-1, 1, 1)
            std  = torch.from_numpy(self.output_norm_std).view(-1, 1, 1)
            src_t = (src_t - mean) / std
            tgt_t = (tgt_t - mean) / std

        freq_norm    = (freq - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)
        d_over_lambda = z_tgt / lam  # 目标位置的绝对 d/λ
        # 超网络输入：相对于源场（0.1λ）的距离差
        source_d      = 0.1
        dist_input    = d_over_lambda - source_d  # 从 0.1λ 出发传播了多远

        return {
            "source":   src_t,
            "target":   tgt_t,
            "freq":     torch.tensor([freq_norm],  dtype=torch.float32),
            "dist":     torch.tensor([dist_input], dtype=torch.float32),
            "freq_ghz": freq,
            "target_d": d_over_lambda,
            "z_tgt_mm": z_tgt,
            "name":     s["name"],
            "category": s["category"],
        }

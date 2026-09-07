"""
多平面散射场数据集（单平面预测 + PINN 约束版本）

主预测目标：z=3mm 的散射场
PINN 约束：同时返回 z=4mm 的散射场，用于计算 ∂/∂z

每条记录：(样本, 频率) 二元组，同时返回两个 z 层
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

FREQ_LIST = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
FREQ_MIN, FREQ_MAX = FREQ_LIST[0], FREQ_LIST[-1]

_Z_MM_VOLUME = np.arange(2, 24, 1, dtype=np.float32)

# 主预测层和 PINN 相邻层
MAIN_Z_IDX = 1   # z=3mm
PINN_Z_IDX = 2   # z=4mm（用于差分）

# 目标 z 层（mm）及其在体数据中的索引
_TARGET_Z_MM = [2.0, 3.0, 4.0]
_TARGET_Z_IDX = [0, 1, 2]  # 对应 _Z_MM_VOLUME 中的索引

# 背景场 z 轴（mm），43层，步长 0.5mm
_Z_MM_BG = np.arange(2, 23.1, 0.5, dtype=np.float32)  # [2.0, 2.5, ..., 23.0]


def _load_background_field(bg_path: str) -> np.ndarray:
    """
    加载背景场体数据。

    Returns:
        bg_volume: (9, 43, 64, 64, 12)，频率 x z x H x W x 通道
    """
    return np.load(bg_path, mmap_mode="r")  # (9, 43, 64, 64, 12), 内存映射避免整份读入


def _get_bg_at_z(bg_volume: np.ndarray, freq_idx: int, z_mm: float) -> np.ndarray:
    """
    从背景场体数据中插值得到指定频率、z位置的场平面。

    Args:
        bg_volume: (9, 43, H, W, 12)
        freq_idx:  频率索引 [0,8]
        z_mm:      目标 z 位置（mm）

    Returns:
        (H, W, 12) 背景场平面
    """
    bg_freq = bg_volume[freq_idx]  # (43, H, W, 12)
    n_z, H, W, C = bg_freq.shape
    flat = bg_freq.reshape(n_z, -1)  # (43, H*W*12)
    f = interp1d(_Z_MM_BG, flat, axis=0, kind="linear", bounds_error=False,
                 fill_value="extrapolate")
    return f(z_mm).reshape(H, W, C).astype(np.float32)


class MultiplaneScatterDataset(Dataset):
    """
    多平面散射场数据集。

    每条记录对应 (样本, 频率, z层) 三元组，输出为散射场（总场 - 背景场）。

    Args:
        data_dir:   数据根目录，包含各类别子目录
        bg_path:    背景场文件路径
        max_samples: 最多加载的样本数（按形状计，None=全部）
    """
    def __init__(
        self,
        data_dir: str,
        bg_path: str,
        max_samples: int = None,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.bg_path = os.path.abspath(bg_path)

        # 加载背景场（一次性加载到内存）
        print(f"[MultiplaneScatterDataset] 加载背景场: {self.bg_path}")
        self.bg_volume = _load_background_field(self.bg_path)
        print(f"  背景场形状: {self.bg_volume.shape}")

        # 扫描所有样本
        self.records = []  # list of dict
        self._scan_data(max_samples)

        if not self.records:
            raise ValueError(f"未找到任何样本: {self.data_dir}")

        print(f"[MultiplaneScatterDataset] 共 {len(self.records)} 条记录")
        print(f"  主预测: z={_Z_MM_VOLUME[MAIN_Z_IDX]:.0f}mm, PINN层: z={_Z_MM_VOLUME[PINN_Z_IDX]:.0f}mm")

    def _scan_data(self, max_samples: int = None):
        """扫描数据目录，建立 (样本, 频率, z层) 三元组列表。"""
        shape_count = 0

        def _iter_dirs(path):
            try:
                return sorted([e.path for e in os.scandir(path) if e.is_dir()])
            except (FileNotFoundError, PermissionError):
                return []

        def _process_sample(sample_dir):
            patch_file = os.path.join(sample_dir, "patch_plane.npy")
            volume_file = os.path.join(sample_dir, "field_volume_merged.npy")
            if not os.path.isfile(patch_file) or not os.path.isfile(volume_file):
                return
            # 每个样本只按频率展开，不按 z 层展开
            for freq_idx, freq_ghz in enumerate(FREQ_LIST):
                self.records.append({
                    "patch":    patch_file,
                    "volume":   volume_file,
                    "freq_idx": freq_idx,
                    "freq_ghz": freq_ghz,
                })

        for cat_dir in _iter_dirs(self.data_dir):
            sub_dirs = _iter_dirs(cat_dir)
            if not sub_dirs:
                continue

            # 判断是否有子类别（三级目录）
            first_sub = sub_dirs[0]
            has_patch = os.path.isfile(os.path.join(first_sub, "patch_plane.npy"))
            has_subcat = not has_patch and bool(_iter_dirs(first_sub))

            if has_subcat:
                for subcat_dir in sub_dirs:
                    for sample_dir in _iter_dirs(subcat_dir):
                        if max_samples is not None and shape_count >= max_samples:
                            return
                        _process_sample(sample_dir)
                        shape_count += 1
            else:
                for sample_dir in sub_dirs:
                    if max_samples is not None and shape_count >= max_samples:
                        return
                    _process_sample(sample_dir)
                    shape_count += 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # 输入：SDF + Mask，归一化 SDF
        patch = np.load(rec["patch"], mmap_mode="r").astype(np.float32)
        sdf = patch[:, :, 0]
        sdf = sdf / (np.abs(sdf).max() + 1e-8)
        mask = patch[:, :, 1]
        inp = torch.from_numpy(np.stack([sdf, mask], axis=0))  # (2, H, W)

        volume = np.load(rec["volume"], mmap_mode="r")
        if volume.ndim == 6 and volume.shape[0] == 2:
            volume = volume[0]
            
        fi = rec["freq_idx"]
        freq_ghz = rec["freq_ghz"]

        def get_scatter(z_idx, z_mm):
            total = volume[fi, z_idx]  # (H, W, 12)
            bg    = _get_bg_at_z(self.bg_volume, fi, z_mm)
            return torch.from_numpy((total - bg).transpose(2, 0, 1).copy())  # (12, H, W)

        # 主预测目标：z=3mm
        out_main = get_scatter(MAIN_Z_IDX, float(_Z_MM_VOLUME[MAIN_Z_IDX]))
        # PINN 相邻层：z=4mm（用于差分计算 ∂/∂z）
        out_pinn = get_scatter(PINN_Z_IDX, float(_Z_MM_VOLUME[PINN_Z_IDX]))

        freq_norm = (freq_ghz - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)

        return {
            "input":      inp,                                                  # (2, H, W)
            "output":     out_main,                                             # (12, H, W) z=3mm
            "output_pinn": out_pinn,                                            # (12, H, W) z=4mm
            "freq":       torch.tensor([freq_norm], dtype=torch.float32),
            "freq_ghz":   freq_ghz,
        }

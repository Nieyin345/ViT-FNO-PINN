"""
根据已有的离散 z 位置数据，插值得到指定归一化距离 (d/lambda) 处的场分布。

数据格式：field_volume_merged.npy  shape=(9, 22, 64, 64, 12)
  axis0: 9 个频点 [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0] GHz
  axis1: 22 个 z 位置，从 2mm 到 23mm，步长 1mm
  axis2: x 空间 64
  axis3: y 空间 64
  axis4: 12 个通道 (Ex_r,Ex_i, Ey_r,Ey_i, Ez_r,Ez_i, Hx_r,Hx_i, Hy_r,Hy_i, Hz_r,Hz_i)
"""

import numpy as np
from scipy.interpolate import interp1d

# 频点列表（GHz）
FREQ_LIST = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
C_LIGHT = 3e8  # 光速 m/s

# z 轴位置（mm），从 2mm 到 23mm，步长 1mm
Z_MM = np.arange(2, 24, 1, dtype=np.float32)  # shape (22,)


def get_wavelength_mm(freq_ghz: float) -> float:
    """计算波长（mm）"""
    return C_LIGHT / (freq_ghz * 1e9) * 1e3


def interpolate_at_normalized_distance(
    field_volume: np.ndarray,
    freq_idx: int,
    target_d_over_lambda: float,
    kind: str = "cubic"
) -> np.ndarray:
    """
    对指定频点，插值得到 target_d_over_lambda 处的场分布。

    Args:
        field_volume: shape (9, 22, 64, 64, 12)
        freq_idx: 频点索引 0~8
        target_d_over_lambda: 目标归一化距离，如 0.1 表示 0.1λ
        kind: 插值方式，'linear' 或 'cubic'

    Returns:
        field_at_target: shape (64, 64, 12)
    """
    freq_ghz = FREQ_LIST[freq_idx]
    lam_mm = get_wavelength_mm(freq_ghz)

    # 当前数据对应的归一化距离
    z_normalized = Z_MM / lam_mm  # shape (22,)

    # 目标距离是否在插值范围内
    z_min, z_max = z_normalized[0], z_normalized[-1]
    if target_d_over_lambda < z_min or target_d_over_lambda > z_max:
        raise ValueError(
            f"目标 d/λ={target_d_over_lambda:.4f} 超出范围 "
            f"[{z_min:.4f}, {z_max:.4f}] (freq={freq_ghz}GHz, λ={lam_mm:.1f}mm)"
        )

    # 取该频点的场数据 (22, 64, 64, 12)
    field_freq = field_volume[freq_idx]  # (22, 64, 64, 12)

    # 对每个空间点和通道做插值
    # reshape 成 (22, 64*64*12) 方便批量插值
    n_z = field_freq.shape[0]
    field_flat = field_freq.reshape(n_z, -1)  # (22, 64*64*12)

    interp_func = interp1d(z_normalized, field_flat, axis=0, kind=kind)
    result_flat = interp_func(target_d_over_lambda)  # (64*64*12,)

    return result_flat.reshape(64, 64, 12).astype(np.float32)


def extract_plane_at_normalized_distance(
    field_volume: np.ndarray,
    target_d_over_lambda: float,
    kind: str = "cubic"
) -> np.ndarray:
    """
    对所有频点，分别插值得到各自 target_d_over_lambda 处的场分布。
    不同频率对应不同的物理距离（mm）。

    Args:
        field_volume: shape (9, 22, 64, 64, 12)
        target_d_over_lambda: 目标归一化距离，如 0.1
        kind: 插值方式

    Returns:
        planes: shape (9, 64, 64, 12)，每个频点对应的场平面
    """
    n_freq = field_volume.shape[0]
    planes = []
    for fi in range(n_freq):
        plane = interpolate_at_normalized_distance(
            field_volume, fi, target_d_over_lambda, kind=kind
        )
        planes.append(plane)
    return np.stack(planes, axis=0)  # (9, 64, 64, 12)


def preprocess_sample(
    sample_dir: str,
    target_d_over_lambda: float = 0.1,
    kind: str = "cubic"
) -> dict:
    """
    处理单个样本，提取 target_d_over_lambda 处的场平面。

    Returns:
        {
            'patch': (64, 64, 2),       SDF + Mask
            'field_plane': (9, 64, 64, 12),  各频点在 target_d/λ 处的场
            'freq_list': [8.0, ..., 12.0],
            'target_d_over_lambda': 0.1,
            'z_mm_per_freq': [3.75, ...],   各频点对应的物理距离(mm)
        }
    """
    import os
    field_path = os.path.join(sample_dir, "field_volume_merged.npy")
    patch_path = os.path.join(sample_dir, "patch_plane.npy")

    field_volume = np.load(field_path)  # (9, 22, 64, 64, 12)
    patch = np.load(patch_path)         # (64, 64, 2)

    field_plane = extract_plane_at_normalized_distance(
        field_volume, target_d_over_lambda, kind=kind
    )

    z_mm_per_freq = [
        target_d_over_lambda * get_wavelength_mm(f) for f in FREQ_LIST
    ]

    return {
        "patch": patch,
        "field_plane": field_plane,
        "freq_list": FREQ_LIST,
        "target_d_over_lambda": target_d_over_lambda,
        "z_mm_per_freq": z_mm_per_freq,
    }


if __name__ == "__main__":
    # 测试
    import os

    sample_dir = "data/data/1_rectangle/1_rectangle_1"
    print("测试插值...")

    result = preprocess_sample(sample_dir, target_d_over_lambda=0.1)
    print(f"patch shape:       {result['patch'].shape}")
    print(f"field_plane shape: {result['field_plane'].shape}")
    print(f"各频点对应物理距离(mm):")
    for f, z in zip(result['freq_list'], result['z_mm_per_freq']):
        print(f"  {f} GHz -> {z:.2f} mm")

    # 验证范围
    print("\n验证各频点的 d/λ 范围:")
    for fi, f in enumerate(FREQ_LIST):
        lam = get_wavelength_mm(f)
        z_min = Z_MM[0] / lam
        z_max = Z_MM[-1] / lam
        print(f"  {f} GHz: {z_min:.3f}λ ~ {z_max:.3f}λ")

"""
根据已有的离散 z 位置数据，插值得到指定归一化距离 (d/lambda) 处的场分布。

数据格式：
  field_volume_merged.npy
    - 单极化: (9, N_Z, 64, 64, 12)
    - 双极化: (2, 9, N_Z, 64, 64, 12)

axis 含义：
  - 双极化首轴: 2 个互易极化激励
  - 9 个频点 [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0] GHz
  - N_Z 个 z 层，相对贴片从 2mm 起、步长 1mm（旧 22 层或新 3 层）
  - 12 通道 (Ex_r,Ex_i, Ey_r,Ey_i, Ez_r,Ez_i, Hx_r,Hx_i, Hy_r,Hy_i, Hz_r,Hz_i)
"""

import numpy as np
from scipy.interpolate import interp1d

FREQ_LIST = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
C_LIGHT = 3e8  # 光速 m/s
Z_MM_START = 2
Z_MM_STEP = 1


def get_wavelength_mm(freq_ghz: float) -> float:
    """计算波长（mm）"""
    return C_LIGHT / (freq_ghz * 1e9) * 1e3


def _z_mm_for_nz(n_z: int) -> np.ndarray:
    return np.arange(Z_MM_START, Z_MM_START + n_z, Z_MM_STEP,
                     dtype=np.float32)


def _interpolate_single_polarization(field_freq, target_d_lam, lam_mm, kind):
    """field_freq: (N_Z, H, W, C)，返回 (H, W, C)。"""
    n_z, H, W, C = field_freq.shape
    z_mm = _z_mm_for_nz(n_z)
    z_norm = z_mm / lam_mm

    flat = field_freq.reshape(n_z, -1)
    # 短 z 轴用线性插值，避免 cubic 在 3 层数据上震荡。
    interp_kind = kind if n_z >= 4 else "linear"
    f = interp1d(z_norm, flat, axis=0, kind=interp_kind,
                 bounds_error=False, fill_value="extrapolate")
    return f(target_d_lam).reshape(H, W, C).astype(np.float32)


def interpolate_at_normalized_distance(
    field_volume: np.ndarray,
    freq_idx: int,
    target_d_over_lambda: float,
    kind: str = "linear",
    polarization: int = None
) -> np.ndarray:
    """
    对指定频点，插值得到 target_d_over_lambda 处的场分布。

    Args:
        field_volume: 5D 单极化或 6D 双极化体数据
        freq_idx: 频点索引 0~8
        target_d_over_lambda: 目标归一化距离，如 0.1 表示 0.1λ
        kind: 插值方式，'linear' 或 'cubic'
        polarization: 6D 数据指定极化；None 时返回所有极化 (P,H,W,C)

    Returns:
        field_at_target: (H, W, 12) 或 (P, H, W, 12)
    """
    freq_ghz = FREQ_LIST[freq_idx]
    lam_mm = get_wavelength_mm(freq_ghz)

    if field_volume.ndim == 6:
        n_pol = field_volume.shape[0]
        if polarization is None:
            planes = [
                _interpolate_single_polarization(field_volume[pi, freq_idx],
                                                 target_d_over_lambda,
                                                 lam_mm, kind)
                for pi in range(n_pol)
            ]
            return np.stack(planes, axis=0)
        return _interpolate_single_polarization(
            field_volume[polarization, freq_idx],
            target_d_over_lambda, lam_mm, kind)

    return _interpolate_single_polarization(
        field_volume[freq_idx], target_d_over_lambda, lam_mm, kind)


def extract_plane_at_normalized_distance(
    field_volume: np.ndarray,
    target_d_over_lambda: float,
    kind: str = "linear"
) -> np.ndarray:
    """
    对所有频点，分别插值得到各自 target_d_over_lambda 处的场分布。

    Args:
        field_volume: 5D 单极化或 6D 双极化体数据
        target_d_over_lambda: 目标归一化距离，如 0.1
        kind: 插值方式

    Returns:
        planes:
          - 5D 输入: (9, H, W, 12)
          - 6D 输入: (2, 9, H, W, 12)
    """
    n_freq = field_volume.shape[1] if field_volume.ndim == 6 else field_volume.shape[0]
    planes = []
    for fi in range(n_freq):
        plane = interpolate_at_normalized_distance(
            field_volume, fi, target_d_over_lambda, kind=kind
        )
        planes.append(plane)
    planes = np.stack(planes, axis=0)
    # 6D 输入时把极化维放到最前：(2, 9, H, W, 12)
    if field_volume.ndim == 6:
        planes = np.transpose(planes, (1, 0, 2, 3, 4))
    return planes


def preprocess_sample(
    sample_dir: str,
    target_d_over_lambda: float = 0.1,
    kind: str = "linear"
) -> dict:
    """
    处理单个样本，提取 target_d_over_lambda 处的场平面。

    Returns:
        {
            'patch': (64, 64, 2),       SDF + Mask
            'field_plane': 5D/6D 各频点（含极化）在 target_d/λ 处的场
            'freq_list': [8.0, ..., 12.0],
            'target_d_over_lambda': 0.1,
            'z_mm_per_freq': [3.75, ...],   各频点对应的物理距离(mm)
        }
    """
    import os
    field_path = os.path.join(sample_dir, "field_volume_merged.npy")
    patch_path = os.path.join(sample_dir, "patch_plane.npy")

    field_volume = np.load(field_path)
    patch = np.load(patch_path)

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
    import os

    sample_dir = "data/data/rectangle/rectangle_1"
    print("测试插值...")

    result = preprocess_sample(sample_dir, target_d_over_lambda=0.1)
    print(f"patch shape:       {result['patch'].shape}")
    print(f"field_plane shape: {result['field_plane'].shape}")
    print("各频点对应物理距离(mm):")
    for f, z in zip(result['freq_list'], result['z_mm_per_freq']):
        print(f"  {f} GHz -> {z:.2f} mm")

    print("\n验证各频点的 d/λ 范围:")
    for fi, f in enumerate(FREQ_LIST):
        lam = get_wavelength_mm(f)
        print(f"  {f} GHz: 2.0/λ={2.0/lam:.3f}, (2+N_Z-1)/λ 以实际 NPY 层数为准")

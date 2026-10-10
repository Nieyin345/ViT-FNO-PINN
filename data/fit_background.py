"""
拟合背景场（无贴片）的驻波参数，存成解析公式的系数。

背景场满足：E(z) = A * sin(k*z + phi) + offset
其中 k = 2*pi*f/c，A/phi/offset 由拟合得到。

运行一次后生成 data/blank/background_params.npz，
之后可直接用公式计算任意 z 位置的背景场，无需加载大文件。
"""

import numpy as np
from scipy.optimize import curve_fit

FREQS_GHZ = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
C_MM_S = 3e11  # 光速 mm/s
Z_MM = np.arange(2, 23.1, 0.5)  # 43 个 z 层，步长 0.5mm

CH_NAMES = ['Ex_r', 'Ex_i', 'Ey_r', 'Ey_i', 'Ez_r', 'Ez_i',
            'Hx_r', 'Hx_i', 'Hy_r', 'Hy_i', 'Hz_r', 'Hz_i']


def _orthogonal_polarization_field(field):
    """把极化 0 的背景场旋转成极化 1：Ex/Ey 互换、Hx/Hy 互换。

    **E 组要额外取负号，H 组不取** —— 两个极化的激励方向相差 90°，旋转后的
    横向分量符号由旋转矩阵决定。实测（115 个双极化样本、3 个频点，复最小二乘）：

        pol1.Ex = c · pol0.Ey,  c = -0.9567  (相位 ~180°)   → 交换 + 负
        pol1.Ey = c · pol0.Ex,  c = -0.8921  (相位 ~180°)   → 交换 + 负
        pol1.Hx = c · pol0.Hy,  c = +0.9473  (相位   0°)    → 纯交换
        pol1.Hy = c · pol0.Hx,  c = +0.9821  (相位   0°)    → 纯交换

    原先两处都是纯交换（E 少了负号），后果：pol1 的散射场（总场 − 背景）里
    混进约两倍幅值的入射波，实测能量比 pol1/pol0 中位 1.479（正确应为 1.000），
    残余与入射波相关 0.925（正确 0.810）。影响 DATASET_POLARIZATIONS=[0,1]
    下约 1/3 的训练记录，且只在数值上体现、终端无异常。
    """
    out = np.asarray(field).astype(np.float32).copy()
    if out.shape[0] >= 12:
        out[[0, 1, 2, 3]] = -field[[2, 3, 0, 1]]   # E: 交换 + 负号
        out[[6, 7, 8, 9]] = field[[8, 9, 6, 7]]    # H: 纯交换
    return out


def fit_all_channels(volume_path="data/blank/blank/field_volume_merged.npy",
                     out_path="data/blank/background_params.npz"):
    """
    拟合背景场参数。

    支持：
      - 单极化: (9, N_Z, 64, 64, 12)
      - 双极化: (2, 9, N_Z, 64, 64, 12)
    """
    d = np.load(volume_path)
    n_pol = int(d.shape[0]) if d.ndim == 6 else 1

    n_freq = len(FREQS_GHZ)
    n_ch = len(CH_NAMES)

    if n_pol > 1:
        A_arr = np.zeros((n_pol, n_freq, n_ch), dtype=np.float64)
        phi_arr = np.zeros((n_pol, n_freq, n_ch), dtype=np.float64)
        offset_arr = np.zeros((n_pol, n_freq, n_ch), dtype=np.float64)
        rmse_arr = np.zeros((n_pol, n_freq, n_ch), dtype=np.float64)
    else:
        A_arr = np.zeros((n_freq, n_ch), dtype=np.float64)
        phi_arr = np.zeros((n_freq, n_ch), dtype=np.float64)
        offset_arr = np.zeros((n_freq, n_ch), dtype=np.float64)
        rmse_arr = np.zeros((n_freq, n_ch), dtype=np.float64)

    for po in range(n_pol):
        vol = d[po] if n_pol > 1 else d
        tag = f" pol{po}" if n_pol > 1 else ""

        for fi, f in enumerate(FREQS_GHZ):
            k = 2 * np.pi * f * 1e9 / C_MM_S

            def standing_wave(z, A, phi, offset):
                return A * np.sin(k * z + phi) + offset

            for ci in range(n_ch):
                # 取中心点（空间均匀，取哪个点都一样）
                y = vol[fi, :, 32, 32, ci].astype(np.float64)
                amp_guess = (y.max() - y.min()) / 2
                try:
                    popt, _ = curve_fit(standing_wave, Z_MM, y,
                                        p0=[amp_guess, 0.0, y.mean()],
                                        maxfev=10000)
                    if n_pol > 1:
                        A_arr[po, fi, ci] = popt[0]
                        phi_arr[po, fi, ci] = popt[1]
                        offset_arr[po, fi, ci] = popt[2]
                        rmse_arr[po, fi, ci] = np.sqrt(np.mean(
                            (y - standing_wave(Z_MM, *popt)) ** 2))
                    else:
                        A_arr[fi, ci] = popt[0]
                        phi_arr[fi, ci] = popt[1]
                        offset_arr[fi, ci] = popt[2]
                        rmse_arr[fi, ci] = np.sqrt(np.mean(
                            (y - standing_wave(Z_MM, *popt)) ** 2))
                except Exception as e:
                    print(f"  拟合失败{tag} {f}GHz {CH_NAMES[ci]}: {e}")

    np.savez(out_path,
             freqs=np.array(FREQS_GHZ),
             A=A_arr, phi=phi_arr, offset=offset_arr, rmse=rmse_arr)

    print(f"已保存: {out_path}")
    print(f"参数维度: A={A_arr.shape} 极化数={n_pol}")
    return A_arr, phi_arr, offset_arr


_BG_NPZ_CACHE = {}


def get_background_field(freq_ghz: float, z_mm: float,
                         params_path="data/blank/background_params.npz",
                         polarization: int = 0) -> np.ndarray:
    """
    根据拟合参数计算指定频率、z 位置和极化的背景场。

    Returns:
        field: (12,) 各通道的背景场值
    """
    p = _BG_NPZ_CACHE.get(params_path)
    if p is None:
        with np.load(params_path) as raw_p:
            p = {k: raw_p[k] for k in raw_p.files}
        _BG_NPZ_CACHE[params_path] = p
    freqs = p['freqs']
    fi = np.argmin(np.abs(freqs - freq_ghz))

    k = 2 * np.pi * freq_ghz * 1e9 / C_MM_S
    A = p['A']
    phi = p['phi']
    offset = p['offset']

    if A.ndim == 3:
        pol = int(min(max(int(polarization), 0), A.shape[0] - 1))
        field = (A[pol, fi] * np.sin(k * z_mm + phi[pol, fi])
                 + offset[pol, fi])
    else:
        field = A[fi] * np.sin(k * z_mm + phi[fi]) + offset[fi]
        if int(polarization) != 0:
            field = _orthogonal_polarization_field(field)
    return np.asarray(field, dtype=np.float32)


def get_background_plane(freq_ghz: float, z_mm: float, H: int = 64, W: int = 64,
                         params_path="data/blank/background_params.npz",
                         polarization: int = 0) -> np.ndarray:
    """
    返回指定频率、z 位置和极化的背景场平面（空间均匀，广播到 H×W）。

    Returns:
        plane: (H, W, 12)
    """
    field = get_background_field(freq_ghz, z_mm, params_path,
                                 polarization=polarization)
    return np.broadcast_to(field, (H, W, 12)).copy()


if __name__ == "__main__":
    fit_all_channels()

    print("\n验证（8GHz, z=10mm）:")
    bg0 = get_background_field(8.0, 10.0, polarization=0)
    bg1 = get_background_field(8.0, 10.0, polarization=1)
    print("  极化0:", bg0.round(2))
    print("  极化1:", bg1.round(2))

    # 和原始数据对比
    d = np.load("data/blank/blank/field_volume_merged.npy")
    z_arr = np.arange(2, 23.1, 0.5)
    zi = np.argmin(np.abs(z_arr - 10.0))
    orig = d[0, zi, 32, 32, :]
    print("  原始值:", orig.round(2))
    print("  误差:  ", np.abs(bg0 - orig).round(4))

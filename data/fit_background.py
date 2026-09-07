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

CH_NAMES = ['Ex_r','Ex_i','Ey_r','Ey_i','Ez_r','Ez_i',
            'Hx_r','Hx_i','Hy_r','Hy_i','Hz_r','Hz_i']


def fit_all_channels(volume_path="data/blank/blank/field_volume_merged.npy",
                     out_path="data/blank/background_params.npz"):
    d = np.load(volume_path)  # (9, 43, 64, 64, 12)

    n_freq = len(FREQS_GHZ)
    n_ch   = len(CH_NAMES)

    A_arr      = np.zeros((n_freq, n_ch), dtype=np.float64)
    phi_arr    = np.zeros((n_freq, n_ch), dtype=np.float64)
    offset_arr = np.zeros((n_freq, n_ch), dtype=np.float64)
    rmse_arr   = np.zeros((n_freq, n_ch), dtype=np.float64)

    for fi, f in enumerate(FREQS_GHZ):
        k = 2 * np.pi * f * 1e9 / C_MM_S

        def standing_wave(z, A, phi, offset):
            return A * np.sin(k * z + phi) + offset

        for ci in range(n_ch):
            # 取中心点（空间均匀，取哪个点都一样）
            y = d[fi, :, 32, 32, ci].astype(np.float64)
            amp_guess = (y.max() - y.min()) / 2
            try:
                popt, _ = curve_fit(standing_wave, Z_MM, y,
                                    p0=[amp_guess, 0.0, y.mean()],
                                    maxfev=10000)
                A_arr[fi, ci]      = popt[0]
                phi_arr[fi, ci]    = popt[1]
                offset_arr[fi, ci] = popt[2]
                pred = standing_wave(Z_MM, *popt)
                rmse_arr[fi, ci] = np.sqrt(np.mean((y - pred) ** 2))
            except Exception as e:
                print(f"  拟合失败 {f}GHz {CH_NAMES[ci]}: {e}")

    np.savez(out_path,
             freqs=np.array(FREQS_GHZ),
             A=A_arr, phi=phi_arr, offset=offset_arr, rmse=rmse_arr)

    print(f"已保存: {out_path}")
    print(f"最大 RMSE: {rmse_arr.max():.4f}  通道: {CH_NAMES[rmse_arr.argmax() % n_ch]}")
    return A_arr, phi_arr, offset_arr


def get_background_field(freq_ghz: float, z_mm: float,
                         params_path="data/blank/background_params.npz") -> np.ndarray:
    """
    根据拟合参数计算指定频率和 z 位置的背景场。

    Returns:
        field: (12,) 各通道的背景场值
    """
    p = np.load(params_path)
    freqs = p['freqs']
    fi = np.argmin(np.abs(freqs - freq_ghz))

    k = 2 * np.pi * freq_ghz * 1e9 / C_MM_S
    field = p['A'][fi] * np.sin(k * z_mm + p['phi'][fi]) + p['offset'][fi]
    return field.astype(np.float32)


def get_background_plane(freq_ghz: float, z_mm: float, H: int = 64, W: int = 64,
                         params_path="data/blank/background_params.npz") -> np.ndarray:
    """
    返回指定频率和 z 位置的背景场平面（空间均匀，广播到 H×W）。

    Returns:
        plane: (H, W, 12)
    """
    field = get_background_field(freq_ghz, z_mm, params_path)
    return np.broadcast_to(field, (H, W, 12)).copy()


if __name__ == "__main__":
    fit_all_channels()

    # 验证
    print("\n验证（8GHz, z=10mm）:")
    bg = get_background_field(8.0, 10.0)
    print("  背景场:", bg.round(2))

    # 和原始数据对比
    d = np.load("data/blank/blank/field_volume_merged.npy")
    z_arr = np.arange(2, 23.1, 0.5)
    zi = np.argmin(np.abs(z_arr - 10.0))
    orig = d[0, zi, 32, 32, :]
    print("  原始值:", orig.round(2))
    print("  误差:  ", np.abs(bg - orig).round(4))

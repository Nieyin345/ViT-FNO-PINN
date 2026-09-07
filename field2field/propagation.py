"""平面波谱（PWS）场传播核心函数。"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from constants import C0


def angular_spectrum_propagate(
    field_xy: np.ndarray,
    dx: float,
    dy: float,
    freq_hz: float,
    dz: float,
    n_medium: float = 1.0,
) -> np.ndarray:
    """
    单向 PWS 传播：只传播波（截断倏逝波），适用于单方向传播的场。
    E(x,y,z+dz) = IFFT2( FFT2(E(x,y,z)) * exp(j*kz*dz) )
    """
    nx, ny = field_xy.shape
    k = 2.0 * math.pi * freq_hz * n_medium / C0
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * math.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kt2 = KX ** 2 + KY ** 2

    prop = kt2 < k ** 2
    kz = np.where(prop, np.sqrt(np.maximum(k ** 2 - kt2, 0.0)), 0.0)
    H = np.where(prop, np.exp(1j * kz * dz), 0.0)

    return np.fft.ifft2(np.fft.fft2(field_xy) * H)


def propagate_em_twoway_from_source(
    e_source: np.ndarray,
    h_source: np.ndarray,
    dx: float,
    dy: float,
    freq_hz: float,
    dz: float,
    n_medium: float = 1.0,
    lp_cutoff: float = 0.8,  # 低通截止，相对于 k，1.0=不滤波
) -> Tuple[np.ndarray, np.ndarray]:
    """
    E+H 联合双向 PWS 传播。
    lp_cutoff: 传播前对源场做低通滤波，去除网格噪声，默认截断 0.8k 以上的分量。
    """
    mu0 = 4.0e-7 * math.pi
    omega = 2.0 * math.pi * freq_hz
    k = 2.0 * math.pi * freq_hz * n_medium / C0
    ommu = omega * mu0

    _, nx, ny = e_source.shape
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * math.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kt2 = KX ** 2 + KY ** 2

    prop = (kt2 < k ** 2)
    kz_r = np.where(prop, np.sqrt(np.maximum(k ** 2 - kt2, 0.0)), 0.0)
    # 掠射角数值不稳定，额外截断
    prop = prop & (kz_r >= 0.001 * k)
    kz_r = np.where(prop, kz_r, 0.0)
    kz_s = np.where(prop, kz_r, 1.0)   # 避免除零

    # 倏逝波：指数衰减而非置零
    evan = ~prop  # 倏逝波掩码
    kz_evan = np.where(evan, np.sqrt(np.maximum(kt2 - k ** 2, 0.0)), 0.0)
    # 限制最大衰减，避免数值下溢（衰减超过 60dB 就截断）
    max_decay = 10.0  # exp(-10) ≈ 4.5e-5
    decay = np.where(evan, np.exp(-np.minimum(kz_evan * abs(dz), max_decay)), 0.0)
    # 低通滤波：截断 kt > lp_cutoff*k 的极高频噪声（默认3倍k，保留倏逝波）
    if lp_cutoff < float('inf'):
        lp_mask = (kt2 < (lp_cutoff * k) ** 2).astype(float)
    else:
        lp_mask = np.ones_like(kt2)

    ex = np.fft.fft2(e_source[0]) * lp_mask
    ey = np.fft.fft2(e_source[1]) * lp_mask
    hx = np.fft.fft2(h_source[0]) * lp_mask
    hy = np.fft.fft2(h_source[1]) * lp_mask

    ep_x = np.where(prop, (ex * kz_s - hy * ommu) / (2.0 * kz_s), 0j)
    em_x = np.where(prop, (ex * kz_s + hy * ommu) / (2.0 * kz_s), 0j)
    ep_y = np.where(prop, (ey * kz_s + hx * ommu) / (2.0 * kz_s), 0j)
    em_y = np.where(prop, (ey * kz_s - hx * ommu) / (2.0 * kz_s), 0j)

    p = np.where(prop, np.exp( 1j * kz_r * dz), 0j)
    m = np.where(prop, np.exp(-1j * kz_r * dz), 0j)

    ep_x_t = ep_x * p;  em_x_t = em_x * m
    ep_y_t = ep_y * p;  em_y_t = em_y * m

    ex_t = ep_x_t + em_x_t
    ey_t = ep_y_t + em_y_t
    ez_t = np.where(prop, -(KX * ex_t + KY * ey_t) / kz_s, 0j)

    hx_t = np.where(prop, ( kz_r * (ep_y_t - em_y_t) - KY * ez_t) / ommu, 0j)
    hy_t = np.where(prop, (-kz_r * (ep_x_t - em_x_t) + KX * ez_t) / ommu, 0j)
    hz_t = np.where(prop, (KX * ey_t - KY * ex_t) / ommu, 0j)

    # 加上倏逝波贡献（指数衰减，不置零）
    ex_t = ex_t + np.where(evan, ex * decay, 0j)
    ey_t = ey_t + np.where(evan, ey * decay, 0j)
    ez_t = ez_t + np.where(evan, np.fft.fft2(e_source[2]) * decay, 0j)
    hx_t = hx_t + np.where(evan, hx * decay, 0j)
    hy_t = hy_t + np.where(evan, hy * decay, 0j)
    hz_t = hz_t + np.where(evan, np.fft.fft2(h_source[2]) * decay, 0j)

    e_out = np.empty_like(e_source)
    h_out = np.empty_like(h_source)
    e_out[0] = np.fft.ifft2(ex_t)
    e_out[1] = np.fft.ifft2(ey_t)
    e_out[2] = np.fft.ifft2(ez_t)
    h_out[0] = np.fft.ifft2(hx_t)
    h_out[1] = np.fft.ifft2(hy_t)
    h_out[2] = np.fft.ifft2(hz_t)
    return e_out, h_out

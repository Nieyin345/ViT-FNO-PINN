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
    lp_cutoff: float = float("inf"),  # 低通截止（相对 k）；inf=不滤波（默认）
) -> Tuple[np.ndarray, np.ndarray]:
    """
    E+H 联合双向 PWS 传播。
    lp_cutoff: 传播前对源场做低通滤波（截断 kt > lp_cutoff·k），默认 inf = 不滤波。

    **为什么默认不能是 0.8**：周期元胞 L 的 Floquet 谐波 kt = 2π·m/L，
    kt/k = m·λ/L。对 15 mm 元胞、8~12 GHz（λ=25~37.5 mm），所有非零阶
    谐波的 kt/k ≥ 1.67，0.8 的截止会把它们全部滤掉，传播结果只剩均匀平面波
    （零阶）。这些倏逝谐波在 z 靠近源面时仍占显著份额（e^{-κz}，κ≈0.3~0.5/mm）。
    倏逝分量本身按 e^{-κ|dz|} 自然衰减，HFSS 网格噪声不会被放大，
    所以不需要额外低通。确需滤波时显式传入。

    Ez/Hz 的符号约定（已由 field2field/test_propagation.py 的 Maxwell 解析解校验）：
    时间因子 e^{+jωt}（HFSS 约定），numpy FFT 的空间因子 e^{+jK·r}，
    因而物理波矢 k = (−KX, −KY, ±kz)。
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
    # 限制指数上限，仅为避免 exp 下溢告警
    # 旧值 10.0 会把高阶倏逝谐波的衰减"封顶"在 4.5e-5，等于在远场留下一层
    # 非物理的高频底噪（弱分量 Ez/Hz 在 z≥15 mm 处误差可达 10%~100%）。
    # exp(-700) 才接近 float64 下溢，真实衰减本来就该趋于 0。
    max_decay = 700.0
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

    # 前/后向波分解（精确版）。命名沿用历史：em = 前向(+z)波，因子 e^{-jkz·dz}；
    # ep = 后向(−z)波，因子 e^{+jkz·dz}。
    # 对每个 (KX,KY)，测得的切向 (Ex,Ey,Hx,Hy) 满足
    #     u ≡ ωμ·(Hy, −Hx) = (kz·I + K·Kᵀ/kz)·d,   d = a − b（前向减后向切向 E）
    # 解得 d = (u − K(K·u)/k²)/kz，于是 a=(E_t+d)/2，b=(E_t−d)/2。
    # 旧式 (Ex·kz ± ωμ·Hy)/(2kz) 是 K=0（垂直入射）时的特例，对斜向传播波忽略了
    # Ez 对 H 的贡献；本项目 15 mm 元胞只有零阶传播（K=0），两者数值相同。
    ux = hy * ommu
    uy = -hx * ommu
    ku = (KX * ux + KY * uy) / (k ** 2)
    d_x = (ux - KX * ku) / kz_s
    d_y = (uy - KY * ku) / kz_s
    em_x = np.where(prop, 0.5 * (ex + d_x), 0j)
    ep_x = np.where(prop, 0.5 * (ex - d_x), 0j)
    em_y = np.where(prop, 0.5 * (ey + d_y), 0j)
    ep_y = np.where(prop, 0.5 * (ey - d_y), 0j)

    p = np.where(prop, np.exp( 1j * kz_r * dz), 0j)
    m = np.where(prop, np.exp(-1j * kz_r * dz), 0j)

    ep_x_t = ep_x * p;  em_x_t = em_x * m
    ep_y_t = ep_y * p;  em_y_t = em_y * m

    ex_t = ep_x_t + em_x_t
    ey_t = ep_y_t + em_y_t
    # ∇·E=0：前向(em, +z)波 Ez=+(KX·Ex+KY·Ey)/kz，后向(ep, −z)波符号相反。
    # 旧写法对两者都用 −(…)/kz（且未区分方向），符号错，仅在 kt≠0 的传播谐波
    # 上才暴露（15 mm 元胞下只有零阶传播，KX=KY=0，所以没被发现）。
    ez_t = np.where(
        prop,
        (KX * (em_x_t - ep_x_t) + KY * (em_y_t - ep_y_t)) / kz_s,
        0j,
    )

    hx_t = np.where(prop, ( kz_r * (ep_y_t - em_y_t) - KY * ez_t) / ommu, 0j)
    hy_t = np.where(prop, (-kz_r * (ep_x_t - em_x_t) + KX * ez_t) / ommu, 0j)
    hz_t = np.where(prop, (KY * ex_t - KX * ey_t) / ommu, 0j)

    # 加上倏逝波贡献（指数衰减，不置零）
    ex_t = ex_t + np.where(evan, ex * decay, 0j)
    ey_t = ey_t + np.where(evan, ey * decay, 0j)
    ez_t = ez_t + np.where(evan, np.fft.fft2(e_source[2]) * lp_mask * decay, 0j)
    hx_t = hx_t + np.where(evan, hx * decay, 0j)
    hy_t = hy_t + np.where(evan, hy * decay, 0j)
    hz_t = hz_t + np.where(evan, np.fft.fft2(h_source[2]) * lp_mask * decay, 0j)

    e_out = np.empty_like(e_source)
    h_out = np.empty_like(h_source)
    e_out[0] = np.fft.ifft2(ex_t)
    e_out[1] = np.fft.ifft2(ey_t)
    e_out[2] = np.fft.ifft2(ez_t)
    h_out[0] = np.fft.ifft2(hx_t)
    h_out[1] = np.fft.ifft2(hy_t)
    h_out[2] = np.fft.ifft2(hz_t)
    return e_out, h_out

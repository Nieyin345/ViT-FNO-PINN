"""
2D FDFD 求解器 — TM 模式 (Ez 极化)

控制方程 (e^{-iωt} 时间约定):
    ∂/∂x(∂Ez/∂x) + ∂/∂y(∂Ez/∂y) + ω²·ε₀·ε_r·Ez = -Jz

PML 通过复数坐标拉伸实现 (Berenger 1994, Chew & Weedon 1994):
    s_w(w) = 1 - j·σ_w(w) / (ω·ε₀)

离散化参考 fdfdpy (Hugues et al., Stanford Fan Group):
    A = Dxf·Dxb + Dyf·Dyb + ω²·ε₀·ε_r
    其中 Dxf = diag(1/sx_f) · Dx_forward_bare
         Dxb = diag(1/sx_b) · Dx_backward_bare

注意: 本求解器使用归一化单位，令 ε₀=μ₀=c=1，
      因此 ω = 2π·freq，波长 λ = 1/freq。
"""

import numpy as np
from scipy.sparse import diags, eye, kron, csc_matrix
from scipy.sparse.linalg import spsolve


# ─── PML 拉伸因子 ──────────────────────────────────────────────────────────────

def _pml_sfactor(n: int, n_pml: int, omega: float, dl: float,
                 s: str) -> np.ndarray:
    """
    计算 PML 拉伸因子数组 1/s_w，长度为 n。

    Args:
        n:     格点总数
        n_pml: PML 格点数
        omega: 角频率
        dl:    格点间距
        s:     'f' 前向（半格点偏移 0.5）或 'b' 后向（整格点）

    Returns:
        inv_s: 1/s_w 数组，shape (n,)，复数
    """
    # PML 厚度（物理长度）
    dw = n_pml * dl

    # 最大衰减系数：m=4 阶多项式，目标反射率 R = e^{lnR}
    m = 4
    lnR = -12.0
    sig_max = -(m + 1) * lnR / (2 * dw)   # 归一化单位下 ε₀=1

    inv_s = np.ones(n, dtype=complex)

    for i in range(n):
        # 前向差分：采样点在 i+0.5 处
        # 后向差分：采样点在 i 处
        offset = 0.5 if s == 'f' else 0.0

        # 左侧 PML（i 从 0 到 n_pml-1）
        if i < n_pml:
            l = (n_pml - i - offset) * dl   # 距内边界的距离
            if l > 0:
                sig = sig_max * (l / dw) ** m
                inv_s[i] = 1.0 / (1.0 - 1j * sig / omega)

        # 右侧 PML（i 从 n-n_pml 到 n-1）
        elif i >= n - n_pml:
            l = (i - (n - n_pml) + offset) * dl
            if l > 0:
                sig = sig_max * (l / dw) ** m
                inv_s[i] = 1.0 / (1.0 - 1j * sig / omega)

    return inv_s


# ─── 裸差分算子（无 PML）─────────────────────────────────────────────────────

def _Dxf_bare(nx: int, dx: float) -> csc_matrix:
    """x 方向前向差分，Dirichlet 边界（最后一列置零）"""
    d0 = -np.ones(nx) / dx
    d1 =  np.ones(nx - 1) / dx
    return diags([d0, d1], [0, 1], shape=(nx, nx), format="csc", dtype=complex)


def _Dxb_bare(nx: int, dx: float) -> csc_matrix:
    """x 方向后向差分，Dirichlet 边界（第一列置零）"""
    d0 =  np.ones(nx) / dx
    dm1 = -np.ones(nx - 1) / dx
    return diags([d0, dm1], [0, -1], shape=(nx, nx), format="csc", dtype=complex)


def _Dyf_bare(ny: int, dy: float) -> csc_matrix:
    """y 方向前向差分"""
    d0 = -np.ones(ny) / dy
    d1 =  np.ones(ny - 1) / dy
    return diags([d0, d1], [0, 1], shape=(ny, ny), format="csc", dtype=complex)


def _Dyb_bare(ny: int, dy: float) -> csc_matrix:
    """y 方向后向差分"""
    d0 =  np.ones(ny) / dy
    dm1 = -np.ones(ny - 1) / dy
    return diags([d0, dm1], [0, -1], shape=(ny, ny), format="csc", dtype=complex)


# ─── 主求解函数 ────────────────────────────────────────────────────────────────

def solve_helmholtz(
    eps_r: np.ndarray,
    source: np.ndarray,
    freq: float = 1.0,
    phys_size: float = 2.0,
    n_pml: int = 12,
) -> np.ndarray:
    """
    求解 2D TM 亥姆霍兹方程，返回复数 Ez 场。

    方程: (Dxf·Dxb + Dyf·Dyb + ω²·ε_r) Ez = -Jz

    Args:
        eps_r:     介电常数分布 (H, W)，物理值（背景=1，介质=ε_r）
        source:    源项 Jz(x,y)，(H, W)，实数或复数
        freq:      归一化频率，ω = 2π·freq，波长 λ = 1/freq
        phys_size: 计算域物理尺寸（正方形边长）
        n_pml:     PML 格点数

    Returns:
        Ez: 复数场 (H, W)
    """
    ny, nx = eps_r.shape
    dx = phys_size / nx
    dy = phys_size / ny
    omega = 2.0 * np.pi * freq

    # ── PML 拉伸因子（1/s，对角矩阵）─────────────────────────────────────────
    # x 方向：前向和后向各一组
    inv_sx_f = _pml_sfactor(nx, n_pml, omega, dx, 'f')   # (nx,)
    inv_sx_b = _pml_sfactor(nx, n_pml, omega, dx, 'b')   # (nx,)
    inv_sy_f = _pml_sfactor(ny, n_pml, omega, dy, 'f')   # (ny,)
    inv_sy_b = _pml_sfactor(ny, n_pml, omega, dy, 'b')   # (ny,)

    # ── 带 PML 的差分算子（1D）────────────────────────────────────────────────
    # Dxf = diag(1/sx_f) · Dx_forward_bare，shape (nx, nx)
    Sxf_1d = diags(inv_sx_f, format="csc", dtype=complex)
    Sxb_1d = diags(inv_sx_b, format="csc", dtype=complex)
    Syf_1d = diags(inv_sy_f, format="csc", dtype=complex)
    Syb_1d = diags(inv_sy_b, format="csc", dtype=complex)

    Dxf_1d = Sxf_1d @ _Dxf_bare(nx, dx)
    Dxb_1d = Sxb_1d @ _Dxb_bare(nx, dx)
    Dyf_1d = Syf_1d @ _Dyf_bare(ny, dy)
    Dyb_1d = Syb_1d @ _Dyb_bare(ny, dy)

    # ── 扩展到 2D（Kronecker 积）──────────────────────────────────────────────
    # 存储顺序：先 x 后 y，即 index = iy*nx + ix（C 顺序）
    # x 方向算子：kron(Iy, Dxf_1d)
    # y 方向算子：kron(Dyf_1d, Ix)
    Ix = eye(nx, format="csc", dtype=complex)
    Iy = eye(ny, format="csc", dtype=complex)

    Dxf = kron(Iy, Dxf_1d, format="csc")   # (ny*nx, ny*nx)
    Dxb = kron(Iy, Dxb_1d, format="csc")
    Dyf = kron(Dyf_1d, Ix, format="csc")
    Dyb = kron(Dyb_1d, Ix, format="csc")

    # ── ω²·ε_r 对角矩阵 ───────────────────────────────────────────────────────
    eps_flat = eps_r.flatten().astype(complex)   # C 顺序，与 kron 一致
    K = diags(omega**2 * eps_flat, format="csc", dtype=complex)

    # ── 系数矩阵 A = Dxf·Dxb + Dyf·Dyb + ω²·ε_r ─────────────────────────────
    A = Dxf @ Dxb + Dyf @ Dyb + K

    # ── 右端项 b = -Jz ────────────────────────────────────────────────────────
    b = -source.flatten().astype(complex)

    # ── 稀疏直接求解 ───────────────────────────────────────────────────────────
    Ez_flat = spsolve(A, b)
    return Ez_flat.reshape(ny, nx)

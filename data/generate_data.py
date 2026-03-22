"""
数据生成脚本 — 2D TM 点源散射，基于 ceviche FDFD 求解器

场景:
    - 计算域: 64×64 格点，物理尺寸 2×2（坐标 [-1, 1]）
    - 激励:   固定高斯点源，位置 (-0.6, 0.0)
    - 边界:   四周 PML 吸收边界
    - 介质:   单个正方形/圆形，大小和位置随机，ε_r=4
    - 模式:   TM (Ez 极化)，频域直接求解

用法:
    python data/generate_data.py --n_samples 100 --shape square
    python data/generate_data.py --n_samples 100 --shape circle --append
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ceviche
from ceviche import fdfd_ez   # TM 模式：Ez 极化


# ─── 物理参数（SI 单位，ceviche 要求）────────────────────────────────────────

GRID_SIZE    = 64
C_0          = 3e8           # 光速 m/s
WAVELENGTH   = 1e-6          # 波长 1 微米
FREQ         = C_0 / WAVELENGTH          # 3e14 Hz
OMEGA        = 2 * np.pi * FREQ          # 角频率 rad/s

# 物理域：4λ × 4λ，域内可见多个完整波前
PHYS_SIZE    = 4 * WAVELENGTH            # 4 微米
DL           = PHYS_SIZE / GRID_SIZE     # 格点间距 ~62.5 nm，λ/16

N_PML        = 10       # PML 格点数
EPS_R        = 4.0      # 正方形介质介电常数（如 Si，n≈2）

# 固定点源位置（格点坐标），物理位置约在域左侧 1/4 处
SOURCE_IX    = GRID_SIZE // 4   # 16
SOURCE_IY    = GRID_SIZE // 2   # 32

# 正方形参数范围（归一化到 [0,1]，0=域左边，1=域右边）
SQUARE_SIZE_MIN = 0.10   # 最小边长（相对域尺寸）
SQUARE_SIZE_MAX = 0.50
SQUARE_POS_MIN  = 0.20   # 中心位置范围（覆盖更大区域）
SQUARE_POS_MAX  = 0.85


# ─── 几何采样 ──────────────────────────────────────────────────────────────────

def sample_geometries(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Returns: (N, 3) — [cx, cy, size]，值域均为 [0,1]（相对格点坐标）
    cx, cy: 正方形中心（格点归一化），size: 边长（格点归一化）
    """
    rng  = np.random.default_rng(seed)
    size = rng.uniform(SQUARE_SIZE_MIN, SQUARE_SIZE_MAX, n_samples)
    cx   = rng.uniform(SQUARE_POS_MIN,  SQUARE_POS_MAX,  n_samples)
    cy   = rng.uniform(SQUARE_POS_MIN,  SQUARE_POS_MAX,  n_samples)
    return np.stack([cx, cy, size], axis=1).astype(np.float32)


# ─── 输入图生成 ────────────────────────────────────────────────────────────────

def make_grid():
    """返回归一化格点坐标 [0,1] 的网格 (H, W)"""
    xs = np.linspace(0, 1, GRID_SIZE)
    ys = np.linspace(0, 1, GRID_SIZE)
    return np.meshgrid(xs, ys)   # XX[iy, ix], YY[iy, ix]


def make_source_map() -> np.ndarray:
    """点源位置的 one-hot 图，用于可视化输入"""
    s = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    s[SOURCE_IY, SOURCE_IX] = 1.0
    return s


def make_eps_map(cx, cy, size, shape="square") -> np.ndarray:
    """
    介质分布，物理值（背景=1，介质=EPS_R）。
    cx, cy, size 均为归一化坐标 [0,1]。
    """
    eps  = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    cx_g = cx * GRID_SIZE
    cy_g = cy * GRID_SIZE
    half = (size * GRID_SIZE) / 2
    iy, ix = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]

    if shape == "square":
        x0, x1 = max(0, int(cx_g - half)), min(GRID_SIZE, int(cx_g + half))
        y0, y1 = max(0, int(cy_g - half)), min(GRID_SIZE, int(cy_g + half))
        eps[y0:y1, x0:x1] = EPS_R

    elif shape == "circle":
        mask = (ix - cx_g) ** 2 + (iy - cy_g) ** 2 <= half ** 2
        eps[mask] = EPS_R

    elif shape == "ring":
        r2 = (ix - cx_g) ** 2 + (iy - cy_g) ** 2
        mask = (r2 >= (half * 0.5) ** 2) & (r2 <= half ** 2)
        eps[mask] = EPS_R

    elif shape == "open_ring":
        r2 = (ix - cx_g) ** 2 + (iy - cy_g) ** 2
        ring_mask = (r2 >= (half * 0.5) ** 2) & (r2 <= half ** 2)
        gap_angle = (size * 11.3) % (2 * np.pi)
        angle = np.arctan2(iy - cy_g, ix - cx_g)
        diff = (angle - gap_angle + np.pi) % (2 * np.pi) - np.pi
        gap_mask = np.abs(diff) < np.pi / 3
        eps[ring_mask & ~gap_mask] = EPS_R

    elif shape == "rect_ring":
        x0, x1 = max(0, int(cx_g - half)), min(GRID_SIZE, int(cx_g + half))
        y0, y1 = max(0, int(cy_g - half)), min(GRID_SIZE, int(cy_g + half))
        eps[y0:y1, x0:x1] = EPS_R
        inner = half * 0.5
        ix0, ix1 = max(0, int(cx_g - inner)), min(GRID_SIZE, int(cx_g + inner))
        iy0, iy1 = max(0, int(cy_g - inner)), min(GRID_SIZE, int(cy_g + inner))
        eps[iy0:iy1, ix0:ix1] = 1.0

    elif shape == "cross":
        arm_w = max(1, int(half * 2 / 3))
        # 横臂
        x0, x1 = max(0, int(cx_g - half)), min(GRID_SIZE, int(cx_g + half))
        y0, y1 = max(0, int(cy_g - arm_w // 2)), min(GRID_SIZE, int(cy_g + arm_w // 2 + 1))
        eps[y0:y1, x0:x1] = EPS_R
        # 竖臂
        x0, x1 = max(0, int(cx_g - arm_w // 2)), min(GRID_SIZE, int(cx_g + arm_w // 2 + 1))
        y0, y1 = max(0, int(cy_g - half)), min(GRID_SIZE, int(cy_g + half))
        eps[y0:y1, x0:x1] = EPS_R

    elif shape == "sector":
        r2 = (ix - cx_g) ** 2 + (iy - cy_g) ** 2
        rot = (size * 7.3) % (2 * np.pi)
        angle = np.arctan2(iy - cy_g, ix - cx_g)
        diff = (angle - rot + np.pi) % (2 * np.pi) - np.pi
        mask = (r2 <= half ** 2) & (np.abs(diff) <= np.pi / 4)
        eps[mask] = EPS_R

    elif shape == "ellipse":
        a, b = half, half * 0.6
        mask = ((ix - cx_g) / a) ** 2 + ((iy - cy_g) / b) ** 2 <= 1.0
        eps[mask] = EPS_R

    elif shape == "triangle":
        angles = np.array([2 * np.pi * k / 3 - np.pi / 2 for k in range(3)])
        vx = cx_g + half * np.cos(angles)
        vy = cy_g + half * np.sin(angles)
        eps[_polygon_mask(ix, iy, vx, vy)] = EPS_R

    elif shape == "pentagon":
        angles = np.array([2 * np.pi * k / 5 - np.pi / 2 for k in range(5)])
        vx = cx_g + half * np.cos(angles)
        vy = cy_g + half * np.sin(angles)
        eps[_polygon_mask(ix, iy, vx, vy)] = EPS_R

    return eps


def _polygon_mask(ix, iy, vx, vy):
    """射线法判断点是否在多边形内，返回 bool 数组 (H, W)"""
    px = ix.ravel().astype(float)
    py = iy.ravel().astype(float)
    inside = np.zeros(len(px), dtype=bool)
    n = len(vx)
    j = n - 1
    for i_v in range(n):
        xi, yi = vx[i_v], vy[i_v]
        xj, yj = vx[j], vy[j]
        cond = ((yi > py) != (yj > py)) & \
               (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i_v
    return inside.reshape(ix.shape)


def make_input_maps(params: np.ndarray, shape: str = "square") -> np.ndarray:
    """
    Returns: inputs (N, 2, H, W)
        channel 0: source 位置图（one-hot）
        channel 1: permittivity，归一化到 [0,1]
    """
    N = len(params)
    H = W = GRID_SIZE
    inputs = np.zeros((N, 2, H, W), dtype=np.float32)
    source_map = make_source_map()

    for i, (cx, cy, size) in enumerate(params):
        inputs[i, 0] = source_map
        eps_map = make_eps_map(cx, cy, size, shape=shape)
        inputs[i, 1] = ((eps_map - 1.0) / (EPS_R - 1.0)).astype(np.float32)

    return inputs


# ─── ceviche FDFD 仿真 ─────────────────────────────────────────────────────────

def run_fdfd(eps_map: np.ndarray) -> np.ndarray:
    """
    用 ceviche fdfd_ez 求解 TM 模式，返回归一化复数 Ez 场 (H, W)。

    ceviche 约定:
        - eps_r shape: (Nx, Ny) = (列数, 行数)，与 numpy (行, 列) 相反，需转置
        - 返回 Ez shape: (Nx, Ny)，同样需转置回 (H, W)
    """
    # ceviche 用 (Nx, Ny) 顺序，转置
    eps_ceviche = eps_map.T.astype(np.complex128)

    sim = fdfd_ez(OMEGA, DL, eps_ceviche, [N_PML, N_PML])

    src = np.zeros((GRID_SIZE, GRID_SIZE), dtype=complex)
    src[SOURCE_IX, SOURCE_IY] = 1.0   # 注意：ceviche 坐标也是 (ix, iy)

    _, _, Ez = sim.solve(src)

    Ez = Ez.T   # 转回 (H, W)

    # 归一化：除以最大幅值
    amp_max = np.abs(Ez).max()
    if amp_max > 0:
        Ez = Ez / amp_max

    return Ez, float(amp_max)   # complex (H, W)，归一化场 + 原始幅值


# ─── 主函数 ───────────────────────────────────────────────────────────────────

ALL_SHAPES = ["square", "circle", "cross", "ring", "open_ring",
              "rect_ring", "sector", "ellipse", "triangle", "pentagon"]


def generate_one_shape(shape, n_samples, seed, out_dir, append):
    """生成单个形状的数据并保存/追加"""
    print(f"\n{'='*50}")
    print(f"形状: {shape}  n={n_samples}  seed={seed}")
    print(f"{'='*50}")

    params = sample_geometries(n_samples, seed=seed)
    inputs = make_input_maps(params, shape=shape)

    outputs = np.zeros((n_samples, 2, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    scales  = np.zeros(n_samples, dtype=np.float64)
    s_effs  = np.zeros((n_samples, 2), dtype=np.float64)

    for i, (cx, cy, size) in enumerate(params):
        eps_map = make_eps_map(cx, cy, size, shape=shape)
        Ez, scale = run_fdfd(eps_map)
        outputs[i, 0] = Ez.real.astype(np.float32)
        outputs[i, 1] = Ez.imag.astype(np.float32)
        scales[i]     = scale

        eps_norm = (eps_map - 1.0) / (EPS_R - 1.0)
        def lap2d(u):
            u_pad = np.pad(u, 1, mode='constant', constant_values=0)
            return u_pad[2:,1:-1]+u_pad[:-2,1:-1]+u_pad[1:-1,2:]+u_pad[1:-1,:-2]-4*u
        k_nd = 2 * np.pi * (4.0 / GRID_SIZE)
        eps_r_nd = 1.0 + (EPS_R - 1.0) * eps_norm
        lap_c = lap2d(Ez.real) + 1j * lap2d(Ez.imag)
        R_no_src = lap_c + k_nd**2 * eps_r_nd * (Ez.real + 1j * Ez.imag)
        s_eff = -R_no_src[SOURCE_IY, SOURCE_IX]
        s_effs[i, 0] = s_eff.real
        s_effs[i, 1] = s_eff.imag

        if (i + 1) % 50 == 0 or i == n_samples - 1:
            print(f"  [{i+1}/{n_samples}]  scale={scale:.4e}  |s_eff|={abs(s_eff):.4f}")

    # 保存
    meta_path    = os.path.join(out_dir, "metadata.npy")
    inputs_path  = os.path.join(out_dir, "inputs.npy")
    outputs_path = os.path.join(out_dir, "outputs.npy")
    scales_path  = os.path.join(out_dir, "scales.npy")
    seffs_path   = os.path.join(out_dir, "s_effs.npy")

    if append and os.path.exists(outputs_path):
        params  = np.concatenate([np.load(meta_path),    params],  axis=0)
        inputs  = np.concatenate([np.load(inputs_path),  inputs],  axis=0)
        outputs = np.concatenate([np.load(outputs_path), outputs], axis=0)
        scales  = np.concatenate([np.load(scales_path),  scales],  axis=0)
        s_effs  = np.concatenate([np.load(seffs_path),   s_effs],  axis=0)

    np.save(meta_path,    params)
    np.save(inputs_path,  inputs)
    np.save(outputs_path, outputs)
    np.save(scales_path,  scales)
    np.save(seffs_path,   s_effs)
    print(f"  → 累计总样本: {len(outputs)}  |s_eff|均值={np.abs(s_effs[:,0]+1j*s_effs[:,1]).mean():.4f}")
    return len(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",   type=int, default=500)
    parser.add_argument("--out_dir",     type=str, default="data/em_data_meep")
    parser.add_argument("--seed",        type=int, default=1000)
    parser.add_argument("--inputs_only", action="store_true")
    parser.add_argument("--shape",       type=str, default="all",
                        choices=ALL_SHAPES + ["all"])
    parser.add_argument("--append",      action="store_true", help="追加到已有数据集")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    shapes = ALL_SHAPES if args.shape == "all" else [args.shape]

    for i, shape in enumerate(shapes):
        seed = args.seed + i * 100   # 每种形状用不同 seed
        append = args.append or (i > 0)   # 第一种形状看 --append，后续自动追加
        generate_one_shape(shape, args.n_samples, seed, args.out_dir, append)

    # 最终预览
    outputs = np.load(os.path.join(args.out_dir, "outputs.npy"))
    inputs  = np.load(os.path.join(args.out_dir, "inputs.npy"))
    params  = np.load(os.path.join(args.out_dir, "metadata.npy"))
    _visualize(inputs, outputs, params,
               save_path=os.path.join(args.out_dir, "samples_preview.png"))
    print(f"\n全部完成！总样本数: {len(outputs)}")


def _visualize(inputs, outputs, params, n_show=4, save_path="samples_preview.png"):
    n_show = min(n_show, len(inputs))
    fig, axes = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        cx, cy, size = params[i]
        vmax = max(np.abs(outputs[i]).max(), 1e-6)

        axes[i, 0].imshow(inputs[i, 0], cmap="hot", origin="lower")
        axes[i, 0].set_title("Source")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(inputs[i, 1], cmap="viridis", origin="lower")
        axes[i, 1].set_title(f"Permittivity\ncx={cx:.2f} cy={cy:.2f} s={size:.2f}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(outputs[i, 0], cmap="RdBu", origin="lower",
                          vmin=-vmax, vmax=vmax)
        axes[i, 2].set_title("Ez real")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(outputs[i, 1], cmap="RdBu", origin="lower",
                          vmin=-vmax, vmax=vmax)
        axes[i, 3].set_title("Ez imag")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"预览图: {save_path}")


if __name__ == "__main__":
    main()

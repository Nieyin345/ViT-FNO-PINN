"""
验证 ceviche 生成的真值场是否满足复数亥姆霍兹方程

∇²Ez + k²·ε_r·Ez + S = 0   (Ez 为复数，S 为实数点源)

用法: python verify_helmholtz.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID_SIZE = 64
N_PML     = 10
EPS_MIN   = 1.0
EPS_MAX   = 4.0
K_PINN    = 2 * np.pi * (4.0 / 64)   # 与 losses.py 一致
SOURCE_IX = GRID_SIZE // 4            # 16
SOURCE_IY = GRID_SIZE // 2            # 32


def fd_laplacian_np(u: np.ndarray) -> np.ndarray:
    """中心差分拉普拉斯，边界零填充，支持复数数组"""
    u_pad = np.pad(u, 1, mode="constant", constant_values=0)
    return (u_pad[2:,  1:-1]
          + u_pad[:-2, 1:-1]
          + u_pad[1:-1, 2:]
          + u_pad[1:-1, :-2]
          - 4 * u)


def helmholtz_residual(Ez_real, Ez_imag, eps_norm, k, s_eff=1.0):
    """
    复数亥姆霍兹残差：R = ∇²Ez + k²·ε_r·Ez - S_eff
    ceviche 约定：∇²Ez + k²·ε_r·Ez = -S  →  R = ... - S_eff
    s_eff: 等效源强度 = 1/amp_max（归一化后）
    """
    eps_r = EPS_MIN + (EPS_MAX - EPS_MIN) * eps_norm
    Ez_c  = Ez_real + 1j * Ez_imag
    lap   = fd_laplacian_np(Ez_real) + 1j * fd_laplacian_np(Ez_imag)

    S = np.zeros_like(Ez_real)
    S[SOURCE_IY, SOURCE_IX] = s_eff

    return lap + k**2 * eps_r * Ez_c - S  # ceviche 约定：∇²Ez + k²ε_r Ez = -S


def main():
    data_dir = "data/em_data_meep"
    inputs  = np.load(f"{data_dir}/inputs.npy").astype(np.float32)
    outputs = np.load(f"{data_dir}/outputs.npy").astype(np.float32)
    s_effs  = np.load(f"{data_dir}/s_effs.npy")   # (N, 2) complex source strength
    N = len(inputs)
    print(f"数据集: {N} 个样本")
    print(f"k = {K_PINN:.6f}  (= 2pi * 4/64)")
    print()

    # ── 批量统计 ──────────────────────────────────────────────────────────────
    p = N_PML
    rel_res_all = []
    for i in range(N):
        s_eff = s_effs[i,0] + 1j*s_effs[i,1]
        R    = helmholtz_residual(outputs[i,0], outputs[i,1], inputs[i,1], K_PINN, s_eff)
        # 排除源点（源点处有点源激励，残差不为零是正常的）
        R_masked = R.copy()
        R_masked[SOURCE_IY, SOURCE_IX] = 0
        Ez_c = (outputs[i,0] + 1j * outputs[i,1])[p:-p, p:-p]
        rel  = (np.sqrt(np.mean(np.abs(R_masked[p:-p, p:-p])**2))
                / (np.sqrt(np.mean(np.abs(Ez_c)**2)) + 1e-10))
        rel_res_all.append(rel)

    rel_res_all = np.array(rel_res_all)
    print("=" * 50)
    print("复数亥姆霍兹相对残差（内部区域，排除 PML）")
    print("=" * 50)
    print(f"  均值:   {rel_res_all.mean():.6f}")
    print(f"  中位数: {np.median(rel_res_all):.6f}")
    print(f"  最小值: {rel_res_all.min():.6f}")
    print(f"  最大值: {rel_res_all.max():.6f}")
    print()
    print("  < 0.01 : 非常好，真值完全满足亥姆霍兹")
    print("  0.01~0.1: 可接受，PML 边界引入的误差")
    print("  > 0.1  : 有问题，k 值或源项设置不对")

    # ── 扫描 k，找最优值 ──────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("k 值扫描（用前 20 个样本）")
    print("=" * 50)
    k_candidates = np.linspace(K_PINN * 0.5, K_PINN * 2.0, 40)
    mean_res_by_k = []
    for k_try in k_candidates:
        res_list = []
        for i in range(min(20, N)):
            s_eff = s_effs[i,0] + 1j*s_effs[i,1]
            R    = helmholtz_residual(outputs[i,0], outputs[i,1], inputs[i,1], k_try, s_eff)
            R_masked = R.copy(); R_masked[SOURCE_IY, SOURCE_IX] = 0
            Ez_c = (outputs[i,0] + 1j * outputs[i,1])[p:-p, p:-p]
            rel  = (np.sqrt(np.mean(np.abs(R_masked[p:-p, p:-p])**2))
                    / (np.sqrt(np.mean(np.abs(Ez_c)**2)) + 1e-10))
            res_list.append(rel)
        mean_res_by_k.append(np.mean(res_list))

    best_idx = np.argmin(mean_res_by_k)
    best_k   = k_candidates[best_idx]
    cur_idx  = np.argmin(np.abs(k_candidates - K_PINN))
    print(f"  最优 k = {best_k:.6f}  (残差={mean_res_by_k[best_idx]:.6f})")
    print(f"  PINN k = {K_PINN:.6f}  (残差={mean_res_by_k[cur_idx]:.6f})")
    print(f"  比值 best_k / K_PINN = {best_k / K_PINN:.4f}")

    # ── 可视化 ────────────────────────────────────────────────────────────────
    n_show = 3
    fig, axes = plt.subplots(n_show, 5, figsize=(17, 3.5 * n_show))
    for row in range(n_show):
        s_eff = s_effs[row,0] + 1j*s_effs[row,1]
        Ez_r = outputs[row, 0]
        Ez_i = outputs[row, 1]
        eps  = inputs[row, 1]
        R    = helmholtz_residual(Ez_r, Ez_i, eps, K_PINN, s_eff)
        R_masked = R.copy(); R_masked[SOURCE_IY, SOURCE_IX] = 0
        Ez_c = Ez_r + 1j * Ez_i
        rel  = (np.sqrt(np.mean(np.abs(R_masked[p:-p, p:-p])**2))
                / (np.sqrt(np.mean(np.abs(Ez_c[p:-p, p:-p])**2)) + 1e-10))

        vmax = np.abs(Ez_c).max()
        rmax = np.abs(R_masked).max() + 1e-10

        axes[row,0].imshow(eps, cmap="viridis", origin="lower")
        axes[row,0].set_title(f"Sample {row}\nPermittivity"); axes[row,0].axis("off")

        axes[row,1].imshow(Ez_r, cmap="RdBu", origin="lower", vmin=-vmax, vmax=vmax)
        axes[row,1].set_title("Ez real"); axes[row,1].axis("off")

        axes[row,2].imshow(Ez_i, cmap="RdBu", origin="lower", vmin=-vmax, vmax=vmax)
        axes[row,2].set_title("Ez imag"); axes[row,2].axis("off")

        axes[row,3].imshow(np.abs(R_masked), cmap="hot", origin="lower", vmin=0, vmax=rmax)
        axes[row,3].set_title(f"|Residual|\nrel={rel:.4f}"); axes[row,3].axis("off")

        axes[row,4].imshow(R_masked.real, cmap="RdBu", origin="lower", vmin=-rmax, vmax=rmax)
        axes[row,4].set_title("Residual real"); axes[row,4].axis("off")

    plt.suptitle(f"Complex Helmholtz Residual  k={K_PINN:.4f}\n"
                 f"mean rel residual = {rel_res_all.mean():.4f}", fontsize=11)
    plt.tight_layout()
    plt.savefig("checkpoints/helmholtz_verify.png", dpi=150)
    plt.close()

    # k 扫描曲线
    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_candidates, mean_res_by_k, "b-o", markersize=4)
    ax.axvline(K_PINN, color="r", linestyle="--", label=f"PINN k={K_PINN:.4f}")
    ax.axvline(best_k, color="g", linestyle="--", label=f"best k={best_k:.4f}")
    ax.set_xlabel("k"); ax.set_ylabel("mean rel residual")
    ax.set_title("Helmholtz residual vs k")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("checkpoints/k_scan.png", dpi=150)
    plt.close()

    print("\ncheckpoints/helmholtz_verify.png")
    print("checkpoints/k_scan.png")


if __name__ == "__main__":
    main()

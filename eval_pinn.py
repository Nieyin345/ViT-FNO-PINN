"""
PINN 预测可视化 + 误差分布

输出:
  eval_preview.png    — 逐样本对比图（真值 / 预测 / 误差分布）
  eval_error_dist.png — 全验证集 rel-L2 直方图 + CDF

用法: python eval_pinn.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import FNO2d
from data import EMDataset, get_splits, get_stratified_splits
import config as C

CKPT      = f"{C.SAVE_DIR}/best.pth"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
VAL_RATIO = 0.2
SEED      = 42   # 必须和 train_pinn.py 里的 seed 一致


def load_model():
    model = FNO2d(
        modes1=C.MODES1, modes2=C.MODES2, width=C.WIDTH, n_layers=C.N_LAYERS,
        out_channels=2, use_geo_encoder=C.USE_GEO_ENCODER,
        geo_embed_dim=C.GEO_EMBED_DIM, geo_depth=C.GEO_DEPTH,
        patch_size=C.PATCH_SIZE,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"模型加载: {CKPT}")
    return model


def predict_one(model, dataset, idx):
    item   = dataset[idx]
    source = item["source"].unsqueeze(0).to(DEVICE)
    eps    = item["permittivity"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(torch.cat([source, eps], dim=1))[0].cpu().numpy()
    return {
        "pred_r": out[0],
        "pred_i": out[1],
        "true_r": item["Ez_true"][0].numpy(),
        "true_i": item["Ez_true"][1].numpy(),
        "eps":    item["permittivity"][0].numpy(),
    }


def rel_l2_from_dict(d):
    pred = d["pred_r"] + 1j * d["pred_i"]
    true = d["true_r"] + 1j * d["true_i"]
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-8))


def draw_row(axes, row, d, label):
    pred    = d["pred_r"] + 1j * d["pred_i"]
    true    = d["true_r"] + 1j * d["true_i"]
    err_map = np.abs(pred - true)
    rel_l2  = float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-8))
    vmax    = max(np.abs(d["true_r"]).max(), np.abs(d["pred_r"]).max(), 1e-6)

    axes[row, 0].imshow(d["eps"],    cmap="viridis", origin="lower")
    axes[row, 0].set_title(f"{label}\nPermittivity"); axes[row, 0].axis("off")

    axes[row, 1].imshow(d["true_r"], cmap="RdBu", origin="lower", vmin=-vmax, vmax=vmax)
    axes[row, 1].set_title("Ez real true"); axes[row, 1].axis("off")

    axes[row, 2].imshow(d["pred_r"], cmap="RdBu", origin="lower", vmin=-vmax, vmax=vmax)
    axes[row, 2].set_title("Ez real pred"); axes[row, 2].axis("off")

    axes[row, 3].imshow(d["true_i"], cmap="RdBu", origin="lower", vmin=-vmax, vmax=vmax)
    axes[row, 3].set_title("Ez imag true"); axes[row, 3].axis("off")

    axes[row, 4].imshow(d["pred_i"], cmap="RdBu", origin="lower", vmin=-vmax, vmax=vmax)
    axes[row, 4].set_title(f"Ez imag pred\nrel-L2={rel_l2:.3f}"); axes[row, 4].axis("off")

    im = axes[row, 5].imshow(err_map, cmap="hot", origin="lower")
    axes[row, 5].set_title(f"|pred-true|\nmax={err_map.max():.3f}"); axes[row, 5].axis("off")
    plt.colorbar(im, ax=axes[row, 5], fraction=0.046, pad=0.04)

    return rel_l2


def eval_val_set(model, dataset, val_indices):
    """对验证集全部样本计算 per-sample rel-L2"""
    rel_l2_list = []
    model.eval()
    for idx in val_indices:
        d = predict_one(model, dataset, idx)
        rel_l2_list.append(rel_l2_from_dict(d))
    return np.array(rel_l2_list)


def plot_error_dist(rel_l2_arr, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(rel_l2_arr, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(rel_l2_arr.mean(),    color="red",    linestyle="--",
                    label=f"mean={rel_l2_arr.mean():.3f}")
    axes[0].axvline(np.median(rel_l2_arr), color="orange", linestyle="--",
                    label=f"median={np.median(rel_l2_arr):.3f}")
    axes[0].set_xlabel("rel-L2"); axes[0].set_ylabel("count")
    axes[0].set_title("Val rel-L2 Distribution")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    sorted_err = np.sort(rel_l2_arr)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    axes[1].plot(sorted_err, cdf, color="steelblue", linewidth=2)
    axes[1].axvline(rel_l2_arr.mean(), color="red", linestyle="--",
                    label=f"mean={rel_l2_arr.mean():.3f}")
    axes[1].set_xlabel("rel-L2"); axes[1].set_ylabel("CDF")
    axes[1].set_title("Cumulative Distribution")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"N={len(rel_l2_arr)}  mean={rel_l2_arr.mean():.4f}  "
        f"std={rel_l2_arr.std():.4f}  "
        f"p50={np.percentile(rel_l2_arr,50):.4f}  "
        f"p90={np.percentile(rel_l2_arr,90):.4f}",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"误差分布图: {save_path}")


def main():
    dataset = EMDataset(n_total=C.N_TOTAL)
    model   = load_model()

    # 用和训练完全相同的 split
    if C.USE_STRATIFIED_SPLIT:
        actual_group = len(dataset) // (C.N_TOTAL // C.GROUP_SIZE)
        train_indices, val_indices = get_stratified_splits(
            len(dataset),
            group_size=actual_group,
            val_ratio=VAL_RATIO,
            seed=SEED,
        )
    else:
        train_indices, val_indices = get_splits(len(dataset), val_ratio=VAL_RATIO, seed=SEED)
    print(f"[Split] 训练: {len(train_indices)}  验证: {len(val_indices)}"
          + (f"  (分层: {C.N_TOTAL//C.GROUP_SIZE}组×{int(C.GROUP_SIZE*C.VAL_RATIO)})" if C.USE_STRATIFIED_SPLIT else ""))

    # ── 1. 逐样本对比图：从训练集取4个，验证集取4个
    n_show   = 4
    show_tr  = train_indices[:n_show]
    show_val = val_indices[:n_show]
    indices  = show_tr + show_val
    labels   = [f"[Train] idx={i}" for i in show_tr] + [f"[Val] idx={i}" for i in show_val]
    n_rows   = len(indices)

    fig, axes = plt.subplots(n_rows, 6, figsize=(21, 3.2 * n_rows))
    errors_train, errors_val = [], []

    for row, (idx, label) in enumerate(zip(indices, labels)):
        d   = predict_one(model, dataset, idx)
        err = draw_row(axes, row, d, label)
        (errors_train if "Train" in label else errors_val).append(err)

    # 分隔线
    fig.add_artist(plt.Line2D(
        [0.02, 0.98], [1 - n_show / n_rows, 1 - n_show / n_rows],
        transform=fig.transFigure, color="orange", linewidth=2, linestyle="--"
    ))
    plt.suptitle(
        f"Train rel-L2: {np.mean(errors_train):.4f}    "
        f"Val rel-L2: {np.mean(errors_val):.4f}",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(f"{C.SAVE_DIR}/eval_preview.png", dpi=150)
    plt.close()
    print(f"预览图: {C.SAVE_DIR}/eval_preview.png")
    print(f"Train rel-L2 (4样本): {np.mean(errors_train):.4f}  "
          f"Val rel-L2 (4样本): {np.mean(errors_val):.4f}")

    # ── 2. 全验证集误差分布
    print(f"\n计算全验证集误差分布 (N={len(val_indices)})...")
    rel_l2_arr = eval_val_set(model, dataset, val_indices)
    print(f"  mean={rel_l2_arr.mean():.4f}  std={rel_l2_arr.std():.4f}")
    print(f"  min={rel_l2_arr.min():.4f}   max={rel_l2_arr.max():.4f}")
    print(f"  p50={np.percentile(rel_l2_arr,50):.4f}  p90={np.percentile(rel_l2_arr,90):.4f}")
    plot_error_dist(rel_l2_arr, f"{C.SAVE_DIR}/eval_error_dist.png")

    # ── 3. 高误差样本分析
    print(f"\n[高误差样本分析] top-10 最差样本:")
    metadata_path = f"{C.DATA_DIR}/metadata.npy"
    has_meta = os.path.exists(metadata_path)
    if has_meta:
        metadata = np.load(metadata_path)  # (N, 3) — [cx, cy, size]

    sorted_idx = np.argsort(rel_l2_arr)[::-1]
    print(f"{'rank':<5} {'val_idx':<8} {'rel-L2':<10}", end="")
    if has_meta:
        print(f"{'cx':<8} {'cy':<8} {'size':<8}", end="")
    print()
    for rank, i in enumerate(sorted_idx[:10]):
        vidx = val_indices[i]
        print(f"{rank+1:<5} {vidx:<8} {rel_l2_arr[i]:<10.4f}", end="")
        if has_meta:
            cx, cy, sz = metadata[vidx]
            print(f"{cx:<8.3f} {cy:<8.3f} {sz:<8.3f}", end="")
        print()

    # 统计高误差样本的几何分布
    if has_meta:
        threshold = np.percentile(rel_l2_arr, 90)
        hard_mask = rel_l2_arr > threshold
        hard_vidx = [val_indices[i] for i in np.where(hard_mask)[0]]
        hard_meta = metadata[hard_vidx]
        easy_vidx = [val_indices[i] for i in np.where(~hard_mask)[0]]
        easy_meta = metadata[easy_vidx]
        print(f"\n[几何分布对比] p90 阈值={threshold:.4f}")
        print(f"  困难样本 (top 10%): cx={hard_meta[:,0].mean():.3f}±{hard_meta[:,0].std():.3f}  "
              f"cy={hard_meta[:,1].mean():.3f}±{hard_meta[:,1].std():.3f}  "
              f"size={hard_meta[:,2].mean():.3f}±{hard_meta[:,2].std():.3f}")
        print(f"  容易样本 (bot 90%): cx={easy_meta[:,0].mean():.3f}±{easy_meta[:,0].std():.3f}  "
              f"cy={easy_meta[:,1].mean():.3f}±{easy_meta[:,1].std():.3f}  "
              f"size={easy_meta[:,2].mean():.3f}±{easy_meta[:,2].std():.3f}")

    # ── 4. 高复杂度低误差样本可视化（场波动大但预测准）
    print(f"\n生成高复杂度低误差样本可视化...")
    # 用真值场的空间高频能量衡量复杂度（拉普拉斯算子响应的 RMS）
    complexity = []
    for idx in val_indices:
        item = dataset[idx]
        ez_r = item["Ez_true"][0].numpy()
        ez_i = item["Ez_true"][1].numpy()
        # 简单用相邻差分的 std 衡量高频振荡强度
        hf = np.std(np.diff(ez_r, axis=0)) + np.std(np.diff(ez_r, axis=1)) \
           + np.std(np.diff(ez_i, axis=0)) + np.std(np.diff(ez_i, axis=1))
        complexity.append(hf)
    complexity = np.array(complexity)

    # 高复杂度阈值：top 30%
    complexity_thresh = np.percentile(complexity, 70)
    high_complex_mask = complexity > complexity_thresh

    # 在高复杂度样本中，找误差最小的 10 个
    masked_err = rel_l2_arr.copy()
    masked_err[~high_complex_mask] = np.inf
    good_complex_idx = np.argsort(masked_err)[:10]

    fig3, axes3 = plt.subplots(10, 6, figsize=(21, 3.2 * 10))
    for row, i in enumerate(good_complex_idx):
        vidx  = val_indices[i]
        d     = predict_one(model, dataset, vidx)
        label = f"[GoodComplex #{row+1}] val_idx={vidx}  rel-L2={rel_l2_arr[i]:.4f}  complexity={complexity[i]:.4f}"
        draw_row(axes3, row, d, label)
    plt.suptitle(
        f"Top-10 High-Complexity Low-Error Samples\n"
        f"(complexity > p70={complexity_thresh:.4f}, sorted by rel-L2 asc)",
        fontsize=12
    )
    plt.tight_layout()
    good_path = f"{C.SAVE_DIR}/eval_good_complex.png"
    plt.savefig(good_path, dpi=150)
    plt.close()
    print(f"高复杂度低误差图: {good_path}")

    # ── 5. top-10 最差样本可视化
    print(f"\n生成 top-10 最差样本可视化...")
    n_worst = 10
    worst_indices = sorted_idx[:n_worst]
    fig2, axes2 = plt.subplots(n_worst, 6, figsize=(21, 3.2 * n_worst))
    for row, i in enumerate(worst_indices):
        vidx  = val_indices[i]
        d     = predict_one(model, dataset, vidx)
        label = f"[Worst #{row+1}] val_idx={vidx}  rel-L2={rel_l2_arr[i]:.4f}"
        draw_row(axes2, row, d, label)
    plt.suptitle(
        f"Top-10 Worst Samples  (mean rel-L2={rel_l2_arr[worst_indices].mean():.4f})",
        fontsize=12
    )
    plt.tight_layout()
    worst_path = f"{C.SAVE_DIR}/eval_worst10.png"
    plt.savefig(worst_path, dpi=150)
    plt.close()
    print(f"最差样本图: {worst_path}")

    # ── 5. 汇总（和训练日志对齐的数字）
    print(f"\n[汇总] 全验证集 mean rel-L2 = {rel_l2_arr.mean():.4f}")


if __name__ == "__main__":
    main()

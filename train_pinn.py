"""
PINN 训练入口 — 联合训练（data loss + pde 正则）

用法:
    python train_pinn.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import FNO2d, PINNLoss
from data import EMDataset, get_splits, get_stratified_splits
import config as C

CFG = {
    "modes1":   C.MODES1,
    "modes2":   C.MODES2,
    "width":    C.WIDTH,
    "n_layers": C.N_LAYERS,

    "n_total":      C.N_TOTAL,

    "n_epochs":   C.N_EPOCHS,
    "batch_size": C.BATCH_SIZE,
    "lr":         C.LR,
    "val_every":  C.VAL_EVERY,

    "early_stop_patience":  C.EARLY_STOP_PATIENCE,
    "early_stop_min_delta": C.EARLY_STOP_MIN_DELTA,

    "lambda_data":    C.LAMBDA_DATA    if C.USE_DATA    else 0.0,
    "lambda_pde":     C.LAMBDA_PDE     if C.USE_PDE     else 0.0,
    "lambda_bc":      C.LAMBDA_BC      if C.USE_BC      else 0.0,
    "lambda_src":     C.LAMBDA_SRC     if C.USE_SRC     else 0.0,
    "lambda_nonzero": C.LAMBDA_NONZERO if C.USE_NONZERO else 0.0,
    "lambda_amp":     C.LAMBDA_AMP     if C.USE_AMP     else 0.0,

    "val_ratio": 0.2,
    "seed":      42,
    "save_dir":  C.SAVE_DIR,
    "device":    "cuda" if torch.cuda.is_available() else "cpu",
}


def make_splits(dataset, cfg):
    if C.USE_STRATIFIED_SPLIT:
        actual_group = len(dataset) // (C.N_TOTAL // C.GROUP_SIZE)
        train_indices, val_indices = get_stratified_splits(
            len(dataset),
            group_size=actual_group,
            val_ratio=cfg["val_ratio"],
            seed=cfg["seed"],
        )
    else:
        train_indices, val_indices = get_splits(
            len(dataset), val_ratio=cfg["val_ratio"], seed=cfg["seed"]
        )
    train_ds = Subset(dataset, train_indices)
    val_ds   = Subset(dataset, val_indices)
    print(f"[Split] 训练: {len(train_ds)}  验证: {len(val_ds)}"
          + (f"  (分层: {C.N_TOTAL//C.GROUP_SIZE}组×{int(C.GROUP_SIZE*C.VAL_RATIO)})" if C.USE_STRATIFIED_SPLIT else ""))
    return train_ds, val_ds


def make_loader(ds, batch_size, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # DataParallel 包装时取 .module
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state, path)


def compute_rel_l2(model, loader, device):
    """per-sample rel-L2 的均值，和 eval 脚本一致"""
    model.eval()
    rel_l2_list = []
    with torch.no_grad():
        for batch in loader:
            source  = batch["source"].to(device)
            eps     = batch["permittivity"].to(device)
            Ez_true = batch["Ez_true"].to(device)
            Ez      = model(torch.cat([source, eps], dim=1))
            diff    = Ez[:, :2] - Ez_true[:, :2]
            # per-sample
            num = (diff[:, 0]**2 + diff[:, 1]**2).flatten(1).sum(dim=1)
            den = (Ez_true[:, 0]**2 + Ez_true[:, 1]**2).flatten(1).sum(dim=1).clamp(min=1e-8)
            rel_l2_list.append((num / den).sqrt())
    return torch.cat(rel_l2_list).mean().item()


def plot_curve(history, save_path, n_sup):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    epochs_tr  = list(range(1, len(history["train_loss"]) + 1))
    epochs_val = history["val_epochs"]

    axes[0].plot(epochs_tr, history["train_loss"], label="train loss (total)", alpha=0.7, linewidth=1)
    axes[0].plot(epochs_val, history["val_rel_l2"], label="val rel-L2",
                 alpha=0.9, marker="o", markersize=3, linewidth=1.5)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss / rel-L2")
    axes[0].set_title(f"Training Curve  (sup={n_sup})")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_tr, history["train_pde"], label="pde loss", color="orange", alpha=0.8, linewidth=1)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("pde loss")
    axes[1].set_title("PDE Residual Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train():
    torch.manual_seed(CFG["seed"])
    device = torch.device(CFG["device"])
    print(f"[Train] 设备: {device}  总样本: {CFG['n_total']}")
    active = [f"data×{CFG['lambda_data']}"]
    if CFG["lambda_pde"]     > 0: active.append(f"pde×{CFG['lambda_pde']}")
    if CFG["lambda_bc"]      > 0: active.append(f"bc×{CFG['lambda_bc']}")
    if CFG["lambda_src"]     > 0: active.append(f"src×{CFG['lambda_src']}")
    if CFG["lambda_nonzero"] > 0: active.append(f"nonzero×{CFG['lambda_nonzero']}")
    if CFG["lambda_amp"]     > 0: active.append(f"amp×{CFG['lambda_amp']}")
    print(f"[Loss] {' + '.join(active)}")

    dataset  = EMDataset(n_total=CFG["n_total"])
    train_ds, val_ds = make_splits(dataset, CFG)
    train_loader = make_loader(train_ds, CFG["batch_size"], shuffle=True)
    val_loader   = make_loader(val_ds,   CFG["batch_size"], shuffle=False)

    model = FNO2d(
        modes1=CFG["modes1"], modes2=CFG["modes2"],
        width=CFG["width"],   n_layers=CFG["n_layers"],
        out_channels=2,
        use_geo_encoder=C.USE_GEO_ENCODER,
        geo_embed_dim=C.GEO_EMBED_DIM,
        geo_depth=C.GEO_DEPTH,
        patch_size=C.PATCH_SIZE,
    ).to(device)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"[Model] DataParallel × {n_gpus} GPUs")
    print(f"[Model] 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 加载预训练权重（fine-tune）
    lr = CFG["lr"]
    if C.PRETRAIN_CKPT and os.path.exists(C.PRETRAIN_CKPT):
        model.load_state_dict(torch.load(C.PRETRAIN_CKPT, map_location=device, weights_only=True))
        lr = C.FINETUNE_LR
        print(f"[Fine-tune] 加载权重: {C.PRETRAIN_CKPT}  lr={lr}")

    criterion = PINNLoss(
        k=C.K, eps_min=C.EPS_MIN, eps_max=C.EPS_MAX, pml_width=C.PML_WIDTH,
        lambda_pde=1.0,  # 由 uncertainty weighting 接管
        lambda_bc=CFG["lambda_bc"],
        lambda_src=CFG["lambda_src"], lambda_nonzero=CFG["lambda_nonzero"],
        lambda_amp=CFG["lambda_amp"], lambda_data=1.0,  # 同上
        lambda_var=0.0,
    )
    criterion.lambda_src_pde = C.LAMBDA_SRC_PDE if C.USE_SRC_PDE else 0.0

    # 课程学习 or 固定权重
    use_uw = False
    log_weights = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["n_epochs"], eta_min=1e-5)

    best_val      = float("inf")
    no_improve    = 0          # 连续未改善的验证次数
    history       = {"train_loss": [], "train_pde": [], "val_epochs": [], "val_rel_l2": []}

    for epoch in range(1, CFG["n_epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_pde  = 0.0

        for batch in train_loader:
            source  = batch["source"].to(device)
            eps     = batch["permittivity"].to(device)
            Ez_true = batch["Ez_true"].to(device)
            s_eff   = batch["s_eff"].to(device)
            Ez      = model(torch.cat([source, eps], dim=1))

            losses = criterion(Ez, eps, source, Ez_true=Ez_true, s_eff=s_eff)

            if C.USE_CURRICULUM and C.USE_PDE and CFG["lambda_pde"] > 0:
                # sigmoid 爬升：epoch < T0 时 w_pde≈0，之后逐渐升到 LAMBDA_PDE
                progress = (epoch - C.CURRICULUM_T0) / max(C.CURRICULUM_RAMP, 1)
                w_pde  = CFG["lambda_pde"] / (1 + math.exp(-progress))
                w_data = CFG["lambda_data"]
                loss = w_data * losses["data"] + w_pde * losses["pde"]
            elif use_uw:
                w = torch.softmax(log_weights, dim=0)
                loss = w[0] * losses["data"] + w[1] * losses["pde"]
            else:
                loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + ([log_weights] if use_uw else []),
                max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()
            total_pde  += losses["pde"].item()

        scheduler.step()
        n_bat = max(len(train_loader), 1)
        history["train_loss"].append(total_loss / n_bat)
        history["train_pde"].append(total_pde / n_bat)

        if epoch % CFG["val_every"] == 0:
            val_rel_l2 = compute_rel_l2(model, val_loader, device)
            history["val_epochs"].append(epoch)
            history["val_rel_l2"].append(val_rel_l2)

            improved = val_rel_l2 < best_val - CFG["early_stop_min_delta"]
            if improved:
                best_val   = val_rel_l2
                no_improve = 0
                save_checkpoint(model, f"{CFG['save_dir']}/best.pth")
            else:
                no_improve += 1

            pde_str = f"  pde={total_pde/n_bat:.3e}" if CFG["lambda_pde"] > 0 else ""
            if C.USE_CURRICULUM and C.USE_PDE:
                progress = (epoch - C.CURRICULUM_T0) / max(C.CURRICULUM_RAMP, 1)
                w_pde_cur = CFG["lambda_pde"] / (1 + math.exp(-progress))
                uw_str = f"  w_pde={w_pde_cur:.4f}"
            else:
                uw_str = (f"  w_data={torch.softmax(log_weights,0)[0].item():.3f}"
                          f"  w_pde={torch.softmax(log_weights,0)[1].item():.3f}") if use_uw else ""
            stop_str = f"  [no improve {no_improve}/{CFG['early_stop_patience']}]" if no_improve > 0 else ""
            print(f"Epoch {epoch:4d} | loss={total_loss/n_bat:.3e}{pde_str}{uw_str}  "
                  f"val rel-L2={val_rel_l2:.4f}  best={best_val:.4f}{stop_str}")

            if no_improve >= CFG["early_stop_patience"]:
                print(f"\n[Early Stop] 验证集连续 {no_improve} 次未改善，停止训练 (epoch={epoch})")
                break

    plot_curve(history, f"{CFG['save_dir']}/pinn_losses.png", CFG["n_total"])
    print(f"\n[Done] 最佳模型: {CFG['save_dir']}/best.pth  (val={best_val:.4f})")


if __name__ == "__main__":
    train()

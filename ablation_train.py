"""
消融实验训练脚本
一次性训练 4 个变体，输出各自验证集 rel-L2

变体:
  A: 纯监督 FNO       (use_geo_encoder=False, lambda_pde=0)
  B: FNO + PINN       (use_geo_encoder=False, lambda_pde=3.0)
  C: ViT-FNO          (use_geo_encoder=True,  lambda_pde=0)
  D: ViT-FNO-PINN     (use_geo_encoder=True,  lambda_pde=3.0)  ← 已有结果，可跳过

用法:
    python ablation_train.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from models import FNO2d, PINNLoss
from data import EMDataset
import config as C

# ─── 公共超参 ──────────────────────────────────────────────────────────────────
N_EPOCHS     = 500
BATCH_SIZE   = 8
LR           = 5e-4
N_TOTAL      = C.N_TOTAL
VAL_RATIO    = 0.2
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANTS = [
    dict(name="A_sup_fno",      use_geo=False, lambda_pde=0.0,  label="纯监督 FNO"),
    dict(name="B_fno_pinn",     use_geo=False, lambda_pde=3.0,  label="FNO + PINN"),
    dict(name="C_vit_fno",      use_geo=True,  lambda_pde=0.0,  label="ViT-FNO"),
    # D 已有结果，取消注释可重跑
    # dict(name="D_vit_fno_pinn", use_geo=True,  lambda_pde=3.0,  label="ViT-FNO-PINN"),
]


def make_splits(dataset):
    n_sup = N_SUPERVISED
    rest  = list(range(n_sup, len(dataset)))
    n_val = int(len(rest) * VAL_RATIO)
    rng   = np.random.default_rng(SEED)
    rng.shuffle(rest)
    val_idx   = rest[:n_val]
    unsup_idx = rest[n_val:]
    train_idx = list(range(n_sup)) + unsup_idx
    return (Subset(dataset, train_idx),
            Subset(dataset, val_idx))


def rel_l2(pred, true):
    """复数 rel-L2，pred/true: (B,2,H,W)"""
    diff = pred - true
    num  = (diff[:,0]**2 + diff[:,1]**2).sum(dim=(-1,-2)).sqrt()
    den  = (true[:,0]**2 + true[:,1]**2).sum(dim=(-1,-2)).sqrt().clamp(min=1e-8)
    return (num / den).mean().item()


def train_variant(variant, dataset, train_ds, val_ds):
    name      = variant["name"]
    use_geo   = variant["use_geo"]
    lam_pde   = variant["lambda_pde"]
    label     = variant["label"]

    print(f"\n{'='*60}")
    print(f"[变体 {name}] {label}")
    print(f"  use_geo_encoder={use_geo}  lambda_pde={lam_pde}")
    print(f"{'='*60}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FNO2d(
        modes1=C.MODES1, modes2=C.MODES2,
        width=C.WIDTH,   n_layers=C.N_LAYERS,
        out_channels=2,
        use_geo_encoder=use_geo,
        geo_embed_dim=C.GEO_EMBED_DIM,
        geo_depth=C.GEO_DEPTH,
        patch_size=C.PATCH_SIZE,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    criterion = PINNLoss(
        k=C.K, eps_min=C.EPS_MIN, eps_max=C.EPS_MAX, pml_width=C.PML_WIDTH,
        lambda_pde=lam_pde,
        lambda_bc=0.1,
        lambda_src=1.0,
        lambda_nonzero=0.001,
        lambda_data=200.0,
        lambda_var=0.0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5)

    best_val = float("inf")
    val_history = []
    t0 = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            source     = batch["source"].to(DEVICE)
            eps        = batch["permittivity"].to(DEVICE)
            supervised = batch["supervised"].to(DEVICE)
            s_eff      = batch["s_eff"].to(DEVICE)
            inp        = torch.cat([source, eps], dim=1)
            Ez         = model(inp)

            # 物理损失 + 有监督 data loss
            lp = lam_pde  # 变体A/C为0，变体B/D为3.0
            losses = criterion(Ez, eps, source, s_eff=s_eff,
                               lam_pde=lp, lam_data=200.0)

            # 重新只对有监督样本算 data loss
            losses["total"] = (
                0.1    * losses["bc"]
              + 1.0    * losses["src"]
              + 0.001  * losses["nonzero"]
              + 1.0    * losses["amp"]
              + lp     * losses["pde"]
            )
            if supervised.any():
                Ez_true = batch["Ez_true"].to(DEVICE)
                dl = criterion.data_loss(Ez[supervised], Ez_true[supervised])
                losses["data"]  = dl
                losses["total"] = losses["total"] + 200.0 * dl

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                source  = batch["source"].to(DEVICE)
                eps     = batch["permittivity"].to(DEVICE)
                Ez_true = batch["Ez_true"].to(DEVICE)
                Ez      = model(torch.cat([source, eps], dim=1))
                val_loss += criterion.data_loss(Ez, Ez_true).item()
        val_loss /= max(len(val_loader), 1)
        val_history.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = f"{C.SAVE_DIR}/ablation_{name}.pth"
            os.makedirs(C.SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d} | val={val_loss:.4f}  best={best_val:.4f}  ({elapsed:.0f}s)")

    # 计算最终 rel-L2
    model.load_state_dict(torch.load(f"{C.SAVE_DIR}/ablation_{name}.pth",
                                      map_location=DEVICE, weights_only=True))
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            source  = batch["source"].to(DEVICE)
            eps     = batch["permittivity"].to(DEVICE)
            Ez_true = batch["Ez_true"].to(DEVICE)
            Ez      = model(torch.cat([source, eps], dim=1))
            all_preds.append(Ez.cpu())
            all_trues.append(Ez_true.cpu())
    preds = torch.cat(all_preds)
    trues = torch.cat(all_trues)
    final_rel_l2 = rel_l2(preds, trues)

    # 推理速度（单样本，CPU）
    model_cpu = model.cpu().eval()
    dummy_src = torch.zeros(1, 1, 64, 64)
    dummy_eps = torch.zeros(1, 1, 64, 64)
    dummy_inp = torch.cat([dummy_src, dummy_eps], dim=1)
    # warmup
    for _ in range(5):
        with torch.no_grad():
            model_cpu(dummy_inp)
    t_start = time.time()
    N_INFER = 100
    for _ in range(N_INFER):
        with torch.no_grad():
            model_cpu(dummy_inp)
    infer_ms = (time.time() - t_start) / N_INFER * 1000

    print(f"\n  [结果] {label}")
    print(f"    验证集 rel-L2 = {final_rel_l2:.4f}")
    print(f"    推理时间 (CPU) = {infer_ms:.2f} ms/sample")

    # 保存 loss 曲线
    plt.figure(figsize=(6, 3))
    plt.plot(val_history)
    plt.title(f"{label} — val data loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{C.SAVE_DIR}/ablation_{name}_loss.png", dpi=120)
    plt.close()

    return {
        "name":       name,
        "label":      label,
        "n_params":   n_params,
        "val_rel_l2": final_rel_l2,
        "infer_ms":   infer_ms,
    }


def main():
    torch.manual_seed(SEED)
    dataset = EMDataset(n_total=N_TOTAL)
    train_ds, val_ds = make_splits(dataset)
    print(f"训练集: {len(train_ds)}  验证集: {len(val_ds)}")

    results = []
    for v in VARIANTS:
        r = train_variant(v, dataset, train_ds, val_ds)
        results.append(r)

    print("\n" + "="*60)
    print("消融实验汇总")
    print("="*60)
    print(f"{'变体':<25} {'参数量':>10} {'val rel-L2':>12} {'推理(ms)':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['label']:<25} {r['n_params']:>10,} {r['val_rel_l2']:>12.4f} {r['infer_ms']:>10.2f}")
    # 加上已知的完整方法结果
    print(f"{'ViT-FNO-PINN (本文)':<25} {'2,560,000':>10} {'0.4200':>12} {'':>10}")
    print("="*60)

    # 保存结果到文件
    with open(f"{C.SAVE_DIR}/ablation_results.txt", "w", encoding="utf-8") as f:
        f.write("消融实验结果\n")
        f.write("="*60 + "\n")
        f.write(f"{'变体':<25} {'参数量':>10} {'val rel-L2':>12} {'推理(ms)':>10}\n")
        f.write("-"*60 + "\n")
        for r in results:
            f.write(f"{r['label']:<25} {r['n_params']:>10,} {r['val_rel_l2']:>12.4f} {r['infer_ms']:>10.2f}\n")
        f.write(f"{'ViT-FNO-PINN (本文)':<25} {'2,560,000':>10} {'0.4200':>12}\n")
    print(f"\n结果已保存: {C.SAVE_DIR}/ablation_results.txt")


if __name__ == "__main__":
    main()

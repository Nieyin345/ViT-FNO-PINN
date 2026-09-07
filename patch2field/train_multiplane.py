# -*- coding: utf-8 -*-
"""
多平面散射场训练脚本

从贴片形状（SDF+Mask）预测指定 z 位置（2mm、3mm、4mm）的散射场，
同时加入 PINN 约束（麦克斯韦方程 ∂Ey/∂z = -jωμ₀Hx）。

配置直接写在文件顶部，不依赖 experiment_config.py。
"""

import os
import sys

# 将项目根目录加入 sys.path，以便访问 data/ 模块
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_P2F = os.path.abspath(os.path.dirname(__file__))
if _P2F not in sys.path:
    sys.path.insert(0, _P2F)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import config as C  # 读取 patch2field/config.py

# =============================================================================
# 配置（从 config.py 读取，可在这里覆盖）
# =============================================================================
DATA_DIR  = os.path.abspath(os.path.join(_ROOT, "data", "data"))
BG_PATH   = os.path.abspath(os.path.join(_ROOT, "data", "blank", "blank", "field_volume_merged.npy"))
SAVE_DIR  = os.path.abspath(os.path.join(_ROOT, "checkpoints", "multiplane"))
TRAIN_CATEGORIES = getattr(C, "TRAIN_CATEGORIES", None)
MAX_SAMPLES      = None

# 模型结构
MODES1    = 12
MODES2    = 12
WIDTH     = 64
N_LAYERS  = 5
FREQ_EMB_DIM = 32
Z_EMB_DIM    = 32

# 训练
N_EPOCHS          = 500
BATCH_SIZE        = 16
LR                = 1e-3
LR_WARMUP_EPOCHS  = 20
WEIGHT_DECAY      = 1e-4
GRAD_CLIP_MAX_NORM = 1.0
NUM_WORKERS       = 0
PIN_MEMORY        = True
VAL_RATIO         = 0.2
SHUFFLE_SEED      = 42
EARLY_STOP_PATIENCE   = 100
EARLY_STOP_MIN_DELTA  = 1e-5

# 损失权重
PINN_WEIGHT = 0.1   # 总 loss = geom_loss + PINN_WEIGHT * pinn_loss

# =============================================================================

import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

from data.dataset_multiplane import MultiplaneScatterDataset, FREQ_LIST, FREQ_MIN, FREQ_MAX
from patch2field.models.losses import field_geom_loss

# 物理常数
MU0 = 4 * math.pi * 1e-7   # H/m
DZ  = 1e-3                  # 1mm = 1e-3 m（z=3mm 到 z=4mm 的间距）


def build_split(dataset):
    """
    按形状（patch 文件路径）分组，同一形状的所有 (频率, z层) 记录
    保持在同一集合，避免数据泄露。
    """
    # 按 patch 路径分组
    shape_groups = defaultdict(list)
    for i, rec in enumerate(dataset.records):
        shape_groups[rec["patch"]].append(i)

    shapes = sorted(shape_groups.keys())
    n_shapes = len(shapes)
    n_val = max(1, int(n_shapes * VAL_RATIO))

    rng = torch.Generator().manual_seed(SHUFFLE_SEED)
    perm = torch.randperm(n_shapes, generator=rng).tolist()
    val_shapes = set(shapes[i] for i in perm[:n_val])

    train_idx, val_idx = [], []
    for shape_path, idxs in shape_groups.items():
        if shape_path in val_shapes:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    print(f"[Split] 形状总数: {n_shapes}, 训练: {n_shapes - n_val}, 验证: {n_val}")
    print(f"[Split] 记录数 — 训练: {len(train_idx)}, 验证: {len(val_idx)}")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def pinn_loss_fn(pred_z3, tgt_z3, tgt_z4, freq, device):
    """
    PINN 约束：麦克斯韦方程 ∂Ey/∂z = -jωμ₀Hx

    用真实数据的差分计算 ∂Ey/∂z（不需要额外前向传播）：
      ∂Ey/∂z ≈ (Ey_true_z4 - Ey_true_z3) / dz

    约束预测的 Hx（z=3mm）满足麦克斯韦方程。

    Args:
        pred_z3: (B, 12, H, W) 模型预测的 z=3mm 场
        tgt_z3:  (B, 12, H, W) 真实 z=3mm 场
        tgt_z4:  (B, 12, H, W) 真实 z=4mm 场
        freq:    (B, 1) 归一化频率
    """
    # 用真实数据差分计算 ∂Ey/∂z
    Ey_z3_true = torch.complex(tgt_z3[:, 2], tgt_z3[:, 3])  # (B, H, W)
    Ey_z4_true = torch.complex(tgt_z4[:, 2], tgt_z4[:, 3])
    dEy_dz = (Ey_z4_true - Ey_z3_true) / DZ  # 真实差分，不依赖模型

    # 预测的 Hx（z=3mm）
    Hx_pred = torch.complex(pred_z3[:, 6], pred_z3[:, 7])  # (B, H, W)

    # 频率
    freq_ghz = freq[:, 0] * (FREQ_MAX - FREQ_MIN) + FREQ_MIN
    omega = 2 * math.pi * freq_ghz * 1e9
    omega_mu0 = (omega * MU0).to(device)

    # rhs = -jωμ₀Hx_pred
    neg_j_Hx = torch.complex(Hx_pred.imag, -Hx_pred.real)
    rhs = omega_mu0[:, None, None] * neg_j_Hx

    # PINN loss = |∂Ey/∂z - rhs|² / |∂Ey/∂z|²
    residual = dEy_dz - rhs
    loss_num = residual.abs().pow(2).mean()
    loss_den = dEy_dz.abs().pow(2).mean().clamp(min=1e-12)
    return loss_num / loss_den


def geom_loss_fn(pred, true):
    """电场 + 磁场几何损失（各取前6/后6通道）。"""
    v_e, _ = field_geom_loss(pred[:, :6], true[:, :6])
    v_h, _ = field_geom_loss(pred[:, 6:12], true[:, 6:12])
    return (v_e + v_h) * 0.5


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] 设备: {self.device}")

        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"[Trainer] 保存路径: {SAVE_DIR}")

        # 数据集
        full_dataset = MultiplaneScatterDataset(
            data_dir=DATA_DIR,
            bg_path=BG_PATH,
        )

        # 按类别过滤
        if TRAIN_CATEGORIES is not None:
            full_dataset.records = [
                r for r in full_dataset.records
                if any(
                    os.path.basename(os.path.dirname(os.path.dirname(r["patch"]))) == cat
                    or os.path.basename(os.path.dirname(os.path.dirname(r["patch"]))).startswith(cat + "/")
                    for cat in TRAIN_CATEGORIES
                )
            ]
            print(f"[数据] 过滤类别 {TRAIN_CATEGORIES}，剩余 {len(full_dataset.records)} 条")
        train_ds, val_ds = build_split(full_dataset)

        self.train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )

        # 模型：只用频率条件，不需要 z 输入
        from patch2field.models.fno2d_film import FNO2dFiLM
        self.model = FNO2dFiLM(
            modes1=MODES1, modes2=MODES2, width=WIDTH, n_layers=N_LAYERS,
            in_channels=2, out_channels=12,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[模型] 参数量: {n_params:,}")

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        # Warmup + Cosine annealing
        def lr_lambda(epoch):
            if epoch < LR_WARMUP_EPOCHS:
                return (epoch + 1) / LR_WARMUP_EPOCHS
            progress = (epoch - LR_WARMUP_EPOCHS) / max(1, N_EPOCHS - LR_WARMUP_EPOCHS)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.history = {"train_total": [], "train_geom": [], "train_pinn": [],
                        "val_total": [], "val_geom": []}
        self.best_val = float("inf")
        self.patience = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = total_geom = total_pinn = 0.0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False):
            inp      = batch["input"].to(self.device)        # (B, 2, H, W)
            tgt      = batch["output"].to(self.device)       # (B, 12, H, W) z=3mm
            tgt_pinn = batch["output_pinn"].to(self.device)  # (B, 12, H, W) z=4mm
            freq     = batch["freq"].to(self.device)         # (B, 1)

            # 主预测（只预测 z=3mm，不需要 z 输入）
            pred = self.model(inp, freq)

            # Geom loss（电场 + 磁场）
            g_loss = geom_loss_fn(pred, tgt)

            # PINN loss（差分约束，需要额外两次前向）
            p_loss = pinn_loss_fn(pred, tgt, tgt_pinn, freq, self.device)

            loss = g_loss + PINN_WEIGHT * p_loss

            if not torch.isfinite(loss):
                print(f"\n[警告] Epoch {epoch}: loss={loss.item():.4f}，跳过此 batch")
                continue

            self.optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP_MAX_NORM > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_MAX_NORM)
            self.optimizer.step()

            total_loss += loss.item()
            total_geom += g_loss.item()
            total_pinn += p_loss.item()

        n = max(1, len(self.train_loader))
        return total_loss / n, total_geom / n, total_pinn / n

    def validate(self):
        self.model.eval()
        total_loss = total_geom = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                inp  = batch["input"].to(self.device)
                tgt  = batch["output"].to(self.device)
                freq = batch["freq"].to(self.device)

                pred = self.model(inp, freq)
                g_loss = geom_loss_fn(pred, tgt)
                total_loss += g_loss.item()
                total_geom += g_loss.item()

        n = max(1, len(self.val_loader))
        return total_loss / n, total_geom / n

    def run(self):
        print(f"\n{'='*50}\n开始训练 {N_EPOCHS} epochs\n{'='*50}\n")

        for epoch in range(1, N_EPOCHS + 1):
            train_total, train_geom, train_pinn = self.train_epoch(epoch)
            self.scheduler.step()

            val_total, val_geom = self.validate()

            self.history["train_total"].append(train_total)
            self.history["train_geom"].append(train_geom)
            self.history["train_pinn"].append(train_pinn)
            self.history["val_total"].append(val_total)
            self.history["val_geom"].append(val_geom)

            is_best = val_geom < self.best_val - EARLY_STOP_MIN_DELTA
            tag = " [BEST]" if is_best else ""
            print(
                f"Epoch {epoch}/{N_EPOCHS}  "
                f"Loss(train/val): {train_total:.4f}/{val_total:.4f}  "
                f"Geom(train/val): {train_geom:.4f}/{val_geom:.4f}  "
                f"PINN: {train_pinn:.4f}{tag}"
            )

            # 保存最新
            ckpt = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_geom": val_geom,
                "use_output_standardization": False,
                "config": {
                    "modes1": MODES1, "modes2": MODES2, "width": WIDTH,
                    "n_layers": N_LAYERS, "freq_emb_dim": FREQ_EMB_DIM,
                    "z_emb_dim": Z_EMB_DIM,
                },
            }
            torch.save(ckpt, os.path.join(SAVE_DIR, "latest.pth"))

            if is_best:
                self.best_val = val_geom
                self.patience = 0
                torch.save(ckpt, os.path.join(SAVE_DIR, "best.pth"))
            else:
                self.patience += 1
                if self.patience >= EARLY_STOP_PATIENCE:
                    print(f"\n早停 (patience={EARLY_STOP_PATIENCE})")
                    break

        self._save_curve()
        print(f"\n训练完成！最佳验证 Geom: {self.best_val:.6f}")
        print(f"模型保存在: {SAVE_DIR}")

    def _save_curve(self):
        if not HAS_MATPLOTLIB:
            return
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.history["train_total"], label="Train Total")
        axes[0].plot(self.history["val_total"], label="Val Total")
        axes[0].set_title("Total Loss"); axes[0].grid(True); axes[0].legend()

        axes[1].plot(self.history["train_geom"], label="Train Geom")
        axes[1].plot(self.history["val_geom"], label="Val Geom")
        axes[1].set_title("Geom Loss"); axes[1].grid(True); axes[1].legend()

        axes[2].plot(self.history["train_pinn"], label="Train PINN")
        axes[2].set_title("PINN Loss"); axes[2].grid(True); axes[2].legend()

        plt.tight_layout()
        curve_path = os.path.join(SAVE_DIR, "training_curve.png")
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"[保存] {curve_path}")


if __name__ == "__main__":
    Trainer().run()

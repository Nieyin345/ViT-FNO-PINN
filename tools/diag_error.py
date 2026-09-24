# -*- coding: utf-8 -*-
"""误差分解诊断：定位第一阶段预测的真实瓶颈。

回答三个问题（全部用实测数字，不靠设计意图推断）：
  Q1  val_geom 那 0.22 是**方向**错、**相位**错、还是**幅值**错？
  Q2  误差集中在哪些通道/频点？大分量是否淹没了小分量？
  Q3  模型的瓶颈是"不知道对多大幅值"还是"不知道相位"？——
      用两个神谕上限对照：
        oracle_mag  : |pred| 换成 |true|，相位保留预测的
        oracle_phase: 相位换成 true 的，幅值保留预测的
      哪个一换误差就塌到近零，哪一半就是模型已经学好的；反之就是瓶颈。

另含两个参照系（确认模型确实强于平凡解）：
        zero    : 预测全零
        mean    : 预测整个留出集/已见集的平均场
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

import models  # noqa: F401
from core.checkpoint import load as load_checkpoint
from configs import base as C
from data.dataset import EMMultiFreqDataset
from models.losses import field_geom_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FREQS_GHZ = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
CH_NAMES = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]


def hr(t=""):
    print("-" * 78)
    if t:
        print(t)


def decompose(pred, true):
    """把 geom 损失拆成 方向/相位/幅值 三项（各自用默认权重 1）。"""
    total, parts = field_geom_loss(pred, true, w_dir=1.0, w_phase=1.0, w_mag=1.0)
    return float(total), {k: float(v) for k, v in parts.items()}


def component_stats(pred, true):
    """逐通道（Ex/Ey/Ez/Hx/Hy/Hz）统计：幅值比、相对L2、径向(幅值)误差、相位角。"""
    rows = []
    for g, off in (("E", 0), ("H", 6)):
        for ci in range(3):
            pr = torch.complex(pred[:, off + 2 * ci], pred[:, off + 2 * ci + 1])
            tr = torch.complex(true[:, off + 2 * ci], true[:, off + 2 * ci + 1])
            # 只在真值有意义的点上统计（避免近零处相位无定义）
            mask = tr.abs() > 0.05 * tr.abs().mean()
            if mask.sum() < 10:
                continue
            pm, tm = pr[mask], tr[mask]
            mag_ratio = (pm.abs().mean() / tm.abs().mean()).item()
            rel_l2 = ((pm - tm).norm() / tm.norm().clamp_min(1e-12)).item()
            # 径向误差（幅值）：|pred|-|true| 相对 |true|
            radial = ((pm.abs() - tm.abs()).abs() / tm.abs().clamp_min(1e-12)).mean().item()
            # 相位误差（角度）
            cosphi = ((pm * tm.conj()).real /
                      (pm.abs() * tm.abs()).clamp_min(1e-12)).clamp(-1, 1)
            ang = torch.acos(cosphi) * 180.0 / np.pi
            rows.append(dict(name=CH_NAMES[ci + (3 if g == "H" else 0)],
                             group=g, mag_ratio=mag_ratio, rel_l2=rel_l2,
                             radial=radial, ang_deg=ang.mean().item(),
                             t_norm=tm.abs().mean().item()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", "-c", required=True)
    ap.add_argument("--category", default="rectangle")
    ap.add_argument("--holdout", default="10.0,11.0",
                    help="留出频点（逗号分隔 GHz）；传 none 表示无留出（全频点训练模型）")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    holdout = () if args.holdout.strip().lower() == "none" else \
        tuple(float(x) for x in args.holdout.split(",") if x.strip())

    lc = load_checkpoint(args.checkpoint)
    model = lc.model.to(DEVICE).eval()
    print("=" * 78)
    print(f"误差分解诊断: {args.checkpoint}")
    print(f"  模型 {lc.model_name}  参数 {sum(p.numel() for p in model.parameters()):,}")
    print(f"  ckpt 记录 val_geom={lc.meta.get('val_geom')} val_rel={lc.meta.get('val_rel')}")
    print("=" * 78)

    ds = EMMultiFreqDataset(
        data_dir=C.DATA_DIR, target_d_over_lambda=C.TARGET_D_OVER_LAMBDA,
        use_scatter=C.USE_SCATTER_FIELD, bg_params_path=C.BACKGROUND_PARAMS_PATH,
        polarizations=C.DATASET_POLARIZATIONS,
    )
    if lc.normalizer is not None:
        ds.set_output_normalization(lc.normalizer)

    idx = [i for i, s in enumerate(ds.samples)
           if s["category"].split("/")[0] == args.category]
    print(f"\n过滤到 category={args.category}: {len(idx)} 条")
    all_pred, all_true, all_freq, all_pol = [], [], [], []
    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    with torch.no_grad():
        for b in loader:
            inp = b["input"].to(DEVICE)
            freq = b["freq"].to(DEVICE)
            pred = model(inp, freq)
            if lc.normalizer is not None:
                pred = lc.normalizer.denormalize(pred)
            all_pred.append(pred.cpu())
            all_true.append(b["output"].cpu())
            all_freq.append(b["freq"][:, 0].cpu())
            all_pol.append(b["freq"][:, 1].cpu())
    P = torch.cat(all_pred); T = torch.cat(all_true)
    F = torch.cat(all_freq); PL = torch.cat(all_pol)
    fGHz = 8.0 + 4.0 * F

    seen = ~torch.tensor([any(abs(f.item() - h) < 1e-6 for h in holdout) for f in fGHz])
    print(f"  已见 {int(seen.sum())} 条 / 留出 {int((~seen).sum())} 条")

    # ---------- Q1: val_geom 三项分解 ----------
    hr("\n[Q1] val_geom 的方向/相位/幅值 三项分解（E 组，默认权重，低幅值点降权）")
    print(f"{'子集':<12}{'总数':>8}{'dir':>9}{'phase':>9}{'mag':>9}   {'占最大项':>10}")
    for lab, sel in (("已见频点", seen), ("留出频点", ~seen)):
        if sel.sum() == 0:
            continue
        tot, parts = decompose(P[sel][:, :6], T[sel][:, :6])
        mx = max(parts.values())
        who = max(parts, key=parts.get)
        print(f"{lab:<12}{tot:>8.4f}{parts['dir_loss']:>9.4f}"
              f"{parts['phase_loss']:>9.4f}{parts['mag_loss']:>9.4f}   "
              f"{who} {mx / max(tot, 1e-9):.0%}")

    hr("\n[Q1b] 同口径 H 组")
    for lab, sel in (("已见频点", seen), ("留出频点", ~seen)):
        if sel.sum() == 0:
            continue
        tot, parts = decompose(P[sel][:, 6:12], T[sel][:, 6:12])
        mx = max(parts.values()); who = max(parts, key=parts.get)
        print(f"{lab:<12}{tot:>8.4f}{parts['dir_loss']:>9.4f}"
              f"{parts['phase_loss']:>9.4f}{parts['mag_loss']:>9.4f}   "
              f"{who} {mx / max(tot, 1e-9):.0%}")

    # ---------- Q2: 逐通道 ----------
    hr("\n[Q2] 逐通道误差（幅值比 |pred|/|tgt| 应≈1；径向=幅值错；角度=相位错）")
    for lab, sel in (("已见频点", seen), ("留出频点", ~seen)):
        if sel.sum() == 0:
            continue
        print(f"  -- {lab} --")
        print(f"    {'通道':<6}{'真值幅值':>11}{'幅值比':>9}{'相对L2':>9}{'径向(幅值)':>11}{'相位角°':>9}")
        rows = component_stats(P[sel], T[sel])
        for r in rows:
            print(f"    {r['name']:<6}{r['t_norm']:>11.3g}{r['mag_ratio']:>9.3f}"
                  f"{r['rel_l2']:>9.3f}{r['radial']:>11.3f}{r['ang_deg']:>9.2f}")

    # ---------- Q2c: 逐频点 ----------
    hr("\n[Q2b] 逐频点 geom（E 组）")
    for fg in FREQS_GHZ:
        sel = (fGHz - fg).abs() < 1e-6
        if sel.sum() == 0:
            continue
        tot, parts = decompose(P[sel][:, :6], T[sel][:, :6])
        tag = "  <-- 留出" if any(abs(fg - h) < 1e-6 for h in holdout) else ""
        print(f"    {fg:5.1f} GHz  geom {tot:.4f}  (dir {parts['dir_loss']:.4f} "
              f"phase {parts['phase_loss']:.4f} mag {parts['mag_loss']:.4f}){tag}")

    # ---------- Q3: 神谕上限 ----------
    hr("\n[Q3] 神谕上限：单独替换幅值 / 单独替换相位（E 组）")
    for lab, sel in (("已见频点", seen), ("留出频点", ~seen)):
        if sel.sum() == 0:
            continue
        p6, t6 = P[sel][:, :6], T[sel][:, :6]
        tot0, base = decompose(p6, t6)
        res = {}
        for mode in ("mag", "phase"):
            outs = []
            for ci in range(3):
                pr = torch.complex(p6[:, 2 * ci], p6[:, 2 * ci + 1])
                tr = torch.complex(t6[:, 2 * ci], t6[:, 2 * ci + 1])
                if mode == "mag":
                    c = pr * (tr.abs() / pr.abs().clamp_min(1e-12))
                else:
                    c = pr.abs() * (tr / tr.abs().clamp_min(1e-12))
                outs += [c.real, c.imag]
            _, pp = decompose(torch.stack(outs, dim=1), t6)
            res[mode] = pp
        print(f"  -- {lab} --  原始 geom {tot0:.4f}")
        print(f"    只换幅值(oracle_mag)  : dir {res['mag']['dir_loss']:.4f} "
              f"phase {res['mag']['phase_loss']:.4f} mag {res['mag']['mag_loss']:.4f}")
        print(f"    只换相位(oracle_phase): dir {res['phase']['dir_loss']:.4f} "
              f"phase {res['phase']['phase_loss']:.4f} mag {res['phase']['mag_loss']:.4f}")
        print(f"    → 幅值已学好? {res['mag']['mag_loss'] < 0.5 * base['mag_loss']}   "
              f"相位已学好? {res['phase']['phase_loss'] < 0.5 * base['phase_loss']}")

    # ---------- 参照系 ----------
    hr("\n[Q4] 平凡参照系（E 组 geom）")
    for lab, sel in (("已见频点", seen), ("留出频点", ~seen)):
        if sel.sum() == 0:
            continue
        t6 = T[sel][:, :6]
        z = torch.zeros_like(t6)
        m = t6.mean(dim=0, keepdim=True).expand_as(t6)
        tz, _ = decompose(z, t6)
        tm, _ = decompose(m, t6)
        tp, _ = decompose(P[sel][:, :6], t6)
        print(f"  {lab}: 模型 {tp:.4f} | 全零 {tz:.4f} | 均值场 {tm:.4f} "
              f"| 模型优于均值 {((tm - tp) / tm):.1%}")
    hr()
    print("诊断完成。")


if __name__ == "__main__":
    main()

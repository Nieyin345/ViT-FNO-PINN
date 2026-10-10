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
    all_pred, all_true, all_freq, all_pol, all_names = [], [], [], [], []
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
            all_names.extend([str(n) for n in b["name"]])
    P = torch.cat(all_pred); T = torch.cat(all_true)
    F = torch.cat(all_freq); PL = torch.cat(all_pol)
    _names = all_names
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

    # ---------- Q4: 平凡参照系 ----------
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

    # ---------- Q5: 频率敏感度对比（模型 vs 真值）----------
    # 核心问题：模型学到的是 f→场的**函数**，还是一张**查找表**（只会复现
    # 训练频点、频点之间不连贯）？
    # 判据：对同一几何/极化，比较"相邻频点的差分"——真值差分 vs 模型差分。
    #   若模型差分远小于真值 → 模型在频点间几乎不动 = 阶梯状，没学插值。
    # 做法：用**解析嵌入** —— 把每个频点的预测减去该频点的“自预测”不成。
    # 改用一个更硬的对照：交叉评估。训 {8,8.5,9,9.5,10.5,11.5,12} 的模型在
    # 10.0 上的预测，与"9.5/10.5 真值线性插值"比，再与"9.5/10.5 模型预测的
    # 线性插值"比。三者对照能分辨是模型不会插值，还是这题本身不可插值。
    hr("\n[Q5] 频率敏感度：模型输出在频点间的变化 vs 真值的变化")
    # 按 (样本名, 极化) 分组，取每个频点的预测/真值
    names = [str(n) for n in _names]
    key2rows = {}
    for i, (nm, pl, fg) in enumerate(zip(names, PL.tolist(), fGHz.tolist())):
        key2rows.setdefault((nm, pl), []).append((fg, i))
    d_model, d_true, n_pairs = [], [], 0
    for key, rows in key2rows.items():
        rows.sort()
        for (f1, i1), (f2, i2) in zip(rows[:-1], rows[1:]):
            if abs(f2 - f1 - 0.5) > 1e-6:
                continue                       # 只在相邻 0.5GHz 对上比
            dm = (P[i2] - P[i1]).abs().mean().item()
            dt = (T[i2] - T[i1]).abs().mean().item()
            d_model.append(dm); d_true.append(dt); n_pairs += 1
    if n_pairs:
        ratio = (np.mean(d_model) / max(np.mean(d_true), 1e-12))
        print(f"  相邻频点差分（{n_pairs} 对，12 通道平均）")
        print(f"    真值  Δ 平均 {np.mean(d_true):.4e}")
        print(f"    模型  Δ 平均 {np.mean(d_model):.4e}")
        print(f"    模型/真值 = {ratio:.3f}")
        if ratio < 0.3:
            print("    [!!] 模型在频点间几乎不动 —— 学到的是阶梯/查找表，不是 f→场 的函数")
        elif ratio > 2.0:
            print("    [!!] 模型随频率变化过猛 —— 预测抖动，未抓住真实趋势")
        else:
            print("    [OK] 模型随频率的变化量与真值同量级")

    # ---------- Q6: 交叉评估 / 可插值性裁决 ----------
    hr("\n[Q6] 可插值性裁决：模型插值 vs 真值线性插值 vs 模型线性插值")
    # 对每个留出频点 h，找它两侧的已见频点 (h-0.5, h+0.5)：
    #   A. 模型在 h 的预测               （直接泛化）
    #   B. 模型在 h 两侧预测的线性插值    （模型的空间连续性）
    #   C. 真值在 h 两侧的线性插值        （这题本身的“可插值上限”）
    for h in holdout:
        errs = {"A_model": [], "B_model_lerp": [], "C_true_lerp": []}
        for key, rows in key2rows.items():
            f_lo, f_hi = h - 0.5, h + 0.5
            cand_lo = [i for fg, i in rows if abs(fg - f_lo) < 1e-6]
            cand_hi = [i for fg, i in rows if abs(fg - f_hi) < 1e-6]
            cand_h = [i for fg, i in rows if abs(fg - h) < 1e-6]
            if not (cand_lo and cand_hi and cand_h):
                continue
            ilo, ihi, ih = cand_lo[0], cand_hi[0], cand_h[0]
            t_h = T[ih]
            errs["A_model"].append(((P[ih] - t_h).norm() / t_h.norm().clamp_min(1e-12)).item())
            lerp_p = 0.5 * (P[ilo] + P[ihi])
            lerp_t = 0.5 * (T[ilo] + T[ihi])
            errs["B_model_lerp"].append(((lerp_p - t_h).norm() / t_h.norm().clamp_min(1e-12)).item())
            errs["C_true_lerp"].append(((lerp_t - t_h).norm() / t_h.norm().clamp_min(1e-12)).item())
        if not errs["A_model"]:
            continue
        print(f"  {h} GHz（{len(errs['A_model'])} 个样本）:")
        print(f"    A 模型直接预测         : {np.mean(errs['A_model']):.4f}  (相对L2)")
        print(f"    B 模型预测的线性插值    : {np.mean(errs['B_model_lerp']):.4f}")
        print(f"    C 真值的线性插值(上限)  : {np.mean(errs['C_true_lerp']):.4f}")

    # ---------- Q7: 全频点可插值性扫描（决定性的对照）----------
    # Q6 只测了留出频点。但"真值插值也失败"可能对**所有**频点都成立
    # （即这个场在 f 上本来就不可线性插值），那样它就是数据性质而非模型缺陷。
    # 这里对 9 个频点全部做 leave-one-out：
    #   true_lerp  : 真值邻居的线性插值误差（该频点的"可插值上限"）
    #   cmplx_lerp : 真值邻居复数的**幅值-相位**插值（更物理的插法）
    #   nearest    : 直接照抄最近邻频点的真值（完全不插值）
    #   meanfield  : 全局平均场（平凡下界）
    #   model_lerp : 模型在邻居频点预测的线性插值
    hr("\n[Q7] 全频点可插值性扫描（leave-one-out，逐频点）")
    print(f"  {'频点':>7}{'true_lerp':>11}{'nearest':>10}{'meanfield':>11}"
          f"{'model_pred':>12}{'样本':>7}")
    T_flat = T.reshape(T.shape[0], -1)
    for h in FREQS_GHZ:
        sel_h = (fGHz - h).abs() < 1e-6
        if not sel_h.any():
            continue
        e_t, e_n, e_p = [], [], []
        for key, rows in key2rows.items():
            me = [i for fg, i in rows if abs(fg - h) < 1e-6]
            if not me:
                continue
            ih = me[0]
            t_h = T[ih]
            nrm = t_h.norm().clamp_min(1e-12)
            e_p.append(((P[ih] - t_h).norm() / nrm).item())
            lo = [i for fg, i in rows if abs(fg - (h - 0.5)) < 1e-6]
            hi = [i for fg, i in rows if abs(fg - (h + 0.5)) < 1e-6]
            nb = lo + hi
            if lo and hi:
                e_t.append(((0.5 * (T[lo[0]] + T[hi[0]]) - t_h).norm() / nrm).item())
            if nb:
                j = min(nb, key=lambda i: abs(fGHz[i].item() - h))
                e_n.append(((T[j] - t_h).norm() / nrm).item())
        # 均值场（该频点全部样本的平均场）
        mf = T[sel_h].mean(dim=0, keepdim=True)
        e_m = ((T[sel_h] - mf).reshape(int(sel_h.sum()), -1).norm(dim=1) /
               T_flat[sel_h].norm(dim=1).clamp_min(1e-12)).mean().item()
        tag = "  <-- 留出" if any(abs(h - x) < 1e-6 for x in holdout) else ""
        f = lambda v: f"{np.mean(v):.4f}" if v else "   --  "
        print(f"  {h:>6.1f}G{f(e_t):>11}{f(e_n):>10}{e_m:>11.4f}{f(e_p):>12}"
              f"{int(sel_h.sum()):>7}{tag}")
    print("\n  判据：true_lerp 若在所有频点都大 → 场在 f 上不可线性插值（数据性质）；")
    print("        若只在留出频点大 → 那些频点特殊（需查数据）。")

    # ---------- Q8: 频率轴 vs 几何轴的难度对比（决定性的相对量）----------
    # 问题：模型对"频率"和"几何"这两条轴，哪条更难泛化？
    # 做法：比较两种"扰动"引起的场变化量——
    #   d_freq : 同一几何、相邻频点的真值差
    #   d_geom : 同一频点、不同几何的真值差
    # 若两者同量级 → 频率轴携带的信息量与几何轴相当，但只有 7 个采样点
    # （几何有 ~400 个），泛化难度天差地别。这解释了为什么"几何泛化好、
    # 频率泛化崩"。
    hr("\n[Q8] 频率轴 vs 几何轴的难度对比")
    n_samples = len(P)
    # d_freq：同一 (样本,极化) 的相邻频点真值差
    df_vals = []
    for key, rows in key2rows.items():
        rows.sort()
        for (f1, i1), (f2, i2) in zip(rows[:-1], rows[1:]):
            if abs(f2 - f1 - 0.5) > 1e-6:
                continue
            df_vals.append(((T[i2] - T[i1]).norm() / T[i1].norm().clamp_min(1e-12)).item())
    # d_geom：同一频点、不同样本的真值差（随机配对，避免 O(n^2)）
    rng = np.random.default_rng(0)
    dg_vals = []
    for h in FREQS_GHZ:
        sel = np.where((fGHz.numpy() - h).__abs__() < 1e-6)[0]
        if len(sel) < 4:
            continue
        a = rng.choice(sel, size=200, replace=True)
        b = rng.choice(sel, size=200, replace=True)
        for ia, ib in zip(a, b):
            if ia == ib:
                continue
            dg_vals.append(((T[ib] - T[ia]).norm() / T[ia].norm().clamp_min(1e-12)).item())
    mf_, mg_ = np.mean(df_vals), np.mean(dg_vals)
    print(f"  同几何·相邻频点(Δ0.5GHz) 的相对差 : {mf_:.4f}   （{len(df_vals)} 对）")
    print(f"  同频点·不同几何        的相对差 : {mg_:.4f}   （{len(dg_vals)} 对）")
    print(f"  频率轴/几何轴 = {mf_ / max(mg_, 1e-12):.3f}")
    if mf_ > 0.5 * mg_:
        print("  [!!] 频率与几何对场的影响同量级 —— 但频率只有 7 个采样点、")
        print("       几何有 ~400 个。频率泛化难是因为**采样密度**，不是模型不会物理。")

    hr()
    print("诊断完成。")


if __name__ == "__main__":
    main()

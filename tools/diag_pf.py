# -*- coding: utf-8 -*-
"""Per-frequency geom with and without PhaseFrame, to prove the eval-path bug.

train.py applies PhaseFrame inside Trainer._fwd. Every other consumer
(diag_error / eval / probe / visualize / propagate) calls model(...) directly,
so for PHASE_FRAME models they compare a de-rotated prediction against rotated
truth. At 10 GHz the rotation is identity (table is anchored there), which is
exactly why the hold-out number looked fine while seen frequencies looked awful.
"""
import os, sys, argparse
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader, Subset
import models  # noqa
from core.checkpoint import load as load_checkpoint
from core.phase_frame import PhaseFrame
from configs import base as C
from data.dataset import EMMultiFreqDataset, FREQ_MIN, FREQ_MAX
from models.losses import field_geom_loss_eh, field_geom_loss, channel_group_balanced_l2

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", "-c", required=True)
ap.add_argument("--category", default="rectangle")
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--max-per-freq", type=int, default=400)
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lc = load_checkpoint(a.checkpoint)
model = lc.model.to(dev).eval()
nm = lc.normalizer
pf = PhaseFrame(nm, FREQ_MIN, FREQ_MAX, dev, scale=1.0)

ds = EMMultiFreqDataset(data_dir=C.DATA_DIR, target_d_over_lambda=C.TARGET_D_OVER_LAMBDA,
                        use_scatter=C.USE_SCATTER_FIELD, bg_params_path=C.BACKGROUND_PARAMS_PATH,
                        polarizations=C.DATASET_POLARIZATIONS)
ds.set_output_normalization(nm)
idx = [i for i, s in enumerate(ds.samples) if s["category"].split("/")[0] == a.category]
loader = DataLoader(Subset(ds, idx), batch_size=a.batch_size, shuffle=False, num_workers=2)

acc = {}  # fGHz -> lists
with torch.no_grad():
    for b in loader:
        inp, freq = b["input"].to(dev), b["freq"].to(dev)
        tgt_n = b["output_norm"].to(dev) if "output_norm" in b else None
        raw = model(inp, freq)
        with_pf = pf(raw, freq)
        p_raw = nm.denormalize(raw)
        p_pf = nm.denormalize(with_pf)
        true = b["output"].to(dev)
        f = (8.0 + 4.0 * freq[:, 0]).round(decimals=2)
        for fv in f.unique():
            m = f == fv
            k = float(fv)
            d = acc.setdefault(k, {"n": 0, "g_pf": 0.0, "g_raw": 0.0, "rel_pf": 0.0, "rel_raw": 0.0})
            n = int(m.sum())
            d["n"] += n
            d["g_pf"] += float(field_geom_loss_eh(p_pf[m], true[m])) * n
            d["g_raw"] += float(field_geom_loss_eh(p_raw[m], true[m])) * n
            d["rel_pf"] += float(channel_group_balanced_l2(p_pf[m], true[m])) * n
            d["rel_raw"] += float(channel_group_balanced_l2(p_raw[m], true[m])) * n
            # 分解：方向 / 相位 / 幅值（E 组、正确口径）
            _, parts = field_geom_loss(p_pf[m][:, :6], true[m][:, :6])
            for key in ("dir_loss", "phase_loss", "mag_loss"):
                d[key] = d.get(key, 0.0) + float(parts[key]) * n
            # 幅值比 |pred|/|true|（逐通道复幅值的均值比），以及"整体缩放校正后"的 geom
            pa = (p_pf[m][:, 0::2] ** 2 + p_pf[m][:, 1::2] ** 2).sqrt()
            ta = (true[m][:, 0::2] ** 2 + true[m][:, 1::2] ** 2).sqrt()
            d["amp_ratio"] = d.get("amp_ratio", 0.0) + float((pa.mean() / ta.mean().clamp(min=1e-9))) * n
            # 每样本一个最优全局缩放 s = <p,t>/<p,p>，看"只是整体缩小了"能解释多少
            pm, tm = p_pf[m], true[m]
            num = (pm * tm).sum(dim=(1, 2, 3))
            den = (pm * pm).sum(dim=(1, 2, 3)).clamp(min=1e-12)
            s_opt = (num / den)[:, None, None, None]
            d["s_opt"] = d.get("s_opt", 0.0) + float(s_opt.mean()) * n
            d["g_scaled"] = d.get("g_scaled", 0.0) + float(field_geom_loss_eh(pm * s_opt, tm)) * n

print(f"\n{'freq':>6} {'n':>5} {'geom':>8} {'dir':>8} {'phase':>8} {'mag':>8} {'ampR':>7} {'s_opt':>7} {'geom@s':>8} {'relL2':>8} {'geom_noPF':>10}")
tot = {"n": 0, "g_pf": 0.0, "g_raw": 0.0, "rel_pf": 0.0, "rel_raw": 0.0}
for k in sorted(acc):
    d = acc[k]
    for key in tot:
        tot[key] += d[key]
    nn_ = d['n']
    print(f"{k:6.1f} {nn_:5d} {d['g_pf']/nn_:8.4f} {d['dir_loss']/nn_:8.4f} {d['phase_loss']/nn_:8.4f} "
          f"{d['mag_loss']/nn_:8.4f} {d['amp_ratio']/nn_:7.3f} {d['s_opt']/nn_:7.3f} "
          f"{d['g_scaled']/nn_:8.4f} {d['rel_pf']/nn_:8.4f} {d['g_raw']/nn_:10.4f}")
n = tot["n"]
print(f"{'ALL':>6} {n:5d} {tot['g_pf']/n:8.4f} {'':8} {'':8} {'':8} {'':7} {'':7} {'':8} {tot['rel_pf']/n:8.4f} {tot['g_raw']/n:10.4f}")
print("\nampR=|pred|/|true| 的均值比；s_opt=每样本最优全局缩放；geom@s=乘上该缩放后的 geom")


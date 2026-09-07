#!/usr/bin/env python3
"""
散射场 PWS 传播：总场 - base场 = 贴片散射场，双向传播后加回 base场。

用法：
  python propagate_scatter.py \
    --source-z-mm 3 \
    --targets-mm 5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43 \
    --freq-ghz 12 \
    --out-dir out_result
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from constants import C0
from field_io import (
    ensure_same_grid,
    load_fld,
    nearest_index,
    parse_targets_mm,
    write_fld_plane,
)
from metrics_utils import collect_metrics
from propagation import angular_spectrum_propagate, propagate_em_twoway_from_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e-total",     default="e_field.fld",  help="总场 E 文件")
    parser.add_argument("--h-total",     default="h_field.fld",  help="总场 H 文件")
    parser.add_argument("--e-base",      default="e_base.fld",   help="base场 E 文件（无贴片）")
    parser.add_argument("--h-base",      default="h_base.fld",   help="base场 H 文件（无贴片）")
    parser.add_argument("--source-z-mm", type=float, default=3.0,   help="源截面 z 坐标 (mm)")
    parser.add_argument("--targets-mm",  type=str,   default="5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43")
    parser.add_argument("--freq-ghz",    type=float, default=12.0,  help="频率 (GHz)")
    parser.add_argument("--n-medium",    type=float, default=1.0,   help="传播介质折射率")
    parser.add_argument("--out-dir",     type=str,   default="out_result")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_mm = parse_targets_mm(args.targets_mm)
    targets_z  = [v * 1e-3 for v in targets_mm]
    source_z   = args.source_z_mm * 1e-3
    freq_hz    = args.freq_ghz * 1e9

    e_tot  = load_fld(Path(args.e_total), "E_total")
    h_tot  = load_fld(Path(args.h_total), "H_total")
    e_base = load_fld(Path(args.e_base),  "E_base")
    h_base = load_fld(Path(args.h_base),  "H_base")
    ensure_same_grid(e_tot, h_tot)
    ensure_same_grid(e_tot, e_base)
    ensure_same_grid(e_tot, h_base)

    dx = float(e_tot.x[1] - e_tot.x[0])
    dy = float(e_tot.y[1] - e_tot.y[0])

    src_idx = nearest_index(e_tot.z, source_z, tol=1e-9)
    z0 = float(e_tot.z[src_idx])

    # 源截面散射场 = 总场 - base场（贴片的二次辐射，含 +z 和 -z 两个方向）
    e_scat_src = e_tot.data[:, :, :, src_idx] - e_base.data[:, :, :, src_idx]
    h_scat_src = h_tot.data[:, :, :, src_idx] - h_base.data[:, :, :, src_idx]

    metrics_rows = []

    for zt_mm, zt in zip(targets_mm, targets_z):
        dz = float(zt - z0)

        # E+H 联合双向 PWS 传播散射场
        e_scat_t, h_scat_t = propagate_em_twoway_from_source(
            e_scat_src, h_scat_src,
            dx=dx, dy=dy, freq_hz=freq_hz, dz=dz, n_medium=args.n_medium,
        )

        # 目标截面 base场（直接从仿真数据取，不需要传播）
        base_idx = nearest_index(e_base.z, zt, tol=1e-9)
        pred_e = e_scat_t + e_base.data[:, :, :, base_idx]
        pred_h = h_scat_t + h_base.data[:, :, :, base_idx]

        write_fld_plane(out_dir / f"pred_e_z{zt_mm:g}mm.fld", "Ex,Ey,Ez", e_tot.x, e_tot.y, zt, pred_e)
        write_fld_plane(out_dir / f"pred_h_z{zt_mm:g}mm.fld", "Hx,Hy,Hz", h_tot.x, h_tot.y, zt, pred_h)

        try:
            ref_e = e_tot.data[:, :, :, nearest_index(e_tot.z, zt, tol=1e-9)]
            ref_h = h_tot.data[:, :, :, nearest_index(h_tot.z, zt, tol=1e-9)]
            metrics_rows.extend(collect_metrics(pred_e, ref_e, "E", zt_mm))
            metrics_rows.extend(collect_metrics(pred_h, ref_h, "H", zt_mm))
        except ValueError:
            pass

    metrics_path = out_dir / "prediction_metrics.csv"
    with metrics_path.open("w", encoding="ascii", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "target_mm", "metric", "value"])
        for row in metrics_rows:
            writer.writerow([row[0], f"{row[1]:g}", row[2], f"{row[3]:.8e}"])

    print(f"Done. Output: {out_dir}")
    for fl in ("E", "H"):
        for zt_mm in targets_mm:
            r = next((r for r in metrics_rows if r[0] == fl and abs(r[1] - zt_mm) < 1e-9 and r[2] == "mag_relL2"), None)
            if r:
                print(f"  {fl} @ {zt_mm:g}mm  mag_relL2 = {r[3]:.4e}")


if __name__ == "__main__":
    main()

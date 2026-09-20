#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FNO 近场预测 → PWS 物理传播 → 全空间场（补齐项目主线）。

主线：patch2field 的 FNO 从贴片形状预测 0.1λ 近场截面的散射场（12 通道 E+H），
再由 field2field 的 PWS 双向传播把该截面传播到任意 z，加回背景场得到目标 z 的
总场 E/H，并可与仿真体数据对比（验证传播精度与整条链路）。

注意：传播只对散射场做功（总场-背景场），传播后再加回背景场，避免驻波被重复传播。

用法：
  1) 用 checkpoint 预测 + 传播（模型类型由 checkpoint 自动判定）：
     python -m field2field.propagate_fno_prediction \
         --sample-dir data/data/rectangle/rectangle_1 --freq-ghz 12 \
         --ckpt checkpoints/fno_attention/best.pth \
         --targets-mm 5,7,9,11,13,15,17,19,21,23 --out-dir out_fno_prop

  2) 跳过模型，直接传播一个已算好的散射场平面 (H,W,12)：
     python -m field2field.propagate_fno_prediction --from-plane plane.npy \
         --freq-ghz 12 --targets-mm 5,7,9 --out-dir out_fno_prop

重构变更：删掉本地 `_build_registry()`（缺 fno_v2，且未知名字会退回）与
`from eval_generalization import load_model`（strict=False + 从权重反推结构），
改为 `core.checkpoint.load`；`use_scatter=True` 的硬编码改为读 config，
与 train.py 对齐。
"""

import argparse
import os
import sys
from pathlib import Path

_F2F = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_F2F, ".."))
for _p in (_F2F, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from scipy.interpolate import interp1d

from propagation import propagate_em_twoway_from_source

# ---------------- 数据/物理参数（与 field2field/visualize.py 保持一致） ----------------
FREQ_LIST = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
CELL_MM = 15.0                       # 单元尺寸 mm
N_GRID = 64                          # 空间格点数
Z_MM_VOL = np.arange(2, 24, 1, dtype=float)        # 体数据 22 层，2~23mm
Z_MM_BLANK = np.arange(2, 23.1, 0.5, dtype=float)  # blank 43 层，步长 0.5mm
C_MM_S = 3e11                        # 光速 mm/s
_BLANK_VOLUME = None

DATA_DIR = os.path.join(_ROOT, "data", "data")
BG_PARAMS_PATH = os.path.join(_ROOT, "data", "blank", "background_params.npz")
BLANK_VOLUME_PATH = os.path.join(_ROOT, "data", "blank", "blank", "field_volume_merged.npy")


def _plane_to_em(plane):
    """(H,W,12) → E/H 复矢量，各 (3,H,W)。"""
    def comp(c):
        return plane[..., c] + 1j * plane[..., c + 1]
    E = np.stack([comp(0), comp(2), comp(4)])   # (3,H,W)
    H = np.stack([comp(6), comp(8), comp(10)])
    return E, H


def _em_to_plane(E, H):
    """E/H 复矢量 → (H,W,12) 实/虚部平面。"""
    def ri(field, i):
        return field[i].real, field[i].imag
    ch = [None] * 12
    ch[0], ch[1] = ri(E, 0)
    ch[2], ch[3] = ri(E, 1)
    ch[4], ch[5] = ri(E, 2)
    ch[6], ch[7] = ri(H, 0)
    ch[8], ch[9] = ri(H, 1)
    ch[10], ch[11] = ri(H, 2)
    return np.stack(ch, axis=-1).astype(np.float32)  # (H,W,12)


def _wavelength_mm(freq_ghz):
    return C_MM_S / (freq_ghz * 1e9)


def _rotate_plane(plane):
    """把单极化背景场旋转为正交极化：Ex/Ey 互换、Hx/Hy 互换。

    **E 组额外取负号、H 组不取**，与 `data/fit_background.py::_orthogonal_polarization_field`
    保持完全一致（两处必须同步，否则训练与传播的极化口径不同）。
    符号来源见该函数 docstring 的实测数据。
    """
    out = plane.copy()
    out[..., [0, 1, 2, 3]] = -plane[..., [2, 3, 0, 1]]
    out[..., [6, 7, 8, 9]] = plane[..., [8, 9, 6, 7]]
    return out

def _get_bg_plane(fi, z_mm, pol=0):
    """从 blank 体数据插值得到目标 z 的背景场平面 (64,64,12)。"""
    global _BLANK_VOLUME
    if _BLANK_VOLUME is None:
        if not os.path.exists(BLANK_VOLUME_PATH):
            raise FileNotFoundError(f"缺少背景体数据: {BLANK_VOLUME_PATH}")
        _BLANK_VOLUME = np.load(BLANK_VOLUME_PATH)  # (9,43,64,64,12)
    f = interp1d(Z_MM_BLANK, _BLANK_VOLUME[fi].reshape(len(Z_MM_BLANK), -1),
                 axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")
    plane = f(z_mm).reshape(64, 64, 12).astype(np.float32)
    return _rotate_plane(plane) if int(pol) else plane


def _load_gt_volume(sample_dir):
    path = os.path.join(sample_dir, "field_volume_merged.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)  # (9,22,64,64,12)


def _gt_plane_at_z(vol, fi, z_mm, pol=0):
    """体数据在目标 z 的线性插值平面 (64,64,12)。"""
    if vol.ndim == 6:
        if int(pol) >= vol.shape[0]:
            raise ValueError(f"体数据只有 {vol.shape[0]} 个极化，请求 {pol}")
        vol = vol[int(pol)]
    n_z = vol.shape[1]
    z_axis = np.arange(2, 2 + n_z, 1, dtype=float)
    f = interp1d(z_axis, vol[fi].reshape(n_z, -1),
                 axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")
    return f(z_mm).reshape(64, 64, 12).astype(np.float32)


def _rel_l2_complex(pred, gt, eps=1e-9):
    return float(np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + eps))


def predict_scatter_plane(ckpt_path, sample_dir, freq_ghz, d_lam, pol=0, device="cpu"):
    """用 checkpoint 预测 0.1λ 截面的散射场，返回物理空间 (H,W,12)。

    模型类型由 checkpoint 自身记录的 `model_name` 决定（不再由命令行传、也不再
    从一份会漏掉新模型的本地注册表里查）。缺 checkpoint 直接抛错——旧代码在这里
    会用随机初始化的模型继续跑完并输出"预测结果"。
    """
    import models  # noqa: F401  触发自注册
    from core.checkpoint import load as load_checkpoint
    from data.dataset import EMMultiFreqDataset

    lc = load_checkpoint(ckpt_path, map_location=device)
    model = lc.model.to(device).eval()

    ds = EMMultiFreqDataset(
        data_dir=DATA_DIR,
        output_field="all",
        normalize=False,
        target_d_over_lambda=d_lam,
        use_scatter=_use_scatter(),
        bg_params_path=BG_PARAMS_PATH,
    )
    norm_sample = os.path.abspath(sample_dir)
    idxs = [i for i, s in enumerate(ds.samples)
            if os.path.abspath(os.path.dirname(s["patch"])) == norm_sample
            and s["freq_ghz"] == freq_ghz
            and s.get("polarization", 0) == int(pol)]
    if not idxs:
        raise FileNotFoundError(
            f"未在 {DATA_DIR} 找到样本 {sample_dir} 的 {freq_ghz}GHz 记录")
    sample = ds[idxs[0]]

    inp = sample["input"].unsqueeze(0).to(device)
    freq = sample["freq"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp, freq)

    if lc.normalizer is not None:
        pred = lc.normalizer.denormalize(pred)
    print(f"[模型] {lc.model_name}  (epoch={lc.epoch}) → 预测截面 {freq_ghz}GHz")
    return pred[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (H,W,12) 散射场


def _use_scatter():
    """是否用散射场。与 train.py 读同一个配置键，避免两处不一致。"""
    try:
        from configs import base as C
        return bool(getattr(C, "USE_SCATTER_FIELD", False))
    except ImportError:
        return True


def parse_targets(targets_str):
    if not targets_str or targets_str.strip().lower() == "all":
        return list(np.arange(2, 24, 2, dtype=float))
    return [float(v) for v in targets_str.split(",") if v.strip()]


def main():
    ap = argparse.ArgumentParser(description="FNO 近场预测 + PWS 传播到全空间")
    ap.add_argument("--sample-dir", default=None, help="样本目录，如 data/data/circle/circle_1")
    ap.add_argument("--freq-ghz", type=float, required=True, help="频率 8~12 GHz")
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, "checkpoints", "fno_attention", "best.pth"),
                    help="FNO checkpoint 路径（模型类型由其中的 model_name 决定）")
    ap.add_argument("--model-name", default=None,
                    help="已废弃：模型类型现由 checkpoint 自身的 model_name 决定。"
                         "若指定且与 checkpoint 不符会报错。")
    ap.add_argument("--device", default=None, help="cpu/cuda，默认自动判断")
    ap.add_argument("--from-plane", default=None,
                    help="直接使用预计算散射场平面 npy (H,W,12)，跳过模型")
    ap.add_argument("--d-lam", type=float, default=0.1, help="源截面归一化距离 d/λ")
    ap.add_argument("--pol", type=int, default=0, choices=[0, 1], help="极化 0/1（默认 0）")
    ap.add_argument("--targets-mm", default=None,
                    help="目标 z 列表（mm，逗号分隔），默认 2~22 每隔 2mm")
    ap.add_argument("--cell-mm", type=float, default=CELL_MM, help="单元尺寸 mm")
    ap.add_argument("--scatter-out", action="store_true",
                    help="输出散射场而不是加回背景的总场")
    ap.add_argument("--out-dir", default="out_fno_prop", help="输出目录")
    args = ap.parse_args()

    if args.from_plane:
        plane_src = np.load(args.from_plane).astype(np.float32)  # (H,W,12) 散射场
    elif args.sample_dir:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if args.model_name:
            import models  # noqa: F401
            from core.checkpoint import read_meta
            recorded = read_meta(args.ckpt)["model_name"]
            if args.model_name != recorded:
                ap.error(f"--model-name {args.model_name!r} 与 checkpoint 记录的不符 "
                         f"（checkpoint 里是 {recorded!r}）。该参数已废弃，请直接省略。")
            print("[提示] --model-name 已废弃：模型类型现由 checkpoint 自动决定。")
        plane_src = predict_scatter_plane(args.ckpt, args.sample_dir, args.freq_ghz,
                                          args.d_lam, args.pol, device)
    else:
        ap.error("--from-plane 与 --sample-dir 至少提供一个")

    if args.freq_ghz not in FREQ_LIST:
        print(f"[注意] 频率 {args.freq_ghz} 不在 {FREQ_LIST} 中，物理参数按给定频率计算")
    fi = int(np.argmin(np.abs(np.array(FREQ_LIST) - args.freq_ghz)))

    H, W = plane_src.shape[:2]
    dx = args.cell_mm / (W - 1) * 1e-3
    E_src, H_src = _plane_to_em(plane_src)
    lam_mm = _wavelength_mm(args.freq_ghz)
    z_src_mm = args.d_lam * lam_mm
    print(f"[源截面] z={z_src_mm:.3f} mm (d/λ={args.d_lam})  |E|max={np.abs(E_src).max():.3f}")

    targets = parse_targets(args.targets_mm)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = None
    if args.sample_dir and os.path.isdir(args.sample_dir):
        gt = _load_gt_volume(args.sample_dir)

    rel_ev, rel_hv = [], []
    print(f"{'z(mm)':>8} {'rel L2 E':>10} {'rel L2 H':>10}  |E|max  |H|max")
    print("-" * 52)
    for zt in targets:
        dz = (zt - z_src_mm) * 1e-3
        E_t, H_t = propagate_em_twoway_from_source(E_src, H_src, dx=dx, dy=dx,
                                                   freq_hz=args.freq_ghz * 1e9, dz=dz)
        if not args.scatter_out:
            bg = _get_bg_plane(fi, zt, args.pol)
            bgE, bgH = _plane_to_em(bg)
            E_t = E_t + bgE
            H_t = H_t + bgH
        plane_tot = _em_to_plane(E_t, H_t)
        np.save(out_dir / f"pred_plane_{zt:05.1f}mm.npy", plane_tot)

        line = f"{zt:8.2f} "
        if gt is not None:
            gt_plane = _gt_plane_at_z(gt, fi, zt, args.pol)
            gtE, gtH = _plane_to_em(gt_plane)
            re = _rel_l2_complex(E_t, gtE)
            rh = _rel_l2_complex(H_t, gtH)
            rel_ev.append(re); rel_hv.append(rh)
            line += f"{re:10.3f} {rh:10.3f} "
        else:
            line += f"{'--':>10} {'--':>10} "
        line += f"{np.abs(E_t).max():8.1f} {np.abs(H_t).max():8.4f}"
        print(line)

    if rel_ev:
        print(f"\n传播平均 rel L2: E={np.mean(rel_ev):.3f}, H={np.mean(rel_hv):.3f} "
              f"(目标 {len(targets)} 个 z 平面)")
    print(f"已保存平面到: {out_dir}")


if __name__ == "__main__":
    main()



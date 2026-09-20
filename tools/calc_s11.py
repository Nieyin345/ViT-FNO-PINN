"""
从预测的散射场计算 S11，与仿真 S11 对比。

物理模型：
  - 金属背板，反射系数 = -1
  - 背景场（无贴片）= 入射波 + 金属反射波 = 驻波
  - 散射场 = 总场 - 背景场 = 贴片额外引起的散射
  - S11 = 散射场在参考面（z=0）的 0 阶 Floquet 模式 / 入射波幅值

用法（在仓库根执行）：
    python -m tools.calc_s11 --checkpoint checkpoints/fno_lora/best.pth \
        --sample rectangle_1 --out s11_comparison.png

重构变更：原先写死 `FNO2dLoRA(...)` 构造 + `ckpt_path = "../checkpoints/fno_lora/best.pth"`
+ `strict=False` 的兼容形状合并，且 `out_channels` 默认取 6（真实是 12）——
即无论给什么 checkpoint 都会把权重塞进一个 LoRA 模型里，缺的层静默随机初始化。
现在改为 `core.checkpoint.load` 按 checkpoint 自身记录的架构重建，strict=True。
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
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import models  # noqa: F401  触发自注册
from core.checkpoint import load as load_checkpoint
from configs import base as C
from data.dataset import EMMultiFreqDataset
from data.fit_background import get_background_field, FREQS_GHZ

C_MM = 3e11  # 光速 mm/s
TARGET_D_LAM = getattr(C, "TARGET_D_OVER_LAMBDA", 0.1)


def wavelength_mm(freq_ghz):
    return C_MM / (freq_ghz * 1e9)


def calc_s11(scatter_hw12, freq_ghz, z_mm):
    """
    从散射场计算 S11。
    scatter_hw12: (H, W, 12) numpy array，散射场
    freq_ghz: 频率 GHz
    z_mm: 测量面距贴片的距离 mm
    返回：(s11_complex, s11_db)
    """
    lam = wavelength_mm(freq_ghz)
    k = 2 * np.pi / lam

    # 散射场 Ey 空间平均（0 阶 Floquet 模式）
    ey_r = scatter_hw12[:, :, 2].mean()
    ey_i = scatter_hw12[:, :, 3].mean()
    ey_scatter_z = complex(float(ey_r), float(ey_i))

    # 相位补偿：散射场是反射波（向 -z 传播），传播回参考面 z=0
    phase_corr = np.exp(1j * k * z_mm)
    ey_scatter_ref = ey_scatter_z * phase_corr

    # 入射波幅值：背景场 = -2j * A_inc * sin(kz)
    bg = get_background_field(freq_ghz, z_mm)
    ey_bg = complex(float(bg[2]), float(bg[3]))
    sin_kz = np.sin(k * z_mm)
    a_inc = abs(ey_bg) / (2 * abs(sin_kz) + 1e-8)

    s11 = ey_scatter_ref / (a_inc + 1e-8)
    s11_db = 20 * np.log10(abs(s11) + 1e-12)
    return s11, s11_db


def predict_scatter(model, normalizer, sample):
    inp  = sample["input"].unsqueeze(0)
    freq = sample["freq"].unsqueeze(0)
    with torch.no_grad():
        pred = model(inp, freq)
    if normalizer is not None:
        pred = normalizer.denormalize(pred)
    pred_phys = pred[0]  # (12, H, W)
    # 转成 (H, W, 12)
    return pred_phys.permute(1, 2, 0).numpy()


def read_sim_s11(csv_path):
    """读取仿真 S11（dB），返回 {freq_ghz: s11_db}"""
    df = pd.read_csv(csv_path, sep=";")
    result = {}
    for _, row in df.iterrows():
        freq = float(row.iloc[0])
        s11_db = float(row.iloc[1])  # dB(S11) 实部，虚部为0
        result[freq] = s11_db
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="从预测的散射场计算 S11，与仿真 S11 对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", "-c", required=True, help="模型 checkpoint")
    p.add_argument("--sample", default="rectangle_1",
                   help="样本目录名（data/data/<category>/<sample>）")
    p.add_argument("--category", default=None,
                   help="样本所属类别目录；不填则自动在所有类别中查找")
    p.add_argument("--csv", default=None, help="仿真 S 参数 CSV；默认取样本目录同名文件")
    p.add_argument("--out", default="s11_comparison.png", help="输出图片")
    p.add_argument("--d-lam", type=float, default=TARGET_D_LAM,
                   help="散射场截面的归一化距离 d/λ（与训练一致）")
    p.add_argument("--pol", type=int, default=0, choices=[0, 1], help="极化")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"加载模型: {args.checkpoint}")
    lc = load_checkpoint(args.checkpoint, map_location=str(device))
    model = lc.model.to(device)

    print("加载数据集...")
    ds = EMMultiFreqDataset(
        data_dir=C.DATA_DIR,
        output_field="all",
        normalize=False,
        target_d_over_lambda=args.d_lam,
        use_scatter=getattr(C, "USE_SCATTER_FIELD", False),
        bg_params_path=getattr(C, "BACKGROUND_PARAMS_PATH",
                               os.path.join(_ROOT, "data", "blank", "background_params.npz")),
        polarizations=[args.pol],
    )
    idxs = [i for i, s in enumerate(ds.samples) if s["name"] == args.sample
            and (not args.category or s["category"] == args.category)]
    if not idxs:
        print(f"[错误] 数据目录里找不到样本 {args.sample!r}（polarization={args.pol}）")
        return 1
    idxs.sort(key=lambda i: ds.samples[i]["freq_ghz"])
    print(f"找到 {len(idxs)} 个频点: {[ds.samples[i]['freq_ghz'] for i in idxs]}")

    # 仿真 S11
    sample_dir = os.path.dirname(ds.samples[idxs[0]]["patch"])
    csv_path = args.csv or os.path.join(sample_dir, f"{args.sample}_S_Parameters.csv")
    if not os.path.isfile(csv_path):
        print(f"[错误] 找不到仿真 S 参数文件: {csv_path}（可用 --csv 指定）")
        return 1
    sim_s11 = read_sim_s11(csv_path)

    freqs, pred_s11_db, sim_s11_db = [], [], []
    for i in idxs:
        freq = ds.samples[i]["freq_ghz"]
        z_mm = args.d_lam * wavelength_mm(freq)

        scatter = predict_scatter(model, lc.normalizer, ds[i])  # (H, W, 12)
        _, s11_db = calc_s11(scatter, freq, z_mm)

        sim_db = sim_s11.get(freq, float("nan"))
        freqs.append(freq)
        pred_s11_db.append(s11_db)
        sim_s11_db.append(sim_db)
        print(f"  {freq} GHz: pred={s11_db:.2f} dB, sim={sim_db:.2f} dB")

    plt.figure(figsize=(8, 5))
    plt.plot(freqs, sim_s11_db,  "b-o", label="Simulation S11")
    plt.plot(freqs, pred_s11_db, "r--s", label="Predicted S11")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S11 (dB)")
    plt.title(f"S11 Comparison: Simulation vs {lc.model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\n已保存: {args.out}")

    valid = [(p, s) for p, s in zip(pred_s11_db, sim_s11_db) if np.isfinite(s)]
    if valid:
        mae = float(np.mean([abs(p - s) for p, s in valid]))
        print(f"S11 平均绝对误差: {mae:.2f} dB（{len(valid)} 个频点）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

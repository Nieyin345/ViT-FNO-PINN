"""
从预测的散射场计算 S11，与仿真 S11 对比。

物理模型：
  - 金属背板，反射系数 = -1
  - 背景场（无贴片）= 入射波 + 金属反射波 = 驻波
  - 散射场 = 总场 - 背景场 = 贴片额外引起的散射
  - S11 = 散射场在参考面（z=0）的 0 阶 Floquet 模式 / 入射波幅值
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import config as C
from data.dataset import EMMultiFreqDataset
from data.fit_background import get_background_field, FREQS_GHZ
from models.fno2d_lora import FNO2dLoRA

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


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = FNO2dLoRA(
        modes1=C.MODES1, modes2=C.MODES2, width=C.WIDTH,
        n_layers=C.N_LAYERS, out_channels=getattr(C, "OUT_CHANNELS", 6),
        lora_rank=getattr(C, "LORA_RANK", 4)
    )
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    msd = model.state_dict()
    compat = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    msd.update(compat)
    model.load_state_dict(msd, strict=False)
    model.eval()

    norm_mean = torch.tensor(ckpt["output_norm_mean"]).view(1, -1, 1, 1)
    norm_std  = torch.tensor(ckpt["output_norm_std"]).view(1, -1, 1, 1)
    return model, norm_mean, norm_std


def predict_scatter(model, norm_mean, norm_std, sample):
    inp  = sample["input"].unsqueeze(0)
    freq = sample["freq"].unsqueeze(0)
    with torch.no_grad():
        pred = model(inp, freq)
    pred_phys = (pred * norm_std + norm_mean)[0]  # (12, H, W)
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


def main():
    ckpt_path = "../checkpoints/fno_lora/best.pth"
    # 取一个样本做验证
    sample_dir = "../data/data/circle_ring/circle_ring_1"
    csv_path   = f"{sample_dir}/circle_ring_1_S_Parameters.csv"

    print("加载模型...")
    model, norm_mean, norm_std = load_model(ckpt_path)

    print("加载数据集...")
    ds = EMMultiFreqDataset(
        data_dir=C.DATA_DIR,
        output_field="all",
        normalize=False,
        target_d_over_lambda=TARGET_D_LAM,
        use_scatter=getattr(C, "USE_SCATTER_FIELD", False),
        bg_params_path=getattr(C, "BACKGROUND_PARAMS_PATH", "data/blank/background_params.npz"),
    )
    # 找 circle_ring_1 的所有频点
    samples = [s for s in ds.samples if s["name"] == "circle_ring_1"]
    samples.sort(key=lambda s: s["freq_ghz"])
    print(f"找到 {len(samples)} 个频点")

    # 读取仿真 S11
    sim_s11 = read_sim_s11(csv_path)

    freqs, pred_s11_db, sim_s11_db = [], [], []

    for s_meta in samples:
        freq = s_meta["freq_ghz"]
        lam  = wavelength_mm(freq)
        z_mm = TARGET_D_LAM * lam

        # 找对应的 dataset 样本
        idx = next(i for i, s in enumerate(ds.samples)
                   if s["name"] == "circle_ring_1" and s["freq_ghz"] == freq)
        sample = ds[idx]

        # 预测散射场
        scatter = predict_scatter(model, norm_mean, norm_std, sample)  # (H, W, 12)

        # 计算 S11
        _, s11_db = calc_s11(scatter, freq, z_mm)

        freqs.append(freq)
        pred_s11_db.append(s11_db)
        sim_s11_db.append(sim_s11.get(freq, float("nan")))

        print(f"  {freq} GHz: pred={s11_db:.2f} dB, sim={sim_s11.get(freq, float('nan')):.2f} dB")

    # 画图
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, sim_s11_db,  "b-o", label="Simulation S11")
    plt.plot(freqs, pred_s11_db, "r--s", label="Predicted S11")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S11 (dB)")
    plt.title("S11 Comparison: Simulation vs Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("s11_comparison.png", dpi=150)
    print("\n已保存: s11_comparison.png")


if __name__ == "__main__":
    main()

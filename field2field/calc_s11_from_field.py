"""
从近场数据计算 S11，与仿真 S 参数对比。

原理：
  散射场的空间平均 = 0 阶 Floquet 模式（反射平面波）
  S11 = 散射场均值 / 入射波幅值（传播回参考面 z=0）

用法：
  python test/calc_s11_from_field.py --sample data/data/circle/circle_1
"""
import os, sys, math, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from scipy.interpolate import interp1d

FREQ_LIST  = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
Z_MM_BLANK = np.arange(2, 23.1, 0.5, dtype=float)
Z_MM_DATA  = np.arange(2, 24, 1, dtype=float)
C_MM       = 3e11
BLANK_PATH = os.path.join(_HERE, "..", "data", "blank", "blank", "field_volume_merged.npy")

# 加载 blank 数据
_blank = np.load(BLANK_PATH)  # (9, 43, 64, 64, 12)

def get_bg_at_z(freq_idx, z_mm):
    """从 blank 插值得到背景场，返回 (64, 64, 12)。"""
    field_freq = _blank[freq_idx]  # (43, 64, 64, 12)
    flat = field_freq.reshape(43, -1)
    f = interp1d(Z_MM_BLANK, flat, axis=0, kind="cubic")
    return f(z_mm).reshape(64, 64, 12).astype(np.float32)


def calc_s11_single_plane(volume, freq_idx, z_mm=3.0):
    """
    单面方法：只需一个截面的总场 + 背景场，计算 S11。
    
    原理：
      E_bg(z) = -2j * E_inc * sin(kz)  →  E_inc = E_bg / (-2j*sin(kz))
      S11 = (E_total/E_inc - e^(-jkz)) / e^(+jkz)
    """
    freq = FREQ_LIST[freq_idx]
    k = 2 * math.pi * freq * 1e9 / C_MM  # rad/mm

    z_idx = int(np.argmin(np.abs(Z_MM_DATA - z_mm)))
    z = float(Z_MM_DATA[z_idx])

    # 总场 Ey 空间平均
    plane = volume[freq_idx, z_idx]
    ey_total = (plane[:,:,2] + 1j * plane[:,:,3]).mean()

    # 背景场 Ey 空间平均
    bg = get_bg_at_z(freq_idx, z)
    ey_bg = (bg[:,:,2] + 1j * bg[:,:,3]).mean()

    # 入射波幅值
    sin_kz = math.sin(k * z)
    e_inc = ey_bg / (-2j * sin_kz + 1e-8)

    # S11
    s11 = (ey_total / (e_inc + 1e-8) - np.exp(-1j * k * z)) / np.exp(1j * k * z)
    s11_db = 20 * math.log10(abs(s11) + 1e-12)
    return s11, s11_db
    """
    从总场（有贴片）在两个 z 位置的空间平均值，解出 S11。
    
    总场 = E_inc * (e^(-jkz) + S11 * e^(+jkz))
    在 z1 和 z2 两点联立方程解出 S11。
    """
    freq = FREQ_LIST[freq_idx]
    k = 2 * math.pi * freq * 1e9 / C_MM  # rad/mm

    def get_ey_mean(z_mm):
        z_idx = int(np.argmin(np.abs(Z_MM_DATA - z_mm)))
        plane = volume[freq_idx, z_idx]
        ey = plane[:,:,2] + 1j * plane[:,:,3]
        return ey.mean()

    E1 = get_ey_mean(z1_mm)
    E2 = get_ey_mean(z2_mm)

    # 联立方程：
    # E1 = E_inc * (e^(-jk*z1) + S11 * e^(+jk*z1))
    # E2 = E_inc * (e^(-jk*z2) + S11 * e^(+jk*z2))
    a1 = np.exp(-1j * k * z1_mm)
    b1 = np.exp( 1j * k * z1_mm)
    a2 = np.exp(-1j * k * z2_mm)
    b2 = np.exp( 1j * k * z2_mm)

    # E1/E2 = (a1 + S11*b1) / (a2 + S11*b2)
    # E1*(a2 + S11*b2) = E2*(a1 + S11*b1)
    # S11*(E1*b2 - E2*b1) = E2*a1 - E1*a2
    denom = E1 * b2 - E2 * b1
    if abs(denom) < 1e-10:
        return None, float("nan")
    s11 = (E2 * a1 - E1 * a2) / denom
    s11_db = 20 * math.log10(abs(s11) + 1e-12)
    return s11, s11_db
    """
    从近场数据计算 S11（TE 模式，Ey 主极化）。
    
    z_src_mm: 用于计算的截面位置（mm），建议 3~5mm
    """
    freq = FREQ_LIST[freq_idx]
    k = 2 * math.pi * freq * 1e9 / C_MM  # rad/mm

    # 取最近的 z 层
    z_idx = int(np.argmin(np.abs(Z_MM_DATA - z_src_mm)))
    z_mm  = float(Z_MM_DATA[z_idx])

    # 总场 Ey（通道 2=Ey_r, 3=Ey_i）
    plane = volume[freq_idx, z_idx]  # (64, 64, 12)
    ey_total = plane[:,:,2] + 1j * plane[:,:,3]  # (64, 64)

    # 背景场 Ey
    bg = get_bg_at_z(freq_idx, z_mm)
    ey_bg = bg[:,:,2] + 1j * bg[:,:,3]  # (64, 64)

    # 散射场 Ey
    ey_scatter = ey_total - ey_bg

    # 0 阶 Floquet 模式 = 空间平均
    ey_mean = ey_scatter.mean()

    # 传播回参考面 z=0（散射场是反射波，向 -z 传播，乘以 exp(+jkz)）
    phase_corr = np.exp(1j * k * z_mm)
    ey_ref = ey_mean * phase_corr

    # 入射波幅值：背景场 = -2j * A_inc * sin(kz)
    ey_bg_mean = ey_bg.mean()  # 背景场空间均匀，取均值
    sin_kz = math.sin(k * z_mm)
    a_inc = abs(ey_bg_mean) / (2 * abs(sin_kz) + 1e-8)

    s11 = ey_ref / (a_inc + 1e-8)
    s11_db = 20 * math.log10(abs(s11) + 1e-12)
    return s11, s11_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="data/data/circle/circle_1")
    parser.add_argument("--z-src", type=float, default=3.0, help="计算截面 z (mm)")
    args = parser.parse_args()

    sample_dir = args.sample
    vol_path   = os.path.join(sample_dir, "field_volume_merged.npy")
    csv_files  = [f for f in os.listdir(sample_dir) if f.endswith("_S_Parameters.csv")]

    volume = np.load(vol_path)
    print(f"样本: {os.path.basename(sample_dir)}")
    print(f"计算截面: z = {args.z_src} mm")
    print()

    # 读取仿真 S11
    sim_s11 = {}
    if csv_files:
        df = pd.read_csv(os.path.join(sample_dir, csv_files[0]), sep=";")
        for _, row in df.iterrows():
            f = float(row.iloc[0])
            sim_s11[f] = float(row.iloc[1])  # S(Port1,Port1) dB

    print(f"{'Freq':>6}  {'|S11| (dB)':>12}  {'Phase (deg)':>12}  {'Sim S11 (dB)':>13}  {'Diff':>8}")
    print("-" * 60)
    for fi, freq in enumerate(FREQ_LIST):
        s11, s11_db = calc_s11_single_plane(volume, fi, z_mm=args.z_src)
        phase_deg = math.degrees(math.atan2(s11.imag, s11.real)) if s11 is not None else float("nan")
        sim_db = sim_s11.get(freq, float("nan"))
        diff = s11_db - sim_db if not math.isnan(sim_db) else float("nan")
        print(f"{freq:>6.1f}  {s11_db:>12.4f}  {phase_deg:>12.2f}  {sim_db:>13.4f}  {diff:>8.4f}")


if __name__ == "__main__":
    main()

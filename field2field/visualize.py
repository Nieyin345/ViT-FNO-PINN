#!/usr/bin/env python3
"""
可视化工具（npy 格式）：
  - 选择 field_volume_merged.npy 文件
  - 选择频率
  - 滑块控制目标截面
  - 对比角谱法传播结果和真实场

用法：python test/visualize.py
"""
from __future__ import annotations
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

import math
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider, RadioButtons

from propagation import propagate_em_twoway_from_source
from scipy.interpolate import interp1d

# 数据参数
FREQ_LIST  = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
Z_MM       = np.arange(2, 24, 1, dtype=float)   # 22 层，2~23mm，步长 1mm
CELL_MM    = 15.0                                # 单元尺寸 mm
N_GRID     = 64                                  # 空间格点数
DX         = CELL_MM / (N_GRID - 1) * 1e-3      # 空间步长 m
C0         = 3e8                                 # 光速 m/s
BLANK_PATH = os.path.join(_HERE, "..", "data", "blank", "blank", "field_volume_merged.npy")
Z_MM_BLANK = np.arange(2, 23.1, 0.5, dtype=float)  # blank 43层，步长0.5mm

# 加载 blank 数据并建立插值器
_blank_volume = None
_blank_interp = {}  # {freq_idx: interp1d}

def _get_blank_interp(freq_idx):
    global _blank_volume
    if _blank_volume is None:
        _blank_volume = np.load(BLANK_PATH)  # (9, 43, 64, 64, 12)
    if freq_idx not in _blank_interp:
        field_freq = _blank_volume[freq_idx]  # (43, 64, 64, 12)
        n_z = field_freq.shape[0]
        flat = field_freq.reshape(n_z, -1)
        _blank_interp[freq_idx] = interp1d(Z_MM_BLANK, flat, axis=0, kind="cubic")
    return _blank_interp[freq_idx]

def get_bg_at_z(freq_idx, z_mm):
    """从 blank 数据插值得到指定 z 位置的背景场，返回 (64, 64, 12)。"""
    f = _get_blank_interp(freq_idx)
    return f(z_mm).reshape(64, 64, 12).astype(np.float32)


def get_field_at_z(volume, freq_idx, z_idx):
    """从体数据取指定频率和z层的场，返回 (3, nx, ny) complex，E 或 H。"""
    plane = volume[freq_idx, z_idx]  # (64, 64, 12)
    # 通道顺序：Ex_r,Ex_i, Ey_r,Ey_i, Ez_r,Ez_i, Hx_r,Hx_i, Hy_r,Hy_i, Hz_r,Hz_i
    def to_complex(r, i): return plane[:,:,r] + 1j * plane[:,:,i]
    E = np.stack([to_complex(0,1), to_complex(2,3), to_complex(4,5)])   # (3,64,64)
    H = np.stack([to_complex(6,7), to_complex(8,9), to_complex(10,11)]) # (3,64,64)
    return E, H


def subtract_bg(field_3, freq_idx, z_mm, field_type="E"):
    """减去背景场（从 blank 数据插值），返回散射场。"""
    bg_plane = get_bg_at_z(freq_idx, z_mm)  # (64, 64, 12)
    if field_type == "E":
        bg_c = np.stack([
            bg_plane[:,:,0]+1j*bg_plane[:,:,1],
            bg_plane[:,:,2]+1j*bg_plane[:,:,3],
            bg_plane[:,:,4]+1j*bg_plane[:,:,5],
        ])
    else:
        bg_c = np.stack([
            bg_plane[:,:,6]+1j*bg_plane[:,:,7],
            bg_plane[:,:,8]+1j*bg_plane[:,:,9],
            bg_plane[:,:,10]+1j*bg_plane[:,:,11],
        ])
    return field_3 - bg_c


def get_field_interp(volume, freq_idx, z_mm):
    """从体数据插值得到指定 z 位置的场，返回 (E, H) 各 (3,64,64) complex。"""
    from scipy.interpolate import interp1d
    field_freq = volume[freq_idx]  # (22, 64, 64, 12)
    n_z = field_freq.shape[0]
    flat = field_freq.reshape(n_z, -1)
    f = interp1d(Z_MM, flat, axis=0, kind="cubic", bounds_error=True)
    plane = f(z_mm).reshape(64, 64, 12).astype(np.float32)
    def to_complex(r, i): return plane[:,:,r] + 1j * plane[:,:,i]
    E = np.stack([to_complex(0,1), to_complex(2,3), to_complex(4,5)])
    H = np.stack([to_complex(6,7), to_complex(8,9), to_complex(10,11)])
    return E, H


def propagate(volume, freq_idx, src_z_mm, tgt_z_mm, lp_cutoff=0.8):
    """从源位置传播到目标位置，返回预测的总场 (E, H)。支持任意 z 值（插值）。"""
    freq_hz = FREQ_LIST[freq_idx] * 1e9
    dz = (tgt_z_mm - src_z_mm) * 1e-3  # m

    E_src, H_src = get_field_interp(volume, freq_idx, src_z_mm)
    E_scat = subtract_bg(E_src, freq_idx, src_z_mm, "E")
    H_scat = subtract_bg(H_src, freq_idx, src_z_mm, "H")

    E_scat_t, H_scat_t = propagate_em_twoway_from_source(
        E_scat, H_scat, dx=DX, dy=DX, freq_hz=freq_hz, dz=dz,
        lp_cutoff=lp_cutoff
    )
    bg_plane = get_bg_at_z(freq_idx, tgt_z_mm)
    bg_E = np.stack([
        bg_plane[:,:,0]+1j*bg_plane[:,:,1],
        bg_plane[:,:,2]+1j*bg_plane[:,:,3],
        bg_plane[:,:,4]+1j*bg_plane[:,:,5],
    ])
    bg_H = np.stack([
        bg_plane[:,:,6]+1j*bg_plane[:,:,7],
        bg_plane[:,:,8]+1j*bg_plane[:,:,9],
        bg_plane[:,:,10]+1j*bg_plane[:,:,11],
    ])
    return E_scat_t + bg_E, H_scat_t + bg_H


class VisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("近场传播可视化（npy）")
        self.root.geometry("1400x900")

        self.volume   = None
        self.freq_idx = 4  # 默认 10GHz
        self.src_z_idx = 0  # 源层固定第 0 层（z=2mm）

        self._build_ctrl()
        self._build_plot()

    def _build_ctrl(self):
        ctrl = tk.Frame(self.root, width=220, bg="#1a1a2e")
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        ctrl.pack_propagate(False)

        def lbl(t):
            tk.Label(ctrl, text=t, bg="#1a1a2e", fg="#ccc", anchor="w").pack(fill=tk.X, pady=(6,0))

        lbl("数据文件 (field_volume_merged.npy):")
        self.file_var = tk.StringVar(value="未选择")
        tk.Label(ctrl, textvariable=self.file_var, bg="#1a1a2e", fg="#4a9eff",
                 wraplength=200, justify="left").pack(fill=tk.X)
        tk.Button(ctrl, text="选择文件", command=self._browse,
                  bg="#4a9eff", fg="white").pack(fill=tk.X, pady=4)

        lbl("频率:")
        self.freq_var = tk.StringVar(value="10.0 GHz")
        cb = ttk.Combobox(ctrl, textvariable=self.freq_var,
                          values=[f"{f} GHz" for f in FREQ_LIST], state="readonly")
        cb.pack(fill=tk.X)
        cb.bind("<<ComboboxSelected>>", self._on_freq)

        lbl("源层 (固定):")
        self.src_lbl = tk.StringVar(value=f"z = {Z_MM[0]:.0f} mm")
        tk.Label(ctrl, textvariable=self.src_lbl, bg="#1a1a2e", fg="#ffd93d").pack(fill=tk.X)

        lbl("显示分量:")
        self.field_var = tk.StringVar(value="E")
        for v in ["E", "H"]:
            tk.Radiobutton(ctrl, text=v, variable=self.field_var, value=v,
                           bg="#1a1a2e", fg="#ccc", selectcolor="#333",
                           command=self._redraw).pack(anchor="w")

        self.comp_var = tk.StringVar(value="y")
        for v in ["x", "y", "z"]:
            tk.Radiobutton(ctrl, text=v, variable=self.comp_var, value=v,
                           bg="#1a1a2e", fg="#ccc", selectcolor="#333",
                           command=self._redraw).pack(anchor="w")

        lbl("显示方式:")
        self.disp_var = tk.StringVar(value="magnitude")
        for v in ["magnitude", "phase", "real", "imag"]:
            tk.Radiobutton(ctrl, text=v, variable=self.disp_var, value=v,
                           bg="#1a1a2e", fg="#ccc", selectcolor="#333",
                           command=self._redraw).pack(anchor="w")

        self.status = tk.StringVar(value="请选择数据文件")
        tk.Label(ctrl, textvariable=self.status, bg="#1a1a2e", fg="#aaa",
                 wraplength=200, justify="left").pack(fill=tk.X, pady=8)

    def _build_plot(self):
        frame = tk.Frame(self.root, bg="#1a1a2e")
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(11, 8), facecolor="#1a1a2e")
        self.canvas = FigureCanvasTkAgg(self.fig, frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # z 输入框
        z_frame = tk.Frame(frame, bg="#1a1a2e")
        z_frame.pack(fill=tk.X, padx=10, pady=4)
        tk.Label(z_frame, text="Target z (mm):", bg="#1a1a2e", fg="#ccc").pack(side=tk.LEFT)
        self.z_entry = tk.Entry(z_frame, width=8)
        self.z_entry.insert(0, str(int(Z_MM[10])))
        self.z_entry.pack(side=tk.LEFT, padx=4)
        tk.Button(z_frame, text="Go", command=self._on_z_enter,
                  bg="#4a9eff", fg="white").pack(side=tk.LEFT)
        self.z_entry.bind("<Return>", lambda e: self._on_z_enter())

        # 低通滤波截止输入框
        lp_frame = tk.Frame(frame, bg="#1a1a2e")
        lp_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Label(lp_frame, text="LP cutoff (×k, e.g. 3):", bg="#1a1a2e", fg="#ccc").pack(side=tk.LEFT)
        self.lp_entry = tk.Entry(lp_frame, width=6)
        self.lp_entry.insert(0, "3.0")
        self.lp_entry.pack(side=tk.LEFT, padx=4)
        self.lp_entry.bind("<Return>", lambda e: self._redraw())

        self.tgt_z_idx = 10
        self.tgt_z_mm  = float(Z_MM[10])  # 支持任意 z 值

    def _browse(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.join(_HERE, "..", "data", "data"),
            filetypes=[("NumPy", "*.npy"), ("All", "*.*")])
        if not path:
            return
        try:
            self.volume = np.load(path)
            assert self.volume.shape[1:] == (22, 64, 64, 12), \
                f"期望 shape (9,22,64,64,12)，实际 {self.volume.shape}"
            self.file_var.set(os.path.basename(path))
            self.status.set(f"已加载\nshape: {self.volume.shape}")
            self._redraw()
        except Exception as e:
            self.status.set(f"加载失败: {e}")

    def _on_freq(self, event=None):
        self.freq_idx = FREQ_LIST.index(float(self.freq_var.get().split()[0]))
        self._redraw()

    def _on_z_enter(self):
        try:
            z = float(self.z_entry.get())
            if z < Z_MM[0] or z > Z_MM[-1]:
                self.status.set(f"z 超出范围 [{Z_MM[0]:.0f}, {Z_MM[-1]:.0f}]mm")
                return
            self.tgt_z_mm = z
            self._redraw()
        except ValueError:
            self.status.set("请输入有效的数字")

    def _extract(self, field3, comp, disp):
        cidx = {"x":0,"y":1,"z":2}[comp]
        d = field3[cidx]
        if disp == "magnitude": return np.abs(d)
        if disp == "phase":     return np.angle(d, deg=True)
        if disp == "real":      return d.real
        return d.imag

    def _redraw(self):
        if self.volume is None:
            return
        fi   = self.freq_idx
        si   = self.src_z_idx
        src_z = float(Z_MM[si])
        tgt_z = self.tgt_z_mm
        freq  = FREQ_LIST[fi]

        # 预测场
        try:
            lp = float(self.lp_entry.get()) if hasattr(self, 'lp_entry') else 0.8
            E_pred, H_pred = propagate(self.volume, fi, src_z, tgt_z, lp_cutoff=lp)
        except Exception as e:
            self.status.set(f"传播失败: {e}")
            return

        # 真实场（插值）
        try:
            E_ref, H_ref = get_field_interp(self.volume, fi, tgt_z)
        except Exception as e:
            self.status.set(f"真实场插值失败: {e}")
            return

        ftype = self.field_var.get()
        comp  = self.comp_var.get()
        disp  = self.disp_var.get()

        pred_f = E_pred if ftype == "E" else H_pred
        ref_f  = E_ref  if ftype == "E" else H_ref

        pred_img = self._extract(pred_f, comp, disp)
        ref_img  = self._extract(ref_f,  comp, disp)
        diff_img = np.abs(pred_img - ref_img)
        rel = np.linalg.norm(diff_img) / (np.linalg.norm(ref_img) + 1e-30)

        # 去均值显示空间变化
        pred_ac = pred_img - pred_img.mean()
        ref_ac  = ref_img  - ref_img.mean()
        combined = np.concatenate([pred_ac.ravel(), ref_ac.ravel()])
        vmin = float(np.percentile(combined, 2))
        vmax = float(np.percentile(combined, 98))
        if abs(vmax-vmin) < 1e-9: vmin -= 1; vmax += 1
        cmap = "RdBu_r" if disp in ("phase","real","imag") else "jet"

        x_mm = np.linspace(0, CELL_MM, N_GRID)
        extent = [0, CELL_MM, 0, CELL_MM]

        self.fig.clear()
        gs = gridspec.GridSpec(2, 3, figure=self.fig,
                               left=0.08, right=0.97, top=0.90, bottom=0.15,
                               wspace=0.4, hspace=0.5)
        axes = [self.fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]

        def style(ax):
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#ccc", labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor("#444")

        label = f"{ftype}{'xyz'.index(comp)+1} ({disp})"

        for ax, img, title in [
            (axes[0], pred_ac, f"Predicted {label}  mean={pred_img.mean():.2f}"),
            (axes[1], ref_ac,  f"Reference {label}  mean={ref_img.mean():.2f}"),
        ]:
            style(ax)
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                           origin="lower", extent=extent, aspect="auto")
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="#ccc", labelsize=7)
            ax.set_title(title, color="#eee", fontsize=8)
            ax.set_xlabel("y (mm)", color="#ccc", fontsize=7)
            ax.set_ylabel("x (mm)", color="#ccc", fontsize=7)

        # 误差图
        style(axes[2])
        im_d = axes[2].imshow(diff_img, cmap="hot", origin="lower",
                              extent=extent, aspect="auto")
        self.fig.colorbar(im_d, ax=axes[2], fraction=0.046, pad=0.04).ax.tick_params(colors="#ccc", labelsize=7)
        axes[2].set_title(f"|Pred-Ref|  relL2={rel:.4f}", color="#eee", fontsize=8)
        axes[2].set_xlabel("y (mm)", color="#ccc", fontsize=7)
        axes[2].set_ylabel("x (mm)", color="#ccc", fontsize=7)

        # 误差 vs z 曲线
        style(axes[3])
        errs = []
        for zi in range(1, len(Z_MM)):
            try:
                Ep, Hp = propagate(self.volume, fi, src_z, float(Z_MM[zi]), lp_cutoff=lp)
                Er, Hr = get_field_interp(self.volume, fi, float(Z_MM[zi]))
                F_p = Ep if ftype=="E" else Hp
                F_r = Er if ftype=="E" else Hr
                e = np.linalg.norm((F_p-F_r).ravel()) / (np.linalg.norm(F_r.ravel())+1e-30)
                errs.append((Z_MM[zi], e))
            except:
                pass
        if errs:
            zz, ee = zip(*errs)
            axes[3].plot(zz, ee, color="#4a9eff", lw=1.5)
            axes[3].axvline(tgt_z, color="#ff6b6b", lw=1.5, ls="--")
            axes[3].axvline(src_z, color="#aaa", lw=1, ls=":")
        axes[3].set_xlabel("z (mm)", color="#ccc", fontsize=7)
        axes[3].set_ylabel("relL2", color="#ccc", fontsize=7)
        axes[3].set_title(f"{ftype} relL2 vs z", color="#eee", fontsize=8)
        axes[3].set_ylim(bottom=0)

        # 截面剖面
        style(axes[4])
        mid = N_GRID // 2
        axes[4].plot(x_mm, pred_img[:, mid], color="#4a9eff", lw=1.5, label="Pred")
        axes[4].plot(x_mm, ref_img[:,  mid], color="#ff6b6b", lw=1.5, ls="--", label="Ref")
        axes[4].legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white", edgecolor="#444")
        axes[4].set_xlabel("x (mm)", color="#ccc", fontsize=7)
        axes[4].set_title(f"Cross-section y={x_mm[mid]:.1f}mm", color="#eee", fontsize=8)

        axes[5].axis("off")

        self.fig.suptitle(
            f"Source z={src_z:.0f}mm → Target z={tgt_z:.1f}mm  "
            f"(dz={tgt_z-src_z:+.1f}mm)   {freq}GHz",
            color="white", fontsize=10, y=0.97)
        self.canvas.draw()
        self.status.set(f"relL2={rel:.4f}  z={tgt_z:.1f}mm")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        plt.close("all")
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    VisApp().run()

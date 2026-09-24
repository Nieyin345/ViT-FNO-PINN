# -*- coding: utf-8 -*-
"""FNO2d with Dispersion-Coordinate Spectral Convolution（fno_disp）。

与 fno_film 的唯一结构差别：谱卷积核心从「逐模式自由参数 w1/w2」换成
「色散坐标 ξ=k/m₀(f) 上的共享函数 Wx(ξx)·Wy(ξy)」+ 模式间注意力。
频率以**物理方式**（色散坐标）进入，而不是靠 FiLM 事后调制。

对照组公平性：骨架/lift/pad/残差/Norm/输出头/dropout 与 fno_film 完全一致
（同一 FNOBase 骨架、同一 build_cond、同一 DirectionHead 输出头）。
A/B 时只换这一个模块，其余配置逐项对齐。

extra 键：
  disp_heads       注意力/谱函数的 head 数（默认 4）
  disp_n_basis     三角基函数组数（默认 6，越大 W(ξ) 越有表达力）
  disp_cell_mm     元胞尺寸 mm（默认 15.0，算色散尺度 m0 用）
  disp_attention   是否开模式间注意力（默认 True；False = 纯色散谱卷积消融）
"""

from typing import Optional

import torch
import torch.nn as nn

from core.base import Cond, FNOBase
from core.config import ModelConfig
from core.dispersion import DispersionSpectralConv
from core.encoders import DirectionHead, FourierFreqEncoder, SDFGeomEncoder
from core.registry import register


class DispBlock(nn.Module):
    """谱卷积（色散坐标调制 w0·(1+g(ξ))）+ bypass + BN + GELU + 残差。
    结构与 FNOBlockFiLM 对齐（残差 + bypass + Norm）。
    """

    def __init__(self, width: int, modes1: int, modes2: int, freq_dim: int,
                 extra: dict):
        super().__init__()
        self.spectral = DispersionSpectralConv(
            width, width, modes1, modes2, freq_dim=freq_dim,
            n_basis=int(extra.get("disp_n_basis", 4)),
            cell_mm=float(extra.get("disp_cell_mm", 15.0)),
            use_attention=bool(extra.get("disp_attention", True)),
        )
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.BatchNorm2d(width)

    def forward(self, x, cond=None):
        return nn.functional.gelu(self.norm(self.spectral(x, cond) + self.bypass(x) + x))


@register("fno_disp")
class FNO2dDisp(FNOBase):
    """FNO2d + 色散坐标谱卷积 + 模式间注意力。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        out_channels = cfg.out_channels

        self.use_em_heads = (cfg.head == "em_split")
        if self.use_em_heads and out_channels != 12:
            raise ValueError(
                f"head='em_split' 需要 out_channels=12，得到 {out_channels}。")

        self.blocks = nn.ModuleList([
            DispBlock(cfg.width, cfg.modes1, cfg.modes2, cfg.freq_dim, cfg.extra)
            for _ in range(cfg.n_layers)
        ])

        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=cfg.freq_dim)
        self.geom_encoder = SDFGeomEncoder(n_modes=4, out_dim=cfg.freq_dim)
        self.dropout = nn.Dropout2d(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.shared_proj = nn.Sequential(
            nn.Conv2d(cfg.width, 64, kernel_size=1), nn.GELU()
        )
        if self.use_em_heads:
            self.e_heads = nn.ModuleList([DirectionHead(in_channels=64) for _ in range(3)])
            self.h_heads = nn.ModuleList([DirectionHead(in_channels=64 + 6) for _ in range(3)])
        else:
            self.heads = nn.ModuleList([
                DirectionHead(in_channels=64) for _ in range(cfg.n_heads)
            ])

    def build_cond(self, x, freq):
        if freq is None:
            return None
        # freq = (B,2) = [归一化频率, 极化]。freq_raw 取第 0 列：
        # 色散尺度 m0 = f·L/c 需要原始频率标量（嵌入向量反推不回确定值）。
        return Cond(freq_emb=self.freq_encoder(freq),
                    geom_emb=self.geom_encoder(x),
                    freq_raw=freq[:, 0:1])

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.dropout(self.shared_proj(x))
        if self.use_em_heads:
            e_pred = torch.cat([h(shared) for h in self.e_heads], dim=1)
            h_input = torch.cat([shared, e_pred], dim=1)
            h_pred = torch.cat([h(h_input) for h in self.h_heads], dim=1)
            return torch.cat([e_pred, h_pred], dim=1)
        return torch.cat([h(shared) for h in self.heads], dim=1)

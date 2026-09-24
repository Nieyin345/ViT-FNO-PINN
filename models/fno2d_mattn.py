# -*- coding: utf-8 -*-
"""FNO2d with Mode-Axis Spectral Self-Attention（fno_mattn）。

## 设计动机（每一句都有本仓库实测证据）

1. fno_attention 的注意力在**通道维**做 softmax，实测权重完全均匀
   （有效位置 64/64=100%）、对输出贡献仅 5.3% —— 通道间在谱域没有
   值得注意力的结构，注意力问错了对象。
2. 真正有结构的是**模式 (kx,ky) 轴**：标准 FNO 的 w1/w2 是固定线性
   模式混合器——对所有输入用同一套混合规则。模式间自注意力把混合
   变成逐样本自适应（谱域的 data-dependent filtering）。
3. fno_disp 的失败属于色散坐标假设（谐振非线性 ≠ 谱形状平移），
   与模式注意力无关——两者当时绑死，从未单独测试。

## 结构

= fno_film 原样保留（FiLM scale/shift 条件化，当前最优 0.2265 的骨架）
  + 模式间自注意力修正：对谱卷积输出的低频块，
      token_i = (Re, Im) of mode i（通道平均，N = m1·m2 个 token）
      out_low = out_low + γ · Wv( softmax(QKᵀ/√d) · V )
  γ 初始化 0.1、Wv 零初始化 → ∂L/∂Wv = γ·(·) ≠ 0，梯度链与 film_net
  同构（film 的 film_net 末层零初始化也成功启动，探针 P3 实测）。

γ=0 时逐位等于 fno_film（用同种子权重实测验证过）——所以任何变化
都来自注意力本身，A/B 是纯增量对照。

每层新增参数：Wq/Wk (2·2·32=128) + Wv (6) + γ (1) ≈ 135 个，
总参数量与 fno_film 几乎相同（+675）。
"""

from typing import Optional

import math

import torch
import torch.nn as nn

from core.base import Cond, FNOBase
from core.config import ModelConfig
from core.encoders import DirectionHead, FourierFreqEncoder, SDFGeomEncoder
from core.registry import register


class ModeSelfAttention(nn.Module):
    """谱域模式间自注意力（对低频块做修正）。"""

    def __init__(self, d_k: int = 32):
        super().__init__()
        self.d_k = d_k
        self.Wq = nn.Linear(2, d_k)
        self.Wk = nn.Linear(2, d_k)
        # Wv 零初始化 → 起点无扰动；γ 提供梯度链（同 film_net 的启动机制）
        self.Wv = nn.Linear(2, 2)
        nn.init.zeros_(self.Wv.weight)
        nn.init.zeros_(self.Wv.bias)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, out_low: torch.Tensor) -> torch.Tensor:
        """out_low: (B, C, m1, m2) complex -> 同形状（+修正）"""
        B, C, m1, m2 = out_low.shape
        # token = 每个模式的通道平均复数值（先取实虚，再转 float）
        t = out_low.mean(dim=1)                                    # (B,m1,m2) complex
        feat = torch.stack([t.real.float(), t.imag.float()], dim=-1).reshape(B, m1 * m2, 2)
        q, k = self.Wq(feat), self.Wk(feat)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, feat)                                # (B,N,2)
        corr = self.gamma * self.Wv(ctx)                           # (B,N,2)
        corr = corr.reshape(B, m1, m2, 2)
        corr_ft = torch.complex(corr[..., 0], corr[..., 1]).to(out_low.dtype)
        return out_low + corr_ft.unsqueeze(1)


class SpectralConv2dModeAttn(nn.Module):
    """SpectralConv2dFiLM 的原样复制 + 模式注意力插在 irfft2 之前。

    （不复用/继承 SpectralConv2dFiLM 的 forward，因为注意力必须插在
    irfft2 之前的频域；复制该实现并插入一行，保证其余逐位一致。）
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int,
                 freq_dim: int):
        super().__init__()
        scale = 1 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # ---- FiLM（与 fno_film 的 film_net 完全一致）
        cond_dim = 2 * freq_dim
        self.film_net = nn.Sequential(
            nn.Linear(cond_dim, 256), nn.GELU(),
            nn.Linear(256, out_channels * 2),
        )
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)

        # ---- 模式注意力（本模型唯一新增）
        self.mode_attn = ModeSelfAttention()

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        freq_emb = cond.freq_emb if isinstance(cond, Cond) else None
        geom_emb = cond.geom_emb if isinstance(cond, Cond) else None

        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x.float())

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2)

        shift = None
        if freq_emb is not None and geom_emb is not None:
            c = torch.cat([freq_emb, geom_emb], dim=-1)
            film_params = self.film_net(c)
            scale = film_params[:, :self.out_channels].view(B, self.out_channels, 1, 1)
            shift = film_params[:, self.out_channels:].view(B, self.out_channels, 1, 1)
            # scale 在频域乘（频域逐通道缩放 == 空域逐通道缩放）；
            # shift 留到 irfft2 之后空域加（频域加常数 = 空域 delta 冲激）。
            out_ft = out_ft * (1.0 + scale).to(out_ft.dtype)

        # ---- 模式间注意力（低频块，仅 FiLM 之后、irfft2 之前）
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.mode_attn(out_ft[:, :, :m1, :m2])

        y = torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)
        if shift is not None:
            y = y + shift.to(y.dtype)
        return y


class FNOBlockModeAttn(nn.Module):
    """FiLM 谱卷积 + 模式注意力 + bypass + BN + GELU + 残差（结构与 FNOBlockFiLM 对齐）。"""

    def __init__(self, width: int, modes1: int, modes2: int, freq_dim: int):
        super().__init__()
        self.spectral = SpectralConv2dModeAttn(width, width, modes1, modes2, freq_dim=freq_dim)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.BatchNorm2d(width)

    def forward(self, x, cond=None):
        return nn.functional.gelu(self.norm(self.spectral(x, cond) + self.bypass(x) + x))


@register("fno_mattn")
class FNO2dModeAttn(FNOBase):
    """FNO2d + FiLM 条件化 + 模式间谱自注意力。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        self.blocks = nn.ModuleList([
            FNOBlockModeAttn(cfg.width, cfg.modes1, cfg.modes2, freq_dim=cfg.freq_dim)
            for _ in range(cfg.n_layers)
        ])

        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=cfg.freq_dim)
        self.geom_encoder = SDFGeomEncoder(n_modes=4, out_dim=cfg.freq_dim)
        self.dropout = nn.Dropout2d(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.shared_proj = nn.Sequential(
            nn.Conv2d(cfg.width, 64, kernel_size=1), nn.GELU()
        )
        self.heads = nn.ModuleList([
            DirectionHead(in_channels=64) for _ in range(cfg.n_heads)
        ])

    def build_cond(self, x, freq):
        if freq is None:
            return None
        return Cond(freq_emb=self.freq_encoder(freq), geom_emb=self.geom_encoder(x))

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.dropout(self.shared_proj(x))
        return torch.cat([h(shared) for h in self.heads], dim=1)

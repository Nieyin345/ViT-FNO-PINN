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
    """模式轴自注意力：token = 每个模式的**通道向量**（保留全部通道信息）。

    对照标准 FNO 谱卷积的模式对角性（实测扰动一个输入模式、泄漏到其他模式的
    比例为 0.00e+00）：本模块是每层里唯一的**模式间耦合算子**，补上「层内
    零模式混合」这个结构缺失。

    ### 为什么是 v2（v1 的三处失效，均由探针 P6 实测定位）

    v1 的实现是 token = 通道平均后的 2 维 (Re, Im)、输出广播回所有通道、
    Q/K 未归一化。P6 实测：

      | 症状 | v1 实测 | 本版对应修法 |
      |---|---|---|
      | 修正对输出无影响 | 修正/主干 **0.00~0.10%** | 输出经 `to_out` 映射回通道（不再广播），容量与通道数相称 |
      | softmax 完全饱和 | 熵 **0.000**/4.159，有效位置 **1.6%** | 输入 `LayerNorm` 归一化（同 CLAUDE.md 记录的 `sig_norm` 修法） |
      | token 信息量不足 | token 仅 2 维（通道平均抹平） | token 保留全部 `2C` 维（实虚拼接） |

    起点恒等：`to_out` 零初始化 → 修正项 = 0。梯度链与 film_net 同构
    （末层零初始化 + 残差，`∂L/∂to_out = (∂L/∂x_mixed)·ctxᵀ ≠ 0`），能正常启动。

    刻意**不设 γ 门控**：γ 只是人为压制修正量级（v2 试过 γ=0.1，实测修正仍
    仅 0.1% 量级、梯度 1e-12），而零初始化的 `to_out` 已经保证了起点恒等，
    量级该由训练决定。
    """

    def __init__(self, channels: int, n_heads: int = 2, d_k: int = 16,
                 use_attn: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.use_attn = use_attn
        dim = 2 * channels                       # 实虚拼接后的 token 维度
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, n_heads * d_k)
        self.to_k = nn.Linear(dim, n_heads * d_k)
        self.to_v = nn.Linear(dim, n_heads * d_k)
        self.to_out = nn.Linear(n_heads * d_k, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, spec_low: torch.Tensor) -> torch.Tensor:
        """spec_low: (B, C, m1, m2) complex -> 同形状（+模式间混合残差）"""
        B, C, m1, m2 = spec_low.shape
        N = m1 * m2
        # token: 每个模式的通道向量（实虚拼接，保留全部信息）
        t = torch.cat([spec_low.real, spec_low.imag], dim=1)      # (B,2C,m1,m2)
        t = t.reshape(B, 2 * C, N).transpose(1, 2)                # (B,N,2C)
        t = self.norm(t)

        q = self.to_q(t).reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.to_k(t).reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.to_v(t).reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)

        if self.use_attn:
            attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d_k), dim=-1)
            ctx = attn @ v
        else:
            # 消融对照：跳过 QKᵀ，ctx = v —— 参数**逐位相同**（to_q/to_k 照建
            # 但不参与前向），只是把"模式间成对交换信息"这一步换成逐模式的
            # 独立变换。用来分离「模式混合的贡献」与「多出来的 84K 参数的贡献」。
            ctx = v
        ctx = ctx.transpose(1, 2).reshape(B, N, self.n_heads * self.d_k)
        delta = self.to_out(ctx)                                  # (B,N,2C)，零初始化
        delta = delta.transpose(1, 2).reshape(B, 2 * C, m1, m2)
        delta_ft = torch.complex(delta[:, :C], delta[:, C:]).to(spec_low.dtype)
        return spec_low + delta_ft


class SpectralConv2dModeAttn(nn.Module):
    """SpectralConv2dFiLM 的原样复制 + 模式注意力插在 irfft2 之前。

    （不复用/继承 SpectralConv2dFiLM 的 forward，因为注意力必须插在
    irfft2 之前的频域；复制该实现并插入一行，保证其余逐位一致。）
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int,
                 freq_dim: int, use_attn: bool = True):
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

        # ---- 模式注意力（本模型唯一新增），可关（消融对照）
        self.mode_attn = ModeSelfAttention(in_channels, use_attn=use_attn)

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

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)

        # ---- 模式间注意力：对投影后的低频块做数据依赖的模式混合。
        # 立论：标准 FNO 的 `einsum("bixy,ioxy->boxy")` 只对输入通道求和、模式
        # 索引原样保留 —— 层内**零模式混合**（实测扰动一个输入模式、泄漏到其他
        # 模式的比例 0.00e+00）。这是每层唯一的模式耦合算子。
        # 位置必须在投影**之后**：放在 `w1` 之前时修正量会被 `w1` 的初始尺度
        # （≈2.4e-4）乘一道，梯度被压到 1e-11、根本起不来（实测 v2.1）。
        out_ft_low = self.mode_attn(out_ft[:, :, :m1, :m2])

        shift = None
        if freq_emb is not None and geom_emb is not None:
            c = torch.cat([freq_emb, geom_emb], dim=-1)
            film_params = self.film_net(c)
            scale = film_params[:, :self.out_channels].view(B, self.out_channels, 1, 1)
            shift = film_params[:, self.out_channels:].view(B, self.out_channels, 1, 1)
            out_ft_low = out_ft_low * (1.0 + scale).to(out_ft.dtype)

        out_ft[:, :, :m1, :m2] = out_ft_low

        y = torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)
        if shift is not None:
            y = y + shift.to(y.dtype)
        return y


class FNOBlockModeAttn(nn.Module):
    """FiLM 谱卷积 + 模式注意力 + bypass + BN + GELU + 残差（结构与 FNOBlockFiLM 对齐）。"""

    def __init__(self, width: int, modes1: int, modes2: int, freq_dim: int,
                 use_attn: bool = True):
        super().__init__()
        self.spectral = SpectralConv2dModeAttn(width, width, modes1, modes2,
                                               freq_dim=freq_dim, use_attn=use_attn)
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
            FNOBlockModeAttn(cfg.width, cfg.modes1, cfg.modes2, freq_dim=cfg.freq_dim,
                             use_attn=cfg.extra.get("mattn_use_attn", True))
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

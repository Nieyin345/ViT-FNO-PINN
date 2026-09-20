# -*- coding: utf-8 -*-
"""频域卷积层：标准 FNO 谱卷积 + 可开关的超网络投影矩阵复数交叉注意力。

核心创新（原样保留）：
  1. 超网络直接生成 ΔW_q / ΔW_k，动态调制投影矩阵本身
     W_q_eff = W_q + ΔW_q(freq_emb)   ← 频率条件调制查询投影
     W_k_eff = W_k + ΔW_k(geom_emb)   ← 几何条件调制键投影
  2. Q = X_sig @ W_q_eff，K = X_sig @ W_k_eff
     数据、频率、几何三路信息在投影步骤融合，而非事后加偏置
  3. gamma 初始化 0.1 → 注意力从第 1 个 epoch 就参与训练

重构变更（唯一一处行为差异，见 core/config.py::SpectralConfig）：
  - 原先模块级 `import config as C` 并在 `__init__` 里读 6 个消融开关，
    现在改为从 `SpectralConfig` 显式传入，构造对全局配置零依赖。
  - `attention=False` 时注意力相关模块（W_q/W_k/hyper_*/gamma）**不再创建**。
    原先它们总是被创建但不参与前向，导致消融实验的参数量虚高
    （已核对 checkpoints/ablation/fno_attention_no_attn/best.pth 确实含这些键）。
    默认 attention=True，主模型参数量不变。
"""

import math

import torch
import torch.nn as nn

from .config import SpectralConfig


class SpectralConv2d(nn.Module):
    """频域卷积 + 可开关的超网络投影矩阵交叉注意力。"""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int,
                 cond_dim: int = 32, cfg: SpectralConfig = None):
        super().__init__()
        cfg = cfg or SpectralConfig()
        self.cfg = cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # 1. 频域特征 V 的参数库（标准 FNO 核心）
        scale = 1 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

        # 2. 消融开关
        self.attention_enabled = bool(cfg.attention)
        self.freq_conditioning = bool(cfg.freq_conditioning)
        self.geom_conditioning = bool(cfg.geom_conditioning)

        # 3. 注意力基础投影矩阵 + 超网络 + gamma 门控（仅启用时创建）
        if self.attention_enabled:
            self.n_modes_attn = min(8, modes1)
            self.sig_length = self.n_modes_attn * self.n_modes_attn
            self.d_k = 32
            self.W_q = nn.Parameter(torch.randn(self.sig_length, self.d_k, dtype=torch.cfloat) * 0.02)
            self.W_k = nn.Parameter(torch.randn(self.sig_length, self.d_k, dtype=torch.cfloat) * 0.02)

            self.hyper_wq = nn.Sequential(
                nn.Linear(cond_dim, 256),
                nn.GELU(),
                nn.Linear(256, self.sig_length * self.d_k),
            )
            self.hyper_wk = nn.Sequential(
                nn.Linear(cond_dim, 256),
                nn.GELU(),
                nn.Linear(256, self.sig_length * self.d_k),
            )
            nn.init.zeros_(self.hyper_wq[-1].weight)
            nn.init.zeros_(self.hyper_wq[-1].bias)
            nn.init.zeros_(self.hyper_wk[-1].weight)
            nn.init.zeros_(self.hyper_wk[-1].bias)

            self.gamma = nn.Parameter(torch.tensor(float(cfg.gamma_init), dtype=torch.float32))
            if cfg.gamma_fixed:
                self.gamma.requires_grad_(False)

        # 4. 频域 Butterworth 衰减
        if cfg.decay:
            self.register_buffer("decay_mask", _butterworth_mask(modes1, modes2, cfg))
        else:
            self.decay_mask = None

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        from .base import Cond
        B, C, H, W = x.shape
        freq_emb = cond.freq_emb if isinstance(cond, Cond) else None
        geom_emb = cond.geom_emb if isinstance(cond, Cond) else None

        # 1. 全局 FFT
        x_ft = torch.fft.rfft2(x.float())

        # 阶段 A：提取频域特征 V
        out_ft = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2)

        out_ft_new = out_ft
        if self.attention_enabled:
            # 阶段 B：超网络动态调制投影矩阵，生成 Q / K
            C_in = x_ft.size(1)
            m1 = min(self.n_modes_attn, x_ft.size(2))
            m2 = min(self.n_modes_attn, x_ft.size(3))

            freq_sig_complex = x_ft[:, :, :m1, :m2]
            if m1 < self.n_modes_attn or m2 < self.n_modes_attn:
                pad = torch.zeros(B, C_in, self.n_modes_attn, self.n_modes_attn,
                                  device=x.device, dtype=torch.cfloat)
                pad[:, :, :m1, :m2] = freq_sig_complex
                freq_sig_complex = pad

            X_sig = freq_sig_complex.reshape(B, C_in, -1)  # (B, C_in, sig_length)

            W_q_eff = self.W_q.unsqueeze(0).expand(B, -1, -1)
            W_k_eff = self.W_k.unsqueeze(0).expand(B, -1, -1)

            if freq_emb is not None and self.freq_conditioning:
                delta_Wq = self.hyper_wq(freq_emb).view(B, self.sig_length, self.d_k)
                W_q_eff = W_q_eff + delta_Wq.to(W_q_eff.dtype)

            if geom_emb is not None and self.geom_conditioning:
                delta_Wk = self.hyper_wk(geom_emb).view(B, self.sig_length, self.d_k)
                W_k_eff = W_k_eff + delta_Wk.to(W_k_eff.dtype)

            Q = torch.bmm(X_sig, W_q_eff)
            K = torch.bmm(X_sig, W_k_eff)

            scores_complex = torch.bmm(Q, K.conj().transpose(1, 2)) / math.sqrt(self.d_k)

            scores_mag = torch.abs(scores_complex)
            scores_phase = scores_complex / (scores_mag + 1e-8)
            attn_weight_real = torch.softmax(scores_mag, dim=-1)
            attn_matrix_complex = attn_weight_real.to(out_ft.dtype) * scores_phase

            # 阶段 C：注意力加权重组 + 残差门控
            attn_out_ft = torch.einsum("bij,bjxy->bixy", attn_matrix_complex, out_ft)
            out_ft_new = out_ft + self.gamma * attn_out_ft

        if self.decay_mask is not None:
            mask = self.decay_mask
            out_ft_new[:, :, :self.modes1, :self.modes2] *= mask
            out_ft_new[:, :, -self.modes1:, :self.modes2] *= mask

        return torch.fft.irfft2(out_ft_new, s=(H, W)).to(x.dtype)


def _butterworth_mask(modes1: int, modes2: int, cfg: SpectralConfig) -> torch.Tensor:
    """生成 2D 径向 Butterworth 衰减权重矩阵。"""
    i1 = torch.arange(modes1, dtype=torch.float32)
    i2 = torch.arange(modes2, dtype=torch.float32)
    k = torch.sqrt(i1[:, None] ** 2 + i2[None, :] ** 2)
    return 1.0 / torch.sqrt(1.0 + (k / cfg.decay_k0) ** (2 * cfg.decay_order))

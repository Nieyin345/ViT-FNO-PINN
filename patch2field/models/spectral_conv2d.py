"""
频域卷积层（超网络动态调制投影矩阵的复数交叉注意力机制）

核心创新：
  1. 超网络直接生成 ΔW_q / ΔW_k，动态调制投影矩阵本身
     W_q_eff = W_q + ΔW_q(freq_emb)   ← 频率条件调制查询投影
     W_k_eff = W_k + ΔW_k(geom_emb)   ← 几何条件调制键投影
  2. Q = X_sig @ W_q_eff，K = X_sig @ W_k_eff
     数据、频率、几何三路信息在投影步骤融合，而非事后加偏置
  3. gamma 初始化 0.1 → 注意力从第 1 个 epoch 就参与训练
"""

import math
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C


def _butterworth_mask(modes1: int, modes2: int) -> torch.Tensor:
    """生成 2D 径向 Butterworth 衰减权重矩阵"""
    k0 = getattr(C, "SPECTRAL_DECAY_K0", 6.0)
    n  = getattr(C, "SPECTRAL_DECAY_ORDER", 3)
    i1 = torch.arange(modes1, dtype=torch.float32)
    i2 = torch.arange(modes2, dtype=torch.float32)
    k = torch.sqrt(i1[:, None] ** 2 + i2[None, :] ** 2)
    return 1.0 / torch.sqrt(1.0 + (k / k0) ** (2 * n))


class SpectralConv2d(nn.Module):
    """频域卷积 + 超网络动态投影矩阵交叉注意力"""
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, cond_dim: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # 1. 频域特征 V 的参数库（标准 FNO 核心）
        scale = 1 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

        # 2. 注意力基础投影矩阵
        self.n_modes_attn = min(8, modes1)
        self.sig_length = self.n_modes_attn * self.n_modes_attn
        self.d_k = 32
        self.W_q = nn.Parameter(torch.randn(self.sig_length, self.d_k, dtype=torch.cfloat) * 0.02)
        self.W_k = nn.Parameter(torch.randn(self.sig_length, self.d_k, dtype=torch.cfloat) * 0.02)

        # 3. 超网络：使用 cond_dim 而非硬编码 32
        self.hyper_wq = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.sig_length * self.d_k)
        )
        self.hyper_wk = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.sig_length * self.d_k)
        )
        nn.init.zeros_(self.hyper_wq[-1].weight)
        nn.init.zeros_(self.hyper_wq[-1].bias)
        nn.init.zeros_(self.hyper_wk[-1].weight)
        nn.init.zeros_(self.hyper_wk[-1].bias)

        # 4. gamma 门控：注意力残差强度
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        if getattr(C, "SPECTRAL_DECAY", False):
            mask = _butterworth_mask(modes1, modes2)
            self.register_buffer("decay_mask", mask)
        else:
            self.decay_mask = None

    def forward(self, x: torch.Tensor, freq_emb: torch.Tensor = None, geom_emb: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. 全局 FFT
        x_ft = torch.fft.rfft2(x.float())

        # =========================================================
        # 阶段 A：提取频域特征 V
        # =========================================================
        out_ft = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2)

        # =========================================================
        # 阶段 B：超网络动态调制投影矩阵，生成 Q / K
        # =========================================================
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

        # 静态基础矩阵扩展到 batch 维度
        W_q_eff = self.W_q.unsqueeze(0).expand(B, -1, -1)  # (B, sig_length, d_k)
        W_k_eff = self.W_k.unsqueeze(0).expand(B, -1, -1)  # (B, sig_length, d_k)

        # 超网络生成动态增量并叠加（实数增量广播到复数矩阵）
        if freq_emb is not None:
            delta_Wq = self.hyper_wq(freq_emb).view(B, self.sig_length, self.d_k)  # (B, sig_length, d_k) real
            W_q_eff = W_q_eff + delta_Wq.to(W_q_eff.dtype)

        if geom_emb is not None:
            delta_Wk = self.hyper_wk(geom_emb).view(B, self.sig_length, self.d_k)  # (B, sig_length, d_k) real
            W_k_eff = W_k_eff + delta_Wk.to(W_k_eff.dtype)

        # Q = X_sig @ W_q_eff，K = X_sig @ W_k_eff
        # 数据 × 条件调制矩阵 → 三路信息在此融合
        Q = torch.bmm(X_sig, W_q_eff)  # (B, C_in, d_k)
        K = torch.bmm(X_sig, W_k_eff)  # (B, C_in, d_k)

        # 复数干涉打分 -> (B, C, C)
        scores_complex = torch.bmm(Q, K.conj().transpose(1, 2)) / math.sqrt(self.d_k)

        # 保留幅值（能量）和相位（相移）的全复数注意力矩阵
        scores_mag = torch.abs(scores_complex)
        scores_phase = scores_complex / (scores_mag + 1e-8)
        attn_weight_real = torch.softmax(scores_mag, dim=-1)
        attn_matrix_complex = attn_weight_real.to(out_ft.dtype) * scores_phase

        # =========================================================
        # 阶段 C：注意力加权重组 + 残差门控
        # =========================================================
        attn_out_ft = torch.einsum("bij,bjxy->bixy", attn_matrix_complex, out_ft)
        out_ft_new = out_ft + self.gamma * attn_out_ft

        if self.decay_mask is not None:
            mask = self.decay_mask
            out_ft_new[:, :, :self.modes1,  :self.modes2] *= mask
            out_ft_new[:, :, -self.modes1:, :self.modes2] *= mask

        return torch.fft.irfft2(out_ft_new, s=(H, W)).to(x.dtype)

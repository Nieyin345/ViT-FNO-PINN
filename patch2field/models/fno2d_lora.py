"""
FNO2d with LoRA-style frequency-conditioned spectral kernels.

核心思路：
  用频率/几何条件通过低秩分解直接调制 FNO 的频域卷积核 w1/w2，
  而不是通过注意力矩阵间接调制。
  
  ΔW = U(freq_emb) @ V^T，rank r 很小（默认4），参数量几乎不增加。
  W_eff = W + ΔW，直接影响最核心的频域变换。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from .fno_core import FNOBase, FourierFreqEncoder




class SDFGeomEncoder(nn.Module):
    def __init__(self, n_modes: int = 8, out_dim: int = 64):
        super().__init__()
        self.n_modes = n_modes
        feat_dim = n_modes * n_modes * 4  # SDF幅值+相位, Mask幅值+相位
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.GELU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sdf  = x[:, 0, :, :]
        mask = x[:, 1, :, :]

        def _low_freq_feat(field):
            ft  = torch.fft.rfft2(field)
            low = ft[:, :self.n_modes, :self.n_modes]
            mag   = low.abs().reshape(field.size(0), -1)
            phase = torch.angle(low).reshape(field.size(0), -1)
            return torch.cat([mag, phase], dim=1)

        feat = torch.cat([_low_freq_feat(sdf), _low_freq_feat(mask)], dim=1)
        return self.mlp(feat)


class SpectralConv2dLoRA(nn.Module):
    """
    FNO 频域卷积 + LoRA 风格的条件调制。
    
    W_eff = W + U(cond) @ V^T
    U: (B, in_ch, r)  由条件生成
    V: (out_ch * modes1 * modes2, r)  可学习静态矩阵
    """
    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int, cond_dim: int = 32, rank: int = 4):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.rank   = rank

        scale = 1 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

        flat_dim = out_channels * modes1 * modes2
        self.V1 = nn.Parameter(torch.randn(flat_dim, rank, dtype=torch.cfloat) * 0.01)
        self.V2 = nn.Parameter(torch.randn(flat_dim, rank, dtype=torch.cfloat) * 0.01)

        # 超网络分别接收 freq_emb 和 geom_emb，内部融合后生成 U
        # freq_dim=32, geom_dim=64，合并后 96 维
        fused_dim = cond_dim + 64  # freq_dim + geom_dim
        self.hyper_u1 = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.GELU(),
            nn.Linear(128, in_channels * rank * 2)
        )
        self.hyper_u2 = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.GELU(),
            nn.Linear(128, in_channels * rank * 2)
        )
        nn.init.zeros_(self.hyper_u1[-1].weight)
        nn.init.zeros_(self.hyper_u1[-1].bias)
        nn.init.zeros_(self.hyper_u2[-1].weight)
        nn.init.zeros_(self.hyper_u2[-1].bias)

    def _get_delta_w(self, hyper, V, cond, B):
        """计算 ΔW = U @ V^T，shape: (B, in_ch, out_ch, modes1, modes2)"""
        u_flat = hyper(cond)  # (B, in_ch * rank * 2)
        u_real = u_flat[:, :self.in_channels * self.rank]
        u_imag = u_flat[:, self.in_channels * self.rank:]
        U = torch.complex(
            u_real.view(B, self.in_channels, self.rank),
            u_imag.view(B, self.in_channels, self.rank)
        )  # (B, in_ch, rank)

        # ΔW = U @ V^T: (B, in_ch, rank) @ (rank, flat_dim) → (B, in_ch, flat_dim)
        delta_w = torch.bmm(U, V.T.unsqueeze(0).expand(B, -1, -1))
        return delta_w.view(B, self.in_channels, self.out_channels, self.modes1, self.modes2)

    def forward(self, x: torch.Tensor, freq_emb: torch.Tensor = None, geom_emb: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x.float())

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        if freq_emb is not None and geom_emb is not None:
            # 频率和几何分别传入，在超网络内部融合
            cond = torch.cat([freq_emb, geom_emb], dim=-1)  # (B, freq_dim+geom_dim)
            dw1 = self._get_delta_w(self.hyper_u1, self.V1, cond, B)
            dw2 = self._get_delta_w(self.hyper_u2, self.V2, cond, B)
            # w1_eff[b] = w1 + dw1[b]
            w1_eff = self.w1.unsqueeze(0) + dw1  # (B, in, out, m1, m2)
            w2_eff = self.w2.unsqueeze(0) + dw2

            # 逐样本 einsum
            x1 = x_ft[:, :, :self.modes1, :self.modes2]   # (B, in, m1, m2)
            x2 = x_ft[:, :, -self.modes1:, :self.modes2]
            out_ft[:, :, :self.modes1, :self.modes2]  = torch.einsum("bixy,bioxy->boxy", x1, w1_eff)
            out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,bioxy->boxy", x2, w2_eff)
        else:
            # 无条件时退化为标准 FNO
            out_ft[:, :, :self.modes1, :self.modes2]  = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1)
            out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2)

        return torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)


class FNOBlockLoRA(nn.Module):
    def __init__(self, width, modes1, modes2, cond_dim: int = 32, rank: int = 4):
        super().__init__()
        self.norm    = nn.BatchNorm2d(width)
        self.spectral = SpectralConv2dLoRA(width, width, modes1, modes2, cond_dim=cond_dim, rank=rank)
        self.bypass  = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x, freq_emb=None, geom_emb=None):
        out = self.spectral(x, freq_emb, geom_emb) + self.bypass(x)
        return nn.functional.gelu(self.norm(out + x))


class FNO2dLoRA(FNOBase):
    """FNO2d with LoRA-style frequency/geometry conditioned spectral kernels."""

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 48,
        n_layers: int = 5,
        in_channels: int = 2,
        out_channels: int = 12,
        freq_dim: int = 32,
        lora_rank: int = 4,
    ):
        super().__init__(in_channels=in_channels, width=width, padding=7)

        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=freq_dim)
        self.geom_encoder = SDFGeomEncoder(n_modes=8, out_dim=64)  # 加大到64维

        self.blocks = nn.ModuleList([
            FNOBlockLoRA(width, modes1, modes2, cond_dim=freq_dim, rank=lora_rank)
            for _ in range(n_layers)
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

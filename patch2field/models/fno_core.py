import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# ==============================================================================
# Encoders
# ==============================================================================

class FourierFreqEncoder(nn.Module):
    def __init__(self, n_freqs: int = 16, out_dim: int = 32):
        super().__init__()
        scales = 2 ** torch.linspace(0, 4, n_freqs)
        self.register_buffer("scales", scales)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freqs, 64), nn.GELU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, freq: torch.Tensor) -> torch.Tensor:
        x = freq * self.scales
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.mlp(x)


class SDFGeomEncoder(nn.Module):
    def __init__(self, n_modes: int = 4, out_dim: int = 32):
        super().__init__()
        self.n_modes = n_modes
        feat_dim = n_modes * n_modes * 2  # 实部 + 虚部
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.GELU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sdf = x[:, 0:1, :, :]
        sdf_ft = torch.fft.rfft2(sdf.squeeze(1))
        spec = sdf_ft[:, :self.n_modes, :self.n_modes]
        feat = torch.cat([spec.real, spec.imag], dim=1).reshape(spec.size(0), -1)
        return self.mlp(feat)


class DirectionHead(nn.Module):
    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ==============================================================================
# Base Class (OOP)
# ==============================================================================

class FNOBase(nn.Module):
    """
    FNO 抽象基类。
    统一处理 Lift 提升、Padding 边界处理、Block 级联和头输出。
    子类只需定义 self.blocks 和 output_head 方法。
    """
    def __init__(self, in_channels: int, width: int, padding: int = 7):
        super().__init__()
        self.padding = padding
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList()

    def preprocess_input(self, x: torch.Tensor, freq: torch.Tensor):
        freq_emb = geom_emb = None
        if hasattr(self, 'freq_encoder') and freq is not None:
            freq_emb = self.freq_encoder(freq)
            geom_emb = self.geom_encoder(x)
        return x, freq_emb, geom_emb

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("子类必须实现 output_head 方法")

    def forward(self, x: torch.Tensor, freq: torch.Tensor = None) -> torch.Tensor:
        x_in, freq_emb, geom_emb = self.preprocess_input(x, freq)
        
        out = self.lift(x_in)
        if self.padding > 0:
            out = F.pad(out, [0, self.padding, 0, self.padding])
            
        for block in self.blocks:
            out = block(out, freq_emb=freq_emb, geom_emb=geom_emb)
            
        if self.padding > 0:
            out = out[..., :-self.padding, :-self.padding]
            
        return self.output_head(out)

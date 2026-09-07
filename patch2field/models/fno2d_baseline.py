"""
纯 FNO2d 基线模型（对比实验用）

与 fno2d.py 的区别：
- 无超网络、无条件注意力机制
- 频率信息展开成 2D 常数通道，和 SDF/Mask 一起作为输入
- 标准 FNO 频域卷积（无注意力）
- 输入通道：SDF + Mask + freq_channel = 3 通道
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fno_core import FNOBase

class SpectralConv2dVanilla(nn.Module):
    """标准 FNO 频域卷积，无任何条件机制。"""
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x.float())

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2)

        return torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)


class FNOBlockVanilla(nn.Module):
    """标准 FNO block：频域卷积 + bypass Conv1×1 + BN + GELU。"""
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2dVanilla(width, width, modes1, modes2)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.BatchNorm2d(width)

    def forward(self, x, freq_emb=None, geom_emb=None):
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class FNO2dBaseline(FNOBase):
    """
    纯 FNO2d 基线。
    输入处理：
        freq (B,1) 标量 → 展开成 (B,1,H,W) 常数通道
        与 SDF/Mask (B,2,H,W) 拼接 → (B,3,H,W) 作为网络输入
    """
    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 48,
        n_layers: int = 5,
        out_channels: int = 6,
    ):
        super().__init__(in_channels=3, width=width, padding=7)

        self.blocks = nn.ModuleList([
            FNOBlockVanilla(width, modes1, modes2) for _ in range(n_layers)
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def preprocess_input(self, x: torch.Tensor, freq: torch.Tensor):
        B, C, H, W = x.shape
        if freq is not None:
            freq_map = freq.view(B, 1, 1, 1).expand(B, 1, H, W)
        else:
            freq_map = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        x = torch.cat([x, freq_map], dim=1)
        return x, None, None

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


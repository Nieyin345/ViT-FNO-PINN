"""纯 FNO2d 基线模型（对比实验用）。

与 models/fno2d.py 的区别：
- 无超网络、无条件注意力机制
- 条件信息展开成常数通道，和 SDF/Mask 一起作为输入（"空间通道广播"式条件化）
- 标准 FNO 频域卷积（无注意力）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import FNOBase
from core.config import ModelConfig
from core.registry import register


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

    def forward(self, x, cond=None):
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


@register("fno_baseline")
class FNO2dBaseline(FNOBase):
    """纯 FNO2d 基线。

    输入处理：freq (B,K) 条件向量 → 展开成 (B,K,H,W)，与 SDF/Mask (B,2,H,W) 拼接。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.in_channels = cfg.in_channels
        self.cond_channels = int(cfg.cond_channels)
        # lift 实际接收 in_channels + cond_channels（基座按 cfg.in_channels 建的，这里重建）
        self.lift = nn.Sequential(
            nn.Conv2d(self.in_channels + self.cond_channels, cfg.width, kernel_size=1),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            FNOBlockVanilla(cfg.width, cfg.modes1, cfg.modes2) for _ in range(cfg.n_layers)
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(cfg.width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, cfg.out_channels, kernel_size=1),
        )

    def _broadcast_condition(self, x, freq):
        B, _, H, W = x.shape
        if freq is None:
            return torch.zeros(B, self.cond_channels, H, W, device=x.device, dtype=x.dtype)
        # freq 已由 FNOBase.forward 里的 normalize_freq 规范成 (B, 2)
        n_cond = freq.shape[1]
        if n_cond < self.cond_channels:
            pad = torch.zeros(B, self.cond_channels - n_cond, device=x.device, dtype=x.dtype)
            freq = torch.cat([freq, pad], dim=1)
        cond = freq[:, :self.cond_channels]
        return cond.view(B, self.cond_channels, 1, 1).expand(B, self.cond_channels, H, W)

    def build_input(self, x, freq):
        return torch.cat([x, self._broadcast_condition(x, freq)], dim=1)

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

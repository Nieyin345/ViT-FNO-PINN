"""
FNO2d with FiLM conditioning for multi-plane scattered field prediction.

输入：SDF+Mask (B,2,H,W) + 频率 (B,1) + z位置 (B,1，归一化到[0,1]，对应2~4mm)
输出：该 z 位置的散射场 (B,12,H,W)

条件编码：
  freq_emb(32) + z_emb(32) = 64维
  用 FiLM (scale+shift) 调制每个 FNO block 的输出特征图

结构与 fno2d_lora.py 一致：WIDTH=64, MODES=12, N_LAYERS=5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from .fno_core import FourierFreqEncoder


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: scale + shift 调制特征图。"""
    def __init__(self, cond_dim: int, n_channels: int):
        super().__init__()
        self.scale_net = nn.Linear(cond_dim, n_channels)
        self.shift_net = nn.Linear(cond_dim, n_channels)
        # 初始化为恒等变换（scale=1, shift=0）
        nn.init.zeros_(self.scale_net.weight)
        nn.init.ones_(self.scale_net.bias)
        nn.init.zeros_(self.shift_net.weight)
        nn.init.zeros_(self.shift_net.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), cond: (B, cond_dim)
        scale = self.scale_net(cond)[:, :, None, None]  # (B, C, 1, 1)
        shift = self.shift_net(cond)[:, :, None, None]  # (B, C, 1, 1)
        return x * scale + shift


from .fno2d_baseline import SpectralConv2dVanilla


class FNOBlockFiLM(nn.Module):
    """FNO block + FiLM 调制（作用于 spectral+bypass 之后、残差之前）。"""
    def __init__(self, width: int, modes1: int, modes2: int, cond_dim: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(width)
        self.spectral = SpectralConv2dVanilla(width, width, modes1, modes2)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.film = FiLM(cond_dim, width)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # spectral + bypass
        h = self.spectral(x) + self.bypass(x)
        # FiLM 调制
        h = self.film(h, cond)
        # 残差 + norm + 激活
        return F.gelu(self.norm(h + x))


class FNO2dMultiPlane(nn.Module):
    """
    FNO2d with FiLM conditioning for multi-plane prediction.

    Args:
        modes1, modes2: FNO 截断模式数
        width: 特征通道数
        n_layers: FNO block 数量
        in_channels: 输入通道数（SDF+Mask=2）
        out_channels: 输出通道数（12个场分量）
        freq_emb_dim: 频率编码维度
        z_emb_dim: z位置编码维度
    """
    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        n_layers: int = 5,
        in_channels: int = 2,
        out_channels: int = 12,
        freq_emb_dim: int = 32,
        z_emb_dim: int = 32,
    ):
        super().__init__()
        self.padding = 7
        cond_dim = freq_emb_dim + z_emb_dim  # 64

        # 条件编码器
        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=freq_emb_dim)
        self.z_encoder = FourierFreqEncoder(n_freqs=16, out_dim=z_emb_dim)

        # 输入提升
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1),
            nn.GELU(),
        )

        # FNO blocks with FiLM
        self.blocks = nn.ModuleList([
            FNOBlockFiLM(width, modes1, modes2, cond_dim)
            for _ in range(n_layers)
        ])

        # 输出投影
        self.proj = nn.Sequential(
            nn.Conv2d(width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        freq: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, 2, H, W)  SDF + Mask
            freq: (B, 1)        归一化频率 [0,1]，对应 8~12 GHz
            z:    (B, 1)        归一化 z 位置 [0,1]，对应 2~4 mm

        Returns:
            (B, 12, H, W) 散射场
        """
        # 条件编码并拼接
        freq_emb = self.freq_encoder(freq)   # (B, 32)
        z_emb = self.z_encoder(z)            # (B, 32)
        cond = torch.cat([freq_emb, z_emb], dim=-1)  # (B, 64)

        # 提升 + padding
        x = self.lift(x)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        # FNO blocks
        for block in self.blocks:
            x = block(x, cond)

        # 去 padding + 投影
        x = x[..., :-self.padding, :-self.padding]
        return self.proj(x)

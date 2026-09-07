"""FNO2d with Frequency-Conditioned Spectral Attention."""

import torch
import torch.nn as nn
from .spectral_conv2d import SpectralConv2d
from .fno_core import FNOBase, FourierFreqEncoder, SDFGeomEncoder, DirectionHead

class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2, cond_dim: int = 32):
        super().__init__()
        self.norm = nn.BatchNorm2d(width)
        self.spectral = SpectralConv2d(width, width, modes1, modes2, cond_dim=cond_dim)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x, freq_emb=None, geom_emb=None):
        out = self.spectral(x, freq_emb, geom_emb) + self.bypass(x)
        return nn.functional.gelu(self.norm(out))


class FNO2d(FNOBase):
    """FNO2d with frequency-conditioned spectral attention."""

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 48,
        n_layers: int = 5,
        in_channels: int = 2,
        out_channels: int = 6,
        freq_dim: int = 32,
    ):
        super().__init__(in_channels=in_channels, width=width, padding=7)
        assert out_channels % 2 == 0, "out_channels 必须是偶数（实部+虚部成对）"
        self.n_heads = out_channels // 2
        # 如果是12通道（电磁场），用自回归输出头；否则用普通输出头
        self.use_em_heads = (out_channels == 12)

        self.blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2, cond_dim=freq_dim) for _ in range(n_layers)
        ])

        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=freq_dim)
        self.geom_encoder = SDFGeomEncoder(n_modes=4, out_dim=freq_dim)

        self.shared_proj = nn.Sequential(
            nn.Conv2d(width, 64, kernel_size=1),
            nn.GELU(),
        )
        if self.use_em_heads:
            # 12通道：先预测E场，再用E场生成H场
            self.e_heads = nn.ModuleList([
                DirectionHead(in_channels=64) for _ in range(3)  # Ex, Ey, Ez
            ])
            self.h_heads = nn.ModuleList([
                DirectionHead(in_channels=64 + 6) for _ in range(3)  # Hx, Hy, Hz
            ])
        else:
            # 6通道或其他：普通输出头
            self.heads = nn.ModuleList([
                DirectionHead(in_channels=64) for _ in range(self.n_heads)
            ])

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_proj(x)
        if self.use_em_heads:
            e_pred = torch.cat([h(shared) for h in self.e_heads], dim=1)
            h_input = torch.cat([shared, e_pred], dim=1)
            h_pred = torch.cat([h(h_input) for h in self.h_heads], dim=1)
            return torch.cat([e_pred, h_pred], dim=1)
        else:
            return torch.cat([h(shared) for h in self.heads], dim=1)


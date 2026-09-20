"""FNO2d with Frequency-Conditioned Spectral Attention（主模型 fno_attention）。

条件化：编码器嵌入 —— FourierFreqEncoder(freq) + SDFGeomEncoder(x) → Cond。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import Cond, FNOBase
from core.config import ModelConfig
from core.encoders import DirectionHead, FourierFreqEncoder, SDFGeomEncoder
from core.registry import register
from core.spectral import SpectralConv2d


class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2, cond_dim: int = 32, spectral_cfg=None):
        super().__init__()
        self.norm = nn.BatchNorm2d(width)
        self.spectral = SpectralConv2d(width, width, modes1, modes2,
                                       cond_dim=cond_dim, cfg=spectral_cfg)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x, cond=None):
        out = self.spectral(x, cond) + self.bypass(x)
        return F.gelu(self.norm(out + x))


@register("fno_attention")
class FNO2d(FNOBase):
    """FNO2d with frequency-conditioned spectral attention。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        out_channels = cfg.out_channels

        # 输出头模式由 cfg.head 决定（原先靠全局开关 USE_SINGLE_OUTPUT_HEAD）
        self.use_em_heads = (cfg.head == "em_split")
        if self.use_em_heads and out_channels != 12:
            raise ValueError(
                f"head='em_split' 需要 out_channels=12（3 个 E 头 + 3 个 H 头），"
                f"得到 {out_channels}。请改用 head='single' 或修正 out_channels。"
            )

        self.blocks = nn.ModuleList([
            FNOBlock(cfg.width, cfg.modes1, cfg.modes2,
                     cond_dim=cfg.freq_dim, spectral_cfg=cfg.spectral)
            for _ in range(cfg.n_layers)
        ])

        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=cfg.freq_dim)
        self.geom_encoder = SDFGeomEncoder(n_modes=4, out_dim=cfg.freq_dim)

        self.shared_proj = nn.Sequential(
            nn.Conv2d(cfg.width, 64, kernel_size=1),
            nn.GELU(),
        )
        if self.use_em_heads:
            self.e_heads = nn.ModuleList([DirectionHead(in_channels=64) for _ in range(3)])
            # H 头显式依赖 E 的预测
            self.h_heads = nn.ModuleList([DirectionHead(in_channels=64 + 6) for _ in range(3)])
        else:
            self.single_head = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(64, out_channels, kernel_size=1),
            )

    def build_cond(self, x, freq):
        if freq is None:
            return None
        return Cond(freq_emb=self.freq_encoder(freq), geom_emb=self.geom_encoder(x))

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_proj(x)
        if self.use_em_heads:
            e_pred = torch.cat([h(shared) for h in self.e_heads], dim=1)
            h_input = torch.cat([shared, e_pred], dim=1)
            h_pred = torch.cat([h(h_input) for h in self.h_heads], dim=1)
            return torch.cat([e_pred, h_pred], dim=1)
        return self.single_head(shared)

"""FNO2d + FiLM 条件化模型（对比实验用）。

与主模型的区别：
- 有频率和几何条件编码器（和主模型相同）
- 但条件信息通过 FiLM（Feature-wise Linear Modulation）注入
  即对频域特征图做 scale + shift，而不是动态调制投影矩阵
- 无注意力机制，无超网络生成 ΔW

用途：消融实验，证明"动态投影矩阵"比"FiLM 调制"更有效。

重构修正：原先 `freq_encoder` / `geom_encoder` 的 out_dim 写死 32（收了 freq_dim
却没用），且 FiLM 的 `film_net` 输入维度写死 64。现在全部跟随 `cfg.freq_dim`。
默认 freq_dim=32 时行为与旧版一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import Cond, FNOBase
from core.config import ModelConfig
from core.encoders import DirectionHead, FourierFreqEncoder, SDFGeomEncoder
from core.registry import register


class SpectralConv2dFiLM(nn.Module):
    """标准 FNO 频域卷积 + FiLM 条件调制。"""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int,
                 freq_dim: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

        # FiLM 生成网络：freq_emb + geom_emb -> scale, shift for out_channels
        cond_dim = 2 * freq_dim
        self.film_net = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.GELU(),
            nn.Linear(256, out_channels * 2),  # scale + shift
        )
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        from core.base import Cond
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

        if freq_emb is not None and geom_emb is not None:
            c = torch.cat([freq_emb, geom_emb], dim=-1)
            film_params = self.film_net(c)
            scale = film_params[:, :self.out_channels].view(B, self.out_channels, 1, 1)
            shift = film_params[:, self.out_channels:].view(B, self.out_channels, 1, 1)
            out_ft = out_ft * (1.0 + scale).to(out_ft.dtype) + shift.to(out_ft.dtype)

        return torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)


class FNOBlockFiLM(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int, freq_dim: int = 32):
        super().__init__()
        self.spectral = SpectralConv2dFiLM(width, width, modes1, modes2, freq_dim=freq_dim)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.BatchNorm2d(width)

    def forward(self, x, cond=None):
        return F.gelu(self.norm(self.spectral(x, cond) + self.bypass(x)))


@register("fno_film")
class FNO2dFiLM(FNOBase):
    """FNO2d + FiLM 条件化。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        self.blocks = nn.ModuleList([
            FNOBlockFiLM(cfg.width, cfg.modes1, cfg.modes2, freq_dim=cfg.freq_dim)
            for _ in range(cfg.n_layers)
        ])

        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=cfg.freq_dim)
        self.geom_encoder = SDFGeomEncoder(n_modes=4, out_dim=cfg.freq_dim)
        self.shared_proj = nn.Sequential(
            nn.Conv2d(cfg.width, 64, kernel_size=1), nn.GELU()
        )
        self.heads = nn.ModuleList([
            DirectionHead(in_channels=64) for _ in range(cfg.n_heads)
        ])

    def build_cond(self, x, freq):
        if freq is None:
            return None
        # freq = (B,2) 整列传入：第 0 列走傅里叶特征，第 1 列(极化)作为标量并入。
        return Cond(freq_emb=self.freq_encoder(freq), geom_emb=self.geom_encoder(x))

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_proj(x)
        return torch.cat([h(shared) for h in self.heads], dim=1)

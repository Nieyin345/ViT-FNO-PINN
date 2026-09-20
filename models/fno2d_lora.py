"""FNO2d with LoRA-style frequency-conditioned spectral kernels。

核心思路：
  用频率/几何条件通过低秩分解直接调制 FNO 的频域卷积核 w1/w2，
  而不是通过注意力矩阵间接调制。

  ΔW = U(freq_emb) @ V^T，rank r 很小（默认4），参数量几乎不增加。
  W_eff = W + ΔW，直接影响最核心的频域变换。

重构变更：
  - 原先本文件**自己定义了一份** `SDFGeomEncoder`（n_modes=8、用 SDF+Mask、
    幅值+相位、hidden=128），与 core/encoders.py 里的同名类实现不同。
    现在统一从 core.encoders 导入，参数取原 LoRA 变体的取值。
  - 超网络的 `fused_dim` 原先写死 `cond_dim + 64`，现在由 geom 嵌入的实际维度推出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import Cond, FNOBase
from core.config import ModelConfig
from core.encoders import FourierFreqEncoder, SDFGeomEncoder
from core.registry import register

# LoRA 变体的几何编码器参数（原 fno2d_lora.py 本地版行为）
_LORA_GEOM_MODES = 8
_LORA_GEOM_DIM = 64
_LORA_GEOM_HIDDEN = 128


class SpectralConv2dLoRA(nn.Module):
    """
    FNO 频域卷积 + LoRA 风格的条件调制。

    W_eff = W + U(cond) @ V^T
    U: (B, in_ch, r)  由条件生成
    V: (out_ch * modes1 * modes2, r)  可学习静态矩阵
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int,
                 cond_dim: int = 32, rank: int = 4, fused_dim: int = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.rank = rank

        scale = 1 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

        flat_dim = out_channels * modes1 * modes2
        self.V1 = nn.Parameter(torch.randn(flat_dim, rank, dtype=torch.cfloat) * 0.01)
        self.V2 = nn.Parameter(torch.randn(flat_dim, rank, dtype=torch.cfloat) * 0.01)

        # 超网络接收 concat([freq_emb, geom_emb])，内部融合后生成 U
        if fused_dim is None:
            fused_dim = cond_dim + _LORA_GEOM_DIM
        self.hyper_u1 = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.GELU(),
            nn.Linear(128, in_channels * rank * 2),
        )
        self.hyper_u2 = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.GELU(),
            nn.Linear(128, in_channels * rank * 2),
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
            u_imag.view(B, self.in_channels, self.rank),
        )  # (B, in_ch, rank)
        delta_w = torch.bmm(U, V.T.unsqueeze(0).expand(B, -1, -1))
        return delta_w.view(B, self.in_channels, self.out_channels, self.modes1, self.modes2)

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        from core.base import Cond
        freq_emb = cond.freq_emb if isinstance(cond, Cond) else None
        geom_emb = cond.geom_emb if isinstance(cond, Cond) else None

        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x.float())

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        if freq_emb is not None and geom_emb is not None:
            c = torch.cat([freq_emb, geom_emb], dim=-1)
            dw1 = self._get_delta_w(self.hyper_u1, self.V1, c, B)
            dw2 = self._get_delta_w(self.hyper_u2, self.V2, c, B)
            w1_eff = self.w1.unsqueeze(0) + dw1
            w2_eff = self.w2.unsqueeze(0) + dw2

            x1 = x_ft[:, :, :self.modes1, :self.modes2]
            x2 = x_ft[:, :, -self.modes1:, :self.modes2]
            out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,bioxy->boxy", x1, w1_eff)
            out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,bioxy->boxy", x2, w2_eff)
        else:
            # 无条件时退化为标准 FNO
            out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1)
            out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2)

        return torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)


class FNOBlockLoRA(nn.Module):
    def __init__(self, width, modes1, modes2, cond_dim: int = 32, rank: int = 4,
                 fused_dim: int = None):
        super().__init__()
        self.norm = nn.BatchNorm2d(width)
        self.spectral = SpectralConv2dLoRA(width, width, modes1, modes2,
                                           cond_dim=cond_dim, rank=rank,
                                           fused_dim=fused_dim)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x, cond=None):
        out = self.spectral(x, cond) + self.bypass(x)
        return F.gelu(self.norm(out + x))


@register("fno_lora")
class FNO2dLoRA(FNOBase):
    """FNO2d with LoRA-style frequency/geometry conditioned spectral kernels。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)

        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=cfg.freq_dim)
        self.geom_encoder = SDFGeomEncoder(
            n_modes=_LORA_GEOM_MODES, out_dim=_LORA_GEOM_DIM,
            channels=(0, 1), amp_phase=True, hidden=_LORA_GEOM_HIDDEN,
        )
        fused_dim = cfg.freq_dim + _LORA_GEOM_DIM

        self.blocks = nn.ModuleList([
            FNOBlockLoRA(cfg.width, cfg.modes1, cfg.modes2,
                         cond_dim=cfg.freq_dim, rank=cfg.lora_rank, fused_dim=fused_dim)
            for _ in range(cfg.n_layers)
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(cfg.width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, cfg.out_channels, kernel_size=1),
        )

    def build_cond(self, x, freq):
        if freq is None:
            return None
        # freq = (B,2) 整列传入：第 0 列走傅里叶特征，第 1 列(极化)作为标量并入。
        return Cond(freq_emb=self.freq_encoder(freq), geom_emb=self.geom_encoder(x))

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

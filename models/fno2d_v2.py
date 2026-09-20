"""FNO2dV2 —— 强化纯 FNO 特征的条件化主模型。

设计理念（格林函数视角）：
  电磁场 = 贴片电流与格林函数 G(r,r'; f) 的卷积，频域内对应核乘法。
  FNO 的频域卷积核 w1/w2 正是格林函数在傅里叶域的学习表示。

v2 相比旧主模型（fno_attention）的关键改动：
  1. 去掉"超网络调制注意力投影矩阵"（已被消融证明有害），回归纯 FNO 频域卷积。
  2. 频率信息以【空间正弦嵌入通道】注入输入：
       sin(k·x), cos(k·x), sin(k·y), cos(k·y)，其中 k 由工作频率决定
     —— FNO 的频域卷积可直接加工这些携带频率信息的空间模式。
  3. 频率/极化/几何通过条件向量【直接调制频域卷积核】：
       - "film": 对频域特征图做 per-channel scale+shift（轻量、稳定）
       - "lora": 低秩扰动 ΔW = U(cond)@V^T 直接加到 w1/w2（更强表达）
     —— 让格林函数本身随频率变化（宽频带学习），极化走独立编码分支。
  4. 条件向量 = [freq_emb, pol_emb, geom_emb] 拼接，频率与极化不再平均混叠。

输入:  (B, 2, H, W) SDF+Mask  → 内部拼接 4 个正弦通道 → (B, 6, H, W)
条件:  freq (B, 2) = [归一化频率, 极化]
输出:  (B, 12, H, W) Ex/Ey/Ez/Hx/Hy/Hz 实虚部

重构变更：
  - **改为继承 FNOBase**。原先它不继承基座，把 lift/pad/循环/crop 抄了一遍
    （与 fno_core.FNOBase 重复约 15 行），因为它的块签名 `block(x, cond)` 与
    基座固定的 `block(x, freq_emb=, geom_emb=)` 不兼容。统一为 `Cond` 后可以复用。
  - `PROJ_DROPOUT` 原先在 `__init__` 里 import config 读取，现在从 `cfg.dropout` 传入。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import Cond, FNOBase
from core.config import ModelConfig
from core.encoders import FourierFreqEncoder, SDFGeomEncoder
from core.registry import register

# 频率范围（与 data/dataset.py 保持一致）
_F_MIN = 8.0
_F_MAX = 12.0


class SpectralConv2dV2(nn.Module):
    """标准 FNO 频域卷积 + 可选的核条件化。

    kernel_cond: "film" | "lora" | "none"
    """

    def __init__(self, in_channels, out_channels, modes1, modes2,
                 cond_dim=80, kernel_cond="film", lora_rank=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.kernel_cond = kernel_cond

        scale = 1 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

        if kernel_cond == "film":
            # 对频域特征图做 per-channel scale + shift（初值 0 → 恒等）
            self.film_net = nn.Sequential(
                nn.Linear(cond_dim, 128), nn.GELU(),
                nn.Linear(128, out_channels * 2),
            )
            nn.init.zeros_(self.film_net[-1].weight)
            nn.init.zeros_(self.film_net[-1].bias)
        elif kernel_cond == "lora":
            flat_dim = out_channels * modes1 * modes2
            self.rank = lora_rank
            self.V1 = nn.Parameter(torch.randn(flat_dim, lora_rank, dtype=torch.cfloat) * 0.01)
            self.V2 = nn.Parameter(torch.randn(flat_dim, lora_rank, dtype=torch.cfloat) * 0.01)
            self.hyper_u1 = nn.Sequential(
                nn.Linear(cond_dim, 128), nn.GELU(),
                nn.Linear(128, in_channels * lora_rank * 2),
            )
            self.hyper_u2 = nn.Sequential(
                nn.Linear(cond_dim, 128), nn.GELU(),
                nn.Linear(128, in_channels * lora_rank * 2),
            )
            nn.init.zeros_(self.hyper_u1[-1].weight)
            nn.init.zeros_(self.hyper_u1[-1].bias)
            nn.init.zeros_(self.hyper_u2[-1].weight)
            nn.init.zeros_(self.hyper_u2[-1].bias)

    def _lora_delta_w(self, hyper, V, cond, B):
        u_flat = hyper(cond)  # (B, in*rank*2)
        u_real = u_flat[:, :self.in_channels * self.rank]
        u_imag = u_flat[:, self.in_channels * self.rank:]
        U = torch.complex(u_real.view(B, self.in_channels, self.rank),
                          u_imag.view(B, self.in_channels, self.rank))
        delta = torch.bmm(U, V.T.unsqueeze(0).expand(B, -1, -1))
        return delta.view(B, self.in_channels, self.out_channels, self.modes1, self.modes2)

    def forward(self, x, cond=None):
        c = cond.fused() if isinstance(cond, Cond) else cond

        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x.float())

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        if c is not None and self.kernel_cond == "lora":
            dw1 = self._lora_delta_w(self.hyper_u1, self.V1, c, B)
            dw2 = self._lora_delta_w(self.hyper_u2, self.V2, c, B)
            w1_eff = self.w1.unsqueeze(0) + dw1
            w2_eff = self.w2.unsqueeze(0) + dw2
            out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
                "bixy,bioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], w1_eff)
            out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
                "bixy,bioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], w2_eff)
        else:
            out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1)
            out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2)

        if c is not None and self.kernel_cond == "film":
            fp = self.film_net(c)  # (B, 2*out)
            scale = fp[:, :self.out_channels].view(B, self.out_channels, 1, 1)
            shift = fp[:, self.out_channels:].view(B, self.out_channels, 1, 1)
            out_ft = out_ft * (1.0 + scale).to(out_ft.dtype) + shift.to(out_ft.dtype)

        return torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)


class FNOBlockV2(nn.Module):
    def __init__(self, width, modes1, modes2, cond_dim=80, kernel_cond="film", lora_rank=4):
        super().__init__()
        self.norm = nn.BatchNorm2d(width)
        self.spectral = SpectralConv2dV2(width, width, modes1, modes2,
                                         cond_dim=cond_dim, kernel_cond=kernel_cond,
                                         lora_rank=lora_rank)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x, cond=None):
        out = self.spectral(x, cond) + self.bypass(x)
        return F.gelu(self.norm(out + x))


@register("fno_v2")
class FNO2dV2(FNOBase):
    """FNO2dV2 主模型（强化 FNO + 正弦频率输入 + 核条件化）。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.use_sine_input = cfg.use_sine_input
        self.use_pol_cond = cfg.use_pol_cond
        # 正弦嵌入通道数：x/y 各 sin/cos = 4
        self.n_sine = 4 if cfg.use_sine_input else 0

        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=cfg.freq_dim)
        self.geom_encoder = SDFGeomEncoder(n_modes=4, out_dim=cfg.freq_dim)
        if cfg.use_pol_cond:
            self.pol_mlp = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 16))
        self.cond_dim = cfg.freq_dim + (16 if cfg.use_pol_cond else 0) + cfg.freq_dim

        # lift 实际接收 in_channels + n_sine（基座按 cfg.in_channels 建的，这里重建）
        self.lift = nn.Sequential(
            nn.Conv2d(cfg.in_channels + self.n_sine, cfg.width, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            FNOBlockV2(cfg.width, cfg.modes1, cfg.modes2, cond_dim=self.cond_dim,
                       kernel_cond=cfg.kernel_cond, lora_rank=cfg.lora_rank)
            for _ in range(cfg.n_layers)
        ])
        self.dropout = nn.Dropout2d(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Conv2d(cfg.width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, cfg.out_channels, kernel_size=1),
        )

    def _sine_embedding(self, x, freq):
        """频率相关的空间正弦嵌入：sin/cos(k·x/y)，k 由工作频率决定。"""
        B, _, H, W = x.shape
        f_ghz = _F_MIN + (_F_MAX - _F_MIN) * freq[:, 0:1]          # (B,1) 8~12 GHz
        phase = (math.pi / 10.0) * f_ghz                           # (B,1) rad
        y_n = torch.linspace(-1.0, 1.0, H, device=x.device).view(1, H, 1).expand(1, H, W)
        x_n = torch.linspace(-1.0, 1.0, W, device=x.device).view(1, 1, W).expand(1, H, W)
        p = phase.view(B, 1, 1)
        return torch.stack([
            torch.sin(p * x_n), torch.cos(p * x_n),
            torch.sin(p * y_n), torch.cos(p * y_n),
        ], dim=1).to(x.dtype)                                       # (B,4,H,W)

    def build_input(self, x, freq):
        if not self.use_sine_input:
            return x
        if freq is None:
            raise ValueError("use_sine_input=True 时必须提供 freq（正弦嵌入依赖频率）")
        return torch.cat([x, self._sine_embedding(x, freq)], dim=1)

    def build_cond(self, x, freq):
        if freq is None:
            return None
        freq_emb = self.freq_encoder(freq[:, 0:1])
        if self.use_pol_cond:
            pol_emb = self.pol_mlp(freq[:, 1:2])
        else:
            pol_emb = None
        geom_emb = self.geom_encoder(x)
        return Cond(freq_emb=freq_emb, pol_emb=pol_emb, geom_emb=geom_emb)

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(x))

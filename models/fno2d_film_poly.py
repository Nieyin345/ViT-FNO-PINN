# -*- coding: utf-8 -*-
"""fno_film_poly：频率只通过"K 阶切比雪夫多项式基"进入输出，主体网络不看频率。

动机（PLAN.md 第 10 节）：去掉线性相位后，真值随频率是光滑的（用相邻两个已见频点真值插值，
保留频点 geom=0.40，而不去相位的邻居平均是 1.48）。但普通 FiLM 网络在 8 个训练频点上拟合完
就不再插值，未见频点落到"抄邻居"水平。这里用结构强制"频率方向光滑"：

    u(x; geom, f) = Σ_{k<K} C_k(x; geom) · T_k(2 f_norm − 1)

C_k 是主体网络输出的 K 组系数场（主体看到的频率被固定为常数），T_k 为切比雪夫多项式。
K 小于训练频点数时是真正的低阶多项式插值；频率泛化由基的光滑性保证，而不是靠网络"自觉"。
MODEL_EXTRA: poly_k（默认 4），free_body（默认 True：主体不看频率），
poly_basis_mode（默认 chebyshev；constant 用于纯频率基消融，参数形状保持不变）。
"""
import torch

from core.config import ModelConfig
from core.base import normalize_freq
from core.encoders import DirectionHead
from core.registry import register
import torch.nn as nn
from .fno2d_film import FNO2dFiLM


@register("fno_film_poly")
class FNO2dFiLMPoly(FNO2dFiLM):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        if self.use_em_heads:
            raise ValueError("fno_film_poly 暂不支持 head='em_split'")
        ex = cfg.extra or {}
        self.K = int(ex.get("poly_k", 4))
        if self.K < 1:
            raise ValueError(f"poly_k must be >= 1, got {self.K}")
        self.free_body = bool(ex.get("free_body", True))
        self.poly_basis_mode = str(ex.get("poly_basis_mode", "chebyshev")).lower()
        if self.poly_basis_mode not in ("chebyshev", "constant"):
            raise ValueError(
                f"poly_basis_mode must be 'chebyshev' or 'constant', got {self.poly_basis_mode!r}"
            )
        self.n_out = 2 * len(self.heads)            # 12
        hid = int(ex.get("poly_head_hidden", 0))
        if hid > 0:
            # 共享头：一个 1x1 MLP 直接输出 K*12 通道（比 K*6 个独立头快得多）
            self.heads = nn.ModuleList([nn.Sequential(
                nn.Conv2d(64, hid, kernel_size=1), nn.GELU(),
                nn.Conv2d(hid, self.n_out * self.K, kernel_size=1))])
        else:
            self.heads = nn.ModuleList([DirectionHead(in_channels=64) for _ in range(len(self.heads) * self.K)])

    def _basis(self, f_norm):
        if self.poly_basis_mode == "constant":
            # Keep all K coefficient banks and their parameter count, but remove
            # frequency dependence from the basis; every bank remains active.
            return f_norm.new_full((f_norm.shape[0], self.K), 1.0 / self.K)
        t = 2.0 * f_norm - 1.0
        T = [torch.ones_like(t), t]
        for _ in range(2, self.K):
            T.append(2 * t * T[-1] - T[-2])
        return torch.stack(T[: self.K], dim=1)       # (B,K)

    def forward(self, x, freq=None):
        freq = normalize_freq(freq, x.shape[0])
        fb = freq.clone()
        if self.free_body:
            fb[:, 0] = 0.5
        out = super().forward(x, fb)                # (B, K*12, H, W)
        B, _, H, W = out.shape
        out = out.view(B, self.K, self.n_out, H, W)
        w = self._basis(freq[:, 0]).to(out.dtype)
        return (out * w[:, :, None, None, None]).sum(dim=1)

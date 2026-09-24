# -*- coding: utf-8 -*-
"""WIME 条件化（适配周期超表面垂直入射）——fno_wime。

## 来源

WIME = Wave-Informed element-wise Multiplicative Encoding，
出自 arXiv 2408.02971《Generalizing Deep Surrogate Solvers for Broadband
Electromagnetic Field Prediction at Unseen Wavelengths》。该工作针对的正是
「训练数据只有稀疏离散波长、要在未见波长上泛化」——与本项目同构。

它的两个概念：
  1. **Spectral Consistency（谱一致性）**：条件嵌入的**空间频率结构**应与目标场
     的空间频率结构一致（依据 ν = 1/λ：短波长 → 场更细）。
  2. **WIME**：把条件嵌入做成 **2D 空间图**（而非标量），只经 1×1 卷积
     （**不做任何空间混合**，否则破坏其空间频率结构），在**每一层**与特征图
     逐元素相乘注入。

## 与原文的差异（必须记录的适配，否则物理错误）

原文的波先验是**斜入射平面波相位斜坡**：
    W_x = exp(j·(2π/λ)·x·Δl),  W_y = exp(j·(2π/λ)·y·Δl)

本项目是**周期边界 + 垂直入射**，直接照搬会引入物理上不存在的横向相位梯度。
在周期结构中横向波数被元胞尺寸量子化为 k = 2π(m,n)/L，**与 λ 无关**；真正随
λ 变化的是各 Floquet 模式的**倏逝衰减**：

    W(x,y;λ) = Σ_{m,n} exp(−κ_mn(λ)·d(λ)) · exp(j·2π(m·x̂ + n·ŷ))
    κ_mn(λ) = sqrt( max( (2πρ/L)² − (2π/λ)² , 0 ) ),  ρ = sqrt(m²+n²)

d(λ) = 0.1λ 是观测面深度。频率升高 → k0 升 → κ 降 → 倏逝模衰减更少 →
场的高空间频率成分更丰富。

**这个机制在本项目数据上实测确认**（tools 脚本 verify_spectral_content.py）：
    freq     k_c     高阶模占比
    8.0G   0.0197      0.05%
   12.0G   0.0392      0.83%
谱质心随频率单调上升、高阶模占比涨 16 倍 —— 谱一致性前提成立。

注意：横向相位 2π·m·j/(W−1) 里 x_j/L = j/(W−1)，**与 L 无关**，因为周期边界
把几何固定在"1 个元胞 = 1 个周期"上。λ 只从衰减权重进入，这正是物理。

## 与 fno_film 的关系

fno_wime = fno_film **逐位保留**（含 FiLM 的 scale/shift、骨架、输出头），
**额外加一条 WIME 通路**。所以 A/B 是干净的单变量对照：
    fno_wime(use_wime=True)  vs  fno_wime(use_wime=False)  vs  fno_film
前者问"WIME 有没有增益"，后者证明"关掉 WIME 时与 fno_film 等价"。

extra 键：
  wime_max_mode   模式阶数上限（m,n ∈ [−M,M]，默认 2 → 25 个模式）
  wime_cell_mm    元胞尺寸 mm（默认 15.0）
  wime_d_over_lam 观测面深度（单位 λ，默认 0.1，须与数据集一致）
  wime_use        是否启用 WIME 通路（默认 True）
"""

from typing import Optional

import torch
import torch.nn as nn

from core.base import Cond, FNOBase
from core.config import ModelConfig
from core.encoders import DirectionHead, FourierFreqEncoder, SDFGeomEncoder
from core.registry import register

_C_MM_S = 3e11          # 光速，mm/s（λ_mm = c/f，f 用 Hz）


class WavePrior(nn.Module):
    """按频率生成 Floquet 倏逝衰减加权的空间波图样 (B,2,H,W)。

    横向相位是固定的几何量（周期边界，模式 m 在元胞内绕 m 圈），
    λ 只通过衰减权重 exp(−κ_mn·d) 进入 —— 这是物理上正确的λ依赖。
    """

    def __init__(self, max_mode: int = 2, cell_mm: float = 15.0,
                 d_over_lam: float = 0.1):
        super().__init__()
        self.max_mode = int(max_mode)
        self.cell_mm = float(cell_mm)
        self.d_over_lam = float(d_over_lam)

    def forward(self, freq_norm: torch.Tensor, H: int, W: int,
                padding: int, device, dtype) -> torch.Tensor:
        """freq_norm: (B,1) 归一化频率 -> (B,2,H+pad,W+pad) [cos, sin]"""
        B = freq_norm.shape[0]
        f_ghz = 8.0 + 4.0 * freq_norm.reshape(B)          # (B,)
        lam_mm = _C_MM_S / (f_ghz * 1e9)                  # (B,)
        d_mm = self.d_over_lam * lam_mm                   # (B,)

        # 横向相位网格：周期边界下 x_j/L = j/(W-1)，与 L 无关
        jx = torch.arange(W, device=device, dtype=torch.float32) / max(W - 1, 1)
        iy = torch.arange(H, device=device, dtype=torch.float32) / max(H - 1, 1)

        M = self.max_mode
        re = torch.zeros(B, H, W, device=device, dtype=torch.float32)
        im = torch.zeros(B, H, W, device=device, dtype=torch.float32)
        k0 = 2.0 * torch.pi / lam_mm                      # (B,) rad/mm
        kcell = 2.0 * torch.pi / self.cell_mm

        for m in range(-M, M + 1):
            for n in range(-M, M + 1):
                rho = (m * m + n * n) ** 0.5
                if rho == 0.0:
                    # (0,0)：传播模，横向无变化，权重 1
                    re = re + 1.0
                    continue
                kt = kcell * rho
                # κ = sqrt(max(kt² − k0², 0))：倏逝时为正，传播时为 0
                kap = torch.sqrt(torch.clamp(kt * kt - k0 * k0, min=0.0))
                w = torch.exp(-kap * d_mm)                # (B,)
                ph = 2.0 * torch.pi * (m * jx.reshape(1, -1) + n * iy.reshape(-1, 1))
                re = re + w.reshape(B, 1, 1) * torch.cos(ph).reshape(1, H, W)
                im = im + w.reshape(B, 1, 1) * torch.sin(ph).reshape(1, H, W)

        # 逐样本标准化（避免量级压过其它通路）
        out = torch.stack([re, im], dim=1)                # (B,2,H,W)
        scale = out.flatten(2).std(dim=2, keepdim=True).clamp_min(1e-6)
        out = out / scale.reshape(B, 2, 1, 1)
        # 与基座对特征图的 pad 保持一致（右/下补零），否则形状对不上。
        # 补零区的乘法因子 = 1 + proj(0) = 1（proj 末层零初始化），不引入扰动。
        if padding and padding > 0:
            out = torch.nn.functional.pad(out, [0, padding, 0, padding])
        return out.to(dtype)


class WIMELayer(nn.Module):
    """一层 WIME：1×1 卷积抬升条件嵌入 → 与特征图逐元素相乘。

    只用 1×1 卷积（逐点、无空间混合）是原文的硬性要求：一旦做空间卷积，
    波先验的空间频率结构就被破坏，spectral consistency 失效。
    末层零初始化 → 起点 x·(1+0) = x，与 fno_film 逐位一致。
    """

    def __init__(self, width: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(2, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + self.proj(prior).to(x.dtype))


class SpectralConv2dWIME(nn.Module):
    """SpectralConv2dFiLM 原样复制 + WIME 乘法注入（在 bypass 之前）。"""

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

        cond_dim = 2 * freq_dim
        self.film_net = nn.Sequential(
            nn.Linear(cond_dim, 256), nn.GELU(),
            nn.Linear(256, out_channels * 2),
        )
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)

        # 始终构建 WIME 层：off 臂也用同一结构（只是先验不随频率变），
        # 这样两臂参数**逐位相同**，差异只剩"先验是否随频率变化"这一个变量。
        self.wime = WIMELayer(out_channels)

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        freq_emb = cond.freq_emb if isinstance(cond, Cond) else None
        geom_emb = cond.geom_emb if isinstance(cond, Cond) else None
        prior = getattr(cond, "wave_prior", None) if isinstance(cond, Cond) else None

        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x.float())

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2)

        shift = None
        if freq_emb is not None and geom_emb is not None:
            c = torch.cat([freq_emb, geom_emb], dim=-1)
            film_params = self.film_net(c)
            scale = film_params[:, :self.out_channels].view(B, self.out_channels, 1, 1)
            shift = film_params[:, self.out_channels:].view(B, self.out_channels, 1, 1)
            # scale 在频域乘；shift 留到 irfft2 之后空域加（频域加常数 = 空域 delta 冲激）
            out_ft = out_ft * (1.0 + scale).to(out_ft.dtype)

        y = torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)
        if shift is not None:
            y = y + shift.to(y.dtype)

        # WIME：波先验以空间图形式逐元素乘入（保留其空间频率结构）
        if prior is not None:
            y = self.wime(y, prior)
        return y


class FNOBlockWIME(nn.Module):
    """WIME 谱卷积 + bypass + BN + GELU + 残差（与 FNOBlockFiLM 结构对齐）。"""

    def __init__(self, width: int, modes1: int, modes2: int, freq_dim: int = 32):
        super().__init__()
        self.spectral = SpectralConv2dWIME(width, width, modes1, modes2,
                                           freq_dim=freq_dim)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.BatchNorm2d(width)

    def forward(self, x, cond=None):
        # WIME 注入在 spectral 内部完成（irfft2 之后、`+x` 之前）
        return nn.functional.gelu(self.norm(self.spectral(x, cond) + self.bypass(x) + x))


@register("fno_wime")
class FNO2dWIME(FNOBase):
    """FNO2d + FiLM（逐位保留）+ WIME 波先验乘法注入。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        out_channels = cfg.out_channels
        ex = cfg.extra
        # on: 先验随频率变（WIME 全量）；off: 先验固定（消融对照）
        self.wime_mode = str(ex.get("wime_mode", "on")).lower()

        self.use_em_heads = (cfg.head == "em_split")
        if self.use_em_heads and out_channels != 12:
            raise ValueError(
                f"head='em_split' 需要 out_channels=12，得到 {out_channels}。")

        self.wave_prior = WavePrior(
            max_mode=int(ex.get("wime_max_mode", 2)),
            cell_mm=float(ex.get("wime_cell_mm", 15.0)),
            d_over_lam=float(ex.get("wime_d_over_lam", 0.1)),
        )

        self.blocks = nn.ModuleList([
            FNOBlockWIME(cfg.width, cfg.modes1, cfg.modes2, freq_dim=cfg.freq_dim)
            for _ in range(cfg.n_layers)
        ])

        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=cfg.freq_dim)
        self.geom_encoder = SDFGeomEncoder(n_modes=4, out_dim=cfg.freq_dim)
        self.dropout = nn.Dropout2d(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.shared_proj = nn.Sequential(
            nn.Conv2d(cfg.width, 64, kernel_size=1), nn.GELU()
        )
        if self.use_em_heads:
            self.e_heads = nn.ModuleList([DirectionHead(in_channels=64) for _ in range(3)])
            self.h_heads = nn.ModuleList([DirectionHead(in_channels=64 + 6) for _ in range(3)])
        else:
            self.heads = nn.ModuleList([
                DirectionHead(in_channels=64) for _ in range(cfg.n_heads)
            ])

    def build_cond(self, x, freq):
        if freq is None:
            return None
        cond = Cond(freq_emb=self.freq_encoder(freq),
                    geom_emb=self.geom_encoder(x))
        if self.wime_mode != "off":
            # 波先验只依赖频率与网格尺寸。off 臂用**固定频率**（10.5GHz，归一化
            # 0.625）生成先验 —— 模块照样建、照样算，参数**逐位相同**，唯一差别
            # 是先验是否随频率变化。这正是要测的变量。（mattnv 的同款做法。）
            f_in = (freq[:, 0:1] if self.wime_mode == "on"
                    else torch.full_like(freq[:, 0:1], 0.625))
            cond.wave_prior = self.wave_prior(
                f_in, x.shape[-2], x.shape[-1], self.padding, x.device, x.dtype)
        return cond

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.dropout(self.shared_proj(x))
        if self.use_em_heads:
            e_pred = torch.cat([h(shared) for h in self.e_heads], dim=1)
            h_input = torch.cat([shared, e_pred], dim=1)
            h_pred = torch.cat([h(h_input) for h in self.h_heads], dim=1)
            return torch.cat([e_pred, h_pred], dim=1)
        return torch.cat([h(shared) for h in self.heads], dim=1)

# -*- coding: utf-8 -*-
"""色散坐标谱卷积（Dispersion-Coordinate Spectral Convolution）。

## 物理动机（论文核心思想）

标准 FNO 的谱权重 w(kx, ky) 是逐模式独立的自由参数，谱权重的「物理形状」
随频率的移动只能靠条件化模块（FiLM/超网络）事后补——泛化到未见频点的
能力完全依赖条件化的插值质量。

频率的作用由色散关系控制：谱特征位置由自由空间波数 k₀ = 2πf/c 决定。
把频率以**无量纲色散坐标** ξ = k·L/(2π) / m₀(f)（m₀ = k₀L/2π）注入谱权重：

    w(k; f) = w₀(k) · (1 + g(ξx, ξy))

- w₀(k)：标准 FNO 的逐模式自由参数（容量不丢，对照 fno_film 的 w1/w2 同规模）
- g：色散坐标上的**共享小函数**（三角基 + 线性系数，连续可求值），
  承载「谱形状随 k/k₀ 移动」这个物理结构。g 零初始化 → 起点严格等价标准 FNO。

与 FiLM 的本质区别：FiLM 的调制是「逐输出通道一个标量」，对所有模式相同，
**不携带任何谱形状信息**；g(ξ) 是「逐模式一个值、按色散坐标组织」，
同一物理特征在不同频率下自动移动到对应的 k 位置——8→12 GHz 的谱形状
插值变成结构自带的性质。

## 模式间注意力（第二个创新点）

每个模式一个 token（特征 = 输入谱值实虚部 + 色散坐标 ξx/ξy），
在**模式维**做 softmax——与主模型 fno_attention 的「通道维注意力」不同轴。
物理含义：谱域里哪些模式相互耦合，由它们的色散坐标与谱幅值共同决定。
频率既进入坐标（W 在哪求值）也进入 token（注意力看多大邻域）。
输出门控 γ·Wv，Wv 零初始化 → 起点无扰动。

## 与对照组的公平性

骨架/lift/pad/残差/Norm/bypass/输出头/dropout 与 fno_film 完全一致；
w0 的形状/初始化与 fno_film 的 w1/w2 一致（逐模式自由、scale·rand）。
增量参数只有 g 的基系数（~百个/层）与注意力（~百个/层），
参数量与 fno_film 几乎相同 —— A/B 是「同容量、同条件、只差频率注入方式」。
"""

import math

import torch
import torch.nn as nn

from core.base import Cond


class DispersionSpectralConv(nn.Module):
    """w0(k)·(1+g(ξ)) 谱卷积 + 模式间注意力。"""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int,
                 freq_dim: int, n_heads: int = 4, n_basis: int = 4,
                 cell_mm: float = 15.0, use_attention: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.n_heads = n_heads
        self.n_basis = n_basis
        self.cell_mm = cell_mm
        self.use_attention = use_attention

        # ---- w0：与 fno_film 的 w1/w2 同规模同初始化（正负频率两块，与标准 FNO 一致）
        scale = 1 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

        # ---- g(ξx,ξy)：色散坐标上的共享函数。
        # 2D 三角基: η2d = η(ξx) ⊗ η(ξy)，n_eta = 2n+1 -> n_eta² 个基。
        # g = <θ, η2d>，θ: (n_eta, n_eta) 实数，全模式共享（"谱形状的物理先验"）。
        self.n_eta = 2 * n_basis + 1
        self.theta = nn.Parameter(torch.zeros(self.n_eta, self.n_eta))

        # ---- 色散尺度修正 δ(freq, geom)：介质基板使有效 k 偏离自由空间，
        # 且贴片形状（几何）会影响有效介质环境。输入与 fno_film 的 film_net
        # 一致用 [freq_emb, geom_emb] 拼接（对照公平）。δ 零初始化 ->
        # 起点严格自由空间色散。
        self.m0_head = nn.Sequential(
            nn.Linear(2 * freq_dim, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.m0_head[-1].weight)
        nn.init.zeros_(self.m0_head[-1].bias)

        # ---- 模式间注意力：token = [Re V, Im V, ξx, ξy]（通道平均）
        self.d_k = 32
        self.Wq = nn.Linear(4, self.d_k)
        self.Wk = nn.Linear(4, self.d_k)
        self.Wv = nn.Linear(2, 2)                    # 零初始化 -> 起点无扰动
        nn.init.zeros_(self.Wv.weight)
        nn.init.zeros_(self.Wv.bias)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def _eta(self, xi: torch.Tensor) -> torch.Tensor:
        """1D 基函数。xi: (B,S) -> (B,S,n_eta)"""
        p = math.pi * xi.unsqueeze(-1)
        parts = [torch.ones_like(p)]
        for n in range(1, self.n_basis + 1):
            parts.append(torch.sin(n * p))
            parts.append(torch.cos(n * p))
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        freq_emb = cond.freq_emb if isinstance(cond, Cond) else None
        geom_emb = cond.geom_emb if isinstance(cond, Cond) else None
        freq_raw = getattr(cond, "freq_raw", None) if isinstance(cond, Cond) else None
        B, C, H, W = x.shape
        W2 = W // 2 + 1
        x_ft = torch.fft.rfft2(x.float())

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W2)

        # ---- 色散坐标 ξ = m / m0(f)（低频块的两个轴都是非负整数频率）
        # m0 = k0·L/(2π) = f·L/c。单位：f_Hz = (8+4·f_norm)·1e9，L=15mm=0.015m，
        # c=3e8 m/s → m0 ≈ 0.4~0.6（λ≈30mm，元胞 15mm → 每波长 2 个元胞）。
        # 注意单位陷阱：f 用 GHz、L 用 mm 时 c=3e11 mm/s，但 GHz≠Hz——
        # 必须 ×1e9 换算，否则 m0 小 9 个量级（实测踩过：m0≈4e-10，
        # ξ 全部 clip 到 4，色散坐标完全失效、频率位移恒 0）。
        if freq_raw is not None:
            f_ghz = 8.0 + 4.0 * freq_raw.reshape(B, 1)
        else:
            f_ghz = torch.full((B, 1), 10.0, device=x.device)
        m0 = f_ghz * 1e9 * (self.cell_mm / 1000.0) / 3e8          # (B,1) 无量纲 ≈0.4~0.6
        if freq_emb is not None and geom_emb is not None:
            d_in = torch.cat([freq_emb, geom_emb], dim=-1)
            m0 = m0 * (1.0 + torch.tanh(self.m0_head(d_in)).reshape(B, 1))
        my = torch.arange(m1, device=x.device, dtype=torch.float32)
        mx = torch.arange(m2, device=x.device, dtype=torch.float32)
        xi_y = (my.reshape(1, m1) / m0).clamp(0.0, 4.0)           # (B,m1)
        xi_x = (mx.reshape(1, m2) / m0).clamp(0.0, 4.0)           # (B,m2)

        # ---- 标准谱卷积（两块低频，与 fno_film 相同）
        low_pos = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.w1)
        low_neg = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.w2)

        # ---- 色散调制 g(ξx,ξy)（乘在两块低频上；零初始化 -> 起点恒等）
        eta_y = self._eta(xi_y)                                   # (B,m1,n_eta)
        eta_x = self._eta(xi_x)                                   # (B,m2,n_eta)
        # 2D 基 = 外积。g[b,i,j] = Σ_{p,q} theta[p,q]·η_y[b,i,p]·η_x[b,j,q]
        g = torch.einsum("pq,bip,bjq->bij",
                         self.theta, eta_y, eta_x)                # (B,m1,m2)
        gmod = (1.0 + g).to(low_pos.dtype).unsqueeze(1)           # (B,1,m1,m2)
        low_pos = low_pos * gmod
        low_neg = low_neg * gmod

        # ---- 模式间注意力
        if self.use_attention and freq_emb is not None:
            V = x_ft[:, :, :m1, :m2]                              # (B,C,m1,m2)
            v_mean = V.mean(dim=1)                                # (B,m1,m2)
            xi_yy = xi_y.reshape(B, m1, 1).expand(B, m1, m2)
            xi_xx = xi_x.reshape(B, 1, m2).expand(B, m1, m2)
            tok = torch.stack([v_mean.real, v_mean.imag, xi_yy, xi_xx], dim=-1)
            tok = tok.reshape(B, m1 * m2, 4)
            q, k = self.Wq(tok), self.Wk(tok)
            scores = torch.einsum("bnd,bmd->bnm", q, k) / math.sqrt(self.d_k)
            attn = torch.softmax(scores, dim=-1)
            v2 = torch.stack([v_mean.real, v_mean.imag], dim=-1).reshape(B, m1 * m2, 2)
            ctx = torch.einsum("bnm,bmd->bnd", attn, v2)          # (B,N,2)
            corr = self.gamma * self.Wv(ctx)                      # (B,N,2) 零初始化->0
            corr = corr.reshape(B, m1, m2, 2)
            corr_ft = torch.complex(corr[..., 0], corr[..., 1])   # (B,m1,m2)
            low_pos = low_pos + corr_ft.unsqueeze(1)

        out_ft = torch.zeros(B, self.out_channels, H, W2, dtype=torch.cfloat, device=x.device)
        out_ft = out_ft.clone()
        out_ft[:, :, :m1, :m2] = low_pos
        out_ft[:, :, -m1:, :m2] = low_neg

        return torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)

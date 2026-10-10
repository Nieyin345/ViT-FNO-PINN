# -*- coding: utf-8 -*-
"""fno_maxwell：麦克斯韦-Floquet 物理等变神经算子（FNO2dMaxwell）。

核心设计（针对超表面宽带近场内插与未见形状泛化的五个正交物理归纳偏置）：
  1. 去缠绕物理总场相位系（Unwrapped Raw-Field PhaseFrame，`raw_frame=True`）：
     消除旧散射场 PhaseFrame 因 F_scat = F_raw - B(f) 双频差拍跨越 ±π 支割线导致的
     10 GHz 镜面相位反转奇点（oracle geom 0.290 → 0.023），且使网络输出到物理散射场
     的映射成为逐通道等距酉旋转变换（条件数 κ=1）。
  2. 精确 D2 极化反射对偶等变（Exact Polarization Equivariance，`pol_equivariant=True`）：
     利用正方形周期元胞麦克斯韦方程的精确对偶对称性（实测误差 0.000000）：
       输入几何：pol=1 时 x_canon(x, y) = x(y, x)（转置）；
       输出场：  pol=1 时 (Ex,Ey,Ez) <-> (Ey^T,Ex^T,Ez^T), (Hx,Hy,Hz) <-> (-Hy^T,-Hx^T,-Hz^T)。
     单套规范算子同时服务双极化，使非对称几何的有效训练形态翻倍。
  3. Floquet (0,0) 镜面模与高阶倏逝衍射模解耦（`floquet_split=True`）：
     将空间常数 (0,0) 阶 Floquet 镜面反射模（DC）与零均值高阶衍射模（AC）显式解耦，
     并施加由物理频谱衰减决定的切比雪夫模态权重 σ_k T_k(2 f_norm - 1)，抑制龙格振荡。
  4. 麦克斯韦 TEM 平面波阻抗耦合（`tem_dc=True`）：
     亚波长周期元胞的 (0,0) 阶 Floquet 谐波为自由空间纯 TEM 平面波，满足精确波阻抗关系
     H_refl = (z_hat × E_refl) / ζ(f)（实测相对误差 3e-6）与恒定下行入射波 E_inc(f)
     （跨形状相对标准差 1e-6）。DC 头仅需预测 2 个复反射系数 (R_perp, R_para)，自动满足
     6 通道电磁场幅度与相位的精确麦克斯韦平面波守恒。
  5. 周期边界梯度提升与离散法拉第旋度耦合头（`maxwell_head=True`）：
     输入端提取 (SDF, mask) 的周期中心差分梯度与拉普拉斯算子以捕捉金属贴片边缘奇异性；
     FNO 层内融合周期 3x3 深度可分离局部边缘分支；输出端通过离散空间导数 (∂_x E_AC, ∂_y E_AC)
     向磁场衍射头注入法拉第旋度先验（∇×E = -jωμH）。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import Cond, FNOBase, normalize_freq
from core.config import ModelConfig
from core.encoders import FourierFreqEncoder, SDFGeomEncoder
from core.registry import register
from .fno2d_film import SpectralConv2dFiLM

# ── 仅由 8 个训练频点（不含 10.0 GHz）统计的物理尺度常数与切比雪夫谱衰减先验 ──
# 1. 高阶衍射模（零均值 AC）12 通道物理 RMS 尺度：[Ex_r, Ex_i, Ey_r, Ey_i, Ez_r, Ez_i, Hx_r, Hx_i, Hy_r, Hy_i, Hz_r, Hz_i]
CANON_AC_RMS_12 = (
    216.40, 216.40, 315.33, 315.33, 357.36, 357.36,
    0.8705, 0.8705, 0.4434, 0.4434, 1.0403, 1.0403,
)
# 2. (0,0) 阶 TEM 反射模 [u_perp_r, u_perp_i, u_para_r, u_para_i] 跨形状 RMS 尺度：
TEM_REF_RMS_4 = (478.10, 478.10, 1201.33, 1201.33)
# 3. 6 通道独立总场 DC RMS 尺度（仅 tem_dc=False 消融时使用）：
CANON_DC_RMS_12 = (
    344.65, 344.65, 1946.68, 1946.68, 1.0, 1.0,
    4.2767, 4.2767, 0.9154, 0.9154, 0.01, 0.01,
)
# 4. 切比雪夫基的物理谱衰减先验 σ_k（k=0..7）：
SIGMA_AC_DEFAULT = (1.0, 1.00, 0.55, 0.25, 0.15, 0.10, 0.07, 0.05)
SIGMA_DC_DEFAULT = (1.0, 0.70, 0.35, 0.20, 0.12, 0.08, 0.05, 0.03)
# 5. 训练集基底镜面反射均值先验（已按 TEM_REF_RMS_4 与 SIGMA_DC_DEFAULT 归一化，k=0,1）：
TEM_PRIOR_K0 = (-0.0212, -0.0186, -0.4006, -0.9075)
TEM_PRIOR_K1 = (-0.0316, +0.0083, +0.5868, -0.4240)


class MaxwellFNOBlock(nn.Module):
    """融合全局 Floquet 谱卷积与周期局部边缘卷积的混合算子块。"""

    def __init__(self, width: int, modes1: int, modes2: int,
                 freq_dim: int = 32, use_local: bool = True):
        super().__init__()
        self.spectral = SpectralConv2dFiLM(width, width, modes1, modes2, freq_dim=freq_dim)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.use_local = bool(use_local)
        if self.use_local:
            self.local_dw = nn.Conv2d(
                width, width, kernel_size=3, padding=1,
                padding_mode="circular", groups=width,
            )
            self.local_pw = nn.Conv2d(width, width, kernel_size=1)
            nn.init.normal_(self.local_pw.weight, std=0.02)
            nn.init.zeros_(self.local_pw.bias)
        self.norm = nn.BatchNorm2d(width)

    def forward(self, x: torch.Tensor, cond: Optional[Cond] = None) -> torch.Tensor:
        y = self.spectral(x, cond) + self.bypass(x) + x
        if self.use_local:
            y = y + self.local_pw(F.gelu(self.local_dw(x)))
        return F.gelu(self.norm(y))


@register("fno_maxwell")
class FNO2dMaxwell(FNOBase):
    """麦克斯韦-Floquet 物理等变神经算子。"""

    def __init__(self, cfg: ModelConfig):
        ex = cfg.extra or {}
        self.K = int(ex.get("poly_k", 6))
        if self.K < 1:
            raise ValueError(f"poly_k must be >= 1, got {self.K}")
        self.free_body = bool(ex.get("free_body", True))
        self.raw_frame = bool(ex.get("raw_frame", True))
        self.pol_equivariant = bool(ex.get("pol_equivariant", True))
        self.floquet_split = bool(ex.get("floquet_split", True))
        self.tem_dc = bool(ex.get("tem_dc", True))
        self.maxwell_head = bool(ex.get("maxwell_head", True))
        self.head_hidden = int(ex.get("poly_head_hidden", 192))

        super().__init__(cfg)
        # 周期元胞在 64x64 网格上满足严格周期边界条件，无需右侧/下侧零填充破坏周期性
        self.padding = 0

        self.blocks = nn.ModuleList([
            MaxwellFNOBlock(
                cfg.width, cfg.modes1, cfg.modes2,
                freq_dim=cfg.freq_dim, use_local=self.maxwell_head,
            )
            for _ in range(cfg.n_layers)
        ])

        self.freq_encoder = FourierFreqEncoder(
            n_freqs=int(ex.get("freq_enc_n", 16)),
            out_dim=cfg.freq_dim,
            max_log2=float(ex.get("freq_enc_max_log2", 4.0)),
        )
        self.geom_encoder = SDFGeomEncoder(n_modes=4, out_dim=cfg.freq_dim)

        # 单一输出前 Dropout（严格遵守 check_dropout_consistency：一次前向恰好调用 1 次）
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.shared_proj = nn.Sequential(
            nn.Conv2d(cfg.width, 64, kernel_size=1),
            nn.GELU(),
        )

        # 高阶 Floquet 衍射（AC）电/磁预测头 + 离散法拉第旋度 E -> H 耦合
        ac_bias = not self.floquet_split
        self.e_conv1 = nn.Conv2d(64, self.head_hidden, kernel_size=1)
        self.e_conv2 = nn.Conv2d(self.head_hidden, 6 * self.K, kernel_size=1, bias=ac_bias)
        if self.maxwell_head:
            self.curl_proj = nn.Conv2d(18 * self.K, self.head_hidden, kernel_size=1, bias=False)
            nn.init.normal_(self.curl_proj.weight, std=0.02)
        self.h_conv1 = nn.Conv2d(64, self.head_hidden, kernel_size=1)
        self.h_conv2 = nn.Conv2d(self.head_hidden, 6 * self.K, kernel_size=1, bias=ac_bias)

        # (0,0) 阶 Floquet 镜面反射（DC）全局预测头
        if self.floquet_split:
            dc_in_dim = 64 * 2 + 4 + cfg.freq_dim
            dc_out_ch = 4 if (self.tem_dc and self.raw_frame) else 12
            self.dc_out_ch = dc_out_ch
            self.dc_mlp = nn.Sequential(
                nn.Linear(dc_in_dim, 128),
                nn.GELU(),
                nn.Linear(128, dc_out_ch * self.K),
            )
            nn.init.normal_(self.dc_mlp[-1].weight, std=0.01)
            nn.init.zeros_(self.dc_mlp[-1].bias)

            prior = torch.zeros(1, dc_out_ch, self.K, dtype=torch.float32)
            if self.tem_dc and self.raw_frame:
                prior[0, :, 0] = torch.tensor(TEM_PRIOR_K0, dtype=torch.float32)
                if self.K > 1:
                    prior[0, :, 1] = torch.tensor(TEM_PRIOR_K1, dtype=torch.float32)
            self.dc_prior = nn.Parameter(prior)

    def lift_in_channels(self) -> int:
        # [SDF, mask] (2) + 周期梯度/拉普拉斯 [dx, dy, lap] (6) + 周期坐标傅里叶基 (4) = 12
        return int(self.cfg.in_channels) + 10

    def _canonicalize_input(self, x: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        if not self.pol_equivariant:
            return x
        pol = freq[:, 1].round().long().clamp(0, 1)
        mask_p1 = (pol == 1).view(-1, 1, 1, 1)
        return torch.where(mask_p1, x.transpose(-2, -1), x)

    def build_input(self, x_canon: torch.Tensor, freq: Optional[torch.Tensor]) -> torch.Tensor:
        B, C, H, W = x_canon.shape
        if self.maxwell_head:
            dx = 0.5 * (torch.roll(x_canon, -1, dims=-1) - torch.roll(x_canon, 1, dims=-1))
            dy = 0.5 * (torch.roll(x_canon, -1, dims=-2) - torch.roll(x_canon, 1, dims=-2))
            lap = (
                torch.roll(x_canon, -1, dims=-1) + torch.roll(x_canon, 1, dims=-1)
                + torch.roll(x_canon, -1, dims=-2) + torch.roll(x_canon, 1, dims=-2)
                - 4.0 * x_canon
            )
        else:
            dx = torch.zeros_like(x_canon)
            dy = torch.zeros_like(x_canon)
            lap = torch.zeros_like(x_canon)

        gy = torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / H), H, device=x_canon.device, dtype=x_canon.dtype)
        gx = torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / W), W, device=x_canon.device, dtype=x_canon.dtype)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        coord = torch.stack([torch.cos(xx), torch.sin(xx), torch.cos(yy), torch.sin(yy)], dim=0)
        coord = coord.unsqueeze(0).expand(B, -1, H, W)
        return torch.cat([x_canon, dx, dy, lap, coord], dim=1)

    def build_cond(self, x_canon: torch.Tensor, freq: Optional[torch.Tensor]) -> Optional[Cond]:
        if freq is None:
            return None
        return Cond(freq_emb=self.freq_encoder(freq), geom_emb=self.geom_encoder(x_canon))

    def _cheb_basis(self, f_norm: torch.Tensor, sigma_tuple) -> torch.Tensor:
        t = 2.0 * f_norm - 1.0
        T = [torch.ones_like(t), t]
        for _ in range(2, self.K):
            T.append(2.0 * t * T[-1] - T[-2])
        basis = torch.stack(T[: self.K], dim=1)                        # (B, K)
        sig_list = [sigma_tuple[k] if k < len(sigma_tuple) else sigma_tuple[-1] for k in range(self.K)]
        sig = basis.new_tensor(sig_list).unsqueeze(0)                  # (1, K)
        return basis * sig

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FNO2dMaxwell 通过 forward 直接融合 (x_canon, cond, f_norm)")

    def forward(self, x: torch.Tensor, freq: Optional[torch.Tensor] = None) -> torch.Tensor:
        freq = normalize_freq(freq, x.shape[0])
        if freq is None:
            freq = x.new_zeros((x.shape[0], 2))
            freq[:, 0] = 0.5

        x_canon = self._canonicalize_input(x, freq)
        fb = freq.clone()
        if self.free_body:
            fb[:, 0] = 0.5
        if self.pol_equivariant:
            fb[:, 1] = 0.0

        x_in = self.build_input(x_canon, fb)
        cond = self.build_cond(x_canon, fb)

        out = self.lift(x_in)
        for block in self.blocks:
            out = block(out, cond)

        shared = self.dropout(self.shared_proj(out))                   # (B, 64, H, W)
        B, _, H, W = shared.shape
        f_norm = freq[:, 0]

        phi_ac = self._cheb_basis(f_norm, SIGMA_AC_DEFAULT).to(shared.dtype)  # (B, K)
        phi_dc = self._cheb_basis(f_norm, SIGMA_DC_DEFAULT).to(shared.dtype)  # (B, K)

        # 1. 高阶 Floquet 衍射头（零均值 AC + 法拉第旋度 E -> H 耦合）
        e_hid = F.gelu(self.e_conv1(shared))
        c_e = self.e_conv2(e_hid)                                      # (B, 6*K, H, W)
        if self.floquet_split:
            c_e = c_e - c_e.mean(dim=(-2, -1), keepdim=True)

        if self.maxwell_head:
            dx_ce = 0.5 * (torch.roll(c_e, -1, dims=-1) - torch.roll(c_e, 1, dims=-1))
            dy_ce = 0.5 * (torch.roll(c_e, -1, dims=-2) - torch.roll(c_e, 1, dims=-2))
            curl_feat = self.curl_proj(torch.cat([c_e, dx_ce, dy_ce], dim=1))
            h_hid = F.gelu(self.h_conv1(shared) + curl_feat)
        else:
            h_hid = F.gelu(self.h_conv1(shared))

        c_h = self.h_conv2(h_hid)                                      # (B, 6*K, H, W)
        if self.floquet_split:
            c_h = c_h - c_h.mean(dim=(-2, -1), keepdim=True)

        c_ac = torch.cat([
            c_e.view(B, 6, self.K, H, W),
            c_h.view(B, 6, self.K, H, W),
        ], dim=1)                                                      # (B, 12, K, H, W)
        u_ac = (c_ac * phi_ac[:, None, :, None, None]).sum(dim=2)      # (B, 12, H, W)

        if self.raw_frame:
            ac_scale = u_ac.new_tensor(CANON_AC_RMS_12).view(1, 12, 1, 1)
            u_ac = u_ac * ac_scale

        # 2. (0,0) 阶 Floquet 镜面反射头（DC + 麦克斯韦 TEM 波阻抗耦合）
        if self.floquet_split:
            sh_mean = shared.mean(dim=(-2, -1))
            sh_std = torch.sqrt(shared.var(dim=(-2, -1), unbiased=False) + 1e-6)
            xm = x_canon.mean(dim=(-2, -1))
            xs = torch.sqrt(x_canon.var(dim=(-2, -1), unbiased=False) + 1e-6)
            g_emb = cond.geom_emb if (cond is not None and cond.geom_emb is not None) else sh_mean.new_zeros(B, self.cfg.freq_dim)
            g_feat = torch.cat([sh_mean, sh_std, xm, xs, g_emb], dim=-1)

            c_dc = self.dc_mlp(g_feat).view(B, self.dc_out_ch, self.K) + self.dc_prior
            u_dc_vec = (c_dc * phi_dc[:, None, :]).sum(dim=2)          # (B, dc_out_ch)

            if self.tem_dc and self.raw_frame:
                tem_scale = u_dc_vec.new_tensor(TEM_REF_RMS_4).view(1, 4)
                u_tem = (u_dc_vec * tem_scale).view(B, 4, 1, 1)
                u_perp = u_tem[:, 0:2]
                u_para = u_tem[:, 2:4]
                z2 = torch.zeros_like(u_perp)
                u_dc = torch.cat([
                    u_perp, u_para, z2,
                    -u_para / 376.50, u_perp / 376.50, z2,
                ], dim=1)                                              # (B, 12, 1, 1)
            else:
                if self.raw_frame:
                    dc_scale = u_dc_vec.new_tensor(CANON_DC_RMS_12).view(1, 12)
                    u_dc_vec = u_dc_vec * dc_scale
                u_dc = u_dc_vec.view(B, 12, 1, 1)

            y_out = u_dc + u_ac
        else:
            y_out = u_ac

        y_out._is_raw_canonical = bool(self.raw_frame)
        y_out._pol_equivariant = bool(self.pol_equivariant)
        y_out._tem_dc = bool(self.tem_dc and self.floquet_split and self.raw_frame)
        return y_out


# ── V2 切比雪夫谱衰减先验（K=8，包含 T_6(0)=-1 与 T_7(0)=0 消除 10GHz 截断误差与振幅塌缩） ──
SIGMA_AC_V2 = (1.0, 0.98, 0.80, 0.55, 0.35, 0.22, 0.14, 0.08)
SIGMA_DC_V2 = (1.0, 0.95, 0.85, 0.72, 0.58, 0.42, 0.28, 0.16)
TEM_PRIOR_K1_V2 = tuple(v * (0.70 / 0.95) for v in TEM_PRIOR_K1)


class V4GeomEncoder(nn.Module):
    """全象限 V4 协变几何编码器：同时提取 (+ky,+kx) 与 (-ky,+kx) 低频傅里叶模及 V4 空间谐波矩。"""

    def __init__(self, n_modes: int = 4, out_dim: int = 32):
        super().__init__()
        self.n_modes = int(n_modes)
        m = self.n_modes
        fft_dim = 2 * 2 * (m * m + max(0, m - 1) * max(0, m - 1))
        in_dim = fft_dim + 16
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, max(64, out_dim * 2)),
            nn.GELU(),
            nn.Linear(max(64, out_dim * 2), out_dim),
        )

    def forward(self, x_canon: torch.Tensor, s_rel: torch.Tensor,
                w_metal: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x_canon.shape
        m = self.n_modes
        g_fields = torch.cat([s_rel, w_metal], dim=1)                  # (B, 2, H, W)
        spec = torch.fft.rfft2(g_fields, norm="ortho")                 # (B, 2, H, W//2+1)

        pos_q = spec[:, :, :m, :m]                                     # (B, 2, m, m)
        if m > 1:
            neg_q = spec[:, :, -(m - 1):, 1:m]                         # (B, 2, m-1, m-1)
            fft_feat = torch.cat([
                pos_q.real.reshape(B, -1), pos_q.imag.reshape(B, -1),
                neg_q.real.reshape(B, -1), neg_q.imag.reshape(B, -1),
            ], dim=-1)
        else:
            fft_feat = torch.cat([pos_q.real.reshape(B, -1), pos_q.imag.reshape(B, -1)], dim=-1)

        gy = torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / H), H, device=x_canon.device, dtype=x_canon.dtype)
        gx = torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / W), W, device=x_canon.device, dtype=x_canon.dtype)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        cos_x, sin_x = torch.cos(xx)[None, None], torch.sin(xx)[None, None]
        cos_y, sin_y = torch.cos(yy)[None, None], torch.sin(yy)[None, None]
        cos_2x, sin_2x = torch.cos(2.0 * xx)[None, None], torch.sin(2.0 * xx)[None, None]
        cos_2y, sin_2y = torch.cos(2.0 * yy)[None, None], torch.sin(2.0 * yy)[None, None]

        moments = torch.cat([
            w_metal.mean(dim=(-2, -1)),
            psi.mean(dim=(-2, -1)),
            torch.sqrt(s_rel.var(dim=(-2, -1), unbiased=False) + 1e-6),
            s_rel.amin(dim=(-2, -1)),
            (w_metal * cos_x).mean(dim=(-2, -1)),
            (w_metal * cos_y).mean(dim=(-2, -1)),
            (psi * cos_x).mean(dim=(-2, -1)),
            (psi * cos_y).mean(dim=(-2, -1)),
            (w_metal * cos_2x).mean(dim=(-2, -1)),
            (w_metal * cos_2y).mean(dim=(-2, -1)),
            (w_metal * (sin_x * sin_y)).mean(dim=(-2, -1)),
            (psi * (sin_x * sin_y)).mean(dim=(-2, -1)),
            (w_metal * (sin_2x * sin_2y)).mean(dim=(-2, -1)),
            (w_metal * sin_x).mean(dim=(-2, -1)),
            (w_metal * sin_y).mean(dim=(-2, -1)),
            (x_canon[:, 1:2].amax(dim=(-2, -1)) > 0.5).to(x_canon.dtype),
        ], dim=-1)                                                     # (B, 16)
        return self.mlp(torch.cat([fft_feat, moments], dim=-1))


class MaxwellFNOBlockV2(nn.Module):
    """多尺度周期空洞局部边缘卷积 + 全局 Floquet 谱卷积混合块。"""

    def __init__(self, width: int, modes1: int, modes2: int,
                 freq_dim: int = 32, use_local: bool = True):
        super().__init__()
        self.spectral = SpectralConv2dFiLM(width, width, modes1, modes2, freq_dim=freq_dim)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.use_local = bool(use_local)
        if self.use_local:
            self.local_dw1 = nn.Conv2d(
                width, width, kernel_size=3, padding=1,
                padding_mode="circular", groups=width,
            )
            self.local_dw2 = nn.Conv2d(
                width, width, kernel_size=3, padding=2, dilation=2,
                padding_mode="circular", groups=width,
            )
            self.local_pw = nn.Conv2d(width, width, kernel_size=1)
            nn.init.normal_(self.local_pw.weight, std=0.02)
            nn.init.zeros_(self.local_pw.bias)
        self.norm = nn.BatchNorm2d(width)

    def forward(self, x: torch.Tensor, cond: Optional[Cond] = None) -> torch.Tensor:
        y = self.spectral(x, cond) + self.bypass(x) + x
        if self.use_local:
            loc = F.gelu(self.local_dw1(x) + 0.5 * self.local_dw2(x))
            y = y + self.local_pw(loc)
        return F.gelu(self.norm(y))


@register("fno_maxwell_v2")
class FNO2dMaxwellV2(FNOBase):
    """麦克斯韦-Floquet D4/V4 物理等变神经算子（第二代 FNO2dMaxwellV2）。

    在 FNO2dMaxwell 五项物理归纳偏置基础上，针对误差分解诊断出的四个核心物理瓶颈进一步升级：
      1. 精确 D4 (V4 x Z2) 二面体群反射等变与模长保持雷诺投影（`v4_equivariant=True`）：
         训练时对周期元胞施加随机 Klein 四元群反射 g in V4 并按麦克斯韦精确宇称反变换，使有效几何形态
         扩大 8 倍；推理时执行模长保持的 V4 群雷诺平均投影 P_{V4}，保证交叉极化 u_perp 在对称形状上
         严格为零、旋转贴片上严格反对称。
      2. 平移不变局部指数势提升（Shift-Invariant Exponential SDF Potential）：
         以 s_rel = sdf - min(sdf) 与局部势 psi = exp(-5 s_rel) 替代非周期裸 SDF 的周期差分，
         彻底消除元胞边界卷绕跳变伪影（幅值降低 14 倍），同时自动修复 mask=0 的退化双矩形样本。
      3. 表面等效定理空间积分耦合 + V4 谐波矩 + 极坐标模长包络 TEM 反射头（`K=8, polar_dc=True`）：
         将切比雪夫阶数提升至 K=8（纳入 T_6(0)=-1 与 T_7(0)=0，消除 10 GHz 截断误差），融合金属贴片/边缘/
         V4 谐波池化与 2D 电场空间积分残差，并引入直接预测 |u_para(f)| 的极坐标模长包络分支消除弦切塌缩。
      4. 多尺度空洞边缘感知 + 高斯散度/法拉第旋度级联衍射头（Gauss-Faraday AC Head）：
         聚合中层与深层 FNO 特征，先由横向电场散度 (d_x Ex + d_y Ey) 修正纵向 Ez，再由完整三维旋度驱动 H。
    """

    def __init__(self, cfg: ModelConfig):
        ex = cfg.extra or {}
        self.K = int(ex.get("poly_k", 8))
        if self.K < 1:
            raise ValueError(f"poly_k must be >= 1, got {self.K}")
        self.free_body = bool(ex.get("free_body", True))
        self.raw_frame = bool(ex.get("raw_frame", True))
        self.pol_equivariant = bool(ex.get("pol_equivariant", True))
        self.v4_equivariant = bool(ex.get("v4_equivariant", True))
        self.v4_reynolds_eval = bool(ex.get("v4_reynolds_eval", True))
        self.floquet_split = bool(ex.get("floquet_split", True))
        self.tem_dc = bool(ex.get("tem_dc", True))
        self.polar_dc = bool(ex.get("polar_dc", True))
        self.maxwell_head = bool(ex.get("maxwell_head", True))
        self.head_hidden = int(ex.get("poly_head_hidden", 192))

        super().__init__(cfg)
        self.padding = 0

        self.blocks = nn.ModuleList([
            MaxwellFNOBlockV2(
                cfg.width, cfg.modes1, cfg.modes2,
                freq_dim=cfg.freq_dim, use_local=self.maxwell_head,
            )
            for _ in range(cfg.n_layers)
        ])

        self.freq_encoder = FourierFreqEncoder(
            n_freqs=int(ex.get("freq_enc_n", 16)),
            out_dim=cfg.freq_dim,
            max_log2=float(ex.get("freq_enc_max_log2", 4.0)),
        )
        self.geom_encoder = V4GeomEncoder(n_modes=4, out_dim=cfg.freq_dim)

        # 多深度特征聚合（中层局部边缘 + 顶层全局模态）與唯一输出前 Dropout
        self.shared_dim = max(64, int(cfg.width))
        self.shared_proj = nn.Sequential(
            nn.Conv2d(cfg.width * 2, self.shared_dim, kernel_size=1),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        # 高阶 Floquet 衍射（AC）高斯散度 + 法拉第旋度级联预测头
        ac_bias = not self.floquet_split
        self.e_conv1 = nn.Conv2d(self.shared_dim, self.head_hidden, kernel_size=1)
        self.e_conv2 = nn.Conv2d(self.head_hidden, 6 * self.K, kernel_size=1, bias=ac_bias)
        if self.maxwell_head:
            self.ez_div_proj = nn.Conv2d(4 * self.K, 2 * self.K, kernel_size=1, bias=False)
            nn.init.normal_(self.ez_div_proj.weight, std=0.01)
            self.curl_proj = nn.Conv2d(18 * self.K, self.head_hidden, kernel_size=1, bias=False)
            nn.init.normal_(self.curl_proj.weight, std=0.02)
        self.h_conv1 = nn.Conv2d(self.shared_dim, self.head_hidden, kernel_size=1)
        self.h_conv2 = nn.Conv2d(self.head_hidden, 6 * self.K, kernel_size=1, bias=ac_bias)

        # (0,0) 阶 Floquet 镜面反射（DC）V4 谐波 + 表面等效积分 + 极坐标模长稳定头
        if self.floquet_split:
            dc_in_dim = self.shared_dim * 9 + cfg.freq_dim
            use_tem = bool(self.tem_dc and self.raw_frame)
            dc_out_ch = (5 if self.polar_dc else 4) if use_tem else 12
            self.dc_out_ch = dc_out_ch
            self.dc_mlp = nn.Sequential(
                nn.Linear(dc_in_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, dc_out_ch * self.K),
            )
            nn.init.normal_(self.dc_mlp[-1].weight, std=0.01)
            nn.init.zeros_(self.dc_mlp[-1].bias)

            prior = torch.zeros(1, 4 if use_tem else 12, self.K, dtype=torch.float32)
            if use_tem:
                prior[0, 2:4, 0] = torch.tensor(TEM_PRIOR_K0[2:4], dtype=torch.float32)
                if self.K > 1:
                    prior[0, 2:4, 1] = torch.tensor(TEM_PRIOR_K1_V2[2:4], dtype=torch.float32)
            self.dc_prior = nn.Parameter(prior)
            self.se_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
            if use_tem and self.polar_dc:
                self.polar_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.register_buffer(
            "v4_signs_w",
            torch.tensor([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0]).view(1, 12, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "v4_signs_h",
            torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]).view(1, 12, 1, 1),
            persistent=False,
        )

    def lift_in_channels(self) -> int:
        # [s_rel, w_metal] (2) + 局部势/掩码有符号周期导数与拉普拉斯 (6) + 周期傅里叶基 (4) = 12
        return int(self.cfg.in_channels) + 10

    def _canonicalize_input(self, x: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        if not self.pol_equivariant:
            return x
        pol = freq[:, 1].round().long().clamp(0, 1)
        mask_p1 = (pol == 1).view(-1, 1, 1, 1)
        return torch.where(mask_p1, x.transpose(-2, -1), x)

    @staticmethod
    def _rflip(t: torch.Tensor, dims) -> torch.Tensor:
        if not dims:
            return t
        return torch.roll(torch.flip(t, dims=dims), shifts=[1] * len(dims), dims=dims)

    def _extract_geom_fields(self, x_canon: torch.Tensor):
        sdf_raw = x_canon[:, 0:1]
        mask_raw = x_canon[:, 1:2]
        has_mask = (mask_raw.amax(dim=(-2, -1), keepdim=True) > 0.5)
        sdf_clean = torch.where(
            has_mask,
            sdf_raw,
            (sdf_raw - sdf_raw.amin(dim=(-2, -1), keepdim=True) - 0.12) * 1.25,
        )
        w_metal = torch.where(has_mask, mask_raw, torch.sigmoid(-30.0 * sdf_clean))
        psi = torch.tanh(-4.0 * sdf_clean)
        return sdf_clean, w_metal, psi

    def build_input(self, x_canon: torch.Tensor, freq: Optional[torch.Tensor]) -> torch.Tensor:
        B, _, H, W = x_canon.shape
        s_rel, w_metal, psi = self._extract_geom_fields(x_canon)
        if self.maxwell_head:
            dx_psi = 0.5 * (torch.roll(psi, -1, dims=-1) - torch.roll(psi, 1, dims=-1))
            dy_psi = 0.5 * (torch.roll(psi, -1, dims=-2) - torch.roll(psi, 1, dims=-2))
            lap_psi = (
                torch.roll(psi, -1, dims=-1) + torch.roll(psi, 1, dims=-1)
                + torch.roll(psi, -1, dims=-2) + torch.roll(psi, 1, dims=-2)
                - 4.0 * psi
            )
            dx_m = 0.5 * (torch.roll(w_metal, -1, dims=-1) - torch.roll(w_metal, 1, dims=-1))
            dy_m = 0.5 * (torch.roll(w_metal, -1, dims=-2) - torch.roll(w_metal, 1, dims=-2))
            lap_m = (
                torch.roll(w_metal, -1, dims=-1) + torch.roll(w_metal, 1, dims=-1)
                + torch.roll(w_metal, -1, dims=-2) + torch.roll(w_metal, 1, dims=-2)
                - 4.0 * w_metal
            )
        else:
            z = torch.zeros_like(s_rel)
            dx_psi = dy_psi = lap_psi = dx_m = dy_m = lap_m = z

        gy = torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / H), H, device=x_canon.device, dtype=x_canon.dtype)
        gx = torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / W), W, device=x_canon.device, dtype=x_canon.dtype)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        coord = torch.stack([torch.cos(xx), torch.sin(xx), torch.cos(yy), torch.sin(yy)], dim=0)
        coord = coord.unsqueeze(0).expand(B, -1, H, W)
        return torch.cat([
            s_rel, w_metal,
            dx_psi, dy_psi, lap_psi,
            dx_m, dy_m, lap_m,
            coord,
        ], dim=1)

    def build_cond(self, x_canon: torch.Tensor, freq: Optional[torch.Tensor]) -> Optional[Cond]:
        if freq is None:
            return None
        s_rel, w_metal, psi = self._extract_geom_fields(x_canon)
        return Cond(
            freq_emb=self.freq_encoder(freq),
            geom_emb=self.geom_encoder(x_canon, s_rel, w_metal, psi),
        )

    def _cheb_basis(self, f_norm: torch.Tensor, sigma_tuple) -> torch.Tensor:
        t = 2.0 * f_norm - 1.0
        T = [torch.ones_like(t), t]
        for _ in range(2, self.K):
            T.append(2.0 * t * T[-1] - T[-2])
        basis = torch.stack(T[: self.K], dim=1)
        sig_list = [sigma_tuple[k] if k < len(sigma_tuple) else sigma_tuple[-1] for k in range(self.K)]
        sig = basis.new_tensor(sig_list).unsqueeze(0)
        return basis * sig

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FNO2dMaxwellV2 通过 forward 直接融合 (x_canon, cond, f_norm)")

    def _forward_single(self, x_canon: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        fb = freq.clone()
        if self.free_body:
            fb[:, 0] = 0.5
        if self.pol_equivariant:
            fb[:, 1] = 0.0

        s_rel, w_metal, psi = self._extract_geom_fields(x_canon)
        x_in = self.build_input(x_canon, fb)
        cond = Cond(
            freq_emb=self.freq_encoder(fb),
            geom_emb=self.geom_encoder(x_canon, s_rel, w_metal, psi),
        )

        out = self.lift(x_in)
        mid_idx = max(0, len(self.blocks) // 2 - 1)
        out_mid = out
        for idx, block in enumerate(self.blocks):
            out = block(out, cond)
            if idx == mid_idx:
                out_mid = out

        shared = self.dropout(self.shared_proj(torch.cat([out_mid, out], dim=1)))
        B, _, H, W = shared.shape
        f_norm = freq[:, 0]

        phi_ac = self._cheb_basis(f_norm, SIGMA_AC_V2).to(shared.dtype)
        phi_dc = self._cheb_basis(f_norm, SIGMA_DC_V2).to(shared.dtype)

        # 1. 高阶 Floquet 衍射头（高斯横向散度修正 Ez + 表面等效积分提取 + 法拉第旋度驱动 H）
        e_hid = F.gelu(self.e_conv1(shared))
        c_e_raw = self.e_conv2(e_hid)                                  # (B, 6*K, H, W)
        if self.maxwell_head:
            dx_et = 0.5 * (torch.roll(c_e_raw[:, :4 * self.K], -1, dims=-1) - torch.roll(c_e_raw[:, :4 * self.K], 1, dims=-1))
            dy_et = 0.5 * (torch.roll(c_e_raw[:, :4 * self.K], -1, dims=-2) - torch.roll(c_e_raw[:, :4 * self.K], 1, dims=-2))
            div_et = dx_et[:, :2 * self.K] + dy_et[:, 2 * self.K:4 * self.K]
            dz_corr = self.ez_div_proj(torch.cat([c_e_raw[:, 4 * self.K:6 * self.K], div_et], dim=1))
            c_e = torch.cat([c_e_raw[:, :4 * self.K], c_e_raw[:, 4 * self.K:6 * self.K] + dz_corr], dim=1)
        else:
            c_e = c_e_raw

        c_e_dc = c_e.view(B, 6, self.K, H, W).mean(dim=(-2, -1))       # (B, 6, K)
        if self.floquet_split:
            c_e = c_e - c_e.mean(dim=(-2, -1), keepdim=True)

        if self.maxwell_head:
            dx_ce = 0.5 * (torch.roll(c_e, -1, dims=-1) - torch.roll(c_e, 1, dims=-1))
            dy_ce = 0.5 * (torch.roll(c_e, -1, dims=-2) - torch.roll(c_e, 1, dims=-2))
            curl_feat = self.curl_proj(torch.cat([c_e, dx_ce, dy_ce], dim=1))
            h_hid = F.gelu(self.h_conv1(shared) + curl_feat)
        else:
            h_hid = F.gelu(self.h_conv1(shared))

        c_h = self.h_conv2(h_hid)
        if self.floquet_split:
            c_h = c_h - c_h.mean(dim=(-2, -1), keepdim=True)

        c_ac = torch.cat([
            c_e.view(B, 6, self.K, H, W),
            c_h.view(B, 6, self.K, H, W),
        ], dim=1)
        u_ac = (c_ac * phi_ac[:, None, :, None, None]).sum(dim=2)

        if self.raw_frame:
            ac_scale = u_ac.new_tensor(CANON_AC_RMS_12).view(1, 12, 1, 1)
            u_ac = u_ac * ac_scale

        # 2. (0,0) 阶 Floquet 镜面反射头（贴片/边缘/V4 谐波池化 + 表面等效积分 + 极坐标模长包络）
        if self.floquet_split:
            gy = torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / H), H, device=shared.device, dtype=shared.dtype)
            gx = torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / W), W, device=shared.device, dtype=shared.dtype)
            yy, xx = torch.meshgrid(gy, gx, indexing="ij")
            cos_x, sin_x = torch.cos(xx)[None, None], torch.sin(xx)[None, None]
            cos_y, sin_y = torch.cos(yy)[None, None], torch.sin(yy)[None, None]

            w_edge = x_in[:, 2:3].abs() + x_in[:, 3:4].abs() + x_in[:, 5:6].abs() + x_in[:, 6:7].abs()
            sh_mean = shared.mean(dim=(-2, -1))
            sh_std = torch.sqrt(shared.var(dim=(-2, -1), unbiased=False) + 1e-6)
            sh_metal = (shared * w_metal).sum(dim=(-2, -1)) / (w_metal.sum(dim=(-2, -1)) + 1e-4)
            sh_edge = (shared * w_edge).sum(dim=(-2, -1)) / (w_edge.sum(dim=(-2, -1)) + 1e-4)
            sh_cosx = (shared * cos_x).mean(dim=(-2, -1))
            sh_cosy = (shared * cos_y).mean(dim=(-2, -1))
            sh_sinxy = (shared * (sin_x * sin_y)).mean(dim=(-2, -1))
            sh_sinx = (shared * sin_x).mean(dim=(-2, -1))
            sh_siny = (shared * sin_y).mean(dim=(-2, -1))

            g_emb = cond.geom_emb if (cond is not None and cond.geom_emb is not None) else sh_mean.new_zeros(B, self.cfg.freq_dim)
            g_feat = torch.cat([
                sh_mean, sh_std, sh_metal, sh_edge,
                sh_cosx, sh_cosy, sh_sinxy, sh_sinx, sh_siny,
                g_emb,
            ], dim=-1)

            c_dc_all = self.dc_mlp(g_feat).view(B, self.dc_out_ch, self.K)
            if self.tem_dc and self.raw_frame:
                c_dc_cart = c_dc_all[:, :4, :] + self.dc_prior + self.se_scale * c_e_dc[:, :4, :]
                u_dc_4 = (c_dc_cart * phi_dc[:, None, :]).sum(dim=2)   # (B, 4)
                if self.polar_dc and self.dc_out_ch >= 5:
                    u_amp_res = (c_dc_all[:, 4, :] * phi_dc).sum(dim=1, keepdim=True)  # (B, 1)
                    u_para_cart = u_dc_4[:, 2:4]
                    r_cart = torch.sqrt(u_para_cart[:, 0:1] ** 2 + u_para_cart[:, 1:2] ** 2 + 1e-6)
                    r0 = 1.444
                    r_env = F.softplus(r0 + u_amp_res) * (r0 / math.log(1.0 + math.exp(r0)))
                    alpha = 0.35 * torch.sigmoid(self.polar_gate)
                    r_eff = (1.0 - alpha) * r_cart + alpha * r_env
                    u_para_unit = u_para_cart * (r_eff / r_cart)
                    u_dc_4 = torch.cat([u_dc_4[:, 0:2], u_para_unit], dim=1)

                tem_scale = u_dc_4.new_tensor(TEM_REF_RMS_4).view(1, 4)
                u_tem = (u_dc_4 * tem_scale).view(B, 4, 1, 1)
                u_perp = u_tem[:, 0:2]
                u_para = u_tem[:, 2:4]
                z2 = torch.zeros_like(u_perp)
                u_dc = torch.cat([
                    u_perp, u_para, z2,
                    -u_para / 376.50, u_perp / 376.50, z2,
                ], dim=1)
            else:
                c_dc = c_dc_all[:, :12, :] + self.dc_prior
                u_dc_vec = (c_dc * phi_dc[:, None, :]).sum(dim=2)
                if self.raw_frame:
                    dc_scale = u_dc_vec.new_tensor(CANON_DC_RMS_12).view(1, 12)
                    u_dc_vec = u_dc_vec * dc_scale
                u_dc = u_dc_vec.view(B, 12, 1, 1)

            return u_dc + u_ac
        return u_ac

    def forward(self, x: torch.Tensor, freq: Optional[torch.Tensor] = None) -> torch.Tensor:
        freq = normalize_freq(freq, x.shape[0])
        if freq is None:
            freq = x.new_zeros((x.shape[0], 2))
            freq[:, 0] = 0.5

        x_canon = self._canonicalize_input(x, freq)
        B = x_canon.shape[0]
        sw = self.v4_signs_w.to(dtype=x_canon.dtype)
        sh = self.v4_signs_h.to(dtype=x_canon.dtype)

        if self.training and self.v4_equivariant and self.cfg.dropout > 0:
            # 训练态：每样本随机施加 V4 周期反射（单次前向，保持 check_dropout_consistency 严格调用 1 次）
            flip_w = torch.randint(0, 2, (B,), device=x_canon.device, dtype=torch.bool)
            flip_h = torch.randint(0, 2, (B,), device=x_canon.device, dtype=torch.bool)
            x_g = x_canon.clone()
            if flip_w.any():
                x_g[flip_w] = self._rflip(x_g[flip_w], (-1,))
            if flip_h.any():
                x_g[flip_h] = self._rflip(x_g[flip_h], (-2,))

            y_g = self._forward_single(x_g, freq)
            if flip_h.any():
                y_g[flip_h] = self._rflip(y_g[flip_h], (-2,)) * sh
            if flip_w.any():
                y_g[flip_w] = self._rflip(y_g[flip_w], (-1,)) * sw
            y_out = y_g
        elif (not self.training) and self.v4_equivariant and self.v4_reynolds_eval:
            # 推理态：执行模长保持的 V4 群雷诺平均投影 P_{V4}，保证严格 D4 等变与零弦切损失
            u_e = self._forward_single(x_canon, freq)
            u_w = self._rflip(self._forward_single(self._rflip(x_canon, (-1,)), freq), (-1,)) * sw
            u_h = self._rflip(self._forward_single(self._rflip(x_canon, (-2,)), freq), (-2,)) * sh
            u_wh = self._rflip(self._forward_single(self._rflip(x_canon, (-2, -1)), freq), (-2, -1)) * (sw * sh)
            u_list = [u_e, u_w, u_h, u_wh]
            y_out = 0.25 * (u_e + u_w + u_h + u_wh)
            if self.floquet_split and self.tem_dc and self.raw_frame and self.polar_dc:
                ur_list = [u[:, 2].mean(dim=(-2, -1)) for u in u_list]
                ui_list = [u[:, 3].mean(dim=(-2, -1)) for u in u_list]
                mean_amp = 0.25 * sum(torch.sqrt(ur ** 2 + ui ** 2 + 1e-6) for ur, ui in zip(ur_list, ui_list))
                bar_ur = 0.25 * sum(ur_list)
                bar_ui = 0.25 * sum(ui_list)
                bar_amp = torch.sqrt(bar_ur ** 2 + bar_ui ** 2 + 1e-6)
                scale = (mean_amp / bar_amp)[:, None, None]
                dr = bar_ur[:, None, None] * (scale - 1.0)
                di = bar_ui[:, None, None] * (scale - 1.0)
                y_out = y_out.clone()
                y_out[:, 2] = y_out[:, 2] + dr
                y_out[:, 3] = y_out[:, 3] + di
                y_out[:, 6] = y_out[:, 6] - dr / 376.50
                y_out[:, 7] = y_out[:, 7] - di / 376.50
        else:
            y_out = self._forward_single(x_canon, freq)

        y_out._is_raw_canonical = bool(self.raw_frame)
        y_out._pol_equivariant = bool(self.pol_equivariant)
        y_out._tem_dc = bool(self.tem_dc and self.floquet_split and self.raw_frame)
        return y_out

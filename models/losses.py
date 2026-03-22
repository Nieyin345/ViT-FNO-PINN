"""
PINN 损失函数 — 2D TM 点源散射，PML 吸收边界

控制方程（格点坐标，dl=1）:
    ∇²Ez + k²·ε_r·Ez = S_eff   (ceviche 约定，S_eff 为复数纯虚数)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fd_laplacian(u: torch.Tensor) -> torch.Tensor:
    """中心差分拉普拉斯，边界零填充，支持实数和复数张量。"""
    if u.is_complex():
        u_r = F.pad(u.real, (1, 1, 1, 1), mode="constant", value=0)
        u_i = F.pad(u.imag, (1, 1, 1, 1), mode="constant", value=0)
        lap_r = (u_r[:, :, 2:,  1:-1] + u_r[:, :, :-2, 1:-1]
               + u_r[:, :, 1:-1, 2:] + u_r[:, :, 1:-1, :-2] - 4 * u.real)
        lap_i = (u_i[:, :, 2:,  1:-1] + u_i[:, :, :-2, 1:-1]
               + u_i[:, :, 1:-1, 2:] + u_i[:, :, 1:-1, :-2] - 4 * u.imag)
        return torch.complex(lap_r, lap_i)
    else:
        u_pad = F.pad(u, (1, 1, 1, 1), mode="constant", value=0)
        return (u_pad[:, :, 2:,  1:-1]
              + u_pad[:, :, :-2, 1:-1]
              + u_pad[:, :, 1:-1, 2:]
              + u_pad[:, :, 1:-1, :-2]
              - 4 * u)


class PINNLoss(nn.Module):

    def __init__(
        self,
        k: float = 2 * np.pi * (4.0 / 64),
        eps_min: float = 1.0,
        eps_max: float = 4.0,
        pml_width: int = 10,
        lambda_pde: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_src: float = 10.0,
        lambda_nonzero: float = 0.1,
        lambda_data: float = 100.0,
        lambda_var: float = 0.0,
        lambda_amp: float = 1.0,
    ):
        super().__init__()
        self.k              = k
        self.eps_min        = eps_min
        self.eps_max        = eps_max
        self.pml_width      = pml_width
        self.lambda_pde     = lambda_pde
        self.lambda_bc      = lambda_bc
        self.lambda_src     = lambda_src
        self.lambda_nonzero = lambda_nonzero
        self.lambda_data    = lambda_data
        self.lambda_var     = lambda_var
        self.lambda_amp     = lambda_amp
        self.lambda_src_pde = 0.0  # 由外部设置

    def pde_loss(self, Ez: torch.Tensor, eps: torch.Tensor, source: torch.Tensor,
                 s_eff: torch.Tensor = None) -> torch.Tensor:
        """
        归一化复数亥姆霍兹残差（内部区域，排除源点附近）。
        源点附近 mask 掉，避免点源奇异性产生错误梯度。
        """
        p = self.pml_width
        eps_r = self.eps_min + (self.eps_max - self.eps_min) * eps  # (B,1,H,W)

        Ez_c = torch.complex(Ez[:, 0:1], Ez[:, 1:2])
        lap  = fd_laplacian(Ez_c)

        if s_eff is not None:
            sr = s_eff[:, 0].to(Ez.device).view(-1, 1, 1, 1)
            si = s_eff[:, 1].to(Ez.device).view(-1, 1, 1, 1)
        else:
            sr = torch.zeros(Ez.shape[0], 1, 1, 1, device=Ez.device)
            si = torch.full((Ez.shape[0], 1, 1, 1), 2.14, device=Ez.device)

        src_c    = torch.complex(sr, si) * source.to(torch.complex64)
        residual = lap + (self.k ** 2) * eps_r * Ez_c - src_c

        # 内部区域
        res_inner = residual[:, :, p:-p, p:-p]

        # 源点 mask：排除源点周围 3×3 区域，避免点源奇异性
        src_inner = source[:, :, p:-p, p:-p]  # (B,1,H',W')
        src_dilated = F.max_pool2d(src_inner, kernel_size=5, stride=1, padding=2)
        mask = (src_dilated < 0.5).float()     # 1=保留, 0=排除源点附近

        n_valid = mask.sum().clamp(min=1)
        src_amp = torch.complex(sr, si).abs().clamp(min=0.1)

        return ((res_inner.abs() / src_amp).pow(2) * mask).sum() / n_valid

    def src_pde_loss(self, Ez: torch.Tensor, eps: torch.Tensor, source: torch.Tensor,
                     s_eff: torch.Tensor = None) -> torch.Tensor:
        """
        源点局部约束：源点附近的亥姆霍兹残差应等于 S_eff。
        R_src = ∇²Ez + k²ε_r·Ez - S_eff ≈ 0  （仅在源点区域）
        """
        eps_r = self.eps_min + (self.eps_max - self.eps_min) * eps

        Ez_c = torch.complex(Ez[:, 0:1], Ez[:, 1:2])
        lap  = fd_laplacian(Ez_c)

        if s_eff is not None:
            sr = s_eff[:, 0].to(Ez.device).view(-1, 1, 1, 1)
            si = s_eff[:, 1].to(Ez.device).view(-1, 1, 1, 1)
        else:
            sr = torch.zeros(Ez.shape[0], 1, 1, 1, device=Ez.device)
            si = torch.full((Ez.shape[0], 1, 1, 1), 2.14, device=Ez.device)

        src_c    = torch.complex(sr, si) * source.to(torch.complex64)
        residual = lap + (self.k ** 2) * eps_r * Ez_c - src_c

        # 只在源点附近 5×5 区域生效
        src_dilated = F.max_pool2d(source, kernel_size=5, stride=1, padding=2)
        mask = (src_dilated > 0.5).float()  # 源点附近=1，其他=0

        n_valid = mask.sum().clamp(min=1)
        src_amp = torch.complex(sr, si).abs().clamp(min=0.1)
        return ((residual.abs() / src_amp).pow(2) * mask).sum() / n_valid

    def bc_loss(self, Ez: torch.Tensor) -> torch.Tensor:
        """PML 四边 Ez ≈ 0（实部+虚部）"""
        p = self.pml_width
        def bnd(u):
            return torch.cat([u[:, :, :p, :].reshape(-1),
                               u[:, :, -p:, :].reshape(-1),
                               u[:, :, :, :p].reshape(-1),
                               u[:, :, :, -p:].reshape(-1)])
        return torch.mean(bnd(Ez) ** 2)

    def src_loss(self, Ez: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """源点处 |Ez| 非零，log barrier 防零解"""
        mask = (source > 0.5)
        if not mask.any():
            return torch.tensor(0.0, device=Ez.device)
        amp = torch.sqrt(Ez[:, 0:1] ** 2 + Ez[:, 1:2] ** 2 + 1e-8)
        return torch.mean(1.0 / (amp[mask] + 1e-6))

    def nonzero_loss(self, Ez: torch.Tensor) -> torch.Tensor:
        """全场 |Ez| 非零，log barrier"""
        amp = torch.sqrt(Ez[:, 0:1] ** 2 + Ez[:, 1:2] ** 2 + 1e-8)
        return torch.mean(1.0 / (amp + 1e-6))

    def amplitude_loss(self, Ez: torch.Tensor) -> torch.Tensor:
        """惩罚幅值超过 1.5 的部分，防止无监督样本幅值爆炸"""
        amp     = torch.sqrt(Ez[:, 0:1] ** 2 + Ez[:, 1:2] ** 2 + 1e-8)
        amp_max = amp.flatten(1).max(dim=1).values
        excess  = torch.clamp(amp_max - 1.5, min=0.0)
        return torch.mean(excess ** 2)

    def data_loss(self, Ez_pred: torch.Tensor, Ez_true: torch.Tensor,
                  source: torch.Tensor = None, src_weight: float = 5.0) -> torch.Tensor:
        """per-sample 归一化 MSE（实部+虚部）"""
        amp   = torch.sqrt(Ez_true[:, 0:1] ** 2 + Ez_true[:, 1:2] ** 2)
        scale = amp.flatten(1).max(dim=1).values.clamp(min=1e-8)[:, None, None, None]
        return torch.mean((Ez_pred / scale - Ez_true[:, 0:2] / scale) ** 2)

    def forward(self, Ez, eps, source, Ez_true=None, s_eff=None,
                lam_pde=None, lam_data=None) -> dict:
        """
        lam_pde / lam_data: 可在外部动态传入，覆盖 self.lambda_*
        用于课程学习阶段动态调整权重。
        """
        lp = lam_pde  if lam_pde  is not None else self.lambda_pde
        ld = lam_data if lam_data is not None else self.lambda_data

        losses = {}
        losses["bc"]      = self.bc_loss(Ez)
        losses["src"]     = self.src_loss(Ez, source)
        losses["nonzero"] = self.nonzero_loss(Ez)
        losses["amp"]     = self.amplitude_loss(Ez)
        losses["pde"]     = self.pde_loss(Ez, eps, source, s_eff)
        losses["src_pde"] = self.src_pde_loss(Ez, eps, source, s_eff)

        total = (self.lambda_bc      * losses["bc"]
               + self.lambda_src     * losses["src"]
               + self.lambda_nonzero * losses["nonzero"]
               + self.lambda_amp     * losses["amp"]
               + lp                  * losses["pde"]
               + self.lambda_src_pde * losses["src_pde"])

        if Ez_true is not None:
            losses["data"] = self.data_loss(Ez_pred=Ez, Ez_true=Ez_true)
            total = total + ld * losses["data"]

        losses["total"] = total
        return losses

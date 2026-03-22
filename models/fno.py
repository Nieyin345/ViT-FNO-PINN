"""
FNO2d with Transformer Geometry Encoder (FiLM modulation)

架构:
  - ViT Patch Encoder: 编码 permittivity 图 → 几何特征向量
  - FNOBlock: 谱卷积 + 旁路卷积，被几何特征 FiLM 调制
  - 输出: (B, 2, H, W) — Ez 实部 + 虚部

FiLM 调制: 每个 FNOBlock 接收 (scale, shift) 对特征做仿射变换
  out = scale * norm(x) + shift
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── 谱卷积 ───────────────────────────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2)
        return torch.fft.irfft2(out_ft, s=(H, W))


# ─── FNOBlock with FiLM ───────────────────────────────────────────────────────

class FNOBlock(nn.Module):
    """
    FNO 层 + FiLM 调制。
    film_vec: (B, 2*width) — 前半是 scale，后半是 shift
    """
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.width   = width
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.bypass   = nn.Conv2d(width, width, kernel_size=1)
        self.norm     = nn.InstanceNorm2d(width, affine=False)

    def forward(self, x: torch.Tensor, film_vec: torch.Tensor = None) -> torch.Tensor:
        h = self.spectral(x) + self.bypass(x)
        h = self.norm(h)

        if film_vec is not None:
            scale = film_vec[:, :self.width].view(-1, self.width, 1, 1)
            shift = film_vec[:, self.width:].view(-1, self.width, 1, 1)
            h = scale * h + shift

        return F.gelu(h)


# ─── ViT Patch Encoder ────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """把 (B,1,H,W) 的 permittivity 图切成 patch，线性嵌入"""
    def __init__(self, img_size: int = 64, patch_size: int = 8, embed_dim: int = 128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) → (B, n_patches, embed_dim)
        x = self.proj(x)                    # (B, embed_dim, H/p, W/p)
        return x.flatten(2).transpose(1, 2) # (B, n_patches, embed_dim)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class GeometryEncoder(nn.Module):
    """
    轻量 ViT，编码 permittivity 图为几何特征向量。
    输出: (B, n_layers * 2 * width) — 每层 FNOBlock 的 FiLM 参数
    """
    def __init__(
        self,
        img_size:   int = 64,
        patch_size: int = 8,
        embed_dim:  int = 128,
        depth:      int = 3,
        n_heads:    int = 4,
        n_fno_layers: int = 4,
        fno_width:  int = 32,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        n_patches = (img_size // patch_size) ** 2

        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 全局池化后映射到每层 FiLM 参数 (scale + shift = 2*width)
        film_dim = n_fno_layers * 2 * fno_width
        self.film_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, film_dim),
        )
        self.n_fno_layers = n_fno_layers
        self.fno_width    = fno_width

        # scale 初始化为 1，shift 初始化为 0（恒等变换）
        nn.init.zeros_(self.film_head[-1].bias)
        with torch.no_grad():
            w = self.film_head[-1].weight
            w.zero_()

    def forward(self, eps: torch.Tensor) -> list:
        """
        eps: (B, 1, H, W)
        返回: list of (B, 2*width)，长度 = n_fno_layers
        """
        x = self.patch_embed(eps) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(dim=1)   # 全局平均池化 (B, embed_dim)

        film = self.film_head(x)       # (B, n_layers * 2 * width)

        # 拆成每层的 film_vec，scale 初始化偏置为 1
        W = self.fno_width
        film_list = []
        for i in range(self.n_fno_layers):
            vec = film[:, i * 2 * W: (i + 1) * 2 * W]
            # scale 部分加 1（初始为恒等）
            scale = vec[:, :W] + 1.0
            shift = vec[:, W:]
            film_list.append(torch.cat([scale, shift], dim=1))

        return film_list


# ─── 完整模型 ─────────────────────────────────────────────────────────────────

class FNO2d(nn.Module):
    """
    FNO2d + Transformer Geometry Encoder (FiLM modulation)

    Args:
        modes1, modes2:  谱卷积保留的频率模态数
        width:           FNO 隐层通道数
        n_layers:        FNO 层数
        in_channels:     输入通道数
        out_channels:    输出通道数
        use_geo_encoder: 是否启用 Transformer 几何编码器
        geo_embed_dim:   Transformer 嵌入维度
        geo_depth:       Transformer 层数
        patch_size:      ViT patch 大小
    """
    def __init__(
        self,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 32,
        n_layers: int = 4,
        in_channels: int = 2,
        out_channels: int = 2,
        use_geo_encoder: bool = True,
        geo_embed_dim: int = 128,
        geo_depth: int = 3,
        patch_size: int = 8,
    ):
        super().__init__()
        self.use_geo_encoder = use_geo_encoder

        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        self.blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2) for _ in range(n_layers)
        ])

        # 输出投影：先 3×3 平滑再 1×1 投影，抑制高频伪影
        self.proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

        if use_geo_encoder:
            self.geo_encoder = GeometryEncoder(
                img_size=64,
                patch_size=patch_size,
                embed_dim=geo_embed_dim,
                depth=geo_depth,
                n_fno_layers=n_layers,
                fno_width=width,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)  — [source, permittivity]
        """
        eps = x[:, 1:2]   # permittivity 通道，送给 Transformer

        film_list = None
        if self.use_geo_encoder:
            film_list = self.geo_encoder(eps)

        x = self.lift(x)
        for i, block in enumerate(self.blocks):
            fv = film_list[i] if film_list is not None else None
            x = block(x, fv)

        return self.proj(x)

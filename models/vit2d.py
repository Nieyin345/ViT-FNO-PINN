"""ViT 基线：空间域 patch 化 + 全局自注意力。

**它证明什么**（和 `cnn` / `fno_baseline` / `afno` 一起才构成完整的对照网）：

| 模型       | 混合范围 | 在哪个域 | 混合方式            |
|-----------|---------|---------|--------------------|
| `mlp`     | 1×1     | 空间    | 逐点 MLP           |
| `cnn`     | 局部     | 空间    | 3×3 卷积堆叠        |
| `unet`    | 多尺度   | 空间    | 编解码 + skip       |
| **`vit`** | **全局** | **空间** | **自注意力**        |
| `afno`    | 全局     | 频域    | 逐频率对的共享 MLP   |
| `fno_*`   | 全局     | 频域    | 每频率对独立的谱卷积 |

`vit` 与 `afno` 是同一件事（全局混合）在两个域的实现；`vit` 与 `fno_baseline`
则分离了"全局混合有用"与"频域参数化有用"。ViT 的归纳偏置最弱（无平移等变性
的显式构造、无频域结构先验），若它在小数据上明显更差，那是意料之中——
但仍值得跑，因为"注意力在小数据上打不过卷积"本身就是一个可引用的结论。

**实现要点**（保持小而可控）：
  - patch 大小由 `extra["vit_patch"]` 决定（默认 8 → 8×8 = 64 个 token）
  - 输入通道通过一个 patch-wise 卷积（stride = patch）直接线性投影成 token，
    **不使用** `FNOBase` 的 lift/pad/块循环骨架（ViT 的 token 化与它不兼容），
    因此 `forward` 整体覆盖，`self.blocks` 仍填入 Transformer 层以保持结构可读
  - 条件化：freq/geom 嵌入 → 一个可学习的 CLS token 初值 + 每个 block 的 adaLN 调制
  - 注意力用 `F.scaled_dot_product_attention`（PyTorch 2.x 自带，无需新依赖）
  - 输出头：token 序列 reshape 回 (B, C, H/p, W/p) → 逐像素线性 → 上采样回原分辨率

extra 键：
  `vit_patch`   patch 大小，默认 8
  `vit_heads`   注意力头数，默认 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from core.base import Cond, FNOBase, normalize_freq
from core.config import ModelConfig
from core.encoders import FourierFreqEncoder, SDFGeomEncoder
from core.registry import register


class ViTBlock(nn.Module):
    """Pre-LN Transformer block + adaLN 条件调制。

    条件向量生成每通道的 (scale, shift)，同时调制 attention 前与 MLP 前的归一化
    —— 这是 DiT / adaLN-Zero 的做法，零初始化时 block 退化为恒等，训练稳定。
    """

    def __init__(self, dim: int, n_heads: int, cond_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        # adaLN：cond -> (6 组 scale/shift) = attn 的 q/k/v 前 + mlp 前
        self.ada = nn.Sequential(nn.Linear(cond_dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim * 6))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        B, N, D = x.shape
        if cond is not None:
            p = self.ada(cond).view(B, 6, D)          # (B,6,D)
            s1, sh1, s2, sh2, g1, g2 = [p[:, i] for i in range(6)]
            s1, sh1 = s1.unsqueeze(1), sh1.unsqueeze(1)
            s2, sh2 = s2.unsqueeze(1), sh2.unsqueeze(1)
            g1, g2 = torch.sigmoid(g1).unsqueeze(1), torch.sigmoid(g2).unsqueeze(1)
        else:
            s1 = sh1 = s2 = sh2 = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
            g1 = g2 = torch.ones(B, 1, D, device=x.device, dtype=x.dtype)

        h = self.norm1(x) * (1.0 + s1) + sh1
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1 * h
        h = self.norm2(x) * (1.0 + s2) + sh2
        return x + g2 * self.mlp(h)


@register("vit")
class ViT2d(FNOBase):
    """Vision Transformer 基线：空间域全局注意力。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.patch = max(1, int(cfg.extra.get("vit_patch", 8)))
        embed_dim = int(cfg.extra.get("vit_dim", cfg.width))
        n_heads = int(cfg.extra.get("vit_heads", 4))
        while embed_dim % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.embed_dim = embed_dim

        # patch 化 = stride 卷积；把 (in_ch + cond_ch) 直接投影到 token 维度
        self.to_tokens = nn.Conv2d(
            cfg.in_channels + int(cfg.cond_channels), embed_dim,
            kernel_size=self.patch, stride=self.patch,
        )

        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=cfg.freq_dim)
        self.geom_encoder = SDFGeomEncoder(n_modes=4, out_dim=cfg.freq_dim)
        cond_dim = 2 * cfg.freq_dim
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, embed_dim * 2), nn.GELU(),
                                       nn.Linear(embed_dim * 2, embed_dim))

        # CLS token 由条件向量**直接生成**（省一次注意力），所以没有可学习的
        # 初值参数。原先这里有个 `self.cls_token = nn.Parameter(zeros(1,1,D))`，
        # 只在 `freq is None` 的兜底分支里用到 —— 而训练/评估/传播全链路都带
        # freq，它永远拿不到梯度，却照样进 state_dict 和优化器。
        # freq=None 是"简化推理"路径，条件本就该是零，直接给零张量即可。

        # 基座的 lift→pad→块循环→crop 骨架在本模型里用不上：token 化要求
        # stride=patch 的卷积，与 padding 不兼容。故整体覆盖 forward，
        # blocks 只用于让基座/检查点看到 Transformer 层。
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, n_heads, cond_dim)
            for _ in range(cfg.n_layers)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)
        # dropout 统一放在输出头（head）之前，与其余 10 个模型同一口径。
        # 原先分散在每个 block 内部（attention dropout + MLP 里两处 Dropout），
        # n_layers=5 时一次 forward 触发 10 次 —— 消融表里混进了"dropout 强度
        # 与位置"这个额外变量。放在 head 前 = 作用在最终 token 表示上，
        # 与 deeponet"作用在 branch 系数上"的定位一致。
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        # 逐 token → 逐像素：把 embed_dim 映射到 patch²×out_channels
        self.head = nn.Linear(embed_dim, self.patch * self.patch * cfg.out_channels)

    def build_lift(self) -> nn.Module:
        """本模型整体覆盖 forward，基座骨架用不上 —— lift 建成恒等。

        否则基座会按 `cfg.in_channels` 建一个 Conv2d，它进 state_dict、
        进优化器，却因为没有调用点而**永远拿不到梯度**。
        """
        return nn.Identity()

    # ViT 不走基座的 lift/pad/块循环，整体自己来（token 化与 padding 不兼容）
    def forward(self, x: torch.Tensor, freq: Optional[torch.Tensor] = None) -> torch.Tensor:
        freq = normalize_freq(freq, x.shape[0])
        B, _, H, W = x.shape
        if H % self.patch or W % self.patch:
            raise ValueError(f"分辨率 {H}×{W} 不能被 patch={self.patch} 整除")

        x_in = torch.cat([x, self._broadcast_condition(x, freq)], dim=1)
        tokens = self.to_tokens(x_in)                                   # (B, D, gh, gw)
        gh, gw = tokens.shape[-2:]
        tokens = tokens.flatten(2).transpose(1, 2)                      # (B, N, D)

        cond_vec = None
        if freq is not None:
            cond_vec = torch.cat([self.freq_encoder(freq), self.geom_encoder(x)], dim=-1)
            cls = self.cond_proj(cond_vec).unsqueeze(1)                 # (B, 1, D)
        else:
            # 无条件兜底路径：CLS 与 adaLN 条件都为 0（ada 末层本就零初始化，
            # 无条件时 block 退化为恒等的标准 Transformer）
            cls = tokens.new_zeros(B, 1, self.embed_dim)
        tokens = torch.cat([cls, tokens], dim=1)                        # (B, 1+N, D)

        for blk in self.blocks:
            tokens = blk(tokens, cond_vec)

        tokens = self.norm_out(tokens)[:, 1:]                           # 丢掉 CLS
        out = self.head(self.dropout(tokens))                           # (B, N, p²·C)
        out = out.reshape(B, gh, gw, self.patch, self.patch, self.cfg.out_channels)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, self.cfg.out_channels, gh * self.patch,
                                                    gw * self.patch)
        if (gh * self.patch, gw * self.patch) != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out

    def _broadcast_condition(self, x, freq):
        B, _, H, W = x.shape
        C = int(self.cfg.cond_channels)
        if freq is None:
            return torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
        n_cond = freq.shape[1]
        if n_cond < C:
            pad = torch.zeros(B, C - n_cond, device=x.device, dtype=x.dtype)
            freq = torch.cat([freq, pad], dim=1)
        return freq[:, :C].view(B, C, 1, 1).expand(B, C, H, W)

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        """ViT 的 forward 已整体覆盖，此方法不会被基座调用（仅为满足抽象协议）。"""
        raise NotImplementedError("ViT2d 在 forward 内完成全部输出映射")

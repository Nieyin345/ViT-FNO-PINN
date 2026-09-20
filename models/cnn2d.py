"""全卷积残差 CNN 基线（与 U-Net 互补的空间域对照）。

**它和 unet 的区别，是这两条曲线一起才能说清的事**：
  - `unet`：有下采样/上采样 + skip。感受野大，但走的是"编码到低分辨率再解码"
    的路子，高频细节靠 skip 补。
  - 本模型：**全程 64×64，尺度不变**。感受野靠堆 3×3 卷积与膨胀（dilation）
    拉大，靠残差连接保证能训深。没有池化、没有上采样、没有 skip。
  - `mlp`：感受野 1×1。

三者构成"感受野 1 → 局部 → 多尺度"的完整谱系。若 `cnn` ≈ `unet`，说明
U-Net 的多尺度结构并非关键，感受野够大就行；若 `unet` 明显更好，说明
超表面响应的多尺度性是真的。

**和 FNO 系列的对照**：同为"宽度 × 层数"的堆叠结构，唯一差别是 3×3 空间卷积
vs 频域全局卷积——这是"频域算子是否必要"最干净的对照（`fno_baseline` 堆的层数
和它一样，但每层是频域卷积）。

条件化：空间通道广播（同 baseline / unet / mlp），`build_cond` 返回 None。

extra 键：
  `cnn_dilation_cycle`  膨胀循环长度，默认 3（1,2,4,1,2,4,...）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from core.base import Cond, FNOBase
from core.config import ModelConfig
from core.registry import register


class ResConvBlock(nn.Module):
    """两个 3×3（可膨胀）卷积 + BN + GELU 的残差块。

    形状不变（padding = dilation），故 64×64 全程保持。
    """

    def __init__(self, width: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, cond: Optional[Cond] = None):
        h = F.gelu(self.norm1(x))
        h = self.conv1(h)
        h = F.gelu(self.norm2(h))
        h = self.conv2(h)
        return x + self.dropout(h)


@register("cnn")
class CNN2d(FNOBase):
    """全卷积残差网络：无下采样、无频域操作。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.in_channels = cfg.in_channels
        self.cond_channels = int(cfg.cond_channels)
        self.dilation_cycle = max(1, int(cfg.extra.get("cnn_dilation_cycle", 3)))

        # 注意 lift 在基座 super().__init__ 里就建好了，这里**不能**再赋值
        # （重建会把基座那个模块留在 _modules 里成为死参数，还会进优化器）。
        self.blocks = nn.ModuleList([
            ResConvBlock(cfg.width, dilation=2 ** (i % self.dilation_cycle), dropout=cfg.dropout)
            for i in range(cfg.n_layers)
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(cfg.width, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, cfg.out_channels, kernel_size=1),
        )

    def lift_in_channels(self) -> int:
        # 基座按 cfg.in_channels 建 lift；本模型额外拼 cond_channels 个条件图。
        # 只读 self.cfg —— 本方法在 super().__init__ 内部被调用。
        return int(self.cfg.in_channels) + int(self.cfg.cond_channels)

    def build_lift(self) -> nn.Module:
        # 与 mlp / baseline 的 1×1 不同，本模型是空间卷积网络，lift 用 3×3
        return nn.Sequential(
            nn.Conv2d(self.lift_in_channels(), self.cfg.width, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def _broadcast_condition(self, x, freq):
        B, _, H, W = x.shape
        if freq is None:
            return torch.zeros(B, self.cond_channels, H, W, device=x.device, dtype=x.dtype)
        # freq 已由 FNOBase.forward 里的 normalize_freq 规范成 (B, 2)
        n_cond = freq.shape[1]
        if n_cond < self.cond_channels:
            pad = torch.zeros(B, self.cond_channels - n_cond, device=x.device, dtype=x.dtype)
            freq = torch.cat([freq, pad], dim=1)
        cond = freq[:, :self.cond_channels]
        return cond.view(B, self.cond_channels, 1, 1).expand(B, self.cond_channels, H, W)

    def build_input(self, x, freq):
        return torch.cat([x, self._broadcast_condition(x, freq)], dim=1)

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

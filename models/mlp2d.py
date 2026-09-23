"""逐像素 MLP 基线（参照系，不是有希望的候选）。

**它证明什么**：把结论钉死成"空间/频域结构确实有用"。
本模型对 64×64 网格上的每个点独立做同一个 MLP 映射，输入只有该点的
(SDF, Mask) 两通道 + 广播进来的频率/极化常数，**完全不看邻居**。
所以它不可能预测出任何由形状边缘、绕射、边缘电流产生的空间结构。

如果它的 val_geom 明显差于 FNO / U-Net / CNN，说明"感受野"是必需的；
如果差距很小，说明我们的指标可能被不能反映空间结构的量（如 DC 分量）主导，
那时该去查指标而不是继续堆模型——这条信息同样有价值。

条件化：和 `fno_baseline` / `unet` 一样走"空间通道广播"，`build_cond` 返回 None。
用 `FNOBase` 是因为它自带 lift/pad/块循环/crop 骨架——每层 1×1 卷积等价于
逐像素 MLP，padding 对 1×1 卷积无影响（cfg.padding 在实验里设成 0 只为省算力）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from core.base import Cond, FNOBase
from core.config import ModelConfig
from core.registry import register


class PointwiseBlock(nn.Module):
    """1×1 卷积（= 逐像素线性层）+ BN + GELU 残差。

    没有任何跨像素操作，这是本基线成立的前提。
    """

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(width)
        self.net = nn.Sequential(
            nn.Conv2d(width, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, width, kernel_size=1),
        )

    def forward(self, x, cond: Optional[Cond] = None):
        return F.gelu(self.norm(x + self.net(x)))


@register("mlp")
class MLP2d(FNOBase):
    """逐像素 MLP 基线：感受野恒为 1×1。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.in_channels = cfg.in_channels
        self.cond_channels = int(cfg.cond_channels)
        # extra 里的同名键优先，缺省取 width 的 2 倍
        self.hidden = int(cfg.extra.get("mlp_hidden", cfg.width * 2))

        # 注意 lift 在基座 super().__init__ 里就建好了，这里**不能**再赋值
        # （重建会把基座那个模块留在 _modules 里成为死参数，还会进优化器）。
        self.blocks = nn.ModuleList([
            PointwiseBlock(cfg.width, self.hidden)
            for _ in range(cfg.n_layers)
        ])
        # dropout 统一放在输出头之前，与其余 10 个模型一致。
        # 原先放在 block 内部（net 序列中间），n_layers=5 时一次 forward 触发
        # 5 次、位置也不同 —— 消融表里就混进了"dropout 强度与位置"这个额外变量。
        self.dropout = nn.Dropout2d(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.proj = nn.Sequential(
            nn.Conv2d(cfg.width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, cfg.out_channels, kernel_size=1),
        )

    def lift_in_channels(self) -> int:
        # 基座按 cfg.in_channels 建 lift；本模型额外拼 cond_channels 个条件图。
        # 只读 self.cfg —— 本方法在 super().__init__ 内部被调用。
        return int(self.cfg.in_channels) + int(self.cfg.cond_channels)

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
        return self.proj(self.dropout(x))

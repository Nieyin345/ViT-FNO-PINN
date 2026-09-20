# -*- coding: utf-8 -*-
"""统一模型基座：`Cond` 条件载体 + `FNOBase` 骨架。

重构背景（原 fno_core.FNOBase）：
    `FNOBase.forward` 固定用 `block(out, freq_emb=..., geom_emb=...)` 调块，
    而 FNO2dV2 的块是 `block(out, cond)`，协议不合，于是 FNO2dV2 只能不继承基座、
    把 lift/pad/循环/crop/head 抄一遍（约 15 行重复）。

现在统一块协议为 `block(x, cond: Cond | None)`，并把模型之间的差异收敛到三个钩子：

    build_input(x, freq) -> 送入 lift 的输入          （V2 在这里拼正弦通道）
    build_cond(x, freq)  -> Cond | None               （各模型在这里产条件嵌入）
    output_head(x)       -> 输出                      （子类必须实现）

`forward(x, freq)` 是全仓库唯一的模型调用签名（所有消费方都已是这个形式）。
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


@dataclass
class Cond:
    """块级条件载体，取代原来的 freq_emb / geom_emb / cond 三套互不兼容的接口。

    - `fused()`：把非 None 的分量按 [freq, pol, geom] 顺序拼接，供
      "核条件化"类块（FNO2dV2 的 SpectralConv2dV2）使用。
    - 各分量单独可取，供 FiLM / LoRA / 注意力类块按需组合。
    """

    freq_emb: Optional[torch.Tensor] = None
    geom_emb: Optional[torch.Tensor] = None
    pol_emb: Optional[torch.Tensor] = None

    def fused(self) -> Optional[torch.Tensor]:
        parts = [e for e in (self.freq_emb, self.pol_emb, self.geom_emb) if e is not None]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)

    def is_empty(self) -> bool:
        return self.freq_emb is None and self.geom_emb is None and self.pol_emb is None


def normalize_freq(freq: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
    """把 `freq` 规范成 (B, 2) = [归一化频率, 极化]。

    权威契约来自 `data/dataset.py`：每条样本给出 (2,)，DataLoader 批成 (B, 2)。
    但手绘推理等路径历史上传过 1D（`torch.tensor([freq_norm])` 或
    `[freq_norm, pol]`），缺 batch 维。窄 slice 的模型（FNO2dV2 的 `freq[:, 1:2]`）
    会直接 IndexError，宽 slice 的模型则靠广播"侥幸跑通"——同一个 bug 时隐时现。

    这里统一补齐：1D 视为**单样本**，补成 (1, 2)；缺极化列时补 0（极化 0）。
    已是 (B, 2) 的原样返回，不改动正常路径的数值。
    """
    if freq is None:
        return None
    if freq.dim() == 1:
        freq = freq.unsqueeze(0)
    elif freq.dim() > 2:
        raise ValueError(f"freq 维度应 <= 2，收到 shape={tuple(freq.shape)}")
    if freq.shape[0] != batch_size:
        raise ValueError(
            f"freq 的 batch 维 ({freq.shape[0]}) 与输入 x ({batch_size}) 不一致；"
            f"freq.shape={tuple(freq.shape)}"
        )
    if freq.shape[1] < 2:
        pad = freq.new_zeros(freq.shape[0], 2 - freq.shape[1])
        freq = torch.cat([freq, pad], dim=1)
    return freq


class FNOBase(nn.Module):
    """FNO 抽象基类：统一 Lift 提升、Padding、Block 级联、头输出。

    子类需要：
      1. `super().__init__(cfg)` 后填充 `self.blocks`
      2. 实现 `output_head`
      3. 按需覆盖 `build_input` / `build_cond`
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.padding = cfg.padding
        self.lift = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.width, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList()

    # ------------------------------------------------------------- 三个钩子
    def build_input(self, x: torch.Tensor, freq: Optional[torch.Tensor]) -> torch.Tensor:
        """送入 lift 之前的输入变换。默认恒等。

        "空间通道广播"式条件化的模型（baseline/unet）在这里把条件拼成常量图；
        FNO2dV2 在这里拼频率正弦嵌入通道。
        """
        return x

    def build_cond(self, x: torch.Tensor, freq: Optional[torch.Tensor]) -> Optional[Cond]:
        """产出块级条件。默认无条件（纯 FNO）。"""
        return None

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("子类必须实现 output_head 方法")

    # ----------------------------------------------------------------- 前向
    def forward(self, x: torch.Tensor, freq: Optional[torch.Tensor] = None) -> torch.Tensor:
        freq = normalize_freq(freq, x.shape[0])
        x_in = self.build_input(x, freq)
        cond = self.build_cond(x, freq)

        out = self.lift(x_in)
        if self.padding > 0:
            out = F.pad(out, [0, self.padding, 0, self.padding])

        for block in self.blocks:
            out = block(out, cond)

        if self.padding > 0:
            out = out[..., :-self.padding, :-self.padding]

        return self.output_head(out)

    # ------------------------------------------------------------- 工具方法
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

# -*- coding: utf-8 -*-
"""条件编码器：所有模型共用的嵌入模块。

重构背景：仓库里曾存在**两个功能不同的同名 `SDFGeomEncoder`**：
  - `fno_core.py` 的：n_modes=4，只用通道 0（SDF），real+imag，hidden=64，出 32 维
  - `fno2d_lora.py` 的：n_modes=8，用通道 0+1（SDF+Mask），mag+phase，hidden=128，出 64 维

同名不同实现是隐蔽的坑（import 错了不会报错，只会悄悄换语义）。
现在合并为一个可参数化实现，两种数值行为都原样保留：

    # 原 fno_core 版行为（FNO2d / FNO2dFiLM / FNO2dV2）
    SDFGeomEncoder(n_modes=4, out_dim=32, channels=(0,), amp_phase=False, hidden=64)

    # 原 fno2d_lora 版行为（FNO2dLoRA）
    SDFGeomEncoder(n_modes=8, out_dim=64, channels=(0, 1), amp_phase=True, hidden=128)
"""

import torch
import torch.nn as nn


class FourierFreqEncoder(nn.Module):
    """频率条件编码：多尺度 sin/cos 傅里叶特征 → MLP。

    接受 (B,1) 旧接口与 (B,2)=[归一化频率, 极化] 新接口。
    多列时**按列分别编码后求平均**（频率与极化被混叠，故 FNO2dV2 改用 pol_mlp 分开处理）。
    """

    def __init__(self, n_freqs: int = 16, out_dim: int = 32):
        super().__init__()
        scales = 2 ** torch.linspace(0, 4, n_freqs)
        self.register_buffer("scales", scales)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freqs, 64), nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, freq: torch.Tensor) -> torch.Tensor:
        if freq.ndim == 1:
            freq = freq.unsqueeze(0)
        x = freq.unsqueeze(-1) * self.scales                      # (B, K, N)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)       # (B, K, 2N)
        return self.mlp(x).mean(dim=1)


class SDFGeomEncoder(nn.Module):
    """几何条件编码：对输入通道做 rfft2，取低频块作为形状特征 → MLP。

    Args:
        n_modes:   低频块边长（取 [:n_modes, :n_modes]）
        out_dim:   输出嵌入维度
        channels:  使用哪些输入通道（0=SDF, 1=Mask）
        amp_phase: True 用 (幅值, 相位)；False 用 (实部, 虚部)
        hidden:    MLP 隐层宽度
    """

    def __init__(self, n_modes: int = 4, out_dim: int = 32,
                 channels=(0,), amp_phase: bool = False, hidden: int = 64):
        super().__init__()
        self.n_modes = int(n_modes)
        self.channels = tuple(int(c) for c in channels)
        self.amp_phase = bool(amp_phase)
        # 每个通道贡献 2 * n_modes² 个实数（两种编码方式数量相同）
        feat_dim = len(self.channels) * self.n_modes * self.n_modes * 2
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def _channel_feat(self, field: torch.Tensor) -> torch.Tensor:
        """field: (B,H,W) -> (B, 2*n_modes²)"""
        ft = torch.fft.rfft2(field)
        low = ft[:, :self.n_modes, :self.n_modes]
        if self.amp_phase:
            first, second = low.abs(), torch.angle(low)
        else:
            first, second = low.real, low.imag
        return torch.cat([
            first.reshape(field.size(0), -1),
            second.reshape(field.size(0), -1),
        ], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self._channel_feat(x[:, c, :, :]) for c in self.channels]
        return self.mlp(torch.cat(feats, dim=1) if len(feats) > 1 else feats[0])


class DirectionHead(nn.Module):
    """单复分量输出头：映射到 2 通道（实部 + 虚部）。"""

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

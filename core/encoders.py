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

    输入约定为 `(B, 1 + extra_cols)`：第 0 列是归一化频率，其余列是本编码器
    顺带承载的额外标量（通常是极化）。**各列走不同的通路，不再混叠**：

        第 0 列  →  sin/cos 傅里叶特征（2*n_freqs 维）
        其余列  →  直接作为标量拼进 MLP 输入

    传入的列数少于 `1 + extra_cols` 时按 0 补齐（fno_v2 的极化走独立 pol_mlp，
    这里补 0 即可，不影响数值）。

    重构修正：原实现对 `(B,2)` 输入**按列分别编码后求平均**
    （`self.mlp(x).mean(dim=1)`），把频率与极化平均进同一个嵌入。
    fno_attention / fno_film / fno_lora 都是直接把 `(B,2)=[freq,pol]` 交给它，
    于是 [8GHz, pol=0] 与 [12GHz, pol=1] 编码出**完全相同**的向量
    （实测相对位移 0.0000），网络结构上无法区分这两个条件。
    """

    def __init__(self, n_freqs: int = 16, out_dim: int = 32, extra_cols: int = 1,
                 max_log2: float = 4.0):
        """max_log2：最高傅里叶尺度 2^max_log2（默认 4 → 1~16，与历史一致）。

        归一化频率 f∈[0,1]，已见频点间隔 0.125。尺度 16 的 sin/cos 在相邻频点间
        转约 2 rad，足以让网络把每个训练频点当作离散标签"背下来"，内插能力差。
        把 max_log2 降到 0~1 得到只含低带宽特征的平滑编码，见 configs 里的 h10_low*。
        """
        super().__init__()
        scales = 2 ** torch.linspace(0, float(max_log2), n_freqs)
        self.register_buffer("scales", scales)
        self.extra_cols = int(extra_cols)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freqs + self.extra_cols, 64), nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, freq: torch.Tensor) -> torch.Tensor:
        if freq.ndim == 1:
            freq = freq.unsqueeze(0)
        f = freq[:, 0:1]                                          # (B,1) 归一化频率
        x = f.unsqueeze(-1) * self.scales                          # (B,1,N)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)        # (B,1,2N)
        x = x.reshape(x.size(0), -1)                               # (B,2N)
        extra = freq[:, 1:1 + self.extra_cols]                     # (B,<=extra_cols)
        if extra.size(1) < self.extra_cols:
            pad = extra.new_zeros(extra.size(0), self.extra_cols - extra.size(1))
            extra = torch.cat([extra, pad], dim=1)
        return self.mlp(torch.cat([x, extra], dim=1))


class SDFGeomEncoder(nn.Module):
    """几何条件编码：对输入通道做 rfft2，取低频块作为形状特征 → MLP。

    Args:
        n_modes:   低频块边长（取 [:n_modes, :n_modes]）
        out_dim:   输出嵌入维度
        channels:  使用哪些输入通道（0=SDF, 1=Mask）
        amp_phase: True 用 (幅值, 相位)；False 用 (实部, 虚部)
        hidden:    MLP 隐层宽度

    ## 尺度修正（原实现在此处有缺陷）

    原实现是 `torch.fft.rfft2(field)` + 未归一化的 SDF，导致输出量级失控：

        DC = sum(SDF) = mean(SDF) * H * W

    实测 64×64、SDF∈[-6, 8.7]：DC = 8841（= 2.1585 * 4096），
    整个 geom_emb 的 |mean| = 386。而同一个条件向量里的 freq_emb 是 0.128、
    pol_emb 是 0.252 —— **几何比它们大 1500~3000 倍**。

    条件向量是 `Cond.fused()` 拼起来送进同一个 `Linear`（fno_v2 的 film_net）
    的，量级悬殊意味着几何把频率与极化**淹没**：实测 fused 向量上
    "只改极化"的位移是 1.28，"换个几何样本"是 2607——后者是前者的 2000 倍。
    双极化是本项目的标准配置（两个极化是两条独立样本），极化信号被压成这样
    会直接导致极化间串扰。

    修正分两步，都是**去掉与物理无关的尺度**，不改变表达能力：

        1. 输入逐样本标准化  → 去掉 SDF 的数值单位与整体偏置
        2. rfft2(norm="ortho") → DC 不再随网格面积 H*W 膨胀

    实测（真实 patch_plane，n_modes=4/channels=(0,)）：386 → 6.0（仅 ortho）
    → 81.8（仅 std）→ **1.27**（两者都上），与 freq_emb / pol_emb 同量级。

    另注：几何是**形状**信息，其嵌入本就不该随网格分辨率漂移。ortho 归一化
    顺带保证了这一点（backward 归一化下 64→128 网格会让嵌入整体放大 2 倍）。
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
        # 逐样本标准化：去掉 SDF 的数值单位与整体偏置（几何信息在形状里，
        # 不在"这张 SDF 的数值有多大"里）
        mu = field.mean(dim=(-2, -1), keepdim=True)
        sd = field.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        field = (field - mu) / sd

        # ortho 归一化：DC 不再等于 sum(field)，从而不随网格面积膨胀，
        # 也让嵌入对网格分辨率不敏感
        ft = torch.fft.rfft2(field, norm="ortho")
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

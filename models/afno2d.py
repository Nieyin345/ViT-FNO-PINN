"""AFNO 基线：频域 MLP + 通道混合，但**没有**频域全局卷积。

**它和 FNO 的关系**（这是引入它的唯一理由）：

    FNO 的 `SpectralConv2d`：out_ft[i, x, y] = Σ_j x_ft[j, x, y] · w[j, i, x, y]
        —— 每个频率对 (x,y) 有**独立**的 in→out 复数矩阵（维度 in×out），
           且没有非线性。它本身就是"全局卷积算子"的参数化，表达力最强。

    AFNO 块：out_ft[·, x, y] = W2 · GELU(W1 · x_ft[·, x, y] + b1) + b2
        —— 所有频率对**共享**同一对复数矩阵 W1(in→hidden) / W2(hidden→out)，
           中间夹一层 GELU（因此是"逐频率对的 MLP"，非线性但共享权重）。

所以 AFNO 是一个**严格弱于** FNO 的频域算子：参数量小得多（in·out·modes² 个
复数 → 两对矩阵），代价是失去"每个频率对独立调权"的能力。它回答的问题是：

    FNO 的收益到底来自"在频域做全局混合"，还是来自"每个频率对都能独立调权"？

    若 `afno` ≈ `fno_baseline`，说明前者就够了，SpectralConv2d 的大量参数是浪费，
    主模型可以大幅瘦身；若 `afno` 明显更差，说明逐频率自由度是必要的。

**和 ViT 的区别**：ViT 在**空间域**做全局自注意力（token = 空间位置）；
AFNO 在**频域**做逐 token 的 MLP 混合（token = 频率对）。两者都是全局混合，
但一个在空间域学，一个在频域学。三个一起才能分离"全局 vs 局部"与
"空间域 vs 频域"这两个因素。

条件化：编码器嵌入 —— FourierFreqEncoder(freq) + SDFGeomEncoder(x) → Cond。
`freq_emb` 经超网络生成**逐通道 scale**（SpectralConv2d 的 FiLM 作用在频域特征上
是对全局标量调制，这里一样），`geom_emb` 走逐频率对的偏置，并把信息带进 MLP 隐层。

extra 键：
  `afno_hidden_ratio`  频域 MLP 隐层相对宽度的倍数，默认 2
  `afno_blocks`        分块数（默认 0 = 不分块，即全局混合；>0 则把频谱切成
                       blocks×blocks 个块，每块用**独立**的 W1/W2 做逐频率对变换，
                       块内共享权重。这是 AFNO 原始论文的 sparse 模式，
                       可用来在"共享权重(FNO 对比)"与"每频率独立(FNO)"之间插值）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from core.base import Cond, FNOBase
from core.config import ModelConfig
from core.encoders import FourierFreqEncoder, SDFGeomEncoder
from core.registry import register


class AFNOBlock(nn.Module):
    """频域 MLP + 通道混合 + 残差。

    形状不变；只在频域做逐频率对的变换与共享权重通道混合。
    """

    def __init__(self, width: int, hidden: int, modes: int, freq_dim: int,
                 blocks: int = 0):
        super().__init__()
        self.width = width
        self.modes = modes          # 参与变换的低频块边长
        self.blocks = blocks        # 0 = 全局共享；>0 = 分块，每块独立权重

        n_groups = blocks * blocks if blocks > 0 else 1
        self.n_groups = n_groups

        # 逐频率对的复数 MLP：所有频率对共享（或按块共享）同一对矩阵
        self.w1 = nn.Parameter(_complex_randn(n_groups, width, hidden))
        self.b1 = nn.Parameter(torch.zeros(n_groups, hidden))
        self.w2 = nn.Parameter(_complex_randn(n_groups, hidden, width))
        self.b2 = nn.Parameter(torch.zeros(n_groups, width))

        # 通道混合（空间域 1×1），AFNO 原论文里承担"跨通道 token 混合"的角色
        self.channel_mix = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.BatchNorm2d(width)

        # 条件：fused(freq+geom) 逐频率对复数偏置（hidden 个复数 = 2·hidden 个实数）；
        # freq_emb 单独做逐通道 scale（width 个复数 = 2·width 个实数）
        self.geom_bias = nn.Sequential(
            nn.Linear(2 * freq_dim, 128), nn.GELU(),
            nn.Linear(128, hidden * 2),
        )
        self.freq_scale = nn.Sequential(
            nn.Linear(freq_dim, 128), nn.GELU(),
            nn.Linear(128, width * 2),
        )
        # 零初始化 → 初始为恒等映射，训练从"纯 FNO 谱变换的退化版本"出发
        for net in (self.geom_bias, self.freq_scale):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward(self, x: torch.Tensor, cond: Optional[Cond] = None):
        fused = cond.fused() if isinstance(cond, Cond) else None
        freq_emb = cond.freq_emb if isinstance(cond, Cond) else None

        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x.float())

        # 参与变换的低频块：**正负 kx 两块都要取**。
        #
        # rfft2 的实轴是完整的 kx（长度 H），负频在末尾。FNO 的 w1 取
        # kx∈{0..m1-1}、w2 取 kx∈{-m1..-1}，两者合起来才是一个**中心对称**的
        # 低频窗。本文件原先只取 `x_ft[:, :, :m1, :m2]` 一块 —— 注释写着
        # "与 FNO 的 modes1/modes2 取法一致"，实际只覆盖了一半。
        # 这不是"少学一点"，而是让 AFNO 与 FNO 的对比**多出一个变量**：
        # AFNO 存在的唯一意义是"共享权重 vs 每频率独立调权"，频域覆盖必须对齐，
        # 否则 afno ≉ fno_baseline 时无法归因。
        #
        # 两块共享同一组 w1/w2（n_groups 与块数无关），所以拼起来做一次变换
        # 与分别做两次**数学上完全等价**，只是省一次 einsum。
        # 注意 `blocks>0` 的分块模式会沿这条拼接缝切块，缝两侧的频率对被归进
        # 同一组 —— 该模式本就是插值用的实验旋钮，不做特殊处理。
        m1 = min(self.modes, x_ft.size(2) // 2)
        m2 = min(self.modes, x_ft.size(3))
        low = torch.cat([x_ft[:, :, :m1, :m2], x_ft[:, :, -m1:, :m2]], dim=2)

        # (B, C, 2·m1, m2) → (B, n_groups, group_size, C)
        grouped = self._to_groups(low, 2 * m1, m2)        # (B, G, S, C)

        # 逐频率对的复数 MLP
        h = torch.einsum("bgsc,gch->bgsh", grouped, self.w1) + self.b1.view(1, self.n_groups, 1, -1)
        if fused is not None:
            gb = self.geom_bias(fused).view(B, 1, 1, -1)   # (B,1,1,2C) = 实+虚
            h = h + torch.complex(gb[..., :h.shape[-1]], gb[..., h.shape[-1]:])
        # 复数激活：PyTorch 的 GELU 不支持 complex，按实部/虚部分别激活。
        # （GELU 非奇函数，但 h 经 w1 后实虚部均值为 0，不会引入直流偏置。）
        h = torch.complex(F.gelu(h.real), F.gelu(h.imag))
        out = torch.einsum("bgsh,ghc->bgsc", h, self.w2) + self.b2.view(1, self.n_groups, 1, -1)

        out = self._from_groups(out, B, C, 2 * m1, m2)     # (B, C, 2·m1, m2) complex

        # 逐通道 scale（频率条件）
        if freq_emb is not None:
            fs = self.freq_scale(freq_emb)                 # (B, 2C)
            scale = torch.complex(fs[:, :C], fs[:, C:]).view(B, C, 1, 1)
            out = out * (1.0 + scale)

        out_ft = x_ft.clone()
        out_ft[:, :, :m1, :m2] = out[:, :, :m1]
        out_ft[:, :, -m1:, :m2] = out[:, :, m1:]

        y = torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)
        y = self.channel_mix(y)
        return F.gelu(self.norm(x + y))

    def _to_groups(self, low: torch.Tensor, m1: int, m2: int) -> torch.Tensor:
        """(B, C, m1, m2) → (B, G, S, C)，把低频块按 blocks×blocks 切分。"""
        B, C = low.shape[0], low.shape[1]
        if self.n_groups == 1:
            return low.reshape(B, 1, m1 * m2, C)
        # 频谱低频块并非整齐可分（m1 可能不是 blocks 的整数倍），用 unfold 切成
        # 近似等大的块；块大小向上取整，边界不足处重复采样（对形状无影响）
        chunk1 = (m1 + self.blocks - 1) // self.blocks
        chunk2 = (m2 + self.blocks - 1) // self.blocks
        pad1 = chunk1 * self.blocks - m1
        pad2 = chunk2 * self.blocks - m2
        if pad1 or pad2:
            low = F.pad(low, (0, pad2, 0, pad1))
        # (B, C, g1, c1, g2, c2) → (B, g1*g2, c1*c2, C)
        x = low.reshape(B, C, self.blocks, chunk1, self.blocks, chunk2)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.n_groups, C, -1)
        return x.permute(0, 1, 3, 2)                      # (B, G, S, C)

    def _from_groups(self, g: torch.Tensor, B: int, C: int, m1: int, m2: int) -> torch.Tensor:
        """`_to_groups` 的逆操作。"""
        if self.n_groups == 1:
            return g.reshape(B, C, m1, m2)
        chunk1 = (m1 + self.blocks - 1) // self.blocks
        chunk2 = (m2 + self.blocks - 1) // self.blocks
        x = g.permute(0, 1, 3, 2).reshape(B, self.blocks, self.blocks, C, chunk1, chunk2)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, chunk1 * self.blocks, chunk2 * self.blocks)
        return x[:, :, :m1, :m2]


def _complex_randn(*shape) -> torch.Tensor:
    """复数参数的小幅随机初始化（实虚部分别 N(0, 0.02²)）。"""
    r = torch.randn(*shape) * 0.02
    i = torch.randn(*shape) * 0.02
    return torch.complex(r, i)


@register("afno")
class AFNO2d(FNOBase):
    """AFNO：频域逐频率对 MLP，共享（或按块共享）权重。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.hidden = max(1, int(cfg.width * float(cfg.extra.get("afno_hidden_ratio", 2.0))))
        self.n_freq_blocks = int(cfg.extra.get("afno_blocks", 0))

        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=cfg.freq_dim)
        self.geom_encoder = SDFGeomEncoder(n_modes=4, out_dim=cfg.freq_dim)

        # 注意：必须填进 self.blocks —— 基座的 forward 遍历的是它
        self.blocks = nn.ModuleList([
            AFNOBlock(cfg.width, self.hidden, cfg.modes1, cfg.freq_dim,
                      blocks=self.n_freq_blocks)
            for _ in range(cfg.n_layers)
        ])
        # dropout 统一放在输出头之前，与其余 10 个模型一致。
        # 原先放在 block 内部（`norm(x + dropout(y))`），n_layers=5 时一次 forward
        # 触发 5 次、位置也不同 —— 消融表里就混进了"dropout 强度与位置"这个额外变量。
        self.dropout = nn.Dropout2d(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.proj = nn.Sequential(
            nn.Conv2d(cfg.width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, cfg.out_channels, kernel_size=1),
        )

    def build_cond(self, x, freq):
        if freq is None:
            return None
        return Cond(freq_emb=self.freq_encoder(freq), geom_emb=self.geom_encoder(x))

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(x))

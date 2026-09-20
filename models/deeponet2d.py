"""DeepONet 风格分支/主干基线（显式算子学习，非 FNO 路线）。

**它代表另一条技术路线**。前面所有模型都是"把几何编码成条件、让网络自己学映射"；
DeepONet 的做法是把算子直接显式分解为两项之积：

    G(几何)(坐标) ≈ Σ_k  B_k(几何) · T_k(坐标)

  - **Branch** `B_k(几何)`：只看贴片形状（SDF+Mask）+ 频率/极化，输出 p 个系数。
    这里用一个小的卷积编码器 + 全局池化实现（不能是逐点 MLP，否则退化成 mlp 基线）。
  - **Trunk** `T_k(坐标)`：只看空间坐标 (x,y)，输出 p 个基函数在该点的取值。
    **不接收任何几何信息**——这是 DeepONet 的关键约束。

乘积再求和得到每个输出通道的场。换句话说：网络输出的空间模式是几何无关的
基函数之线性组合。**若目标场中存在"场分布形状随几何大幅改变"的成分，
这个结构就是错的**（比如金属边缘处的局域绕射结构无法由全局基函数张成），
它会体现为明显更差的 val_geom。若它反而接近 FNO，说明任务的场分布
主要由少数几个与形状弱相关的空间模式构成——那是个很强的结论。

条件化：freq/geom 进 branch；trunk 只吃坐标。`build_cond` 返回 None。

**实现说明**：本类的数学输出 = 逐通道的 Σ_k B_k(geo)·T_k(x,y)，与基座的
"lift→pad→blocks→crop→head" 无关，因此整体覆盖 `forward`；
`self.blocks` 仍填入 DeepONet 层以便 checkpoint 保存结构（它们由本类自行调用）。

extra 键：
  `deeponet_basis`  basis 个数 p，默认 128
"""

import torch
import torch.nn as nn
from typing import Optional

from core.base import FNOBase, normalize_freq
from core.config import ModelConfig
from core.encoders import FourierFreqEncoder
from core.registry import register


class BranchNet(nn.Module):
    """分支网络：几何 (+条件) → p 个系数。输出 (B, C_out, p)。"""

    def __init__(self, in_channels: int, cond_dim: int, width: int, p: int,
                 out_channels: int):
        super().__init__()
        self.p = p
        self.cond_dim = cond_dim
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )
        # 全局池化 → 条件拼接 → p×out_channels 系数
        self.mlp = nn.Sequential(
            nn.Linear(width + cond_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, out_channels * p),
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv(x).mean(dim=(-2, -1))                # (B, width) 全局池化
        B = x.size(0)
        if cond is None:
            # 无条件路径（freq=None）也必须能跑：补零向量，保持 mlp 输入维度一致
            cond = x.new_zeros(B, self.cond_dim)
        h = torch.cat([h, cond], dim=-1)
        return self.mlp(h).view(B, self.out_channels, self.p)


class TrunkNet(nn.Module):
    """主干网络：坐标 (x,y) → p 个基函数值。**不接收几何信息**。"""

    def __init__(self, p: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, p),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 2) → (N, p)"""
        return self.net(coords)


@register("deeponet")
class DeepONet2d(FNOBase):
    """DeepONet 基线：branch(几何) ⊗ trunk(坐标) 的显式算子分解。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.p = int(cfg.extra.get("deeponet_basis", 128))

        # 频率编码器在 forward 里显式调用：cond = self.freq_encoder(freq)，
        # 再交给 branch 拼进 mlp 输入。**不要**误读成"branch 自己编条件"而删掉它。
        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=cfg.freq_dim)
        cond_dim = cfg.freq_dim

        self.branch = BranchNet(cfg.in_channels, cond_dim, cfg.width, self.p, cfg.out_channels)
        self.trunk = TrunkNet(self.p)
        # 输出前的 dropout：作用在 branch 系数上（本模型没有独立输出头，
        # 乘积求和就是输出映射）。与其他模型"输出前 dropout"的口径一致。
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        # 基座骨架（lift→pad→blocks→crop→head）在本模型里完全用不上：输出是
        # branch·trunk 的乘积求和，没有"逐层堆块"这一步。故 blocks 留空，
        # 不去注册会重复出现在 state_dict 里的别名。
        self.blocks = nn.ModuleList()

        # trunk 输出偏置（初始化的经典做法，避免乘积全为正/全为负）
        self.trunk_bias = nn.Parameter(torch.zeros(self.p))

        # 记录网格尺寸，坐标只依赖分辨率，与 batch 无关
        self._coords_cache = {}

    def _coords(self, H: int, W: int, device, dtype) -> torch.Tensor:
        """归一化到 [-1,1]² 的网格坐标 (H·W, 2)。"""
        # dtype 也进 key：缓存的是张量，dtype 变了不能复用
        key = (H, W, str(device), str(dtype))
        if key not in self._coords_cache:
            ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            self._coords_cache[key] = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        return self._coords_cache[key]

    def build_lift(self) -> nn.Module:
        """本模型整体覆盖 forward，基座骨架用不上 —— lift 建成恒等。

        否则基座会按 `cfg.in_channels` 建一个 Conv2d，它进 state_dict、
        进优化器，却因为没有调用点而**永远拿不到梯度**（实测就是这两个参数）。
        """
        return nn.Identity()

    # DeepONet 的输出映射（乘积求和）与基座骨架不同，整体覆盖 forward
    def forward(self, x: torch.Tensor, freq: Optional[torch.Tensor] = None) -> torch.Tensor:
        freq = normalize_freq(freq, x.shape[0])
        B, _, H, W = x.shape

        cond = self.freq_encoder(freq) if freq is not None else None
        coeff = self.dropout(self.branch(x, cond))            # (B, C_out, p)
        basis = self.trunk(self._coords(H, W, x.device, x.dtype)) + self.trunk_bias
        # (B, C_out, p) @ (p, H·W) → (B, C_out, H·W)
        out = torch.einsum("bcp,np->bcn", coeff, basis).reshape(B, self.cfg.out_channels, H, W)
        return out

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        """forward 已整体覆盖，此方法不会被调用（仅为满足抽象协议）。"""
        raise NotImplementedError("DeepONet2d 在 forward 内完成乘积求和")

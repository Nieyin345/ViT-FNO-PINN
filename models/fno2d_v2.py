"""FNO2dV2 —— 强化纯 FNO 特征的条件化主模型。

设计理念（格林函数视角）：
  电磁场 = 贴片电流与格林函数 G(r,r'; f) 的卷积，频域内对应核乘法。
  FNO 的频域卷积核 w1/w2 正是格林函数在傅里叶域的学习表示。

v2 相比旧主模型（fno_attention）的关键改动：
  1. 去掉"超网络调制注意力投影矩阵"（已被消融证明有害），回归纯 FNO 频域卷积。
  2. 频率信息以【空间正弦嵌入通道】注入输入：
       sin(k·x), cos(k·x), sin(k·y), cos(k·y)，其中 k 由工作频率决定
     —— FNO 的频域卷积可直接加工这些携带频率信息的空间模式。
  3. 频率/极化/几何通过条件向量【直接调制频域卷积核】：
       - "film": 对频域特征图做 per-channel scale+shift（轻量、稳定）
       - "lora": 低秩扰动 ΔW = U(cond)@V^T 直接加到 w1/w2（更强表达）
     —— 让格林函数本身随频率变化（宽频带学习），极化走独立编码分支。
  4. 条件向量 = [freq_emb, pol_emb, geom_emb] 拼接，频率与极化不再平均混叠。

输入:  (B, 2, H, W) SDF+Mask  → 内部拼接 4 个正弦通道 → (B, 6, H, W)
条件:  freq (B, 2) = [归一化频率, 极化]
输出:  (B, 12, H, W) Ex/Ey/Ez/Hx/Hy/Hz 实虚部

重构变更：
  - **改为继承 FNOBase**。原先它不继承基座，把 lift/pad/循环/crop 抄了一遍
    （与 fno_core.FNOBase 重复约 15 行），因为它的块签名 `block(x, cond)` 与
    基座固定的 `block(x, freq_emb=, geom_emb=)` 不兼容。统一为 `Cond` 后可以复用。
  - `PROJ_DROPOUT` 原先在 `__init__` 里 import config 读取，现在从 `cfg.dropout` 传入。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import Cond, FNOBase
from core.config import ModelConfig
from core.encoders import FourierFreqEncoder, SDFGeomEncoder
from core.registry import register

# 频率范围（与 data/dataset.py 保持一致）
_F_MIN = 8.0
_F_MAX = 12.0

# 正弦嵌入用的空间相位：phase = k0 * L，k0 = 2π f / c（真空），L = 单元尺寸。
#
# 原实现写死 `phase = (math.pi / 10.0) * f_ghz`。验算后它确实等于 k0*L：
#     π/10 * f_GHz = 2π * f*1e9 / 2.998e8 * 0.015   （比值恒为 0.9993）
# 但那个常数把**光速与单位换算都烧进了字面量**，且 L=15mm 是隐含的。
# 元胞尺寸一变（例如 10mm），正确常数该是 0.2094 而代码仍是 0.3142，
# **静默失配**——正弦通道携带的就不再是物理波数了。
#
# 现在从物理量算：光速改为精确值 2.99792458e8（原 2.998e8 让比值停在
# 0.9993，改用精确值后与 π/10 的差别是 0.07%，数值上几乎等价）。
_C_LIGHT = 2.99792458e8          # m/s
_CELL_MM_DEFAULT = 15.0          # 元胞/单元尺寸，(与 CLAUDE.md 记录的 cell_mm=15 一致)


def _sine_phase(freq_norm: torch.Tensor, cell_mm: float) -> torch.Tensor:
    """归一化频率 (B,1) → 空间相位 k0*L，单位 rad。

    Args:
        freq_norm: (B,1) 归一化频率（0 → _F_MIN，1 → _F_MAX）
        cell_mm:   单元尺寸（mm）

    Returns:
        (B,1) 相位，量级 ~2.5（8GHz）~3.8（12GHz）
    """
    f_hz = (_F_MIN + (_F_MAX - _F_MIN) * freq_norm) * 1e9
    return 2.0 * math.pi * f_hz * (cell_mm * 1e-3) / _C_LIGHT


class SpectralConv2dV2(nn.Module):
    """标准 FNO 频域卷积 + 可选的核条件化。

    kernel_cond: "film" | "lora" | "none"
    """

    def __init__(self, in_channels, out_channels, modes1, modes2,
                 cond_dim=80, kernel_cond="film", lora_rank=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.kernel_cond = kernel_cond

        scale = 1 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

        if kernel_cond == "film":
            # 对频域特征图做 per-channel scale + shift（初值 0 → 恒等）
            self.film_net = nn.Sequential(
                nn.Linear(cond_dim, 128), nn.GELU(),
                nn.Linear(128, out_channels * 2),
            )
            nn.init.zeros_(self.film_net[-1].weight)
            nn.init.zeros_(self.film_net[-1].bias)
        elif kernel_cond == "lora":
            flat_dim = out_channels * modes1 * modes2
            self.rank = lora_rank
            self.V1 = nn.Parameter(torch.randn(flat_dim, lora_rank, dtype=torch.cfloat) * 0.01)
            self.V2 = nn.Parameter(torch.randn(flat_dim, lora_rank, dtype=torch.cfloat) * 0.01)
            self.hyper_u1 = nn.Sequential(
                nn.Linear(cond_dim, 128), nn.GELU(),
                nn.Linear(128, in_channels * lora_rank * 2),
            )
            self.hyper_u2 = nn.Sequential(
                nn.Linear(cond_dim, 128), nn.GELU(),
                nn.Linear(128, in_channels * lora_rank * 2),
            )
            nn.init.zeros_(self.hyper_u1[-1].weight)
            nn.init.zeros_(self.hyper_u1[-1].bias)
            nn.init.zeros_(self.hyper_u2[-1].weight)
            nn.init.zeros_(self.hyper_u2[-1].bias)

    def _lora_delta_w(self, hyper, V, cond, B):
        u_flat = hyper(cond)  # (B, in*rank*2)
        u_real = u_flat[:, :self.in_channels * self.rank]
        u_imag = u_flat[:, self.in_channels * self.rank:]
        U = torch.complex(u_real.view(B, self.in_channels, self.rank),
                          u_imag.view(B, self.in_channels, self.rank))
        delta = torch.bmm(U, V.T.unsqueeze(0).expand(B, -1, -1))
        return delta.view(B, self.in_channels, self.out_channels, self.modes1, self.modes2)

    def forward(self, x, cond=None):
        c = cond.fused() if isinstance(cond, Cond) else cond

        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x.float())

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        if c is not None and self.kernel_cond == "lora":
            dw1 = self._lora_delta_w(self.hyper_u1, self.V1, c, B)
            dw2 = self._lora_delta_w(self.hyper_u2, self.V2, c, B)
            w1_eff = self.w1.unsqueeze(0) + dw1
            w2_eff = self.w2.unsqueeze(0) + dw2
            out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
                "bixy,bioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], w1_eff)
            out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
                "bixy,bioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], w2_eff)
        else:
            out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1)
            out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2)

        if c is not None and self.kernel_cond == "film":
            fp = self.film_net(c)  # (B, 2*out)
            scale = fp[:, :self.out_channels].view(B, self.out_channels, 1, 1)
            shift = fp[:, self.out_channels:].view(B, self.out_channels, 1, 1)
            # FiLM 的两个动作分别作用在**不同的域**，这是本处修正的核心：
            #
            #   scale：逐通道缩放。频域缩放 == 空域缩放（傅里叶变换是线性的），
            #          所以直接乘在 out_ft 上，语义正确，保持原样。
            #
            #   shift：逐通道平移。**频域上没有"平移"这个操作**——在频域加常数，
            #          反变换回空域是一个 **delta 冲激**（实测原点能量占比 1.0000），
            #          即每层都往特征图原点打一个尖峰，而不是"整体抬高/压低"。
            #          原实现写的是 `out_ft * (1+scale) + shift`，正是这个错误。
            #
            # 错误注释的历史（两版都错，记在这里免得再踩）：
            #   第一版说"给 DC 加常数 = 原点冲激"——反了：只改 DC 得到的是**均匀
            #     常数**（实测 std 1.8e-7）；得到冲激的是"给**全部**频点加常数"。
            #   第二版改成了"只排除 DC 那一个点"——依然错：只排除 DC 后原点能量
            #     占比仍是 0.9998，冲激还在，只是被减掉了一个均匀量。排除**整个
            #     kx=0 行**同理（原点占比 0.9844），只是把冲激沿 y 展平。
            #   三种写法都消不掉冲激，因为问题不在"改哪些频点"，而在**加法本身
            #   就不该发生在频域**。
            #
            # 现在的做法：scale 留在频域乘，shift 挪到 irfft2 之后**空域相加**——
            # 这才是 FiLM 语义下的逐通道平移（与 fno2d_film 的骨架一致）。
            # 零初始化时 scale=shift=0，仍然**恒等**，起点行为不变。
            out_ft = out_ft * (1.0 + scale).to(out_ft.dtype)
        else:
            shift = None

        y = torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)
        if shift is not None:
            # (B, C, 1, 1) 广播到 (B, C, H, W)：空域逐通道加偏置
            y = y + shift.to(y.dtype)
        return y


class FNOBlockV2(nn.Module):
    def __init__(self, width, modes1, modes2, cond_dim=80, kernel_cond="film", lora_rank=4):
        super().__init__()
        self.norm = nn.BatchNorm2d(width)
        self.spectral = SpectralConv2dV2(width, width, modes1, modes2,
                                         cond_dim=cond_dim, kernel_cond=kernel_cond,
                                         lora_rank=lora_rank)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x, cond=None):
        out = self.spectral(x, cond) + self.bypass(x)
        return F.gelu(self.norm(out + x))


@register("fno_v2")
class FNO2dV2(FNOBase):
    """FNO2dV2 主模型（强化 FNO + 正弦频率输入 + 核条件化）。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.use_sine_input = cfg.use_sine_input
        self.use_pol_cond = cfg.use_pol_cond
        # 正弦嵌入用的单元尺寸（mm）。走 cfg.extra 扩展口，不占 ModelConfig 字段；
        # 换元胞尺寸时在实验条目里写 MODEL_EXTRA={"CELL_MM": 10.0} 即可。
        self.cell_mm = float(cfg.extra.get("CELL_MM", _CELL_MM_DEFAULT))
        # 正弦嵌入通道数：x/y 各 sin/cos = 4
        self.n_sine = 4 if cfg.use_sine_input else 0

        self.freq_encoder = FourierFreqEncoder(n_freqs=16, out_dim=cfg.freq_dim)
        self.geom_encoder = SDFGeomEncoder(n_modes=4, out_dim=cfg.freq_dim)
        if cfg.use_pol_cond:
            self.pol_mlp = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 16))
        self.cond_dim = cfg.freq_dim + (16 if cfg.use_pol_cond else 0) + cfg.freq_dim

        self.blocks = nn.ModuleList([
            FNOBlockV2(cfg.width, cfg.modes1, cfg.modes2, cond_dim=self.cond_dim,
                       kernel_cond=cfg.kernel_cond, lora_rank=cfg.lora_rank)
            for _ in range(cfg.n_layers)
        ])
        self.dropout = nn.Dropout2d(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Conv2d(cfg.width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, cfg.out_channels, kernel_size=1),
        )

    def lift_in_channels(self) -> int:
        # 基座按 cfg.in_channels 建 lift；本模型额外拼 n_sine 个正弦通道。
        # **只读 self.cfg**：本方法在 super().__init__ 内部被调用，
        # 那时本类 __init__ 里的 self.n_sine 还没赋值。
        return int(self.cfg.in_channels) + (4 if self.cfg.use_sine_input else 0)

    def _sine_embedding(self, x, freq):
        """频率相关的空间正弦嵌入：sin/cos(k0·L·x/y)，k0 由工作频率决定。

        相位取物理量 k0*L（L=单元尺寸），见 `_sine_phase`。
        空间坐标归一化到 [-1,1]，因此这里的"相位跨度"是 2*k0*L，
        即网格跨越约 0.8（8GHz）~1.2（12GHz）个周期——这组通道提供的是
        **全局低频调制**（频率线索），不是精细空间结构，与设计意图一致。
        """
        B, _, H, W = x.shape
        p = _sine_phase(freq[:, 0:1], self.cell_mm).view(B, 1, 1)   # (B,1,1)
        y_n = torch.linspace(-1.0, 1.0, H, device=x.device).view(1, H, 1).expand(1, H, W)
        x_n = torch.linspace(-1.0, 1.0, W, device=x.device).view(1, 1, W).expand(1, H, W)
        return torch.stack([
            torch.sin(p * x_n), torch.cos(p * x_n),
            torch.sin(p * y_n), torch.cos(p * y_n),
        ], dim=1).to(x.dtype)                                       # (B,4,H,W)

    def build_input(self, x, freq):
        if not self.use_sine_input:
            return x
        if freq is None:
            raise ValueError("use_sine_input=True 时必须提供 freq（正弦嵌入依赖频率）")
        return torch.cat([x, self._sine_embedding(x, freq)], dim=1)

    def build_cond(self, x, freq):
        if freq is None:
            return None
        freq_emb = self.freq_encoder(freq[:, 0:1])
        if self.use_pol_cond:
            pol_emb = self.pol_mlp(freq[:, 1:2])
        else:
            pol_emb = None
        geom_emb = self.geom_encoder(x)
        return Cond(freq_emb=freq_emb, pol_emb=pol_emb, geom_emb=geom_emb)

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(x))

# -*- coding: utf-8 -*-
"""fno_film_pole：用**极点-留数（有理）基**而不是多项式基来承载频率依赖。

## 为什么换基

`fno_film_poly` 把频率写成 K 阶切比雪夫多项式：u(x;f) = Σ_k C_k(x) T_k(f̃)。
诊断（`tools/diag_pf.py`，A6ema seed1，rectangle）暴露了它的毛病——留出的
10 GHz 上 geom 0.388，其中**幅值项占 0.294（76%）**，而 |pred|/|true| = 0.787、
每样本最优全局缩放 1.311：**多项式在没见过的频点上系统性地把场缩小了**。
相位反而几乎是对的（0.050）。

这正是多项式基的典型行为。超表面近场在 8–12 GHz 内是**谐振**响应：一两个极点
主导，幅值随 f 走洛伦兹峰。低阶多项式穿过 8 个训练频点后，在峰附近只能走割线，
于是峰值被削平——留出点恰好落在被削的那一段。阶数调高（K=7,8）又会在未见频点
振荡（项目实测 K=7/8 的未见频 geom 反而涨到 0.466/0.477）。这是 Runge 现象，
不是容量问题，加数据也救不了。

有理基把同一件事写成电磁里标准的形式（vector fitting / AWE 在做的事）：

    u(x; f) = Σ_p  R_p(x) / (f̃ − s_p)  +  Σ_{k<K0} C_k(x) T_k(f̃)

* `R_p(x)` 是复**留数场**，由主体网络输出（主体仍然看不到频率，`free_body`）；
* `s_p = a_p + i b_p` 是复**极点**，谐振频率 a_p 与阻尼 b_p（Q ≈ |a|/2|b|）；
* 后一项是低阶多项式**背景**，承载非谐振的缓变部分。

好处有三，每一条都能直接对上诊断里的现象：

1. **幅值**：洛伦兹峰是有理函数的本征形状，不再靠割线逼近，未见频点不会被削平。
2. **相位**：复留数除以复数，天然给出随 f 旋转的相位。`PhaseFrame` 那张**硬编码**的
   旋转表（`AC_STEPS` 八个实测常数、以 10 GHz 为锚）做的就是这件事，但它
   ① 是拟合出来的魔数，审稿人会问从哪来；② 在留出频点附近那两步是**猜的**
   （注释里写明 9.5→10.5 取邻居均值），诊断里 10.5 GHz 的相位误差 0.237 是邻居的
   5 倍，正是这个猜值的代价。有理基把相位旋转变成**学出来的、对 f 连续**的东西。
3. **可解释**：学到的极点就是谐振频率与 Q 值，可以和 HFSS 的 S 参数共振峰对照——
   论文里这是一张能讲物理的图，不是又一个精度表格。

## 极点怎么参数化

- 全局极点 `P` 个（整个数据集共享的\"模态基\"），加上按几何预测的**小偏移**
  （`pole_delta=True` 时启用，tanh 限幅）。理由：谐振频率确实随几何移动，但让网络
  从零预测极点很容易跑飞；共享基 + 小偏移既保留物理，又好训。
- 阻尼 `b_p` 用 `softplus` 保证 > `pole_b_min`，**防止极点压到实轴上**——那会让
  分母在某个训练频点附近趋零，梯度炸掉。这不是数值洁癖：vector fitting 的实现里
  同样要把极点钉在半平面。

## 与 A6ema 的参数量对照

K=6 的多项式头输出 6×12=72 通道；P=4、K0=2 的有理头输出 (4+2)×12=72 通道。
**输出头逐通道相同**，所以这组对照是干净的\"同参数量、换频率基\"。
"""
import torch
import torch.nn as nn

from core.config import ModelConfig
from core.base import normalize_freq
from core.registry import register
from .fno2d_film import FNO2dFiLM


@register("fno_film_pole")
class FNO2dFiLMPole(FNO2dFiLM):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        if self.use_em_heads:
            raise ValueError("fno_film_pole 暂不支持 head='em_split'")
        ex = cfg.extra or {}
        self.P = int(ex.get("pole_p", 4))            # 极点个数
        self.K0 = int(ex.get("pole_k0", 2))          # 多项式背景阶数（0 = 不要背景）
        if self.P < 1:
            raise ValueError(f"pole_p must be >= 1, got {self.P}")
        if self.K0 < 0:
            raise ValueError(f"pole_k0 must be >= 0, got {self.K0}")
        self.free_body = bool(ex.get("free_body", True))
        self.pole_b_min = float(ex.get("pole_b_min", 0.05))
        self.use_pole_delta = bool(ex.get("pole_delta", True))
        self.pole_delta_max = float(ex.get("pole_delta_max", 0.25))

        self.n_out = 2 * len(self.heads)              # 12 个实通道 = 6 个复分量
        self.n_bank = self.P + self.K0                # 系数场的组数

        # 极点初值：实部在 [-1,1] 上等距铺开（覆盖整个频带），阻尼取中等 Q。
        # 等距而不是随机:谐振可能落在任何地方,随机初值会让两个极点一开始就挤在一起,
        # 退化成一个。
        a0 = torch.linspace(-0.8, 0.8, self.P)
        self.pole_a = nn.Parameter(a0)
        # b = softplus(raw) + b_min；raw 取 softplus 的反函数,使初始 b ≈ 0.3
        b_init = 0.3 - self.pole_b_min
        self.pole_b_raw = nn.Parameter(torch.full((self.P,), float(torch.log(torch.expm1(torch.tensor(max(b_init, 1e-3)))))))

        hid = int(ex.get("poly_head_hidden", 256))
        if hid > 0:
            self.heads = nn.ModuleList([nn.Sequential(
                nn.Conv2d(64, hid, kernel_size=1), nn.GELU(),
                nn.Conv2d(hid, self.n_out * self.n_bank, kernel_size=1))])
        else:
            from core.encoders import DirectionHead
            self.heads = nn.ModuleList(
                [DirectionHead(in_channels=64) for _ in range(len(self.heads) * self.n_bank)])

        # 几何相关的极点偏移：从几何编码出 (P,2)，tanh 限幅后加到全局极点上。
        if self.use_pole_delta:
            self.pole_delta_net = nn.Sequential(
                nn.Linear(cfg.freq_dim, 64), nn.GELU(), nn.Linear(64, self.P * 2))
            nn.init.zeros_(self.pole_delta_net[-1].weight)
            nn.init.zeros_(self.pole_delta_net[-1].bias)

    # ── 频率基 ────────────────────────────────────────────────────────────
    def _cheb(self, t, K):
        """t: (B,) → (B,K) 切比雪夫基（背景项用）。"""
        if K == 0:
            return t.new_zeros((t.shape[0], 0))
        T = [torch.ones_like(t)]
        if K > 1:
            T.append(t)
        for _ in range(2, K):
            T.append(2 * t * T[-1] - T[-2])
        return torch.stack(T[:K], dim=1)

    def _poles(self, geom_emb, B, device, dtype):
        """返回 (B,P) 复极点。"""
        a = self.pole_a.to(device=device, dtype=dtype)[None].expand(B, -1)
        b = (nn.functional.softplus(self.pole_b_raw.to(device=device, dtype=dtype))
             + self.pole_b_min)[None].expand(B, -1)
        if self.use_pole_delta and geom_emb is not None:
            d = torch.tanh(self.pole_delta_net(geom_emb)).view(B, self.P, 2) * self.pole_delta_max
            a = a + d[:, :, 0]
            # 阻尼只允许**乘性**微调且恒正(exp)，不让它被推到 0 以下。
            b = b * torch.exp(d[:, :, 1])
        return torch.complex(a, b)

    def forward(self, x, freq=None):
        freq = normalize_freq(freq, x.shape[0])
        fb = freq.clone()
        if self.free_body:
            fb[:, 0] = 0.5
        out = super().forward(x, fb)                      # (B, n_bank*12, H, W)
        B, _, H, W = out.shape
        out = out.view(B, self.n_bank, self.n_out, H, W)

        t = 2.0 * freq[:, 0] - 1.0                        # f̃ ∈ [-1,1]
        geom_emb = self.geom_encoder(x) if self.use_pole_delta else None
        s = self._poles(geom_emb, B, out.device, out.dtype)          # (B,P) complex
        # 1/(f̃ − s)：实部虚部各自是一个随 f 变化的权重。留数场的 (re,im) 两半
        # 当作复数,乘上这个复权重 —— 这一步就是\"相位随频率旋转\"的来源。
        denom = (t[:, None].to(s.dtype) - s)
        w = 1.0 / denom                                               # (B,P) complex
        res = out[:, : self.P]                                        # (B,P,12,H,W)
        re, im = res[:, :, 0::2], res[:, :, 1::2]                     # (B,P,6,H,W)
        wr, wi = w.real[:, :, None, None, None], w.imag[:, :, None, None, None]
        o_re = (re * wr - im * wi).sum(dim=1)                         # (B,6,H,W)
        o_im = (re * wi + im * wr).sum(dim=1)
        y = torch.stack([o_re, o_im], dim=2).reshape(B, self.n_out, H, W)

        if self.K0 > 0:
            bg = out[:, self.P:]                                      # (B,K0,12,H,W)
            cb = self._cheb(t, self.K0).to(out.dtype)                 # (B,K0)
            y = y + (bg * cb[:, :, None, None, None]).sum(dim=1)
        return y

    # 给诊断脚本用：把学到的极点换算回 GHz
    @torch.no_grad()
    def pole_table(self, f_min=8.0, f_max=12.0):
        a = self.pole_a.detach().cpu()
        b = (nn.functional.softplus(self.pole_b_raw.detach().cpu()) + self.pole_b_min)
        half = 0.5 * (f_max - f_min)
        f_res = f_min + half * (a + 1.0)           # 谐振频率 GHz
        bw = b * half                              # 半带宽 GHz
        q = f_res / (2.0 * bw).clamp(min=1e-9)
        return [{"f_res_GHz": float(f_res[i]), "halfwidth_GHz": float(bw[i]), "Q": float(q[i])}
                for i in range(self.P)]


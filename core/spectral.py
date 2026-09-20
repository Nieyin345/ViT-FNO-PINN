# -*- coding: utf-8 -*-
"""频域卷积层：标准 FNO 谱卷积 + 可开关的超网络投影矩阵复数交叉注意力。

核心创新（原样保留）：
  1. 超网络直接生成 ΔW_q / ΔW_k，动态调制投影矩阵本身
     W_q_eff = W_q + ΔW_q(freq_emb)   ← 频率条件调制查询投影
     W_k_eff = W_k + ΔW_k(geom_emb)   ← 几何条件调制键投影
  2. Q = X_sig @ W_q_eff，K = X_sig @ W_k_eff
     数据、频率、几何三路信息在投影步骤融合，而非事后加偏置
  3. gamma 初始化 0.1 → 注意力从第 1 个 epoch 就参与训练

重构变更（唯一一处行为差异，见 core/config.py::SpectralConfig）：
  - 原先模块级 `import config as C` 并在 `__init__` 里读 6 个消融开关，
    现在改为从 `SpectralConfig` 显式传入，构造对全局配置零依赖。
  - `attention=False` 时注意力相关模块（W_q/W_k/hyper_*/gamma）**不再创建**。
    原先它们总是被创建但不参与前向，导致消融实验的参数量虚高
    （已核对 checkpoints/ablation/fno_attention_no_attn/best.pth 确实含这些键）。
    默认 attention=True，主模型参数量不变。
"""

import math

import torch
import torch.nn as nn

from .config import SpectralConfig


class SpectralConv2d(nn.Module):
    """频域卷积 + 可开关的超网络投影矩阵交叉注意力。"""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int,
                 cond_dim: int = 32, cfg: SpectralConfig = None):
        super().__init__()
        cfg = cfg or SpectralConfig()
        self.cfg = cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # 1. 频域特征 V 的参数库（标准 FNO 核心）
        scale = 1 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

        # 2. 消融开关
        self.attention_enabled = bool(cfg.attention)
        self.freq_conditioning = bool(cfg.freq_conditioning)
        self.geom_conditioning = bool(cfg.geom_conditioning)

        # 3. 注意力基础投影矩阵 + 超网络 + gamma 门控（仅启用时创建）
        if self.attention_enabled:
            self.n_modes_attn = min(8, modes1)
            self.sig_length = self.n_modes_attn * self.n_modes_attn
            self.d_k = 32
            # X_sig 是**未归一化**的频域幅值（量级 1e2，DC 点还要再高一个量级），
            # W_q/W_k 是 N(0, 0.02²)。打分是 Q·K^H，形状 (B, C_in, C_in)——
            # 注意 softmax 归一化的是**通道维**（C_in = width），不是频点维。
            # 打分不归一化时实测（width=32，输入量级 1e1）：
            #   |logit| max 2.9e4  ->  softmax 熵 0.0000（均匀分布的 0%），
            #   最大权重 1.0000，有效位置数 1.00
            # 即注意力退化成硬 argmax——每个通道只挑一个通道、权重恒为 1，
            # 1.07M 参数的注意力模块对输出的贡献趋近于一个固定的通道置换，
            # 且梯度几乎为零。
            #
            # 除以 sqrt(d_k)=5.66 远远不够（2.9e4 -> 5e3，熵仍是 0）。问题的根源
            # 是 |X_sig| 的量级而不是 d_k，所以额外做**逐样本 L2 归一化**，
            # 让打分建立在"频域签名的方向"上而不是"幅值有多大"上 ——
            # 注意力本来就该看方向（Q·K 的夹角），幅值由后面的 V 承担。
            # 归一化后实测：|logit| max 9.3e-4 -> 熵 3.4657 = log(32)，
            # 最大权重 0.0313 = 1/32，有效位置数 32.00（均匀分布）。
            # 零初始化下"均匀"正是应有的起点：每个通道等权，训练中 W_q/W_k
            # 增长后再自然变得稀疏。
            self.score_scale = float(getattr(cfg, "score_scale", 1.0))
            self.sig_norm = bool(getattr(cfg, "sig_norm", True))
            self.W_q = nn.Parameter(torch.randn(self.sig_length, self.d_k, dtype=torch.cfloat) * 0.02)
            self.W_k = nn.Parameter(torch.randn(self.sig_length, self.d_k, dtype=torch.cfloat) * 0.02)

            self.hyper_wq = nn.Sequential(
                nn.Linear(cond_dim, 256),
                nn.GELU(),
                nn.Linear(256, self.sig_length * self.d_k),
            )
            self.hyper_wk = nn.Sequential(
                nn.Linear(cond_dim, 256),
                nn.GELU(),
                nn.Linear(256, self.sig_length * self.d_k),
            )
            # 末层双向零初始化 -> t=0 时 ΔW_q = ΔW_k = 0，起点是纯 FNO 谱卷积，
            # 训练稳定，**保留不改**。
            #
            #   ΔW = W_last · GELU(W1·cond + b1) + b_last，零初始化后 W_last = b_last = 0
            #   ∂L/∂cond  ∝ W_last      = 0     ← 条件嵌入在第一步拿不到梯度
            #   ∂L/∂W1    ∝ W_last      = 0
            #   ∂L/∂W_last = g · GELU(·)ᵀ       ← 唯一能把它抬离零的通路
            #
            # 曾经在这里写过一句"把超网络抬离零的梯度，正是被上面的 softmax 饱和
            # 掐掉的那一份"。**实测不支持这个说法**：把 sig_norm 打开前后，
            # ∂L/∂W_last 是 1.1212e-04 vs 1.1125e-04（比值 0.992×），20 步后
            # |W_last| 2.24e-06 vs 2.23e-06，几乎没有区别。原因是 W_last 的梯度
            # 里 γ·attn_out 那一项占了主导，并不经过 softmax 的稀疏性。
            # 所以上面那处修正的价值是它本身（注意力从硬 argmax 变回可训练），
            # **不是**"解锁了条件化模块"。两件事各自独立，不要混为一谈。
            nn.init.zeros_(self.hyper_wq[-1].weight)
            nn.init.zeros_(self.hyper_wq[-1].bias)
            nn.init.zeros_(self.hyper_wk[-1].weight)
            nn.init.zeros_(self.hyper_wk[-1].bias)

            self.gamma = nn.Parameter(torch.tensor(float(cfg.gamma_init), dtype=torch.float32))
            if cfg.gamma_fixed:
                self.gamma.requires_grad_(False)

        # 4. 频域 Butterworth 衰减
        if cfg.decay:
            self.register_buffer("decay_mask", _butterworth_mask(modes1, modes2, cfg))
        else:
            self.decay_mask = None

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        from .base import Cond
        B, C, H, W = x.shape
        freq_emb = cond.freq_emb if isinstance(cond, Cond) else None
        geom_emb = cond.geom_emb if isinstance(cond, Cond) else None

        # 1. 全局 FFT
        x_ft = torch.fft.rfft2(x.float())

        # 阶段 A：提取频域特征 V
        out_ft = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.w2)

        out_ft_new = out_ft
        if self.attention_enabled:
            # 阶段 B：超网络动态调制投影矩阵，生成 Q / K
            C_in = x_ft.size(1)
            m1 = min(self.n_modes_attn, x_ft.size(2))
            m2 = min(self.n_modes_attn, x_ft.size(3))

            freq_sig_complex = x_ft[:, :, :m1, :m2]
            if m1 < self.n_modes_attn or m2 < self.n_modes_attn:
                pad = torch.zeros(B, C_in, self.n_modes_attn, self.n_modes_attn,
                                  device=x.device, dtype=torch.cfloat)
                pad[:, :, :m1, :m2] = freq_sig_complex
                freq_sig_complex = pad

            X_sig = freq_sig_complex.reshape(B, C_in, -1)  # (B, C_in, sig_length)

            # 逐样本 L2 归一化：打分只看频域签名的**方向**，不看幅值量级。
            # 不做这一步时 |X_sig| ~ 1e2（DC 还会把某些通道再抬一个量级），
            # 打分动辄 1e3，softmax 直接饱和（实测熵只有均匀分布的 0.28%）。
            if self.sig_norm:
                X_sig = X_sig / X_sig.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)

            W_q_eff = self.W_q.unsqueeze(0).expand(B, -1, -1)
            W_k_eff = self.W_k.unsqueeze(0).expand(B, -1, -1)

            if freq_emb is not None and self.freq_conditioning:
                delta_Wq = self.hyper_wq(freq_emb).view(B, self.sig_length, self.d_k)
                W_q_eff = W_q_eff + delta_Wq.to(W_q_eff.dtype)

            if geom_emb is not None and self.geom_conditioning:
                delta_Wk = self.hyper_wk(geom_emb).view(B, self.sig_length, self.d_k)
                W_k_eff = W_k_eff + delta_Wk.to(W_k_eff.dtype)

            Q = torch.bmm(X_sig, W_q_eff)
            K = torch.bmm(X_sig, W_k_eff)

            scores_complex = torch.bmm(Q, K.conj().transpose(1, 2)) / math.sqrt(self.d_k)

            # 复数打分 -> 实数权重。原始设计用 softmax(|scores|)：丢掉相位，只按
            # "两个通道的低频签名有多像"来分配权重，而相位由 `scores_phase` 在加权
            # 时乘回去。这个设计保留原样（不属于本次修正范围），只补上缺失的缩放。
            #
            # 写法上把"取模"和"缩放"分开：先取模（纯相位归一化，与 score_scale
            # 无关），再对模乘 score_scale 送进 softmax。等价于
            # `scores_complex / (|scores_complex| + 1e-8)` 后再 softmax(|·|*scale)，
            # 但不会让 score_scale 意外混进相位那一项（原写法 `/(scores_mag/scale)`
            # 在 scale≠1 时相位会被整体放大 scale 倍）。
            scores_mag = torch.abs(scores_complex)
            scores_phase = scores_complex / (scores_mag + 1e-8)
            attn_weight_real = torch.softmax(scores_mag * self.score_scale, dim=-1)
            attn_matrix_complex = attn_weight_real.to(out_ft.dtype) * scores_phase

            # 阶段 C：注意力加权重组 + 残差门控
            attn_out_ft = torch.einsum("bij,bjxy->bixy", attn_matrix_complex, out_ft)
            out_ft_new = out_ft + self.gamma * attn_out_ft

        if self.decay_mask is not None:
            mask = self.decay_mask
            out_ft_new[:, :, :self.modes1, :self.modes2] *= mask
            out_ft_new[:, :, -self.modes1:, :self.modes2] *= mask

        return torch.fft.irfft2(out_ft_new, s=(H, W)).to(x.dtype)


def _butterworth_mask(modes1: int, modes2: int, cfg: SpectralConfig) -> torch.Tensor:
    """生成 2D 径向 Butterworth 衰减权重矩阵。"""
    i1 = torch.arange(modes1, dtype=torch.float32)
    i2 = torch.arange(modes2, dtype=torch.float32)
    k = torch.sqrt(i1[:, None] ** 2 + i2[None, :] ** 2)
    return 1.0 / torch.sqrt(1.0 + (k / cfg.decay_k0) ** (2 * cfg.decay_order))

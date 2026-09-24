# -*- coding: utf-8 -*-
"""探针：诊断模型"条件化是否真的在起作用"。

训练的 loss 下降**不能**证明条件化有效 —— 模型完全可能忽略频率/极化输入，
靠几何 + 频点索引的记忆把训练集拟合得很好，而在未见频点上崩掉。
本脚本用一组**因果干预**探针把这件事变成可测的数字。

四个探针：

[P1] 条件敏感度 —— 只改频率（几何不变）/ 只改极化 / 只改几何，
     看输出位移的量级。若"只改频率"的位移 ≈ 0，说明模型没在用频率条件。

[P2] 条件嵌入的可分性 —— 9 个频点 × 2 个极化的嵌入两两距离。
     若 [8GHz,pol0] 与 [12GHz,pol1] 的嵌入距离 ≈ 0，说明编码器把它们
     混叠了（历史上真发生过：按列平均导致这两者编码完全相同）。

[P3] 频域调制强度 —— 超网络生成的 ΔW_q / ΔW_k 相对基础 W 的量级。
     零初始化起步，若训练后仍 ≈ 0，说明条件化模块是**死的**（表面有参数，
     实际不参与前向），这类问题不掉精度、不报错。

[P4] 注意力健康度 —— softmax 熵 / 最大权重 / 有效位置数。
     饱和（熵→0）时注意力退化成硬 argmax，对输出贡献趋近固定置换。

用法:
    python -m tools.probe -c checkpoints/basemain/best.pth
    python -m tools.probe -c checkpoints/basemain/best.pth --freqs 8,10,12
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math

import numpy as np
import torch

import models  # noqa: F401  触发自注册
from core.checkpoint import load as load_checkpoint
from configs import base as C
from data.dataset import EMMultiFreqDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hr(t=""):
    print("-" * 72)
    if t:
        print(t)


def get_sample(sample_dir="data/data/rectangle/rectangle_1"):
    """取一个真实样本的几何（patch_plane）。"""
    path = sample_dir
    if not os.path.isabs(path):
        path = os.path.join(_ROOT, path)
    patch = np.load(os.path.join(path, "patch_plane.npy"))   # (64,64,2)
    x = torch.from_numpy(patch).permute(2, 0, 1).float()      # (2,64,64)
    return x.unsqueeze(0).to(DEVICE)


def p1_condition_sensitivity(model, x):
    """只改一个条件，看输出位移。三个量级应可比（历史 bug 是差 2000 倍）。"""
    model.eval()
    print("\n[P1] 条件敏感度（只改一个条件时的输出位移）")
    base_freq = torch.tensor([[0.375, 0.0]], device=DEVICE)   # 10GHz 居中

    with torch.no_grad():
        y0 = model(x, base_freq)
        # 只改频率：8GHz <-> 12GHz
        y_f1 = model(x, torch.tensor([[0.0, 0.0]], device=DEVICE))
        y_f2 = model(x, torch.tensor([[1.0, 0.0]], device=DEVICE))
        # 只改极化
        y_p1 = model(x, torch.tensor([[0.375, 1.0]], device=DEVICE))
        # 阈值：位移相对输出自身尺度
    scale = y0.abs().mean().item()
    d_freq = (y_f2 - y_f1).abs().mean().item()
    d_pol = (y_p1 - y0).abs().mean().item()
    print(f"  输出平均幅值        : {scale:.4e}")
    print(f"  只改频率(8G vs 12G) : {d_freq:.4e}   ({d_freq / (scale + 1e-12):.3%} of output)")
    print(f"  只改极化(0 vs 1)    : {d_pol:.4e}   ({d_pol / (scale + 1e-12):.3%} of output)")
    if d_freq < 1e-6 * scale:
        print("  [!!] 频率条件几乎无影响 —— 模型可能没用频率输入")
    if d_pol < 1e-6 * scale:
        print("  [!!] 极化条件几乎无影响 —— 模型可能没用极化输入")
    return {"d_freq": d_freq, "d_pol": d_pol, "scale": scale}


def p2_embedding_separability(model, n_freqs=9):
    """条件嵌入的两两距离：检查频率/极化是否被混叠。"""
    if not hasattr(model, "freq_encoder"):
        print("\n[P2] 模型没有 freq_encoder，跳过")
        return None
    print("\n[P2] 条件嵌入可分性（9 频点 × 2 极化）")
    freqs = torch.linspace(0, 1, n_freqs, device=DEVICE)
    embs, labels = [], []
    with torch.no_grad():
        for f in freqs:
            for p in (0.0, 1.0):
                cond = torch.tensor([[f.item(), p]], device=DEVICE)
                embs.append(model.freq_encoder(cond))
                labels.append(f"{8 + 4 * f.item():.1f}G,p{int(p)}")
    E = torch.cat(embs, 0)                       # (18, D)
    D = torch.cdist(E, E)                        # 两两距离

    # 最接近的异频对
    same_ft = []
    for i in range(n_freqs):
        for j in range(i + 1, n_freqs):
            d = min(D[2 * i, 2 * j].item(), D[2 * i, 2 * j + 1].item(),
                    D[2 * i + 1, 2 * j].item(), D[2 * i + 1, 2 * j + 1].item())
            same_ft.append((d, labels[2 * i], labels[2 * j]))
    same_ft.sort()
    # 同频不同极化
    pol_d = [D[2 * i, 2 * i + 1].item() for i in range(n_freqs)]

    print(f"  嵌入平均范数           : {E.norm(dim=-1).mean().item():.4f}")
    print(f"  同频不同极化的距离     : 最小 {min(pol_d):.4f} / 中位 {np.median(pol_d):.4f}")
    print(f"  最近的异频对           : {same_ft[0][0]:.4f}  ({same_ft[0][1]} vs {same_ft[0][2]})")
    print(f"  最远的异频对           : {same_ft[-1][0]:.4f}  ({same_ft[-1][1]} vs {same_ft[-1][2]})")
    # 判定：异频最近距离应显著大于 0
    if same_ft[0][0] < 1e-4:
        print("  [!!] 有异频条件的嵌入几乎重合 —— 编码器混叠了频率")
    if min(pol_d) < 1e-4:
        print("  [!!] 极化方向在嵌入里分不出来")
    return {"min_hetero_freq": same_ft[0][0], "min_pol": min(pol_d)}


def p3_modulation_strength(model):
    """超网络生成的 ΔW 相对基础 W 的量级。零初始化起步，训练后仍≈0 即死模块。"""
    print("\n[P3] 频域调制强度（超网络 ΔW / 基础 W）")
    found = False
    for name, mod in model.named_modules():
        if not hasattr(mod, "W_q"):
            continue
        found = True
        base_q = mod.W_q.abs().mean().item()
        base_k = mod.W_k.abs().mean().item()
        hyper_q = getattr(mod, "hyper_wq", None)
        hyper_k = getattr(mod, "hyper_wk", None)
        print(f"  {name}:")
        print(f"    |W_q| 基础值 {base_q:.4e}   |W_k| 基础值 {base_k:.4e}")
        if hyper_q is not None:
            last_w = hyper_q[-1].weight
            last_b = hyper_q[-1].bias
            print(f"    hyper_wq 末层 |W| {last_w.abs().mean().item():.4e}  |b| {last_b.abs().mean().item():.4e}")
            if last_w.abs().max().item() < 1e-8 and last_b.abs().max().item() < 1e-8:
                print("    [!!] 末层仍恒零 —— ΔW_q 恒为 0，频率调制是死的")
            if hasattr(mod, "gamma"):
                print(f"    gamma = {mod.gamma.item():.6f}")
    if not found:
        print("  未找到带 W_q 的谱卷积层（模型可能未启用注意力）")


def p4_attention_health(model, x):
    """注意力 softmax 的熵 / 最大权重 / 有效位置数。饱和 = 退化成 argmax。"""
    print("\n[P4] 注意力健康度")
    stats = []
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            # 重算一次 softmax 统计（forward 内部已算过，这里复算便于观测）
            stats.append(name)
        return hook

    for name, mod in model.named_modules():
        if hasattr(mod, "W_q"):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        model(x, torch.tensor([[0.375, 0.0]], device=DEVICE))
    for h in hooks:
        h.remove()
    print(f"  启用注意力的谱卷积层数: {len(stats)}")
    if not stats:
        print("  模型未启用注意力")
        return

    # 手动复算熵：取第一层，观察 scores 分布
    for name, mod in model.named_modules():
        if not hasattr(mod, "W_q"):
            continue
        with torch.no_grad():
            x_ft = torch.fft.rfft2(x.float())
            C_in = x_ft.size(1)
            m1 = min(mod.n_modes_attn, x_ft.size(2))
            m2 = min(mod.n_modes_attn, x_ft.size(3))
            sig = x_ft[:, :, :m1, :m2]
            if m1 < mod.n_modes_attn or m2 < mod.n_modes_attn:
                pad = torch.zeros(x.size(0), C_in, mod.n_modes_attn, mod.n_modes_attn,
                                  device=DEVICE, dtype=torch.cfloat)
                pad[:, :, :m1, :m2] = sig
                sig = pad
            X_sig = sig.reshape(x.size(0), C_in, -1)
            if getattr(mod, "sig_norm", True):
                X_sig = X_sig / X_sig.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
            Q = torch.bmm(X_sig, mod.W_q.unsqueeze(0).expand(x.size(0), -1, -1))
            K = torch.bmm(X_sig, mod.W_k.unsqueeze(0).expand(x.size(0), -1, -1))
            sc = torch.bmm(Q, K.conj().transpose(1, 2)) / math.sqrt(mod.d_k)
            w = torch.softmax(sc.abs() * mod.score_scale, dim=-1)
            ent = -(w * (w + 1e-12).log()).sum(-1).mean().item()
            maxw = w.max(-1).values.mean().item()
            eff = math.exp(ent)
        print(f"  {name}:")
        print(f"    |logit| max {sc.abs().max().item():.4e}")
        print(f"    熵 {ent:.4f}  (均匀={math.log(C_in):.4f})")
        print(f"    最大权重 {maxw:.4f}  (=1/C_in 是 {1.0 / C_in:.4f})")
        print(f"    有效位置数 {eff:.2f}  (={C_in} 表示均匀)")
        if eff < 1.5:
            print("    [!!] 注意力接近硬 argmax（饱和）")
        elif abs(eff - C_in) < 0.5:
            print("    [~] 注意力接近均匀 —— 可能还没学到选择性")
        break   # 只看第一层，避免刷屏


def main():
    ap = argparse.ArgumentParser(description="模型条件化有效性探针")
    ap.add_argument("--checkpoint", "-c", required=True)
    ap.add_argument("--sample-dir", default="data/data/rectangle/rectangle_1")
    args = ap.parse_args()

    lc = load_checkpoint(args.checkpoint)
    model = lc.model.to(DEVICE)
    print("=" * 72)
    print(f"探针: {args.checkpoint}")
    print(f"  模型 {lc.model_name}  参数量 {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 72)

    x = get_sample(args.sample_dir)
    p1_condition_sensitivity(model, x)
    p2_embedding_separability(model)
    p3_modulation_strength(model)
    p4_attention_health(model, x)
    hr()
    print("探针完成。")


if __name__ == "__main__":
    main()

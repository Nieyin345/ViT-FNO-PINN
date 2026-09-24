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

FREQS_GHZ = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]


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


def get_samples(n=4, sample_dir="data/data/rectangle/rectangle_1"):
    """取 n 个**不同几何**的样本（P6 用它量注意力的逐样本变化性）。

    同一目录下的多个 patch_plane 属于不同贴片形状；若目录里只有几何文件，
    按编号递增取。取不到就重复同一个 —— 此时跨样本差异恒 0，会误导判定，
    所以调用方需注意。
    """
    base = sample_dir if os.path.isabs(sample_dir) else os.path.join(_ROOT, sample_dir)
    parent = os.path.dirname(base)
    stem = os.path.basename(base)
    prefix = stem.rsplit("_", 1)[0] + "_"
    out = []
    for i in range(n):
        p = os.path.join(parent, f"{prefix}{i + 1}", "patch_plane.npy")
        if not os.path.exists(p):
            break
        patch = np.load(p)
        out.append(torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE))
    return out or [get_sample(sample_dir)]


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
    """注意力 softmax 的熵 / 最大权重 / 有效位置数。饱和 = 退化成 argmax。

    ★ 必须用**该层真实的输入**来算。SpectralConv2d 拿到的不是模型输入
    （2 通道），而是上一层出来的 (B, width, H, W) —— 用模型输入复算会得到
    C_in=2，注意力在 2 个通道上做 softmax，统计量毫无意义（恒为均匀）。
    所以这里用 forward_hook 把真实输入抓下来。
    """
    print("\n[P4] 注意力健康度")
    records = []

    def hook(mod, inp, out):
        records.append((mod, inp[0]))

    hooks = [mod.register_forward_hook(hook)
             for _, mod in model.named_modules() if hasattr(mod, "W_q")]
    if not hooks:
        print("  模型未启用注意力")
        return

    model.eval()
    with torch.no_grad():
        model(x, torch.tensor([[0.375, 0.0]], device=DEVICE))
    for h in hooks:
        h.remove()

    print(f"  启用注意力的谱卷积层数: {len(records)}")
    worst_eff = None
    for i, (mod, xin) in enumerate(records):
        with torch.no_grad():
            x_ft = torch.fft.rfft2(xin.float())
            C_in = x_ft.size(1)
            m1 = min(mod.n_modes_attn, x_ft.size(2))
            m2 = min(mod.n_modes_attn, x_ft.size(3))
            sig = x_ft[:, :, :m1, :m2]
            if m1 < mod.n_modes_attn or m2 < mod.n_modes_attn:
                pad = torch.zeros(xin.size(0), C_in, mod.n_modes_attn, mod.n_modes_attn,
                                  device=xin.device, dtype=torch.cfloat)
                pad[:, :, :m1, :m2] = sig
                sig = pad
            X_sig = sig.reshape(xin.size(0), C_in, -1)
            if getattr(mod, "sig_norm", True):
                X_sig = X_sig / X_sig.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-6)
            Q = torch.bmm(X_sig, mod.W_q.unsqueeze(0).expand(xin.size(0), -1, -1))
            K = torch.bmm(X_sig, mod.W_k.unsqueeze(0).expand(xin.size(0), -1, -1))
            sc = torch.bmm(Q, K.conj().transpose(1, 2)) / math.sqrt(mod.d_k)
            w = torch.softmax(sc.abs() * mod.score_scale, dim=-1)
            ent = -(w * (w + 1e-12).log()).sum(-1).mean().item()
            maxw = w.max(-1).values.mean().item()
            eff = math.exp(ent)
        ratio = eff / C_in
        print(f"  层{i} (C_in={C_in}): |logit|max {sc.abs().max().item():.3e}  "
              f"熵 {ent:.3f}/{math.log(C_in):.3f}  最大权重 {maxw:.4f}  有效位置 {eff:.2f}/{C_in} "
              f"({ratio:.1%})")
        if worst_eff is None or ratio < worst_eff:
            worst_eff = ratio
    if worst_eff < 0.2:
        print(f"  [!!] 有层注意力饱和（有效位置仅占 {worst_eff:.1%}），接近硬 argmax")
    elif worst_eff > 0.8:
        print(f"  [~] 注意力偏均匀（最低 {worst_eff:.1%}）—— 是否学到了选择性存疑")


def p6_mode_attention(model, xs):
    """[P6] 模式轴注意力健康度 —— 针对 fno_mattn 的 `mode_attn`。

    与 P4 的两个关键差别：
      1. 轴不同：P4 在**通道维** softmax（fno_attention），这里在**模式维**
         （N = m1·m2 个 token）softmax。有效位置数相对 N 才有意义。
      2. 必须同时量「修正的贡献」：‖γ·Wv(ctx)‖ / ‖out_low‖。若这个比例 ≈ 0，
         说明注意力修正在数值上没进入输出，再"健康"的注意力也是摆设。

    另加**逐样本变化性**：同一层在不同几何上的注意力图若有差异，才是
    "自适应滤波"；若完全一致，只是一个静态重参数化，不构成创新点。
    """
    print("\n[P6] 模式轴注意力健康度（fno_mattn.mode_attn）")
    captured = []

    def hook(mod, inp, out):
        captured.append((mod, inp[0].detach()))

    hooks = [mod.register_forward_hook(hook)
             for _, mod in model.named_modules() if hasattr(mod, "Wv") and hasattr(mod, "Wq")]
    if not hooks:
        print("  未找到模式注意力层（模型非 fno_mattn 系列）")
        return None

    model.eval()
    with torch.no_grad():
        for x in xs:
            model(x, torch.tensor([[0.375, 0.0]], device=DEVICE))
    for h in hooks:
        h.remove()

    n_layers = len(captured) // len(xs)
    worst_ratio, worst_eff = None, None
    for li in range(n_layers):
        mod = captured[li][0]
        ratios, ents, effs, maxws, attn_maps = [], [], [], [], []
        for si in range(len(xs)):
            out_low = captured[li + si * n_layers][1]           # (B,C,m1,m2)
            B, C, m1, m2 = out_low.shape
            t = out_low.mean(dim=1)
            feat = torch.stack([t.real.float(), t.imag.float()], dim=-1).reshape(B, m1 * m2, 2)
            q, k = mod.Wq(feat), mod.Wk(feat)
            sc = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(mod.d_k)
            attn = torch.softmax(sc, dim=-1)
            ctx = torch.bmm(attn, feat)
            corr = mod.gamma * mod.Wv(ctx)
            corr_ft = torch.complex(corr[..., 0], corr[..., 1]).reshape(B, m1, m2)
            ratios.append((corr_ft.abs().norm() / out_low.abs().norm().clamp_min(1e-12)).item())
            N = attn.size(-1)
            ent = -(attn * (attn + 1e-12).log()).sum(-1).mean().item()
            ents.append(ent)
            effs.append(math.exp(ent) / N)
            maxws.append(attn.max(-1).values.mean().item())
            attn_maps.append(attn[0])
        # 逐样本变化性：不同几何的注意力图的最大相对差
        var = max((attn_maps[i] - attn_maps[0]).abs().max().item() for i in range(len(xs)))
        print(f"  层{li} (N={m1 * m2}): γ={mod.gamma.item():.4f}  "
              f"|Wv|均{mod.Wv.weight.abs().mean().item():.3e}  "
              f"修正/主干 {np.mean(ratios):.2%}  "
              f"熵 {np.mean(ents):.3f}/{math.log(m1 * m2):.3f}  最大权重 {np.mean(maxws):.4f}  "
              f"有效位置 {np.mean(effs):.1%}  跨样本差异 {var:.2e}")
        if worst_ratio is None or np.mean(ratios) < worst_ratio:
            worst_ratio = np.mean(ratios)
        if worst_eff is None or np.mean(effs) < worst_eff:
            worst_eff = np.mean(effs)
    if worst_ratio < 0.01:
        print(f"  [!!] 注意力修正占主干输出仅 {worst_ratio:.2%} —— 数值上等于没接")
    if worst_eff > 0.9:
        print(f"  [~] 注意力过于均匀（{worst_eff:.1%}）—— 未学到模式选择性")
    return {"min_ratio": worst_ratio, "min_eff": worst_eff}


def p5_frequency_generalization(model, normalizer, holdout=(10.0, 11.0)):
    """留出频点泛化：在**指定频点**上测误差，与其余频点对比。

    ★ 为什么这个探针最重要：现有的 val_geom 是**按形状**划分的，训练/验证的
    9 个频点和 2 个极化完全一样（实测确认），所以它只衡量"新形状"，**测不到
    频率/极化泛化** —— 而这正是本项目的核心目标。用一个不测目标的指标去选
    模型，选出的可能是形状拟合更好、频率插值更差的那个。

    这里复用数据集本身（它负责散射场相减与输出反归一化），只是**只取留出频点**
    的样本算误差，与已见频点对比。比值 = 频率泛化的代价。

    注意：这只是**诊断**，不改变训练/早停口径（那需要动 build_split，见 README）。
    """
    from torch.utils.data import DataLoader, Subset
    print(f"\n[P5] 留出频点泛化诊断（留出 {list(holdout)} GHz）")
    ds = EMMultiFreqDataset(
        data_dir=C.DATA_DIR,
        target_d_over_lambda=C.TARGET_D_OVER_LAMBDA,
        use_scatter=C.USE_SCATTER_FIELD,
        bg_params_path=C.BACKGROUND_PARAMS_PATH,
        polarizations=C.DATASET_POLARIZATIONS,
    )
    if normalizer is not None:
        ds.set_output_normalization(normalizer)

    f = getattr(__import__("models.losses", fromlist=["x"]), "field_geom_loss_eh")
    model.eval()
    stats = {"seen": [], "hold": []}
    for fg in FREQS_GHZ:
        idx = [i for i, s in enumerate(ds.samples) if abs(s["freq_ghz"] - fg) < 1e-6]
        if not idx:
            continue
        loader = DataLoader(Subset(ds, idx), batch_size=16, shuffle=False, num_workers=0)
        errs = []
        with torch.no_grad():
            for b in loader:
                inp = b["input"].to(DEVICE)
                tgt = b["output"].to(DEVICE)
                freq = b["freq"].to(DEVICE)
                pred = model(inp, freq)
                if normalizer is not None:
                    pred = normalizer.denormalize(pred)
                errs.append(f(tgt, pred).item())
        m = float(np.mean(errs))
        key = "hold" if any(abs(fg - h) < 1e-6 for h in holdout) else "seen"
        stats[key].append((fg, m))
        flag = "  <-- 留出" if key == "hold" else ""
        print(f"  {fg:5.1f} GHz: Geom {m:.4f}{flag}")
    if not stats["hold"]:
        print("  留出频点无数据")
        return None
    s_mean = float(np.mean([m for _, m in stats["seen"]]))
    h_mean = float(np.mean([m for _, m in stats["hold"]]))
    print(f"  已见频点平均 {s_mean:.4f}  |  留出频点平均 {h_mean:.4f}  "
          f"|  代价 {h_mean / (s_mean + 1e-12):.3f}x")
    return {"seen": s_mean, "holdout": h_mean, "per_freq": stats}


def main():
    ap = argparse.ArgumentParser(description="模型条件化有效性探针")
    ap.add_argument("--checkpoint", "-c", required=True)
    ap.add_argument("--sample-dir", default="data/data/rectangle/rectangle_1")
    ap.add_argument("--holdout", default="10.0,11.0",
                    help="留出频点（逗号分隔 GHz，用于 P5）")
    args = ap.parse_args()

    lc = load_checkpoint(args.checkpoint)
    model = lc.model.to(DEVICE)
    print("=" * 72)
    print(f"探针: {args.checkpoint}")
    print(f"  模型 {lc.model_name}  参数量 {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 72)

    x = get_sample(args.sample_dir)
    holdout = tuple(float(t) for t in args.holdout.split(",") if t.strip())
    p1_condition_sensitivity(model, x)
    p2_embedding_separability(model)
    p3_modulation_strength(model)
    p4_attention_health(model, x)
    p6_mode_attention(model, get_samples(4, args.sample_dir))
    p5_frequency_generalization(model, lc.normalizer, holdout=holdout)
    hr()
    print("探针完成。")


if __name__ == "__main__":
    main()

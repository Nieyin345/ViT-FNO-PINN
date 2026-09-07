"""
形状泛化评估脚本
训练用 v2 数据的 circle/circle_ring/cross/square_ring（rectangle 留出），
测试在 rectangle（数据目录 "1_rectangle"）上的泛化误差
对比主模型（fno_attention）和基线（fno_baseline）
"""

import sys
import os
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from torch.utils.data import DataLoader
import config as C
from data.dataset import EMMultiFreqDataset
from data.norm_utils import OutputNormalizer
from models.fno2d import FNO2d
from models.fno2d_baseline import FNO2dBaseline
from models.losses import field_geom_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_out_channels(state):
    """从 state dict 推断输出通道数（支持 FNO2d 的 e_heads/h_heads/heads 与基线 proj）。"""
    e_keys = [k for k in state if re.match(r"^e_heads\.\d+\.net\.2\.weight$", k)]
    if e_keys:
        n_e = max(int(k.split(".")[1]) for k in e_keys) + 1
        n_h = 0
        h_keys = [k for k in state if re.match(r"^h_heads\.\d+\.net\.2\.weight$", k)]
        if h_keys:
            n_h = max(int(k.split(".")[1]) for k in h_keys) + 1
        return (n_e + n_h) * 2
    heads = [k for k in state if re.match(r"^heads\.\d+\.net\.2\.weight$", k)]
    if heads:
        n_heads = max(int(k.split(".")[1]) for k in heads) + 1
        return n_heads * 2
    if "proj.2.weight" in state:
        return state["proj.2.weight"].shape[0]
    return None


def _infer_model_kwargs(state):
    """从 checkpoint state dict 推断模型结构参数，避免与训练配置不一致。"""
    kw = {}
    lift = next((k for k in state if k.startswith("lift.") and k.endswith(".weight")), None)
    if lift is not None and state[lift].ndim == 4:
        kw["in_channels"] = int(state[lift].shape[1])
        kw["width"] = int(state[lift].shape[0])
    w1 = next((k for k in state if re.search(r"\.spectral\.w1$", k)), None)
    if w1 is not None and state[w1].ndim == 4:
        kw["modes1"] = int(state[w1].shape[2])
        kw["modes2"] = int(state[w1].shape[3])
    block_keys = [k for k in state if re.match(r"^blocks\.\d+\.", k)]
    if block_keys:
        kw["n_layers"] = max(int(k.split(".")[1]) for k in block_keys) + 1
    hyper = next((k for k in state if re.search(r"\.hyper_wq\.0\.weight$", k)), None)
    if hyper is not None and state[hyper].ndim == 2:
        kw["freq_dim"] = int(state[hyper].shape[1])
    out_ch = _infer_out_channels(state)
    if out_ch is not None:
        kw["out_channels"] = out_ch
    return kw


def load_model(model_cls, ckpt_path, **kwargs):
    if not os.path.exists(ckpt_path):
        print(f"  [警告] checkpoint 不存在: {ckpt_path}")
        return model_cls(**kwargs).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    # 兼容 module. 前缀
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    # 显式 kwargs 优先，未指定的结构参数从 checkpoint 推断
    model_kwargs = {**_infer_model_kwargs(state), **kwargs}
    # 过滤模型不支持的参数（如基线模型不接受 in_channels/freq_dim）
    import inspect
    sig = inspect.signature(model_cls.__init__)
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in sig.parameters}
    model = model_cls(**model_kwargs).to(DEVICE)
    model_sd = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in model_sd and model_sd[k].shape == v.shape}
    model_sd.update(compatible)
    model.load_state_dict(model_sd, strict=False)
    print(f"  加载权重: {ckpt_path}  ({len(compatible)}/{len(state)} 层匹配, "
          f"结构: {model_kwargs})")
    if len(compatible) < len(state):
        print("  [警告] checkpoint 与当前模型架构存在不匹配层 "
              f"({len(state) - len(compatible)} 层)，未匹配权重已随机初始化，"
              "评估结果可能失真；建议用当前代码重新训练后评估。")
    return model


def maybe_denorm(pred, ckpt_path):
    """从 checkpoint 读取归一化统计量并反归一化。"""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    norm = OutputNormalizer.from_checkpoint(ckpt)
    if norm is None:
        return pred
    return norm.denormalize(pred)


def evaluate(model, loader, ckpt_path, model_name, normalizer=None):
    model.eval()
    geom_errors = []
    rel_errors  = []

    with torch.no_grad():
        for batch in loader:
            inp  = batch["input"].to(DEVICE)
            tgt  = batch["output"].to(DEVICE)
            freq = batch["freq"].to(DEVICE)

            pred = model(inp, freq)

            # 反归一化 pred（物理空间评估）
            if normalizer is not None:
                pred = normalizer.denormalize(pred)

            # geom loss（物理空间）
            g, _ = field_geom_loss(pred, tgt)
            geom_errors.append(g.item())

            # 相对 L2
            B = pred.shape[0]
            diff = (pred - tgt).reshape(B, -1).norm(dim=1)
            norm = tgt.reshape(B, -1).norm(dim=1).clamp(min=1e-6)
            rel_errors.append((diff / norm).mean().item())

    geom_mean = np.mean(geom_errors)
    rel_mean  = np.mean(rel_errors)
    print(f"\n  [{model_name}]")
    print(f"    Geom Loss:  {geom_mean:.4f}")
    print(f"    Rel L2:     {rel_mean:.4f}")
    return geom_mean, rel_mean


def main():
    print("=" * 55)
    print("形状泛化评估：训练集 circle/circle_ring/cross/square_ring，测试 rectangle")
    print("=" * 55)

    # 加载 rectangle 数据（不做归一化，在物理空间评估）
    # 注意：必须和训练时保持一致，如果训练用散射场，这里也要用散射场
    dataset = EMMultiFreqDataset(
        data_dir=C.DATA_DIR,
        output_field="all",
        normalize=False,
        use_scatter=getattr(C, "USE_SCATTER_FIELD", False),
        bg_params_path=getattr(C, "BACKGROUND_PARAMS_PATH", "data/blank/background_params.npz"),
    )
    # 矩形类数据在 v2 数据中目录名为 "1_rectangle"
    dataset.samples = [s for s in dataset.samples
                       if "rectangle" in s["category"]]
    print(f"\nRectangle 样本数: {len(dataset.samples)}")
    if len(dataset.samples) == 0:
        print("未找到 rectangle 数据，请检查路径")
        return

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    # ── 参数量说明 ──────────────────────────────────────
    # fno_attention(w64,m8,5层) ≈ 8.05M；fno_baseline_large(w72) ≈ 3.35M；
    # fno_baseline_parity(w112) ≈ 8.10M（与主模型同量级，推荐作为公平对比基线）
    print("\n参数量：fno_attention≈8.05M, fno_baseline_large≈3.35M, fno_baseline_parity≈8.10M")

    # ── 主模型 ──────────────────────────────────────────
    ckpt_attn = "../checkpoints/fno_attention/best.pth"
    print("\n加载主模型 (fno_attention)...")
    model_attn = load_model(FNO2d, ckpt_attn)

    # ── 基线模型 ─────────────────────────────────────────
    candidates = [
        ("fno_baseline_large", "../checkpoints/fno_baseline_large/best.pth", FNO2dBaseline),
        ("fno_baseline_parity", "../checkpoints/fno_baseline_parity/best.pth", FNO2dBaseline),
    ]
    entries = [("fno_attention", model_attn, ckpt_attn)]
    for name, ckpt, cls in candidates:
        if os.path.exists(ckpt):
            print(f"\n加载基线模型 ({name})...")
            entries.append((name, load_model(cls, ckpt), ckpt))
        else:
            print(f"\n[跳过] {name} checkpoint 不存在（{ckpt}），仅保留 baseline_large")

    # ── 读取各自的归一化统计量并评估 ─────────────────────
    def load_normalizer(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        return OutputNormalizer.from_checkpoint(ckpt)

    print("\n开始评估...")
    results = []
    for name, model, ckpt in entries:
        g, r = evaluate(model, loader, ckpt, name, load_normalizer(ckpt))
        results.append((name, g, r))

    # ── 汇总 ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("汇总对比（rectangle 形状泛化）")
    print(f"{'模型':<20} {'Geom Loss':>12} {'Rel L2':>10}")
    print("-" * 45)
    for name, g, r in results:
        print(f"{name:<20} {g:>12.4f} {r:>10.4f}")
    if len(results) >= 2:
        base_name, base_g, base_r = results[1]
        g0, r0 = results[0][1], results[0][2]
        print("-" * 45)
        print(f"{'ratio (attn/base)':<20} {g0/base_g:>12.3f}x {r0/base_r:>10.3f}x")
    print("=" * 55)
    print("< 1.0 表示主模型更好，> 1.0 表示基线更好")


if __name__ == "__main__":
    main()


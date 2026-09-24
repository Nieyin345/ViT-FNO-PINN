# -*- coding: utf-8 -*-
"""泛化评估：在留出形状（rectangle）上评估一个或多个 checkpoint。

用法（在仓库根执行）：
    python -m tools.eval --checkpoint checkpoints/fno_attention/best.pth
    python -m tools.eval --checkpoint a/best.pth b/best.pth --categories rectangle
    python -m tools.eval --checkpoint checkpoints/fno_v2/best.pth --device cpu

重构变更（这是本轮修复的核心症状）：
  原先本文件用 `_infer_model_kwargs` 从权重形状反推模型结构，再
  `load_state_dict(..., strict=False)`，只打印"匹配 N 层，跳过 M 层"。
  实测 bug：fno_v2 的 checkpoint 被推出 in_channels=6（因为它从 `lift.0.weight`
  的输入通道数读，而 FNO2dV2 的 lift 内部已拼了 4 个正弦通道，构造时的语义是 2），
  于是 57/58 层匹配、`lift` 被静默随机初始化——**架构错的运行看起来像成功的运行**。

  现在模型结构随权重一起存在 checkpoint 里（`model_config`），加载走
  `core.checkpoint.load` 并 `strict=True`，架构不符直接抛错。
  同时删掉了写死的 `../checkpoints/fno_attention/best.pth` 与
  `FNO2dBaseline`/`UNet2d` 的身份判断分支。
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader

import models  # noqa: F401  触发自注册
from core.checkpoint import load as load_checkpoint
from configs import base as C
from data.dataset import EMMultiFreqDataset
from models.losses import field_geom_loss_eh

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="在留出形状上评估 checkpoint 的泛化性能",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", "-c", nargs="+", required=True,
                   help="一个或多个 .pth 路径")
    p.add_argument("--categories", default=None,
                   help="逗号分隔的类别过滤（如 rectangle）；不填 = 全部样本")
    p.add_argument("--limit", type=int, default=None,
                   help="只评估前 N 条样本（快速冒烟用；默认全部）")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def build_dataset(categories=None, limit=None):
    """构造评估数据集。

    数据集构造参数必须与 `tools/train.py` 逐项一致（散射场、极化、插值距离），
    否则评估的是分布不同的数据，指标没有可比性。输出不做归一化——归一化器的
    反变换在模型输出之后做。
    """
    dataset = EMMultiFreqDataset(
        data_dir=C.DATA_DIR,
        output_field="all",
        normalize=False,
        target_d_over_lambda=getattr(C, "TARGET_D_OVER_LAMBDA", 0.1),
        use_scatter=getattr(C, "USE_SCATTER_FIELD", False),
        bg_params_path=getattr(C, "BACKGROUND_PARAMS_PATH", "data/blank/background_params.npz"),
        polarizations=getattr(C, "DATASET_POLARIZATIONS", None),
    )
    n_all = len(dataset.samples)
    if categories:
        dataset.samples = [s for s in dataset.samples
                           if any(s["category"] == c or s["category"].startswith(c + "/")
                                  for c in categories)]
        print(f"[数据] 类别 {categories}: {len(dataset.samples)}/{n_all} 条样本")
    else:
        print(f"[数据] 全部样本: {n_all} 条")
    if limit:
        dataset.samples = dataset.samples[:limit]
        print(f"[数据] 截断到前 {len(dataset.samples)} 条")
    return dataset


@torch.no_grad()
def evaluate(model, loader, normalizer, name):
    """物理空间评估。

    ★ 本函数**必须同时报"幅值比"**（|pred| / |tgt|）。只看 Geom/RelL2 会被误导。
    实测（真实数据上人为缩放预测，看 Geom 怎么变）：
        预测×1.0（完美）Geom 0.0000 | 预测×0.5 Geom 0.5035
        预测×0.05       Geom 0.9567 | 预测×0.0 Geom 3.0069
    Geom 在缩放上单调正确，但**缩放到 5% 就能拿到 0.96 这个看似不错的值**——
    而几何损失里的方向项对接近零的预测本就没有意义。幅值比是识别这种情况
    最直接的证据，必须一并打印。

    注意：这里的口径与 `tools/train.py` 的验证口径**不同**：
      - 本文件的 Geom 是 E/H 分组平均（与 Trainer 的 `geom_metric` 同）。
      - 本文件的 RelL2 是 12 通道整体展平的相对 L2；Trainer 用
        `channel_group_balanced_l2`（分组归一后平均）。
    两者都随 checkpoint 一起记录在 `summary.json` / 终端输出里，跨文件比较时
    请认准是哪个口径。**另注**：checkpoint 里的 `val_geom` 是**训练时那批数据**
    上的结果；若本地数据与服务器训练数据不是同一批，这里的数字**不可比**
    （实测本地数据与服务器数据不是同一批，见 `_check_data_match`）。
    """
    model.eval()
    geom_errors, rel_errors, mag_ratios = [], [], []
    for batch in loader:
        inp = batch["input"].to(DEVICE)
        tgt = batch["output"].to(DEVICE)
        freq = batch["freq"].to(DEVICE)

        pred = model(inp, freq)
        if normalizer is not None:
            pred = normalizer.denormalize(pred)

        g = field_geom_loss_eh(pred, tgt)
        geom_errors.append(g.item())

        b = pred.shape[0]
        diff = (pred - tgt).reshape(b, -1).norm(dim=1)
        base = tgt.reshape(b, -1).norm(dim=1).clamp(min=1e-6)
        rel_errors.append((diff / base).mean().item())
        mag_ratios.append((pred.abs().mean() / tgt.abs().mean().clamp(min=1e-12)).item())

    geom_mean = float(np.mean(geom_errors))
    rel_mean = float(np.mean(rel_errors))
    mag_mean = float(np.mean(mag_ratios))
    print(f"\n  [{name}]")
    print(f"    Geom Loss:  {geom_mean:.4f}")
    print(f"    Rel L2:     {rel_mean:.4f}")
    print(f"    幅值比:     {mag_mean:.4f}   (|pred|/|tgt|，应接近 1.0)")
    if mag_mean < 0.5 or mag_mean > 2.0:
        print(f"    [!!] 预测幅值严重偏离目标（{mag_mean:.3f}）——"
              f"Geom/RelL2 在此情况下会失真，先确认模型不是塌陷到 0 或爆炸")
    return geom_mean, rel_mean, mag_mean


def _check_data_match(dataset):
    """警告：评估用的本地数据是否可能存在"与训练数据不是同一批"的风险。

    ★ 为什么需要：2026-09-24 实测发现本地 `data/data/rectangle` 与服务器
    `/proj/.../data/rectangle` **不是同一批数据**——同名样本的场值不同
    （rectangle_199 差 38%），且本地是 115 双极化 + 85 单极化、服务器是
    199 全双极化。于是"用本地数据评估服务器训好的模型"得到的数字毫无意义：
    实测 fno_attention 训练报 val_geom=0.326、本地 eval 报 7.40（差 22 倍），
    而两边数据恰好一致的那次（no_attn）只差 3%。

    这里用**极化构成**做粗判（原始 200 形状里两个极化应当均衡）：
    训练若在均衡数据上做，而评估侧不均衡，指标会被样本多的极化主导。
    """
    pols = {}
    for s in dataset.samples:
        p = int(s.get("polarization", 0))
        pols[p] = pols.get(p, 0) + 1
    print(f"  [数据] {len(dataset.samples)} 条样本，极化构成 {pols}")
    if len(pols) > 1 and min(pols.values()) / max(pols.values()) < 0.8:
        print(f"  [!!] 两个极化的样本数**不均衡**（{pols}）。若模型是在均衡数据上"
              f"训练的，此处的指标会被样本多的极化主导（本地曾是 1800:1035，"
              f"服务器是 1791:1791）——请确认评估数据与训练数据是同一批。")



def main(argv=None):
    args = parse_args(argv)
    global DEVICE
    if args.device:
        DEVICE = torch.device(args.device)

    print("=" * 62)
    print(f"泛化评估  (device={DEVICE})")
    print("=" * 62)

    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None
    dataset = build_dataset(categories, args.limit)
    if len(dataset.samples) == 0:
        print("没有匹配的样本，请检查 --categories / 数据目录")
        return 1
    _check_data_match(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    results = []
    for path in args.checkpoint:
        print(f"\n加载 {path} ...")
        try:
            lc = load_checkpoint(path, map_location=str(DEVICE))
        except (FileNotFoundError, KeyError, RuntimeError) as e:
            print(f"  [失败] {type(e).__name__}: {e}")
            continue
        lc.model.to(DEVICE)
        g, r, mag = evaluate(lc.model, loader, lc.normalizer, lc.model_name)
        # checkpoint 里记的 val_geom 是**训练时数据**上的结果。与本次评估对比，
        # 若差得离谱，多半是"评估数据 != 训练数据"，而不是模型坏了。
        ck_val = (lc.meta or {}).get("val_geom")
        if ck_val is not None:
            ratio = g / (float(ck_val) + 1e-12)
            note = ""
            if ratio > 3 or ratio < 1 / 3:
                note = "   [!!] 与 ckpt 的 val_geom 差 >3x —— 先核对两边数据是否同一批"
            print(f"    ckpt 记录的 val_geom: {float(ck_val):.4f}（训练时数据）"
                  f"  本次/记录 = {ratio:.2f}x{note}")
        results.append({"path": path, "name": lc.model_name,
                        "geom": g, "rel": r, "mag": mag,
                        "params": sum(p.numel() for p in lc.model.parameters())})

    if not results:
        print("\n没有任何 checkpoint 评估成功。")
        return 1

    print("\n" + "=" * 62)
    print(f"汇总（{len(dataset.samples)} 条样本）")
    hdr = f"{'模型':<16}{'参数量':>12}{'Geom Loss':>12}{'Rel L2':>10}{'幅值比':>10}  路径"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda r: r["geom"]):
        print(f"{r['name']:<16}{r['params']:>12,}{r['geom']:>12.4f}{r['rel']:>10.4f}"
              f"{r['mag']:>10.4f}  {r['path']}")
    print("=" * 62)
    print("注：幅值比应接近 1.0。若某模型 Geom 明显更低但幅值比也低（<<1），")
    print("    那是预测塌陷，不是更好的模型；跨数据集的绝对数值不可直接比较。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

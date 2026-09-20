# -*- coding: utf-8 -*-
"""结果报告：把各 run 的 summary.json 汇总成对比表，并按场景代次分组。

用法（仓库根执行）：
    python -m tools.report                       # 汇总 runs/ 与 checkpoints/
    python -m tools.report runs/gen1 runs/gen2   # 指定目录
    python -m tools.report --csv out.csv         # 另存 CSV

## 它解决什么问题

主模型的损失函数要反复改，而每次改完都要回答两个问题：

    A. 我的新主模型，比冻结的基线好了多少？      （必须同场景）
    B. 这次改损失，比上一版主模型好了多少？      （场景同、损失不同）

两个问题的前提都是"尺子没变"。本工具按**场景指纹**分组：

  - 同组内（场景一致）→ 打一张表，按 val_geom 升序，**这组内可直接比大小**；
  - 跨组 → 不合并成一张表，只提示"这些数字不可直接比"，并列出场景差异。

## 为什么基线冻结就够

保存/早停指标不读损失配置（见 configs/scenario.py 与 tools/smoke.py 的守卫），
所以基线在旧损失下选出的 best.pth，在新损失下仍然是那个基线。
换损失**不需要**重跑基线；只有当**场景**变了（模型类变了，旧 checkpoint 加载不进去）
才需要重跑一代。

## 损失食谱的用法

同场景不同损失的两个 run 会并排列出损失指纹，并回答"这次改损失买到了什么"。
不同场景的 run 之间的损失差异只作记录，不作结论。
"""

import argparse
import csv
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_DIRS = [os.path.join(_ROOT, "runs"), os.path.join(_ROOT, "checkpoints")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="按场景代次汇总各 run 的指标",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dirs", nargs="*", default=None,
                   help=f"要扫描的目录，默认 {' '.join(DEFAULT_DIRS)}")
    p.add_argument("--csv", default=None, help="另存一份 CSV 到该路径")
    p.add_argument("--quiet", action="store_true", help="不打印分组明细，只出总表")
    return p.parse_args(argv)


def find_summaries(dirs):
    """递归找 summary.json，跳过明显的临时目录。"""
    out = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, dirnames, filenames in os.walk(d):
            dirnames[:] = [x for x in dirnames
                           if not x.startswith(".") and x not in ("__pycache__",)]
            if "summary.json" in filenames:
                out.append(os.path.join(root, "summary.json"))
    return sorted(out)


def load_runs(paths):
    runs = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                s = json.load(f)
        except (OSError, ValueError) as e:
            print(f"  [跳过] {p}: {type(e).__name__}: {e}")
            continue
        s["_path"] = p
        s["_name"] = s.get("name") or os.path.basename(os.path.dirname(p))
        runs.append(s)
    return runs


def group_key(run):
    """按场景指纹分组。没有场景信息的老 run 归到 'legacy'。"""
    sc = run.get("scenario") or {}
    return sc.get("fingerprint") or "legacy"


def fmt_params(n):
    if not n:
        return "-"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def fmt(v, spec=".4f"):
    if v is None:
        return "-"
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


def mark_stale(runs):
    """标记**已过期**的 run：它的模型类在当前代码下已经重建不出来了。

    判据是机械的：拿存档里的 `model_name` + `model_config` 用**当前代码**
    重建模型，再哈希其 state_dict 签名，与存档记下的 `arch_fingerprint` 比。
    不一致（或根本构造不出来）→ 那条结果的尺子已经换过，不能再与现在的比。

    重要：过期的原因是**改了模型类**，不是改了损失。改损失不影响这里——
    因为保存/早停指标不读损失配置（见 check_metric_independence）。
    """
    from configs import scenario as SC

    for r in runs:
        saved = r.get("arch_fingerprint")
        mn = r.get("model")
        cfg = r.get("model_config")
        if not saved or not mn or not cfg:
            r["_arch_status"] = "unknown"
            continue
        try:
            now = SC.arch_fingerprint_from_cfg(mn, cfg)
        except Exception as e:
            r["_arch_status"] = "gone"
            r["_arch_note"] = f"{type(e).__name__}: {e}"
            continue
        r["_arch_status"] = "ok" if now == saved else "changed"


def print_stale(runs):
    stale = [r for r in runs if r.get("_arch_status") in ("changed", "gone")]
    if not stale:
        return
    print(f"\n[过期] {len(stale)} 个 run 的模型类在当前代码下已重建不出（"
          f"旧结果不可再与现在的比）：")
    for r in stale:
        why = "架构变了" if r["_arch_status"] == "changed" else "构造失败"
        note = f"  {r['_arch_note']}" if r.get("_arch_note") else ""
        print(f"    {r.get('_name'):<28} {r.get('model'):<14} {why}{note}")


def _arch_mark(status):
    """表格里的架构状态标记。

    只用 ASCII：Windows 终端默认 GBK，`✓` 之类的符号会 UnicodeEncodeError。
    """
    return {"ok": "ok", "changed": "STALE", "gone": "GONE"}.get(status, "?")


def print_table(runs, title):
    ok = [r for r in runs if r.get("val_geom") is not None]
    ok.sort(key=lambda r: r["val_geom"])
    if not ok:
        return

    cols = [("name", 28), ("model", 14), ("params", 9),
            ("val_geom", 10), ("val_rel", 9), ("best_ep", 8),
            ("loss", 14), ("epochs", 7), ("arch", 7)]
    header = "".join(c.ljust(w) for c, w in cols)
    print(f"\n{title}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in ok:
        lr = r.get("loss_recipe") or {}
        vals = [
            r.get("_name", ""), r.get("model", ""), fmt_params(r.get("n_params")),
            fmt(r.get("val_geom")), fmt(r.get("val_rel")),
            str(r.get("best_epoch", "")),
            (lr.get("fingerprint") or "-")[:12],
            str(r.get("epochs", "")),
            _arch_mark(r.get("_arch_status")),
        ]
        print("".join(str(v).ljust(w) for v, (_, w) in zip(vals, cols)))
    print("=" * len(header))


def print_scenario_diff(runs):
    """列出各场景代次之间的差异——告诉用户"到底改了什么"。"""
    from configs import scenario as SC

    by_fp = {}
    for r in runs:
        sc = r.get("scenario") or {}
        fp = sc.get("fingerprint")
        if fp and fp not in by_fp:
            by_fp[fp] = sc

    if len(by_fp) < 2:
        return
    fps = sorted(by_fp)
    print(f"\n场景差异（共 {len(fps)} 代）")
    print("-" * 68)
    base = by_fp[fps[0]]
    print(f"  以 {fps[0]} 为基准：")
    for fp in fps[1:]:
        d = SC.diff(base.get("keys", {}), by_fp[fp].get("keys", {}))
        print(f"\n  [{fp}]  与基准相差 {len(d)} 项"
              + (f"，划分指纹 {by_fp[fp].get('split')}" if by_fp[fp].get("split") else ""))
        for k, a, b in d[:12]:
            print(f"      {k}: {a}  ->  {b}")
        if len(d) > 12:
            print(f"      ... 另有 {len(d)-12} 项")


def print_loss_diff(runs):
    """同一场景内，不同损失食谱的差异 —— 回答"这次改损失买到了什么"。"""
    from configs import scenario as SC

    groups = {}
    for r in runs:
        lr = r.get("loss_recipe") or {}
        fp = lr.get("fingerprint")
        if fp:
            groups.setdefault(group_key(r), {})[fp] = lr

    for gk, fps in groups.items():
        if len(fps) < 2:
            continue
        print(f"\n同一场景 [{gk}] 下的损失版本（共 {len(fps)} 版）")
        print("-" * 68)
        keys = sorted(fps)
        base = fps[keys[0]]
        for fp in keys:
            print(f"  [{fp}]  " + "、".join(fps[fp].get("_active") or ["(无)"]))
        print(f"\n  以 {keys[0]} 为基准的差异：")
        for fp in keys[1:]:
            d = SC.diff(base, fps[fp])
            print(f"    [{fp}]  {len(d)} 项变化")
            for k, a, b in d[:8]:
                print(f"        {k}: {a}  ->  {b}")


def write_csv(runs, path):
    fields = ["name", "model", "experiment", "width", "n_layers", "n_params",
              "val_geom", "val_rel", "best_val_loss", "best_epoch", "epochs",
              "scenario_fp", "split_fp", "loss_fp", "loss_active",
              "arch_status", "save_dir", "path"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            sc = r.get("scenario") or {}
            lr = r.get("loss_recipe") or {}
            w.writerow({
                **r,
                "name": r.get("_name"), "path": r.get("_path"),
                "scenario_fp": sc.get("fingerprint"),
                "split_fp": sc.get("split"),
                "loss_fp": lr.get("fingerprint"),
                "loss_active": "、".join(lr.get("_active") or []),
                "arch_status": r.get("_arch_status"),
            })
    print(f"\n[汇总] {path}")


def main(argv=None):
    args = parse_args(argv)
    dirs = args.dirs or DEFAULT_DIRS
    paths = find_summaries(dirs)
    if not paths:
        print(f"没找到 summary.json（扫描了 {dirs}）")
        return 1

    runs = load_runs(paths)
    print(f"共 {len(runs)} 个 run（来自 {len(paths)} 个 summary.json）")

    mark_stale(runs)

    groups = {}
    for r in runs:
        groups.setdefault(group_key(r), []).append(r)

    if "legacy" in groups:
        print(f"\n[注意] {len(groups['legacy'])} 个 run 没有场景指纹"
              f"（早于本次改动，或未走 tools.train），单独列出、不参与同代比较。")

    for gk in sorted(groups):
        tag = "（无场景指纹，不可同代比较）" if gk == "legacy" else ""
        print_table(groups[gk],
                    f"场景 [{gk}]  {len(groups[gk])} 个 run{tag}")

    if not args.quiet and len(groups) > 1:
        print_scenario_diff(runs)
    if not args.quiet:
        print_loss_diff(runs)
        print_stale(runs)

    if len(groups) > 1:
        print("\n[提醒] 上表按场景分代。**跨代的数字不可直接比大小**——"
              "场景变了（数据/划分/训练条件/归一化），尺子就变了。")
        print("        同代内按 val_geom 升序，可直接比较。")

    if args.csv:
        write_csv(runs, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())

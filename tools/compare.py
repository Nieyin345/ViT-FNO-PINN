# -*- coding: utf-8 -*-
"""对比实验运行器：跑一个实验矩阵并出汇总表。

用法（在仓库根执行）：
    python -m tools.compare --experiments fno_v2 fno_baseline --widths 32,64 --layers 3,5 --epochs 15
    python -m tools.compare --experiments fno_v2 --widths 32 --layers 3 --epochs 2 --dry-run
    python -m tools.compare --experiments fno_v2 --resume      # 已有 summary.json 的组合跳过

设计要点：
  - **每个组合起一个子进程**跑 `tools.train`。这是必须的：`configs.base` 的配置项是
    模块级全局值，同进程内反复切换会互相污染（内存、显存、随机状态也都隔离不掉）。
  - 每个 run 落到 `<runs>/<name>/`，其中 `summary.json` 由 `tools.train` 写出。
  - 全部跑完在终端打表，并写 `<runs>/summary.csv`。
  - `--resume` 让中断后的续跑只补没跑完的组合。
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_RUNS_DIR = os.path.join(_ROOT, "runs")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="跑对比实验矩阵并出汇总表",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--experiments", "-e", nargs="+", required=True,
                   help="实验名（configs/experiments.py 的 key）")
    p.add_argument("--widths", default=None,
                   help="逗号分隔的 width 列表，如 32,64；不填则用配置默认值")
    p.add_argument("--layers", default=None,
                   help="逗号分隔的 n_layers 列表，如 3,5；不填则用配置默认值")
    p.add_argument("--extra", action="append", default=[],
                   help="额外透传给 tools.train 的参数，可重复，如 "
                        "--extra=--kernel-cond --extra=lora")
    p.add_argument("--epochs", type=int, required=True, help="每个组合的训练轮数")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda/cpu，默认交给 tools.train 自动判断")
    p.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help="结果根目录")
    p.add_argument("--resume", action="store_true",
                   help="已有 summary.json 的组合跳过")
    p.add_argument("--dry-run", action="store_true", help="只打印将要跑的组合，不执行")
    p.add_argument("--keep-going", action="store_true", default=True,
                   help="某个组合失败后继续跑其余组合（默认开启）")
    p.add_argument("--fail-fast", dest="keep_going", action="store_false")
    return p.parse_args(argv)


def parse_int_list(s):
    if not s:
        return [None]
    return [int(v.strip()) for v in s.split(",") if v.strip()]


def combo_name(exp, width, layers):
    parts = [exp]
    if width is not None:
        parts.append(f"w{width}")
    if layers is not None:
        parts.append(f"l{layers}")
    return "_".join(parts)


def build_matrix(args):
    """展开 (实验, width, layers) 组合。"""
    from configs.experiments import EXPERIMENTS

    unknown = [e for e in args.experiments if e not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"未知实验: {unknown}，可选: {sorted(EXPERIMENTS)}")

    out = []
    for exp in args.experiments:
        for width in parse_int_list(args.widths):
            for layers in parse_int_list(args.layers):
                out.append((exp, width, layers))
    return out


def run_one(exp, width, layers, name, args):
    """起一个子进程训练一个组合，返回 (成功?, summary dict 或 None, 秒数)。"""
    run_dir = os.path.join(args.runs_dir, name)
    cmd = [sys.executable, "-m", "tools.train",
           "--experiment", exp,
           "--epochs", str(args.epochs),
           "--save-dir", run_dir]
    if width is not None:
        cmd += ["--width", str(width)]
    if layers is not None:
        cmd += ["--layers", str(layers)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.num_workers is not None:
        cmd += ["--num-workers", str(args.num_workers)]
    if args.device:
        cmd += ["--device", args.device]
    cmd += args.extra

    print(f"\n{'='*72}\n[{name}] {' '.join(cmd[1:])}\n{'='*72}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=_ROOT)
    dt = time.time() - t0

    if r.returncode != 0:
        print(f"[{name}] 失败（exit={r.returncode}），耗时 {dt:.1f}s")
        return False, None, dt

    sp = os.path.join(run_dir, "summary.json")
    if not os.path.isfile(sp):
        print(f"[{name}] 训练成功但没写出 summary.json: {sp}")
        return False, None, dt
    with open(sp, encoding="utf-8") as f:
        s = json.load(f)
    s["minutes"] = dt / 60.0
    return True, s, dt


def fmt_params(n):
    if n is None:
        return "-"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def print_table(rows, failures):
    """按 val_geom 升序打印（geom 是保存最优模型的指标，越小越好）。"""
    ok = [r for r in rows if r.get("val_geom") is not None]
    ok.sort(key=lambda r: r["val_geom"])

    cols = [("name", 26), ("model", 14), ("w", 4), ("L", 3), ("params", 9),
            ("val_geom", 10), ("val_rel", 9), ("best_ep", 8), ("min", 7)]
    header = "".join(c.ljust(w) for c, w in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in ok:
        vals = [r.get("name", ""), r.get("model", ""), str(r.get("width", "")),
                str(r.get("n_layers", "")), fmt_params(r.get("n_params")),
                f"{r['val_geom']:.4f}", f"{r.get('val_rel', float('nan')):.4f}",
                str(r.get("best_epoch", "")), f"{r.get('minutes', 0):.1f}"]
        print("".join(str(v).ljust(w) for v, (_, w) in zip(vals, cols)))
    print("=" * len(header))
    if failures:
        print(f"\n失败的组合 ({len(failures)}): {', '.join(failures)}")


def write_csv(rows, failures, path):
    fields = ["name", "model", "experiment", "width", "n_layers", "modes1", "modes2",
              "head", "kernel_cond", "lora_rank", "n_params",
              "val_geom", "val_rel", "best_val_loss", "best_epoch", "minutes",
              "epochs", "save_dir", "status"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "status": "ok"})
        for name in failures:
            w.writerow({"name": name, "status": "failed"})
    print(f"\n[汇总] {path}")


def main(argv=None):
    args = parse_args(argv)
    matrix = build_matrix(args)

    names = [combo_name(e, w, l) for e, w, l in matrix]
    print(f"实验矩阵: {len(matrix)} 个组合，每个 {args.epochs} epochs")
    for n in names:
        print(f"  - {n}")
    if args.dry_run:
        print("\n--dry-run：未执行。")
        return 0

    os.makedirs(args.runs_dir, exist_ok=True)

    rows, failures, skipped = [], [], []
    t_all = time.time()
    for (exp, width, layers), name in zip(matrix, names):
        run_dir = os.path.join(args.runs_dir, name)
        sp = os.path.join(run_dir, "summary.json")
        if args.resume and os.path.isfile(sp):
            with open(sp, encoding="utf-8") as f:
                s = json.load(f)
            print(f"\n[跳过] {name}（已有 summary.json）")
            s.setdefault("minutes", 0.0)
            s["name"] = name
            rows.append(s)
            skipped.append(name)
            continue

        ok, s, dt = run_one(exp, width, layers, name, args)
        if ok:
            s["name"] = name
            rows.append(s)
        else:
            failures.append(name)
            if not args.keep_going:
                break

    print_table(rows, failures)
    write_csv(rows, failures, os.path.join(args.runs_dir, "summary.csv"))
    print(f"\n总耗时 {(time.time()-t_all)/60:.1f} 分钟"
          + (f"，跳过 {len(skipped)} 个" if skipped else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

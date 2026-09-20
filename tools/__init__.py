# -*- coding: utf-8 -*-
"""可执行入口：训练 / 评估 / GUI / S11 / 对比实验 / 冒烟测试。

    python -m tools.train --model fno_v2 --epochs 100
    python -m tools.eval --checkpoint checkpoints/fno_v2/best.pth
    python -m tools.compare --experiments fno_v2 fno_baseline --widths 32,64
    python -m tools.smoke

全部从仓库根执行（`python -m` 会把仓库根放进 sys.path）。
"""

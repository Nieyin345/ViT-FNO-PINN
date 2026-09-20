# 电磁超表面场预测

预测 RIS 单元贴片的近场电磁场分布，并用物理传播重建全空间场。

## 工作流

```
贴片形状 (SDF + Mask)
    │  FNO（models/ + core/）
    ▼
近场截面场 (E+H, 12 通道 @ 0.1λ)
    │  PWS 物理传播（field2field）
    ▼
全空间场分布 E(x, y, z)
```

## 目录结构

```
├── configs/           # 可调参数
│   ├── base.py        #   全局默认（损失、训练超参、模型结构默认值）
│   └── experiments.py #   实验总控（切换模型与保存路径）
├── core/              # 被复用的基础设施
│   ├── base.py        #   FNOBase 统一基座 + Cond 条件载体
│   ├── config.py      #   ModelConfig / SpectralConfig 结构参数
│   ├── registry.py    #   唯一的"模型名 → 类"注册表
│   ├── encoders.py    #   频率 / 几何 / 方向编码器
│   ├── spectral.py    #   频域卷积层 SpectralConv2d
│   └── checkpoint.py  #   唯一的 save / load（strict=True）
├── models/            # 模型结构，一个模型一个文件
│   ├── fno2d.py       #   fno_attention（主模型）
│   ├── fno2d_v2.py    #   fno_v2（正弦频率输入 + 核条件化）
│   ├── fno2d_baseline.py / fno2d_film.py / fno2d_lora.py / unet2d.py
│   ├── mlp2d.py / cnn2d.py / vit2d.py / afno2d.py / deeponet2d.py  # 对比基线族
│   └── losses.py      #   损失函数（geom / DC-AC / 频域 / PINN）
├── tools/             # 可直接运行的入口
│   ├── train.py       #   主训练（argparse 驱动）
│   ├── eval.py        #   泛化评估
│   ├── ui_eval.py     #   交互式评估 UI
│   ├── calc_s11.py    #   从预测场算 S11 并与仿真对比
│   ├── compare.py     #   对比实验矩阵 + 汇总表
│   └── smoke.py       #   结构一致性冒烟测试（CPU 可跑）
├── data/              # 数据加载器 + 原始数据（不入库）
├── field2field/       # PWS 传播工具（截面 → 全空间）
│   ├── propagation.py #   角谱法传播核心
│   ├── propagate_scatter.py  # 主入口
│   └── field_io.py    #   HFSS .fld 文件读写
├── checkpoints/       # 模型权重与结果（不入库）
├── MODEL_DOC.md       # 模型架构详细文档
└── LOSS_DESIGN.md     # 自适应损失设计（计划中）
```

## 数据

- 主数据 `data/data/`：<category>/<sample>/，64×64，12 通道（E+H）
- 详细格式见 `MODEL_DOC.md`

## 快速开始

所有命令从**仓库根目录**运行。

```bash
# 安装依赖
pip install -r requirements.txt

# 冒烟测试（CPU 可跑，几十秒）
python -m tools.smoke

# 看有哪些模型 / 实验
python -m tools.train --list-models
python -m tools.train --list-experiments

# 训练（实验自带模型与超参；-e 选实验，--width/--layers 可就地覆盖）
python -m tools.train -e fno_attention

# 对比实验矩阵 + 汇总表
python -m tools.compare -e fno_v2 fno_baseline --widths 32,64 --layers 3,5 --epochs 15

# 评估
python -m tools.eval -c checkpoints/fno_attention/best.pth
python -m tools.ui_eval                                 # 交互式 UI
python -m tools.calc_s11 -c checkpoints/fno_lora/best.pth --sample rectangle_1

# 端到端：FNO 预测近场截面 → PWS 传播到全空间（模型类型由 checkpoint 自动判定）
python -m field2field.propagate_fno_prediction --sample-dir data/data/rectangle/rectangle_1 --freq-ghz 12 --ckpt checkpoints/fno_attention/best.pth --targets-mm 5,7,9,11,13,15,17,19,21,23 --out-dir out_fno_prop

# PWS 传播：从 HFSS 截面场计算全空间（必须在 field2field/ 下运行，用法见 field2field/README.md）
cd field2field && python propagate_scatter.py --source-z-mm 3 --targets-mm 5,7,9 --freq-ghz 12
```

## 加一个新模型

新建 `models/<你的模型>.py`，用 `@register("名字")` 装饰一个继承 `FNOBase`、
签名为 `__init__(self, cfg: ModelConfig)` 的类，然后在 `models/__init__.py` 加一行
import。训练 / 评估 / UI / S11 / 传播 / 对比实验六条链路会自动认得它，无需改动
任何消费方代码。详见 `CLAUDE.md`。

## 说明

- 模型学的是**散射场**（总场 − 无贴片背景场）；传播只对散射场做功，传完再加回背景场
- checkpoints 下的 `.pth`/`.json`/`.npz` 与数据文件已被 `.gitignore` 排除，不入版本库
- 输出归一化默认 `OUTPUT_NORM_MODE="log"`：`sign(x)*log1p(|x|/scale)` 压缩动态范围后逐通道 z-score，解决电场（~1e3）与磁场（~0.7~3）尺度差约 3 个数量级的问题；恢复旧行为可设 `"zscore"`。统计量与模式保存在 checkpoint 的 `normalizer` 键，训练/评估/UI 自动反归一化
- checkpoint 里同时存了 `model_name` 与 `model_config`，加载时按它重建模型并 `strict=True`。**本次结构重构后旧格式权重已作废，需重新训练**
- 详细架构、损失与配置见 `MODEL_DOC.md`，开发约定见 `CLAUDE.md`

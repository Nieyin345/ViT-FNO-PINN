# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

电磁超表面场预测，**两级级联**：

```
贴片几何 (SDF+Mask, 2×64×64) ──FNO──► 近场截面散射场 (12ch E+H) ──PWS 角谱法──► 全空间总场 E(x,y,z)
                                   models/ + core/                    field2field/
```

- `models/` + `core/`：FNO 学习"几何 → 0.1λ 处截面场"。数据驱动。
- `field2field/`：物理传播（PWS/角谱法），**无学习**，把单截面推到任意 z。

关键物理约定（改代码前必须理解）：

- 模型学的是**散射场**（总场 − 无贴片背景场），不是总场。传播只对散射场做功，传完再加回背景场，避免驻波被重复传播。见 `field2field/README.md`。
- 输出 12 通道 = `[Ex_r, Ex_i, Ey_r, Ey_i, Ez_r, Ez_i, Hx_r, Hx_i, Hy_r, Hy_i, Hz_r, Hz_i]`，实虚部成对。
- 频率 8~12 GHz 共 9 点；Floquet 端口垂直入射 + 周期边界。

## 目录结构

```
configs/   可调参数：base.py（全局默认）+ experiments.py（实验覆盖）
core/      被复用的基础设施：base(基座) config(结构参数) registry(注册表)
           encoders spectral(频域层) checkpoint(存取)
models/    模型结构本身，一个模型一个文件 + losses.py
data/      数据加载与预处理（dataset / norm_utils / fit_background / preprocess_v2）
tools/     可直接运行的入口：train eval ui_eval calc_s11 compare smoke
field2field/  PWS 物理传播
```

日常只需要碰 `models/`（加模型）和 `configs/experiments.py`（加实验）。

## 常用命令

统一从**仓库根目录**运行（`python -m tools.xxx`；`tools/` 内已自行把根目录插入 `sys.path`）。

```bash
# ---- 训练 ----
python -m tools.train --list-models                    # 看已注册模型
python -m tools.train --list-experiments               # 看可用实验
python -m tools.train -e fno_v2 --epochs 50             # 跑实验（实验自带 model/超参）
python -m tools.train -e fno_v2 --width 32 --layers 3   # 就地覆盖结构参数
python -m tools.train -m fno_baseline -e fno_baseline --tag myrun   # 模型/实验分开指定

# ---- 对比实验（矩阵 + 汇总表）----
python -m tools.compare -e fno_v2 fno_baseline --widths 32,64 --layers 3,5 --epochs 15
python -m tools.compare -e fno_v2 --widths 32 --layers 3 --epochs 2 --dry-run   # 只列组合
python -m tools.compare -e fno_v2 --epochs 50 --resume        # 已有 summary.json 的跳过
#   结果落 runs/<exp>_w<W>_l<L>/，汇总在 runs/summary.csv，按 val_geom 升序打表

# ---- 冒烟测试（CPU 可跑，不需要 GPU）----
python -m tools.smoke        # 注册表 / 前向反向 / ModelConfig 往返 / strict 往返 / 架构拒绝 / 实验可解析

# ---- 评估 ----
python -m tools.eval -c checkpoints/fno_v2/best.pth                    # 全量
python -m tools.eval -c a/best.pth b/best.pth --categories rectangle   # 多个 ckpt 横向比
python -m tools.eval -c checkpoints/fno_v2/best.pth --limit 8 --device cpu

python -m tools.ui_eval                                # tkinter 交互式 UI
python -m tools.calc_s11 -c checkpoints/fno_lora/best.pth --sample rectangle_1

# ---- 端到端：FNO 预测 → PWS 传播（模型类型由 checkpoint 自动判定）----
python -m field2field.propagate_fno_prediction \
  --sample-dir data/data/rectangle/rectangle_1 --freq-ghz 12 \
  --ckpt checkpoints/fno_v2/best.pth --targets-mm 5,7,9 --out-dir out_fno_prop

# ---- 纯 HFSS 截面传播（必须在 field2field/ 下运行：读写 e_field.fld / out_result 等相对路径）----
cd field2field && python propagate_scatter.py --source-z-mm 3 --targets-mm 5,7,9 --freq-ghz 12

# ---- 数据准备（必须在仓库根目录运行：DATA_DIR 相对解析）----
python data/preprocess_v2.py                   # 生成 field_plane_d0.10.npy 缓存
python data/fit_background.py                  # 生成 data/blank/background_params.npz
```

**本地环境**：基础 conda 环境（`D:\anaconda1`）的 torch 有 `c10_cuda.dll` 加载错误，用 `D:\anaconda1\envs\pytorch\python.exe`。训练在**服务器 GPU** 上做，本地只跑 mini 规模验证。

## 加一个新模型（重构的核心目标）

只有两步，**不需要改任何消费方**：

1. 新建 `models/<你的模型>.py`：

```python
from core.base import FNOBase, Cond
from core.registry import register
from core.config import ModelConfig

@register("fno_myidea")                       # 注册名
class FNO2dMyIdea(FNOBase):
    def __init__(self, cfg: ModelConfig):      # 唯一合法的构造签名
        super().__init__(cfg)
        self.blocks = nn.ModuleList([...])     # 必须填 self.blocks
    def build_cond(self, x, freq):             # 按需覆盖三个钩子
        return Cond(freq_emb=..., geom_emb=...)
    def output_head(self, x):
        return self.head(x)
```

2. 在 `models/__init__.py` 加一行 `from . import <你的模型模块>`。

块（block）的唯一协议是 `forward(self, x, cond: Cond | None = None)`。之后训练 / 评估 / GUI / S11 / 传播 / 对比实验六条链路全部自动认得它。

## 架构要点

### 统一基座（`core/`）

- **`core/config.py::ModelConfig`** 是模型结构参数的**唯一**来源。所有模型统一 `__init__(cfg: ModelConfig)`。`extra` 字典是扩展口：新模型有基座没有的参数塞进去，`ModelConfig` 不用改。`SpectralConfig` 是其中的频域层子配置。
- **`core/base.py::FNOBase`** 统一 lift / pad / block 级联 / crop / head，把模型差异收敛到三个钩子：`build_input`（V2 在这里拼正弦通道）、`build_cond`（产条件）、`output_head`（必填）。
- **`Cond`** 取代原来的 `freq_emb=` / `geom_emb=` / `cond=` 三套互不兼容的块签名。`fused()` 按 `[freq, pol, geom]` 拼接，供核条件化块使用。
- **`core/registry.py`** 是全仓库唯一的"模型名 → 类"映射，模型在自己文件里 `@register` 自注册。**未知名字一律抛错**并列出可选值（不再静默降级）。`build_model(name, cfg)` 是唯一构造入口。
- 三种条件化方式都能落在同一骨架上：编码器嵌入（`fno_attention`/`fno_film`/`fno_lora`）走 `build_cond`；空间通道广播（`fno_baseline`/`unet`）走 `build_input` 拼常量条件图、`build_cond` 返回 `None`；混合（`fno_v2`）两个钩子都用。
- **`freq` 的契约是 `(B, 2) = [归一化频率, 极化]`**，由 `data/dataset.py` 定义（每条样本给 `(2,)`，DataLoader 批成 `(B,2)`）。`normalize_freq()` 在 `FNOBase.forward` 入口和 `UNet2d.forward` 里统一补齐：1D 视为单样本、缺极化列补 0，batch 维不匹配则抛 `ValueError`。**新写调用点时按 `(B,2)` 传**，别依赖 1D 兜底；上一版就是因为各模型对 `freq` 的切片宽度不同（宽的靠广播侥幸跑通、窄的如 `fno_v2` 的 `freq[:, 1:2]` 直接 IndexError），同一个坏输入时崩时不崩。

### 配置的生效顺序

`configs/base.py` 是唯一配置源，模块级全局值：

1. 文件内的字面量
2. `configs/experiments.py::EXPERIMENTS[ACTIVE_EXPERIMENT]` 的覆盖（通过 `apply_experiment()`）
3. `tools/train.py` 的命令行参数（`apply_cli_overrides()` 直接写回 `configs.base` 的全局）
4. 模型结构参数经 `C.model_config_from_globals()` 汇总成 `ModelConfig`

`ACTIVE_EXPERIMENT` 默认 `"fno_attention"`，可用 `TRANSFORMER_EXPERIMENT` 环境变量或 `-e` 指定。**新增实验请在 `configs/experiments.py` 里加条目**（含 `"model"` 键指定用哪个模型），不要改文件顶部的字面量。

`apply_experiment()` 会先把所有大写配置键**重置回默认值**再套用覆盖——否则上一个实验设的 `WIDTH`/`HEAD_MODE` 会泄漏到下一个（这是重构中实测到并修掉的 bug）。因此同进程内反复切实验是安全的；但 `tools/compare.py` 仍然**每个组合起一个子进程**，因为内存/显存/随机状态也需要隔离。

### checkpoint 加载（`core/checkpoint.py`）

checkpoint 里**存构建配置**，加载时不再靠猜：

```python
{"format_version": 2, "model_name": "fno_v2", "model_config": {...},
 "model_state_dict": ..., "normalizer": {...}, "epoch", "val_loss", "val_rel", "val_geom"}
```

`core.checkpoint.load(path)` 按 `model_config` 重建模型并 **`strict=True`** 载入，返回 `LoadedCheckpoint(model, cfg, normalizer, meta)`。架构不符**直接抛错**。

这是本轮重构修掉的核心症状：旧 `eval_generalization._infer_model_kwargs()` 从 state dict 反推结构，对 `FNO2dV2` 会推错（它从 `lift.0.weight` 读出 `in_channels=6`，而 `FNO2dV2.__init__` 的语义是 2，正弦通道内部另加），再配合 `load_state_dict(strict=False)`，于是错配层被**静默随机初始化**，评估结果失真而界面无异常。**任何地方都不要再引入 `strict=False` + 形状过滤的加载路径。**

旧格式（v1）checkpoint 无 `model_name`/`model_config`，`load()` 会明确报错要求重训。本地 pytorch env 是 numpy 1.24，无法反序列化在 numpy 2.x 下存出的 ckpt（pickle 里记的是 `numpy._core`）——遇到 `ModuleNotFoundError: numpy._core` 说明是环境不匹配，不是文件损坏。

### 指标口径有两套（跨文件比较时认准）

| 出处 | Geom | RelL2 |
|---|---|---|
| `tools/train.py`（写入 `summary.json`，也是 `best.pth` 的判据） | E(前6)/H(后6) 分组平均 | `channel_group_balanced_l2`（分组归一后平均） |
| `tools/eval.py` | 12 通道整体 | 12 通道整体展平的相对 L2 |

两者不可直接比大小。`val_geom` 是早停与保存最优模型的判据，不是 `val_rel`。

### 输出归一化（E/H 尺度差 3 个量级）

电场 std ~1e2~1e3，磁场 ~1。默认 `OUTPUT_NORM_MODE="log"`：先 `sign(x)*log1p(|x|/scale)` 压动态范围，再逐通道 z-score；反变换 `sign(y)*expm1(|y|)*scale`。统计量存在 checkpoint 的 `normalizer` 键和 `output_norm_stats.npz`，训练/评估/UI 自动反归一化。

- **训练损失**在归一化空间计算；**保存/早停指标**（`val_rel` / `val_geom`）在**物理空间**计算（先反归一化），避免 log 空间失真方向角/相位。
- 改归一化方式会让旧 checkpoint 失效；`OutputNormalizer.from_checkpoint()` 兼容旧格式（无 `output_norm_mode` 时按 `zscore` 解释）。

### 损失项是位置下标驱动的

`LOSS_ENABLE_VECTOR` / `LOSS_WEIGHT_VECTOR` 是**长度 9 的列表，含义只存在于 `configs/base.py` 的注释里**，且 `models/losses.py::loss_fn` 内部用两套长度守卫（`len(w) > 7` / `len(w) > 8`）。改列表长度或顺序会影响所有实验。`L9`(PINN 麦克斯韦残差) 需要相邻 z 平面，只有 `v2` 体数据能提供；`v2_cache` 下开启会打警告且空转。

### 数据格式

```
data/data/<category>/<sample>/
    patch_plane.npy                 (64, 64, 2)         ch0=SDF, ch1=Mask
    field_plane_d0.10.npy           (P, 9, 64, 64, 12)  v2_cache（训练默认走这条）
    field_volume_merged.npy         (P, 9, N_Z, 64, 64, 12)  v2 体数据
```

`P`=极化数（单极化文件为 1，双极化展开成 2 条记录）。`data/dataset.py` 按缓存文件是否存在自动选 `v2_cache` / `v2` / `v1` 分支。

- **`data/data/`、`data/blank/`、`*.npy`、`*.fld` 全部被 gitignore**，不入库。新环境需自行准备数据。
- 数据划分 `build_split()` 按 `(category, shape)` 分层，同一形状的所有频点/极化进同一集合（防泄漏）。
- 极化 1 的背景场由极化 0 旋转得到（Ex↔Ey, Hx↔Hy），见 `data/fit_background.py`。
- 评估/传播构造数据集时**必须与训练逐项对齐**（`use_scatter` / `polarizations` / `target_d_over_lambda`），否则评的是分布不同的数据。

### 已知的不一致（改相关代码时留意）

- **空间步长有两套值**：`propagate_fno_prediction.py` 用 `cell_mm/(W-1) = 15/63 ≈ 0.2381 mm`，`models/losses.py` 的 PINN 用 `30/64 = 0.4688 mm`。而 HFSS 原始导出（`field2field/e_field.fld` 表头）是 `0.15 mm`、101×101 点。仓库里**没有**从 101 点降到 64×64 的代码，该转换在何处完成未记录。涉及 kz / 物理约束时先确认网格口径。
- **背景场训练/推理口径不同**：训练路径 `fit_background.get_background_field` 每通道返回**标量**（仅在网格中心拟合）；推理路径 `propagate_fno_prediction._get_bg_plane` 返回从 blank 体数据插值的**完整平面**。blank 场并非空间均匀，实测对训练目标的相对影响约 0.4%。

## 代码风格

- 注释与 docstring 用中文；变量/函数名英文。改动时保持这一风格。
- 路径一律用 `_ROOT` / `_HERE` 从 `os.path.dirname(os.path.abspath(__file__))` 派生，不要新增裸相对路径。`configs/base.py` 的 `DATA_DIR`/`SAVE_DIR` 都从自身位置解析，移动文件时注意。
- 提交信息用中文 `type: 描述`（`feat:` / `fix:` / `chore:` / `perf:`），与 `git log` 一致。
- 临时脚本放 `.tmp/`（已 gitignore），用完删除；命名 `fix_xxx.py` / `verify_xxx.py` / `extract_xxx.py`。
- `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` 是必要的（多份 OpenMP 运行时共存），各入口脚本已设置。

## 参考文档

- `MODEL_DOC.md` — 模型架构、损失编号（L1~L9）、checkpoint 格式的权威说明
- `LOSS_DESIGN.md` — 自适应损失设计（**计划中，未实现**）
- `README.md` / `field2field/README.md` — PWS 传播原理与 `.fld` 输入约定

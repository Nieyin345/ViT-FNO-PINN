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
- **极化 2 的背景场 = 极化 1 背景场的旋转变换，且 E 组要取负号、H 组不取**：

  ```
  pol2.Ex = -pol1.Ey     pol2.Hx = +pol1.Hy
  pol2.Ey = -pol1.Ex     pol2.Hy = +pol1.Hx
  ```

  **两处实现必须同步**：`data/fit_background.py::_orthogonal_polarization_field`（训练路径）
  与 `field2field/propagate_fno_prediction.py::_rotate_plane`（推理/传播路径）。
  原先两处都是**纯交换、E 少了负号**，后果是极化 2 的散射场里混进约两倍幅值的入射波
  （实测 pol2/pol1 散射场能量比中位 1.477、均值 22.7、最大 813.9，而正确应为 1.000），
  影响 `DATASET_POLARIZATIONS=[0,1]` 下约 1/3 的训练记录，且终端无任何异常提示。
  符号来自复最小二乘实测（115 个双极化样本 × 3 频点）：

  | 关系 | 系数 c | 相位 |
  |---|---|---|
  | `pol2.Ex = c·pol1.Ey` | −0.9567 | 179.9° |
  | `pol2.Ey = c·pol1.Ex` | −0.8921 | 179.8° |
  | `pol2.Hx = c·pol1.Hy` | +0.9473 | −0.0° |
  | `pol2.Hy = c·pol1.Hx` | +0.9821 | −0.1° |

  **别凭直觉推翻它**：E 与 H 的符号相反（一个取负、一个不取），且 E 的两个通道都取负。
  曾据"合并通道的相关"误判为"整体取负、H 也要负"，直到逐通道复最小二乘才定下来——
  合并相关由幅值占绝对优势的 E（`Ey`~2900 vs `Hx`~6）主导，分辨不出 H。

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
python -m tools.smoke        # 注册表 / 前向反向 / 往返 / 架构拒绝 / 实验 / 指标解耦 / 场景划分 / 架构指纹
# ---- 结果档案（按场景代次汇总，跨代不混表）----
python -m tools.report                          # 扫 runs/ + checkpoints/
python -m tools.report runs/gen1 runs/gen2 --csv out.csv
#   同场景的 run 打一张表（可直接比）；跨代的表分开打，并列出场景差异
#   额外标出：STALE(模型类变了，旧结果作废) / GONE(模型已被删) / ?(无指纹的老 run)

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

块（block）的唯一协议是 `forward(self, x, cond=None)`（**注解写 `Optional[Cond]`，不要写 `Cond | None`——本地环境是 Python 3.8，`X | None` 在运行时求值会 `TypeError`**）。之后训练 / 评估 / GUI / S11 / 传播 / 对比实验六条链路全部自动认得它。

模型专属的结构参数走 `cfg.extra` 字典，**不改 `core/config.py`**：在 `configs/base.py` 的 `MODEL_EXTRA`（或实验条目里的 `"MODEL_EXTRA": {...}`）写键值，模型里用 `cfg.extra.get("你的键", 默认值)` 读。实验条目里也能直接覆盖 `PADDING` 等基座参数（例如 `mlp` 用 1×1 卷积，设 `"PADDING": 0` 省算力）。

已有 11 个模型可作模板：`mlp` / `cnn` 走"空间通道广播 + 基座骨架"；`afno` 走"编码器嵌入 + 基座骨架"；`vit` / `deeponet` 因为 token 化/乘积求和与基座骨架不兼容，**整体覆盖 `forward`**（但仍继承 `FNOBase` 以复用 `self.cfg`、`normalize_freq` 契约与注册机制）。

## 架构要点

### 统一基座（`core/`）

- **`core/config.py::ModelConfig`** 是模型结构参数的**唯一**来源。所有模型统一 `__init__(cfg: ModelConfig)`。`extra` 字典是扩展口：新模型有基座没有的参数塞进去，`ModelConfig` 不用改。`SpectralConfig` 是其中的频域层子配置。
- **`core/base.py::FNOBase`** 统一 lift / pad / block 级联 / crop / head，把模型差异收敛到三个钩子：`build_input`（V2 在这里拼正弦通道）、`build_cond`（产条件）、`output_head`（必填）。
- **`Cond`** 取代原来的 `freq_emb=` / `geom_emb=` / `cond=` 三套互不兼容的块签名。`fused()` 按 `[freq, pol, geom]` 拼接，供核条件化块使用。
- **`core/registry.py`** 是全仓库唯一的"模型名 → 类"映射，模型在自己文件里 `@register` 自注册。**未知名字一律抛错**并列出可选值（不再静默降级）。`build_model(name, cfg)` 是唯一构造入口。
- 三种条件化方式都能落在同一骨架上：编码器嵌入（`fno_attention`/`fno_film`/`fno_lora`/`afno`）走 `build_cond`；空间通道广播（`fno_baseline`/`unet`/`mlp`/`cnn`）走 `build_input` 拼常量条件图、`build_cond` 返回 `None`；混合（`fno_v2`）两个钩子都用；`vit`/`deeponet` 因输出映射方式不同，整体覆盖 `forward`。
- **`ModelConfig.extra` 是模型专属参数的唯一通道**，由 `configs/base.py::MODEL_EXTRA` 传入（见 `model_config_from_globals()`）。它参与 `to_dict`/`from_dict` 与 `__post_init__` 之前的构造，也随 checkpoint 一起存取——所以新增模型参数不需要动 `core/config.py`，但也要注意 **`extra` 里改了默认值会让旧 checkpoint 重建出不同结构**（`strict=True` 会报错，这是期望行为）。
- **`freq` 的契约是 `(B, 2) = [归一化频率, 极化]`**，由 `data/dataset.py` 定义（每条样本给 `(2,)`，DataLoader 批成 `(B,2)`）。`normalize_freq()` 在 `FNOBase.forward` 入口和 `UNet2d.forward` 里统一补齐：1D 视为单样本、缺极化列补 0，batch 维不匹配则抛 `ValueError`。**新写调用点时按 `(B,2)` 传**，别依赖 1D 兜底；上一版就是因为各模型对 `freq` 的切片宽度不同（宽的靠广播侥幸跑通、窄的如 `fno_v2` 的 `freq[:, 1:2]` 直接 IndexError），同一个坏输入时崩时不崩。

### 结果档案：为什么换损失不必重跑基线（`configs/scenario.py`）

主模型的损失函数是**要反复改的变量**，所以配置不能整体冻结。但"改了什么"必须能被机器认出来。
`configs/scenario.py` 把 `configs/base.py` 的大写配置项分成两桶：

| 桶 | 内容 | 改了怎么办 |
|---|---|---|
| `SCENARIO_KEYS` | 数据/划分/轮数/batch/lr/早停/输出归一化 | **换了一代**，旧 run 必须重跑；跨代数字不可比 |
| `LOSS_KEYS` | `LOSS_ENABLE_VECTOR`/`LOSS_WEIGHT_VECTOR`/全部 `FIELD_*` 子权重 | **允许随时改**，改完仍与旧 run 可比；只记录下来解释提升来自哪版 |

两桶之外是模型结构（`WIDTH`/`MODES1`/`HEAD_MODE`/`MODEL_EXTRA`），它由 `ModelConfig.to_dict()` 随 checkpoint 记录。

**为什么损失可以随便改而结果仍可比**：保存/早停指标 `val_geom`/`val_rel` **不读任何损失配置**。
`Trainer.geom_metric` 调 `field_geom_loss_eh(pred, true)` 用子权重默认值 1/1/1（不是训练侧的
`FIELD_DIR_ANGLE_WEIGHT` 等），`rel_l2_error` 调 `channel_group_balanced_l2(pred, true)` 用
`h_boost` 默认 1.0（不是 `FIELD_H_GROUP_WEIGHT`）。所以尺子不随损失漂移，基线的 `best.pth`
是用同一把尺子选出来的，在新损失下**仍然是那个基线**——不必重跑。
`tools/smoke.py::check_metric_independence` 把这条不变式钉死：把损失旋钮全部打到极端值，
指标必须逐位不变；并静态检查 `geom_metric`/`rel_l2_error` 没把 config 值透传进去。

**"要不要重跑"的机械判据是架构指纹**（`SC.arch_fingerprint`）：当前代码重建出的模型，
其 `state_dict` 的 (键, 形状) 签名哈希。改了模型类（增删层/改通道）→ 变 → 旧结果作废；
只改损失/学习率 → 不变 → 旧结果仍有效。这等价于"旧 checkpoint 还能否 `strict=True` 加载"，
但不需要持有 `.pth` 就能算——`tools/report.py` 靠它给历史 run 打 `STALE`/`GONE` 标记。

三个指纹都随 run 落盘（`summary.json` 与 checkpoint 的 `extra` 里）：
`scenario.fingerprint`（含 `split` 划分指纹）、`loss_recipe.fingerprint`、`arch_fingerprint`。
`split_fingerprint` 是"同样训练集、同样验证"最硬的证据——比哈希配置键更硬，因为数据文件
增删改会让划分变化而配置哈希看不出来。

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
| `tools/eval.py` | E(前6)/H(后6) 分组平均（与 Trainer 同，走同一个 `field_geom_loss_eh`） | 12 通道整体展平的相对 L2 |

两者不可直接比大小。`val_geom` 是早停与保存最优模型的判据，不是 `val_rel`。

`geom_metric` 故意用 `field_geom_loss` 的默认等权子权重（1/1/1），而训练项 L3 用
`FIELD_DIR_ANGLE_WEIGHT` / `FIELD_COMPLEX_PHASE_WEIGHT` / `FIELD_TOTAL_MAG_WEIGHT`
（默认 2/2/4）——指标要的是可跨实验横向比较的等权几何误差，不随训练权重漂移。
**改 L3 子权重不会改变 best.pth 的判据。**

`summary.json` 里 `val_geom` / `val_rel` / `best_val_loss` 全部取自**同一个最佳轮**
（`Trainer.best_metrics`）。早期版本 `val_rel` 取的是最后一轮，早停后与 `val_geom`
可能相差几十个 epoch，按它横向比模型等于在比不同 epoch。

`field_geom_loss` 只读传入张量的**前 6 个通道**、当作 3 个复分量 `(0,1),(2,3),(4,5)`。
12 通道张量直接喂进去只会算到电场。需要 E/H 分组时一律用 **`field_geom_loss_eh`**，
不要再在调用点手写切片。

### 输出归一化（E/H 尺度差 3 个量级）

电场 std ~1e2~1e3，磁场 ~1。默认 `OUTPUT_NORM_MODE="log"`：先 `sign(x)*log1p(|x|/scale)` 压动态范围，再逐通道 z-score；反变换 `sign(y)*expm1(|y|)*scale`。统计量存在 checkpoint 的 `normalizer` 键和 `output_norm_stats.npz`，训练/评估/UI 自动反归一化。

- **训练损失**在归一化空间计算；**保存/早停指标**（`val_rel` / `val_geom`）在**物理空间**计算（先反归一化），避免 log 空间失真方向角/相位。
- 改归一化方式会让旧 checkpoint 失效；`OutputNormalizer.from_checkpoint()` 兼容旧格式（无 `output_norm_mode` 时按 `zscore` 解释）。

### 损失项是位置下标驱动的

`LOSS_ENABLE_VECTOR` / `LOSS_WEIGHT_VECTOR` 是**长度 9 的列表，含义只存在于 `configs/base.py` 的注释里**。`loss_fn` 不是 `models/losses.py` 里的函数，而是 **`tools/train.py::Trainer.loss_fn`**；其中 `len(w) > 7` / `len(w) > 8` 两处守卫是**不可达的死代码**（`_get_loss_weight_vector` 保证长度 ≥ 9）。索引本身没有运行期校验，**在列表中间插入一项会让其后所有损失项静默错位**。改列表长度或顺序会影响所有实验。`L9`(PINN 麦克斯韦残差) 需要相邻 z 平面，只有 `v2` 体数据能提供；`v2_cache` 下开启会打警告且空转。另外 `maxwell_pinn_loss` 的 `dx_mm = dy_mm = 30/64 = 0.46875` 与本仓库其余各处的 `15/63 = 0.2381` 不一致（差 1.97 倍），启用 L9 前须先核对网格口径。

### 名义权重 ≠ 实际影响：L8 的通道权重（`LOSS_CH_WEIGHT_POWER`）

**看损失别只看 `LOSS_WEIGHT_VECTOR`，要看梯度贡献。** 在散射场真实数据上实测，
`[1.0, 0, 0.5, 0…, 1.0, 0]` 的三项**梯度贡献是 0.8% / 1.3% / 97.8%** —— 名义上
1.0/0.5/1.0 的三项，实际是 L8 一家独大，L1 和 L3 几乎被关掉。

根因不在外层权重，在 L8 内部：`channel_group_balanced_l2` 对 12 个通道**等权平均**，
但 `rel_c = ||Δc|| / ||t_c||` 使梯度 ∝ `1/||t_c||`，**通道越弱梯度越大**。12 通道的
`||t_c||` 跨 4 个数量级（Ex 109062 → Hz 12.2），于是：最强的 `Ex_r` 拿 **0.00%**，
最弱的 `Hz_i` 拿 **36~45%**，H 组的 Hy/Hz 四个通道合计吃掉约 3/4 的梯度预算。
（这些通道本身是**真物理**，不是噪声：相邻像素相关 0.99+、低频能量占比 0.30~0.99，
对比白噪声的 0.0018 / 0.0085。问题是**权重配比**，不是数据质量。）

修法是 `LOSS_CH_WEIGHT_POWER`：把组内平均从等权换成按 `||t_c||^p` 加权。

| 值 | 含义 |
|---|---|
| `0.0` | 旧行为，**逐位相同**；指标 `val_rel` 固定用这个值 |
| `0.5` | 训练默认；三项梯度贡献变成 ~10% / ~15% / ~75% |
| `1.0` | 绝对误差等权，反过来让 Ey 碾压 Hz |

**它只影响训练。** `Trainer.rel_l2_error` 不传该参数（靠默认值 0.0），所以 `val_rel`
仍是旧口径、可跨代比较；`tools/smoke.py::check_metric_independence` 守着这条——
它额外断言该参数的**默认值必须是 0.0**，因为指标正是靠这个默认值不漂移的。

**仍存在的错配（未解决）**：`best.pth` 由 `val_geom` 选出，而 `val_geom` 对应的训练项
L3 只拿 ~15% 梯度。要真正对齐得**同时**动外层权重和 L3 的子权重——实测 L3 内部
`mag` 占 82~96%、`phase` 只占 1~6%（`1-cos` 在小角度是二次的，天然偏小）。单独把
L3 的外层权重抬上去只会放大这个失衡，不解决问题。

### 基线代次配方：`configs/experiments.py::GEN1` + `check_recipe()`

上节解决的是"**换损失要不要重跑基线**"（答案：不必，因为尺子不随损失漂移）。
这一节解决的是另一半问题：**说好全模型一起重跑，实际只跑了一部分**。

用户的研究循环是「固定一版损失 → 全部模型集体跑一次得到基线 → 只迭代主模型 →
主模型提升后把新损失同步给全部模型再跑一次」。这条循环最怕**单侧漂移**：改了
`LOSS_WEIGHT_VECTOR`（或 `LOSS_CH_WEIGHT_POWER`）却只重跑了几个模型，于是"基线
的分数"和"主模型的分数"不是同一套损失算出来的，而两边在终端上都只是"跑完了"。

`GEN1` 把当前这一代基线的**全部 20 个 `LOSS_KEYS` 逐字写出来**——显式写而不是
"碰巧等于默认值"，是为了让它成为一份可校验的契约。`tools/train.py::main` 在应用完
实验覆盖与 CLI 覆盖之后调 `check_recipe(C)`：**只要有一处对不上就直接报错**，
除非显式传 `--allow-recipe-drift`（此时打警告并声明本次结果不可与基线同表比较）。

**升代次就三步**：改 `GEN1` 的值 → `GEN1_NAME` 同步改（`"gen2"`）→ 全部模型重跑。
`tools/report.py` 按场景/损失指纹分表，跨代不会混在一起。

`tools/smoke.py::check_recipe_consistency`（第 10 项）守着三件事：配方**覆盖全部**
`LOSS_KEYS`（漏一个 = 改它无人监管）、当前配置与该代次**逐键一致**、以及**负向测试**
——故意改一个键必须能被检出，否则守卫等于没写。写这个检查时它当场抓到一个真实缺漏：
`LOWFREQ_PATCH` / `HIGHFREQ_CUTOFF` 在 `LOSS_KEYS` 里但没进 `GEN1`（本代 L5/L6 未启用，
值是惰性的，但"惰性"不等于"可以不受监管"）。

**`GEN1` 不等于 `configs/base.py` 的默认值可以随便动**：`base.py` 的默认值是"新开
一个实验时从哪出发"，`GEN1` 是"这一代基线用的那套"。两者当前一致，但**协议是
改基线损失请改 `GEN1` 并升代次**，而不是改 `base.py` 的默认值——改默认值会让
`check_recipe` 报错，那正是它该做的事。

### 模型结构上的修正（会改变数值行为）

下面第 4 条是**新增**的（逐模型审查那一轮），其余三条是更早一轮修的。

1. **`SDFGeomEncoder` 的尺度**（`core/encoders.py`）：原来 `rfft2(field)` 用默认
   `norm="backward"`，且 SDF 未标准化，导致 `DC = sum(SDF) = mean×H×W = 8841`，
   `geom_emb` 的量级是 `freq_emb` 的 **3190 倍** —— 而它们拼进**同一个** `Linear`。
   后果是几何把频率与极化淹没（"只改极化"的位移 1.28 vs "换几何"的 2607，差 2036 倍），
   而**双极化是本项目的标准配置**。改为逐样本标准化 + `norm="ortho"` 后降到 13.5 倍 /
   35.7 倍。影响 **6 个模型**：`fno_v2` / `fno_attention` / `fno_film` / `fno_lora` /
   `afno` / `vit`。
2. **`fno_v2` 的正弦嵌入相位**（`models/fno2d_v2.py::_sine_phase`）：原来写死
   `(π/10)·f_GHz`，把 15mm 元胞尺寸与光速隐式烘焙进常数（换 10mm 元胞时它会**静默**
   保持原值）。改为由 `_C_LIGHT` 与 `cfg.extra["CELL_MM"]`（默认 15.0）推导。数值上
   与旧行为等价（最大相对差 6.9e-4，即 `π/10·f` 本就是 `k0·L(15mm)` 的近似）。
3. **FiLM 的 `shift` 挪到空域相加**（`models/fno2d_v2.py::SpectralConv2dV2.forward`
   与 `models/fno2d_film.py::SpectralConv2dFiLM.forward`）：原来对整个 `out_ft` 做
   `out_ft*(1+scale)+shift`，**这个 `+shift` 在频域加常数，反变换回空域是一个
   delta 冲激**——实测原点能量占比 1.0000（`std` 5.5e-2，`max` 恰为 3.5），即每层
   都往特征图原点打一个尖峰，而不是 FiLM 语义里的"逐通道平移特征图"。现在
   `scale` 留在频域乘（频域缩放 == 空域缩放），`shift` 挪到 `irfft2` 之后空域相加
   （实测 `std` 0.0、各通道均值恒为 3.5）。零初始化时仍**恒等**（起点行为不变）。
   **该修正不改变任何 key/shape**，所以架构指纹看不见它，见下一节。

   > 这里记下两版**错误**注释，免得再踩：第一版写"给 DC 加常数 = 原点冲激"——
   > 反了，只改 DC 得到的是**均匀常数**（`std` 1.8e-7）；得到冲激的是"给**全部**
   > 频点加常数"。第二版改成"只排除 DC 那一个点"——依然错，原点占比仍是 0.9998，
   > 冲激还在，只是被减掉一个均匀量；排除整个 `kx=0` 行同理（占比 0.9844，把冲激
   > 沿 y 展平）。**三种写法都消不掉冲激**，因为问题不在"改哪些频点"，而在
   > **加法本身就不该发生在频域**。

4. **主模型注意力的 softmax 饱和**（`core/spectral.py::SpectralConv2d`，新增
   `SpectralConfig.sig_norm` / `score_scale`，默认 `True` / `1.0`）：
   打分是 `Q·K^H`，形状 `(B, C_in, C_in)`，**softmax 归一化的是通道维**（`C_in`）。
   `X_sig` 是未归一化的频域幅值，打分不归一化时实测（`width=32`）`|logit|` 最大
   **2.9e4** → 熵 **0.0000**（均匀分布的 0%）、最大权重 **1.0000**、有效位置数 **1.00**
   ——注意力退化成硬 argmax，1.07M 参数的模块对输出的贡献趋近于一个固定的通道置换。
   注意 `scores_complex` 那一行**本来就有** `/sqrt(d_k)`，根因是 `|X_sig|` 的量级
   而非 `d_k`。修法是逐样本 L2 归一化 `X_sig`，让打分看**方向**而非幅值：归一化后
   `|logit|` 最大 9.3e-4 → 熵 **3.4657 = log(32)**、最大权重 **0.0313 = 1/32**、
   有效位置数 **32.00**（均匀）。零初始化下"均匀"正是应有的起点。

   > 同一处曾经写过一句"把超网络抬离零的梯度，正是被饱和的 softmax 掐掉的那一份"。
   > **实测不支持**：开关前后 `∂L/∂W_last` = 1.1212e-04 vs 1.1125e-04（0.992×），
   > 20 步后 `|W_last|` 2.24e-06 vs 2.23e-06。`W_last` 的梯度里 `γ·attn_out`
   > 那一项占主导，不经过 softmax 的稀疏性。**两件事各自独立**，别把修正的价值
   > 挂在"解锁条件化模块"上——它本身就值。

### 逐模型审查修掉的一类通病：**建了却不调用**的模块

`state_dict` 里有一份模块、`model.parameters()` 也把它交给优化器，但它**没有调用点**
（或者被同名的重新赋值盖掉），于是永远 `grad is None`。这类参数不报错、不掉精度，
只是白占显存、白进优化器，并污染任何"按参数做统计/正则"的逻辑。

根因是 `FNOBase.__init__` 无条件建 `self.lift`，而子类有两类绕开它的方式：

| 症状 | 原写法 | 现在 |
|---|---|---|
| 子类 `self.lift = ...` 重建 | 基座那个模块留在 `_modules` 里成为死参数 | 覆盖 `lift_in_channels()` / `build_lift()` 钩子 |
| 子类整体覆盖 `forward`，不走骨架 | `lift` 建了但没人调用 | `build_lift()` 返回 `nn.Identity()` |

已修：`fno_v2`（拼正弦通道）、`fno_baseline` / `mlp`（拼条件图）、`cnn`（拼条件图 +
3×3 核）、`vit` / `deeponet`（整体覆盖 `forward` → `nn.Identity()`）、
`vit` 的 `cls_token`（只在 `freq is None` 兜底分支里用到，训练/评估/传播全链路都带
`freq`，永远拿不到梯度；`freq=None` 本就该是零条件，直接用零张量）。

**新写模型时**：需要额外输入通道就覆盖 `lift_in_channels()`，需要别的卷积核就覆盖
`build_lift()`，**永远不要重新赋值 `self.lift`**。两个钩子都**只读 `self.cfg`**——
它们是在 `super().__init__(cfg)` **内部**被调用的，那时子类自己 `__init__` 里赋的
属性（`self.n_sine` / `self.in_channels` 等）**还不存在**，读了会 `AttributeError`。

### 逐模型审查修掉的另一类：消融对照里混进了额外变量

`afno` 原先只取 `x_ft[:, :, :m1, :m2]` 一块低频，注释写着"与 FNO 的 modes1/modes2
取法一致"——**实际只覆盖了一半**。FNO 的 `w1` 取 `kx∈{0..m1-1}`、`w2` 取
`kx∈{-m1..-1}`，两者合起来才是中心对称的低频窗。AFNO 存在的唯一意义是分离
"共享权重 vs 每频率独立调权"，频域覆盖不对齐就无法归因。现在两块都取（拼起来做
一次变换，因为两块共享同一组 `w1/w2`，与分别做两次数学等价）。
`AFNOBlock._to_groups` 的 `blocks>0` 分块模式会沿这条拼接缝切块，缝两侧的频率对被
归进同一组——该模式本就是插值用的实验旋钮，不做特殊处理。

同类：`fno_baseline` / `fno_film` 的 block 漏了 FNO 原论文的 `+ x` 残差
（别的 block 是 `norm(f(x) + x)`，它们是 `norm(f(x))`）。残差不是"能不能训"的必要
条件，但它让消融的差异里混进了"少一个残差"这个变量。两个都补齐了。

### 一个诚实的局限：架构指纹看不见模型**代码**变了

`SC.arch_fingerprint` 用 `state_dict` 的 (键, 形状) 签名做哈希 —— 上面第 3 条修正
（FiLM 的 `shift` 换域）**不增加也不删除任何 key/shape**，所以指纹**不变**，
`tools/report.py` 会把这些 run 判成 `ok` 而不是 `STALE`。形状相同 ≠ 数值行为相同。
真要捕捉这类改动，得把模型源码哈希并进指纹；当前没有做，**改模型内部计算时请自行
判断是否需要重跑**，不要只信指纹。

（上面那批"建了却不调用"的修正**会**改变 key/shape，所以指纹能看见——`vit` /
`deeponet` 少了 `lift.*`、`vit` 少了 `cls_token`。这些模型此前的 checkpoint 一律
`STALE`。）

### 输出头 dropout 必须在所有模型上一致（`PROJ_DROPOUT`）

`configs/base.py` 的 `PROJ_DROPOUT`（默认 0.05）**只对"输出头"生效**，且历史上
只有 5 个模型真正读了 `cfg.dropout`（`fno_v2`/`afno`/`vit`/`cnn`/`mlp`），另 6 个
建了头却从不使用 → 主模型 `fno_attention` **完全没有正则**，而它的对比对象有。
后果是消融表里混进了"有无 dropout"这个额外变量（实测 train 模式两次前向：主模型
差 0.0000、`fno_v2` 差 1.19）。现在 11 个模型全部接入：在**输出头之前**加
`Dropout2d(cfg.dropout)`（`deeponet` 无独立输出头，dropout 作用在 branch 系数上）。
`dropout=0` 时退化为 `nn.Identity()`，不改变任何 key/shape。

**新模型必须接上**：不接就是"消融对照里混进额外变量"那一类（见上一节）。核对方法
是 `m.train()` 下连续两次 `forward` 的最大差应显著非零，`m.eval()` 下必须**恰好为 0**。

**注意这是架构指纹的盲区**：`Dropout` 无参数，加/去它**不改变 `state_dict` 的
(key, shape)**，所以 `arch_fingerprint` 和 `tools/report.py` 都看不见——实测
`dropout=0` 与 `0.05` 的指纹完全相同。与上面 FiLM 换域是同一类问题，改它请自行判断
是否需要重跑。


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

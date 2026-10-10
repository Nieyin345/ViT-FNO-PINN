# -*- coding: utf-8 -*-
"""全局默认配置（数据、训练、损失）。

模型**结构**参数的唯一来源是 `core.config.ModelConfig`（由本文件的
MODES1/MODES2/WIDTH/N_LAYERS/... 构造，见 `configs/model_config_from_globals()`）。
实验级覆盖见 `configs/experiments.py`。
"""

import os

# =============================================================================
# 1) 数据与划分
# =============================================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE_DIR)

DATA_DIR = os.path.join(_ROOT, "data", "data")
SAVE_DIR = os.path.join(_ROOT, "checkpoints")

# 当前数据目录只有 rectangle 一类（其中部分样本已更新为双极化体数据）。
TRAIN_CATEGORIES = ["rectangle"]
TRAIN_FREQS = None  # 全部 9 个频点

# 数据平滑（保留旧参数兼容）
SMOOTH_SIGMA = 0.0

# 插值目标归一化距离（v2 数据）
TARGET_D_OVER_LAMBDA = 0.1

# 双极化配置：
# - DATASET_POLARIZATIONS=[0,1] 表示对每个双极化文件展开成 2 条样本；
# - 单极化文件只会贡献极化 0，避免因为数据生成不完整而直接报错。
DATASET_POLARIZATIONS = [0, 1]

# ========== FNO2dV2 新主模型（强化 FNO + 正弦频率输入 + 核条件化）==========
# 核条件化方式：film(频域特征图 scale+shift) | lora(低秩扰动 w1/w2) | none(纯 FNO)
KERNEL_COND_MODE = "film"
# 频率以空间正弦嵌入通道注入输入（不用超参数调频）
USE_SINE_INPUT = True
# 极化走独立条件编码分支（不再与频率平均混叠）
USE_POL_COND = True
LORA_RANK = 4

# 散射场：用拟合的理想背景场相减，去除入射波主导成分
USE_SCATTER_FIELD = True
BACKGROUND_PARAMS_PATH = os.path.join(_ROOT, "data", "blank", "background_params.npz")

VAL_RATIO = 0.2
SHUFFLE_SEED = 42
# 训练随机种子（初始化/dropout/增强）。None = 沿用 SHUFFLE_SEED（旧行为）。
# 需要「固定划分、重复多个随机种子」时设它（命令行 --train-seed），划分仍由 SHUFFLE_SEED 定。
TRAIN_SEED = None


# =============================================================================
# 2) 模型结构默认值（喂给 core.config.ModelConfig）
# =============================================================================
# 主模型：fno_attention。其余实验在 configs/experiments.py 里覆盖 MODEL_NAME。
MODEL_NAME = "fno_attention"

MODES1 = 8
MODES2 = 8
WIDTH = 64
N_LAYERS = 5
FREQ_DIM = 32
# 边界 padding（右/下各补 PADDING 圈 0，过完所有 FNO 块后再裁掉）。
# 注意：HFSS 元胞是 Lattice-Pair 周期边界，场本身严格周期，FFT 卷积天然满足；
# 补 0 反而在 [W, W+PADDING) 造出一条非物理的"零值边界"，破坏周期性、
# 且只补右/下不对称。历史实验用 7；新实验用 PADDING=0 对照
# （见 experiments.py 的 film_periodic*）。纯逐点模型（mlp）也可设 0。
PADDING = 7

# 模型条件向量：freq (B,2) = [归一化频率, 极化索引]。
# baseline/unet 通过 COND_CHANNELS 把多列条件广播进空间输入。
COND_CHANNELS = 2

# 输出通道（每个极化仍预测 12 通道 E+H）
OUT_CHANNELS = 12

# 输出头模式：single | em_split | grouped（含义见 core/config.py）
#   single   - 共享头直接输出全部 12 通道（默认）
#   em_split - 旧的 3 个 E 头 + 3 个 H 头（H 显式依赖 E 预测）
HEAD_MODE = "single"

# 模型专属结构参数的通道：整个 dict 原样进 `ModelConfig.extra`。
# 新模型有基座没有的参数（如 AFNO 的分块数、ViT 的 patch 大小）就写在这里，
# 不必改 core/config.py。示例：{"afno_blocks": 16, "vit_patch": 8}
MODEL_EXTRA = {}

# 输出头 dropout（0=关闭，0.05~0.1 适合轻度正则化，缓解小数据过拟合）
PROJ_DROPOUT = 0.05

# 频域 Butterworth 衰减：w(k) = 1 / sqrt(1 + (k / k0)^(2n))
SPECTRAL_DECAY = False
SPECTRAL_DECAY_K0 = 6
SPECTRAL_DECAY_ORDER = 2

# 消融开关：Attention = 超网络投影矩阵 + 复数注意力
SPECTRAL_ATTENTION_ENABLE = True
SPECTRAL_FREQ_CONDITIONING = True
SPECTRAL_GEOM_CONDITIONING = True
SPECTRAL_GAMMA_INIT = 0.1
SPECTRAL_GAMMA_FIXED = False


# =============================================================================
# 3) 训练与优化
# =============================================================================
N_EPOCHS = 1000
BATCH_SIZE = 16
LR = 1e-3
LR_WARMUP_EPOCHS = 10
WEIGHT_DECAY = 1e-3
GRAD_CLIP_MAX_NORM = 1.0

NUM_WORKERS = 2   # Windows + mmap 加载下建议 2~4
PIN_MEMORY = True
VAL_EVERY = 5

EARLY_STOP_PATIENCE = 60
EARLY_STOP_MIN_DELTA = 1e-5

# 频率插值增强：训练时随机对 batch 做频率线性插值（宽带学习的数据增强）
FREQ_INTERP_PROB = 0.3

# 周期平移增强：训练时对 (input, output) 同步做随机 torch.roll。
# 物理依据：周期元胞 + 正入射平面波 ⇒ 体系平移等变（散射场随贴片整体平移）。
# 只作用于训练集；验证集不变。对带 out_zminus/out_zplus 的 batch 自动跳过。
# 注意：SDFGeomEncoder(amp_phase=False) 用低频模的实/虚部，不具平移不变性，
# 需配合 amp_phase=True 或让模型靠增强自己学。
AUG_SHIFT = False
# 相位参考系（core/phase_frame.py）：网络在去旋转系输出，DC/AC 各自乘以随频率线性旋转的相位后再进入损失
PHASE_FRAME = False
TRAIN_SHAPE_FRAC = 1.0   # <1：只用这一比例的训练形状（学习曲线实验；验证集不变）
EMA_DECAY = 0.0   # >0 启用权重 EMA（如 0.995）；验证/保存用 EMA 权重
PHASE_FRAME_SCALE = 1.0   # 仅对照实验用：把相位表整体乘以该系数（0.5=故意错的旋转）

# 频率泛化的真正评估：这些频点**不进训练、也不进常规验证**，而是另开两组评估：
#   unseenF_seenS   = 训练形状 × 保留频点   （只考频率泛化）
#   unseenF_unseenS = 验证形状 × 保留频点   （频率 + 形状都未见）
# 只在"被选为最优的那一轮"上计算并写进 summary.json，不参与选模，避免信息泄漏。
# 用它时把 TRAIN_FREQS 设 None（全部频点），由本键决定保留哪些。空列表 = 旧行为。
HOLDOUT_FREQS = []

# H 组（弱场/非主导分量）损失权重偏置：
# FNO 存在"傅里叶参数化偏差"(SpecB-FNO)，天然偏置主导频率(强 E 场)而淹没非主导的
# H 场。>1.0 时对 H 组相对误差额外加权，促使模型精细化拟合弱场结构（对总场有利）。
FIELD_H_GROUP_WEIGHT = 1.5

# 保存策略
SAVE_LATEST_EVERY = 1
ONLY_SAVE_PTH = True


# =============================================================================
# 4) 输出归一化
# =============================================================================
# - 训练：使用标准化后的 12 通道输出作为监督
# - 推理/评估：从 checkpoint 读取统计量并自动反标准化
# - 极化 1 的散射场会使用旋转后的正交背景场，见 data/fit_background.py
USE_OUTPUT_STANDARDIZATION = True
# 归一化模式：
#   "log"    - sign(x)*log1p(|x|/scale) 逐通道对数压缩动态范围后再 z-score（保留正负号）
#   "zscore" - 逐通道线性 (x-mean)/std（旧行为）
#   "none"   - 不归一化（调试/对比用）
OUTPUT_NORM_MODE = "log"
OUTPUT_STANDARDIZATION_EPS = 1e-8


# =============================================================================
# 5) 损失函数设计专区（统一管理）
# =============================================================================
# 说明：
# - LOSS_ENABLE_VECTOR 与 LOSS_WEIGHT_VECTOR 的下标一一对应
# - LOSS_SPACE: "physical"=主损失在物理空间计算(反归一化后，E/H 分组平衡)；
#               "normalized"=旧行为，在 log 归一化空间计算
# - 当前默认启用：L1(总场 E/H 分组相对L2) + L3(几何) + L8(逐通道分组相对L2)

# 损失计算空间：physical | normalized
LOSS_SPACE = "physical"

# 课程式损失调度
USE_LOSS_CURRICULUM = False
DETAIL_LOSS_START_EPOCH = 40
DETAIL_LOSS_RAMP_EPOCHS = 200

# 损失编号定义（L1 ~ L9）
# [L1/index=0] total_rel
#   对应函数: e_h_total_rel_l2 (physical) / field_rel_l2_loss (normalized)
#   含义: E/H 分组总场模值相对L2的平均（平衡 422:1 的量级差）
#
# [L2/index=1] directional
#   对应函数: dc_ac_directional_mean_loss
#   含义: Ex/Ey/Ez 三方向幅值的 DC/AC 相对误差均值
#
# [L3/index=2] geom
#   对应函数: field_geom_loss
#   含义: 几何一致性（方向角/相位/总场幅值），E/H 各自算再平均
#
# [L4/index=3] ch_dcac_stable
#   对应函数: complex_channel_rel_error
#   含义: 12通道逐通道 DC/AC（稳定版，归一化空间）
#
# [L5/index=4] lowfreq
# [L6/index=5] highfreq
# [L7/index=6] ch_dcac_raw
# [L8/index=7] ch_rel6_mean
#   对应函数: channel_group_balanced_l2 (physical) / channel_mean_rel_l2_error (normalized)
#   含义: 逐通道相对L2，E 组/H 组各自取均值再平均
#
# [L9/index=8] pinn_maxwell
#   对应函数: maxwell_pinn_loss
#   含义: 基于真实数据差分的麦克斯韦方程残差约束

# 向量开关：1=启用，0=关闭（顺序对应 L1~L9）
LOSS_ENABLE_VECTOR = [1, 0, 1, 0, 0, 0, 0, 1, 0]

# 向量权重：顺序对应 L1~L9
LOSS_WEIGHT_VECTOR = [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

# ---- L8 逐通道加权指数（**训练专用**，不参与指标）----
# 背景：L8 名义权重 1.0，与 L1/L3 的 1.0/0.5 看起来"平级"，但实测（散射场、
# 真实数据）三者的**梯度贡献**是 0.8% / 1.3% / 97.8% —— L8 把另外两项完全盖掉。
# 原因见 models/losses.py::channel_group_balanced_l2 的 docstring：逐通道等权
# 平均下，梯度 ∝ 1/||t_c||，于是最弱的 Hz(rms 0.216) 单独拿走 36~45%，
# 而最强的 Ex(rms 1601) 拿 0.00%。**名义权重相等 ≠ 实际影响相等。**
#
#   0.0 → 旧行为，逐位相同（val_rel 指标固定用这个值）
#   0.5 → 训练默认：组内份额跨度从数千倍压到几十倍
#   1.0 → 绝对误差等权，另一个极端
LOSS_CH_WEIGHT_POWER = 0.5

# ---- L2/L7 共享内部参数：DC/AC 子权重 ----
DC_REL_WEIGHT = 1.0
AC_REL_WEIGHT = 1.0

# ---- L3 几何损失内部参数 ----
FIELD_DIR_ANGLE_WEIGHT = 2.0
FIELD_COMPLEX_PHASE_WEIGHT = 2.0
FIELD_TOTAL_MAG_WEIGHT = 4

# ---- L4 稳定版通道 DC/AC 参数 ----
AUX_COMPLEX_CH_DC_WEIGHT = 1.0
AUX_COMPLEX_CH_AC_WEIGHT = 1.0
AUX_COMPLEX_CH_NORMALIZE_BY_TOTAL = True
AUX_COMPLEX_CH_DC_DENOM_FLOOR_RATIO = 0.1
AUX_COMPLEX_CH_LOG_COMPRESS = True

# ---- L5/L6 频域参数 ----
LOWFREQ_PATCH = 4
HIGHFREQ_CUTOFF = 6


# =============================================================================
# 6) 实验覆盖（从 configs/experiments.py 自动注入）
# =============================================================================
# 快照本模块的默认值（约定：配置项全是大写名）。切换实验时先复位到默认值，
# 再叠加覆盖项——否则上一个实验设过的键会**残留**到下一下实验
# （例如 fno_attention_multi 设 HEAD_MODE="em_split"、fno_baseline_parity 设
# WIDTH=112，之后所有实验都会继承它们）。
_CONFIG_KEYS = tuple(k for k in list(globals()) if k.isupper() and not k.startswith("_"))
_DEFAULTS = {k: globals()[k] for k in _CONFIG_KEYS}


def apply_experiment(name: str) -> dict:
    """切到某个实验：先复位默认值，再叠加该实验的覆盖项，返回覆盖字典。

    导入时会自动应用 `ACTIVE_EXPERIMENT`；`--experiment` 与 tools/compare.py 用本
    函数在导入之后再切换实验（下游一律 `getattr(C, key)` 实时读，改完即生效）。
    """
    from .experiments import EXPERIMENTS
    if name not in EXPERIMENTS:
        raise ValueError(f"未知实验: {name!r}，可选: {sorted(EXPERIMENTS)}")

    g = globals()
    for _k in _CONFIG_KEYS:
        g[_k] = _DEFAULTS[_k]
    overrides = EXPERIMENTS[name]
    for _k, _v in overrides.items():
        g[_k] = _v
    g["ACTIVE_EXPERIMENT"] = name
    return dict(overrides)


try:
    from .experiments import ACTIVE_EXPERIMENT
    apply_experiment(ACTIVE_EXPERIMENT)
except ImportError:
    ACTIVE_EXPERIMENT = "fno_attention"


def get(key, default=None):
    """安全读取本模块的全局配置项。"""
    return globals().get(key, default)


def model_config_from_globals(name: str = None):
    """由本模块的全局值构造 `core.config.ModelConfig`。

    这是"扁平全局配置"与"结构化模型配置"之间的唯一桥梁。
    所有消费方都用它，不再各自拼 kwargs 字典。
    """
    from core.config import ModelConfig, SpectralConfig

    # 输出头模式：新键 HEAD_MODE 优先；旧键 USE_SINGLE_OUTPUT_HEAD=False 等价于 "em_split"
    head_mode = str(get("HEAD_MODE", "single"))
    if get("USE_SINGLE_OUTPUT_HEAD") is False:
        head_mode = "em_split"

    cfg = ModelConfig(
        modes1=int(get("MODES1", 8)),
        modes2=int(get("MODES2", 8)),
        width=int(get("WIDTH", 64)),
        n_layers=int(get("N_LAYERS", 5)),
        in_channels=int(get("IN_CHANNELS", 2)),
        out_channels=int(get("OUT_CHANNELS", 12)),
        freq_dim=int(get("FREQ_DIM", 32)),
        padding=int(get("PADDING", 7)),
        cond_channels=int(get("COND_CHANNELS", 2)),
        use_pol_cond=bool(get("USE_POL_COND", True)),
        use_sine_input=bool(get("USE_SINE_INPUT", True)),
        kernel_cond=str(get("KERNEL_COND_MODE", "film")),
        lora_rank=int(get("LORA_RANK", 4)),
        head=head_mode,
        dropout=float(get("PROJ_DROPOUT", 0.0)),
        spectral=SpectralConfig(
            attention=bool(get("SPECTRAL_ATTENTION_ENABLE", True)),
            freq_conditioning=bool(get("SPECTRAL_FREQ_CONDITIONING", True)),
            geom_conditioning=bool(get("SPECTRAL_GEOM_CONDITIONING", True)),
            gamma_init=float(get("SPECTRAL_GAMMA_INIT", 0.1)),
            gamma_fixed=bool(get("SPECTRAL_GAMMA_FIXED", False)),
            decay=bool(get("SPECTRAL_DECAY", False)),
            decay_k0=float(get("SPECTRAL_DECAY_K0", 6)),
            decay_order=int(get("SPECTRAL_DECAY_ORDER", 2)),
        ),
        # 模型专属参数（新模型的扩展口）：实验条目里写 MODEL_EXTRA={...} 即可传入
        extra=dict(get("MODEL_EXTRA", {}) or {}),
    )
    return cfg


def model_name_from_globals() -> str:
    """当前实验要用的模型注册名。新键 `model` 优先，回落到旧的 `MODEL_NAME`。"""
    return str(get("model") or get("MODEL_NAME", "fno_attention"))

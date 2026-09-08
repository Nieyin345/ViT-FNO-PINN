# 全局配置文件

# =============================================================================
# 1) 数据与划分
# =============================================================================
import os
import multiprocessing
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认值，具体实验在 experiment_config.py 里覆盖
DATA_DIR = os.path.abspath(os.path.join(_BASE_DIR, "../data/data"))
SAVE_DIR = os.path.abspath(os.path.join(_BASE_DIR, "../checkpoints"))
MODEL_NAME = "fno_attention"

# 训练类别（v2 数据共 5 类）。矩形类（数据目录 "1_rectangle"）留作
# 形状泛化评估（eval_generalization.py），训练时不纳入。
TRAIN_CATEGORIES = ["circle", "circle_ring", "cross", "square_ring"]
TRAIN_FREQS = None  # 全部9个频点

# 数据平滑
SMOOTH_SIGMA = 0.0

# 插值目标归一化距离（v2 数据）
TARGET_D_OVER_LAMBDA = 0.1

# 散射场：用拟合的理想背景场相减，去除入射波主导成分
USE_SCATTER_FIELD = True
BACKGROUND_PARAMS_PATH = os.path.abspath(os.path.join(_BASE_DIR, "../data/blank/background_params.npz"))

VAL_RATIO = 0.2
SHUFFLE_SEED = 42

# 输出通道全局标准化（训练集统计）
# - 训练：使用标准化后的 12 通道输出作为监督
# - 推理/评估：从 checkpoint 读取统计量并自动反标准化
# - 电场（std~1e2~1e3）与磁场（Hx~3, Hy/Hz~1）尺度相差约 3 个数量级，默认做对数压缩
USE_OUTPUT_STANDARDIZATION = True
# 归一化模式：
#   "log"    - sign(x)*log1p(|x|/scale) 逐通道对数压缩动态范围后再 z-score（推荐，保留正负号）
#   "zscore" - 逐通道线性 (x-mean)/std（旧行为）
#   "none"   - 不归一化（调试/对比用）
OUTPUT_NORM_MODE = "log"
OUTPUT_STANDARDIZATION_EPS = 1e-8
OUTPUT_NORM_STATS_PATH = f"{SAVE_DIR}/output_norm_stats.npz"


# =============================================================================
# 2) 模型结构
# =============================================================================
MODES1 = 8
MODES2 = 8
WIDTH = 64
N_LAYERS = 5
FREQ_DIM = 32
OUT_CHANNELS = 12

# 频域 Butterworth 衰减
# w(k) = 1 / sqrt(1 + (k / k0)^(2n))
SPECTRAL_DECAY = False
SPECTRAL_DECAY_K0 = 6
SPECTRAL_DECAY_ORDER = 2


# =============================================================================
# 3) 训练与优化
# =============================================================================
N_EPOCHS = 1000
BATCH_SIZE = 16
LR = 1e-3
LR_WARMUP_EPOCHS = 10
WEIGHT_DECAY = 1e-3
GRAD_CLIP_MAX_NORM = 1.0

NUM_WORKERS = 2   # Windows + mmap 加载下建议 2~4，过多反而因进程开销变慢
PIN_MEMORY = True
VAL_EVERY = 5

EARLY_STOP_PATIENCE = 60
EARLY_STOP_MIN_DELTA = 1e-5

# 频率插值增强：训练时随机对 batch 做频率线性插值
# 0.0=关闭，0.5=50%概率做插值
FREQ_INTERP_PROB = 0

# 输出头 dropout（0=关闭，0.05~0.1 适合轻度正则化）
PROJ_DROPOUT = 0

# 保存策略
SAVE_LATEST_EVERY = 1
ONLY_SAVE_PTH = True


# =============================================================================
# 4) 损失函数设计专区（统一管理）
# =============================================================================
# 说明：
# - LOSS_ENABLE_VECTOR 与 LOSS_WEIGHT_VECTOR 的下标一一对应
# - 每个损失均给编号，便于实验记录与论文复现实验
# - 当前默认启用：L3（几何损失）+ L8（逐通道相对L2）+ L9（PINN 麦克斯韦残差）

# 课程式损失调度（只作用于向量中 index>=1 的细分项）
USE_LOSS_CURRICULUM = False
DETAIL_LOSS_START_EPOCH = 40
DETAIL_LOSS_RAMP_EPOCHS = 200

# ---------------------------
# 损失编号定义（L1 ~ L8）
# ---------------------------
# [L1/index=0] total_rel
#   对应函数: field_rel_l2_loss
#   含义: 总场模值相对L2（全局主对齐）
#
# [L2/index=1] directional
#   对应函数: dc_ac_directional_mean_loss
#   含义: Ex/Ey/Ez 三方向幅值的 DC/AC 相对误差均值
#
# [L3/index=2] geom
#   对应函数: field_geom_loss
#   含义: 几何一致性（方向角/相位/总场幅值）
#
# [L4/index=3] ch_dcac_stable
#   对应函数: complex_channel_rel_error
#   含义: 6通道逐通道 DC/AC（稳定版：归一化+分母下限+log压缩）
#
# [L5/index=4] lowfreq
#   对应函数: freq_band_loss.low
#   含义: 低频频谱结构对齐
#
# [L6/index=5] highfreq
#   对应函数: freq_band_loss.high
#   含义: 高频能量抑制（当前为“每通道独立计算后再平均”）
#
# [L7/index=6] ch_dcac_raw
#   对应函数: dc_ac_rel_loss
#   含义: 6通道原始 DC/AC（不做稳定化）
#
# [L8/index=7] ch_rel6_mean
#   对应函数: channel_mean_rel_l2_error
#   含义: 6个通道分别算相对L2后取平均（与 SaveRel6 保存口径一致）

# [L9/index=8] pinn_maxwell
#   对应函数: maxwell_pinn_loss
#   含义: 基于真实数据差分的麦克斯韦方程残差约束

# 向量开关：1=启用，0=关闭（顺序对应 L1~L9）
# 关闭高频抑制 (index=5)
# 注意：L9(PINN, index=8) 需要 v2 体数据（field_volume_merged.npy，含相邻 z 平面才能做
# 中心差分）；当前默认数据为 v2_cache 缓存平面（无相邻平面），故默认关闭，改用体数据时再开启。
LOSS_ENABLE_VECTOR = [0, 0, 1, 0, 0, 0, 0, 1, 0]

# 向量权重：顺序对应 L1~L9
LOSS_WEIGHT_VECTOR = [0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

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
# 5) 实验覆盖（从 experiment_config.py 自动注入）
# 不要手动修改这里，去 experiment_config.py 里切换实验
# =============================================================================
try:
    from experiment_config import ACTIVE_EXPERIMENT, EXPERIMENTS
    _overrides = EXPERIMENTS.get(ACTIVE_EXPERIMENT, {})
    for _k, _v in _overrides.items():
        globals()[_k] = _v
    # 依赖 SAVE_DIR 的派生变量需要重新计算
    OUTPUT_NORM_STATS_PATH = f"{SAVE_DIR}/output_norm_stats.npz"
    # DataLoader worker 子进程（Windows spawn）会重新导入本模块，避免重复刷屏
    if multiprocessing.current_process().name == "MainProcess":
        print(f"[Config] 当前实验: {ACTIVE_EXPERIMENT}  保存路径: {SAVE_DIR}")
except ImportError:
    pass

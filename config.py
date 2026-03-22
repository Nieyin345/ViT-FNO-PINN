# 全局配置文件
# 修改模型结构参数后，train_pinn.py 和 eval_pinn.py 会自动同步

# ─── 模型结构 ─────────────────────────────────────────────────────────────────
MODES1   = 12
MODES2   = 12
WIDTH    = 32
N_LAYERS = 4

# ─── Transformer Geometry Encoder ────────────────────────────────────────────
USE_GEO_ENCODER = True
GEO_EMBED_DIM   = 128
GEO_DEPTH       = 3
PATCH_SIZE      = 8

# ─── 数据 ─────────────────────────────────────────────────────────────────────
N_TOTAL      = 5000
GROUP_SIZE   = 500   # 每种形状的样本数，用于分层截取
GRID_SIZE    = 64
DATA_DIR     = "data/em_data_meep"
SAVE_DIR     = "checkpoints"
VAL_RATIO    = 0.2   # 验证集比例
USE_STRATIFIED_SPLIT = True

# ─── 物理参数 ─────────────────────────────────────────────────────────────────
K         = 2 * 3.141592653589793 * (4.0 / 64)
EPS_MIN   = 1.0
EPS_MAX   = 4.0
PML_WIDTH = 10

# ─── 预训练权重（fine-tune 用）────────────────────────────────────────────────
PRETRAIN_CKPT = None  # 设为 None 则从头训练 "checkpoints/best.pth"
FINETUNE_LR   = 1e-4   # fine-tune 时用更小的 lr


N_EPOCHS       = 300
BATCH_SIZE     = 32
LR             = 5e-4
VAL_EVERY      = 10    # 每隔多少 epoch 评估一次验证集

# Early Stopping
EARLY_STOP_PATIENCE = 2   # 连续多少次验证不改善则停止（单位：验证次数）
EARLY_STOP_MIN_DELTA = 1e-4  # 最小改善幅度，小于此视为没有改善

# ─── 损失项开关与权重 ─────────────────────────────────────────────────────────
# data loss（监督，必须开）
USE_DATA     = True
LAMBDA_DATA  = 1.0

# 亥姆霍兹 PDE 残差（内部区域）
USE_PDE      = True
LAMBDA_PDE   = 0.3
USE_UNCERTAINTY_WEIGHTING = False  # False = 使用固定权重 LAMBDA_DATA/LAMBDA_PDE
# 课程学习：pde 权重随训练进度从 0 爬升到 LAMBDA_PDE
USE_CURRICULUM   = True
CURRICULUM_T0    = 50    # 第几个 epoch 开始引入 pde（之前 w_pde=0）
CURRICULUM_RAMP  = 50    # 爬升持续多少 epoch（sigmoid 过渡）

# 源点局部亥姆霍兹约束（源点附近残差=S_eff）
USE_SRC_PDE    = False
LAMBDA_SRC_PDE = 0.05

# PML 边界条件 Ez≈0
USE_BC       = False
LAMBDA_BC    = 0.1

# 源点非零约束（防零解）
USE_SRC      = False
LAMBDA_SRC   = 0.1

# 全场非零约束（防零解）
USE_NONZERO  = False
LAMBDA_NONZERO = 0.0

# 幅值上界约束（防爆炸）
USE_AMP      = False
LAMBDA_AMP   = 0.0

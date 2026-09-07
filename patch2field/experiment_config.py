# =============================================================================
# 实验总控配置
# 在这里切换要训练的模型，保存路径自动隔离，互不影响
# =============================================================================

# 选择当前实验：改这一行即可
# 可选值：见下方 EXPERIMENTS 字典的 key
ACTIVE_EXPERIMENT = "fno_attention"

# =============================================================================
# 实验定义
# 每个实验只需要覆盖与默认 config.py 不同的字段
# =============================================================================
import os
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _abs(path):
    return os.path.abspath(os.path.join(_BASE_DIR, path))

EXPERIMENTS = {

    "fno_attention": {
        "MODEL_NAME": "fno_attention",
        "SAVE_DIR":   _abs("../checkpoints/fno_attention"),
    },
    "fno_baseline": {
        "MODEL_NAME": "fno_baseline",
        "SAVE_DIR":   _abs("../checkpoints/fno_baseline"),
    },

    # “大基线”：width=72 实际约 3.35M 参数（N_LAYERS=5），远小于主模型 8.05M。
    # 对比时需注意参数量差距，公平对比请用 fno_baseline_parity。
    "fno_baseline_large": {
        "MODEL_NAME": "fno_baseline",
        "SAVE_DIR":   _abs("../checkpoints/fno_baseline_large"),
        "WIDTH":      72,
    },

    # 参数量对齐基线：width=112 ≈ 8.10M 参数，与主模型（w64/modes8/5层 ≈ 8.05M）量级一致
    "fno_baseline_parity": {
        "MODEL_NAME": "fno_baseline",
        "SAVE_DIR":   _abs("../checkpoints/fno_baseline_parity"),
        "WIDTH":      112,
    },

    # FNO + LoRA 风格频率条件调制卷积核
    "fno_lora": {
        "MODEL_NAME": "fno_lora",
        "SAVE_DIR":   _abs("../checkpoints/fno_lora"),
        "LORA_RANK":  16,
    },

    "fno_attention_old": {
        "MODEL_NAME":       "fno_attention",
        "SAVE_DIR":         _abs("../checkpoints/fno_attention_old"),
        "DATA_DIR":         _abs("../data/em_data"),
        "OUT_CHANNELS":     6,
        "TRAIN_FREQS":      None,
        "TRAIN_CATEGORIES": ["circle", "cross_base"],
        "WIDTH":            48,
        "SMOOTH_SIGMA":     0.0,
    },
    "fno_baseline_old": {
        "MODEL_NAME":       "fno_baseline",
        "SAVE_DIR":         _abs("../checkpoints/fno_baseline_old"),
        "DATA_DIR":         _abs("../data/em_data"),
        "OUT_CHANNELS":     6,
        "TRAIN_FREQS":      None,
        "TRAIN_CATEGORIES": ["circle", "cross_base"],
        "WIDTH":            48,
        "SMOOTH_SIGMA":     0.0,
    },

    # FNO + FiLM：有条件编码器，但用简单 scale+shift 代替动态投影矩阵
    "fno_film": {
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/fno_film"),
    },

    # U-Net：纯空间域卷积，无频域操作，证明频域操作的必要性
    "unet": {
        "MODEL_NAME": "unet",
        "SAVE_DIR":   _abs("../checkpoints/unet"),
    },
}

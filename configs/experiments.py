# =============================================================================
# 实验总控配置
# 在这里切换要训练的模型，保存路径自动隔离，互不影响
# =============================================================================
"""实验定义表。

每个实验只写与 `configs.base` 默认值不同的字段。约定：
  - `model`      必填，模型注册名（见 `core.registry.available()`），
                 也是旧字段 `MODEL_NAME` 的替代。两者同时存在时以 `model` 为准。
  - `SAVE_DIR`   建议显式写，避免不同实验互相覆盖 checkpoint。

新增一个对比实验 = 在下面加一个条目，无需改动任何代码。
"""

import os

# 选择当前实验：改这一行，或用环境变量 TRANSFORMER_EXPERIMENT 覆盖
ACTIVE_EXPERIMENT = os.environ.get("TRANSFORMER_EXPERIMENT", "fno_attention")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _abs(path):
    return os.path.abspath(os.path.join(_BASE_DIR, path))


# =============================================================================
# 实验定义
# =============================================================================
EXPERIMENTS = {

    "fno_attention": {
        "model": "fno_attention",
        "MODEL_NAME": "fno_attention",
        "SAVE_DIR":   _abs("../checkpoints/fno_attention"),
    },
    # 消融：完全关闭超网络注意力，等价于普通 FNO 频域卷积
    "fno_attention_no_attn": {
        "model": "fno_attention",
        "MODEL_NAME": "fno_attention",
        "SAVE_DIR":   _abs("../checkpoints/fno_attention_no_attn"),
        "SPECTRAL_ATTENTION_ENABLE": False,
    },
    # 消融：保留注意力但只做频率超网络调制，不调制几何 K
    "fno_attention_freq_only": {
        "model": "fno_attention",
        "MODEL_NAME": "fno_attention",
        "SAVE_DIR":   _abs("../checkpoints/fno_attention_freq_only"),
        "SPECTRAL_GEOM_CONDITIONING": False,
    },
    # 消融：保留注意力但只做几何超网络调制，不调制频率 Q
    "fno_attention_geom_only": {
        "model": "fno_attention",
        "MODEL_NAME": "fno_attention",
        "SAVE_DIR":   _abs("../checkpoints/fno_attention_geom_only"),
        "SPECTRAL_FREQ_CONDITIONING": False,
    },
    # 消融：恢复旧的 3 E 头 + 3 H 头，对照单头输出
    "fno_attention_multi": {
        "model": "fno_attention",
        "MODEL_NAME": "fno_attention",
        "SAVE_DIR":   _abs("../checkpoints/fno_attention_multi"),
        "HEAD_MODE":  "em_split",
    },
    # 消融：静态注意力参与，但 gamma 固定为 0，关闭任何条件/全局注意力影响
    "fno_attention_gamma0": {
        "model": "fno_attention",
        "MODEL_NAME": "fno_attention",
        "SAVE_DIR":   _abs("../checkpoints/fno_attention_gamma0"),
        "SPECTRAL_GAMMA_INIT": 0.0,
        "SPECTRAL_GAMMA_FIXED": True,
    },
    "fno_baseline": {
        "model": "fno_baseline",
        "MODEL_NAME": "fno_baseline",
        "SAVE_DIR":   _abs("../checkpoints/fno_baseline"),
    },

    # “大基线”：width=72 实际约 3.35M 参数（N_LAYERS=5），远小于主模型 8.05M。
    # 对比时需注意参数量差距，公平对比请用 fno_baseline_parity。
    "fno_baseline_large": {
        "model": "fno_baseline",
        "MODEL_NAME": "fno_baseline",
        "SAVE_DIR":   _abs("../checkpoints/fno_baseline_large"),
        "WIDTH":      72,
    },

    # 参数量对齐基线：width=112 ≈ 8.10M 参数，与主模型（w64/modes8/5层 ≈ 8.05M）量级一致
    "fno_baseline_parity": {
        "model": "fno_baseline",
        "MODEL_NAME": "fno_baseline",
        "SAVE_DIR":   _abs("../checkpoints/fno_baseline_parity"),
        "WIDTH":      112,
    },

    # FNO + LoRA 风格频率条件调制卷积核
    "fno_lora": {
        "model": "fno_lora",
        "MODEL_NAME": "fno_lora",
        "SAVE_DIR":   _abs("../checkpoints/fno_lora"),
        "LORA_RANK":  16,
    },

    # FNO + FiLM：有条件编码器，但用简单 scale+shift 代替动态投影矩阵
    "fno_film": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/fno_film"),
    },

    # U-Net：纯空间域卷积，无频域操作，证明频域操作的必要性
    "unet": {
        "model": "unet",
        "MODEL_NAME": "unet",
        "SAVE_DIR":   _abs("../checkpoints/unet"),
    },

    # ============ FNO2dV2 新主模型系列 ============
    # 强化纯 FNO + 频率正弦输入 + FiLM 核条件化（默认）
    "fno_v2": {
        "model": "fno_v2",
        "MODEL_NAME": "fno_v2",
        "SAVE_DIR":   _abs("../checkpoints/fno_v2"),
        "KERNEL_COND_MODE": "film",
        "USE_SINE_INPUT":   True,
    },
    # 消融：纯 FNO + 正弦输入，无任何核条件化
    "fno_v2_none": {
        "model": "fno_v2",
        "MODEL_NAME": "fno_v2",
        "SAVE_DIR":   _abs("../checkpoints/fno_v2_none"),
        "KERNEL_COND_MODE": "none",
        "USE_SINE_INPUT":   True,
    },
    # 消融：无正弦输入，仅 FiLM 核条件化
    "fno_v2_nosine": {
        "model": "fno_v2",
        "MODEL_NAME": "fno_v2",
        "SAVE_DIR":   _abs("../checkpoints/fno_v2_nosine"),
        "KERNEL_COND_MODE": "film",
        "USE_SINE_INPUT":   False,
    },
    # 变体：LoRA 低秩扰动 w1/w2 核条件化
    "fno_v2_lora": {
        "model": "fno_v2",
        "MODEL_NAME": "fno_v2",
        "SAVE_DIR":   _abs("../checkpoints/fno_v2_lora"),
        "KERNEL_COND_MODE": "lora",
        "USE_SINE_INPUT":   True,
        "LORA_RANK":        8,
    },
}

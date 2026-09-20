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
    # 消融：完全关闭超网络注意力，等价于普通 FNO 频域卷积。
    # 注意：关掉时注意力模块（W_q/W_k/hyper_*/gamma）**根本不创建**，
    # 所以这是"与主模型同宽同层、只少了注意力机制"的干净消融，不是容量对照。
    # 但参数反而降到 2.66M（与 fno_baseline 同量级），因此同时是"容量对照"之一：
    #   看 fno_attention vs 本实验 → 注意力机制本身是否有效（容量不控）
    #   看 本实验 vs fno_baseline  → 同容量下两者的差异只剩实现细节
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

    # 参数量对齐基线：width=112 ≈ 8.10M 参数，与主模型（w64/modes8/5层 ≈ 8.05M）量级一致。
    # 但它是 fno_baseline 类（无注意力、无条件编码器），只对齐了**参数量**，
    # 没对齐架构——它回答的是"主模型的性能是否只是来自参数量"。
    # 若要"同架构、只去掉注意力"的消融，用 fno_attention_no_attn（2.66M）。
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

    # ============ 对比基线族 ============
    # 设计思路：把"感受野/混合范围"与"混合发生在哪个域"两个因素分开，
    # 让每一对实验之间的差异只有一个变量。
    #
    #   感受野 1×1  →  mlp
    #   局部        →  cnn            （对照 unet：多尺度结构是否必要）
    #   全局(空间域) →  vit            （对照 cnn：全局混合是否必要）
    #   全局(频域)   →  afno / fno_*   （对照 vit：频域参数化是否必要）
    #
    # 所有基线都用与主模型相同的损失/归一化/训练超参，只有结构不同。

    # 逐像素 MLP：感受野 1×1，不能表达任何空间结构。这是"下界参照系"，
    # 用来确认空间/频域结构确实带来了收益（若它不差，该查指标而不是堆模型）。
    # PADDING=0：1×1 卷积下 padding 无作用，去掉省约 48% 算力。
    "mlp": {
        "model": "mlp",
        "MODEL_NAME": "mlp",
        "SAVE_DIR":   _abs("../checkpoints/mlp"),
        "PADDING":    0,
        "MODEL_EXTRA": {"mlp_hidden": 128},
    },

    # 全卷积残差 CNN：全程 64×64，无池化/上采样/skip，感受野靠 dilation 拉大。
    # 与 fno_baseline 层数、宽度相同，唯一差别是 3×3 空间卷积 vs 频域卷积。
    "cnn": {
        "model": "cnn",
        "MODEL_NAME": "cnn",
        "SAVE_DIR":   _abs("../checkpoints/cnn"),
        "PADDING":    0,               # 3×3 卷积自带 padding，基座的额外 pad 不需要
        "MODEL_EXTRA": {"cnn_dilation_cycle": 3},
    },

    # ViT：空间域 patch 化 + 全局自注意力。全局混合的"空间域"一侧。
    "vit": {
        "model": "vit",
        "MODEL_NAME": "vit",
        "SAVE_DIR":   _abs("../checkpoints/vit"),
        "MODEL_EXTRA": {"vit_patch": 8, "vit_heads": 4},
    },

    # AFNO：频域逐频率对 MLP（所有频率对共享权重）+ 通道混合。
    # 严格弱于 FNO 的谱卷积（后者每个频率对有独立矩阵），用来定位
    # "FNO 的收益来自频域全局混合，还是来自逐频率自由度"。
    "afno": {
        "model": "afno",
        "MODEL_NAME": "afno",
        "SAVE_DIR":   _abs("../checkpoints/afno"),
        "MODEL_EXTRA": {"afno_hidden_ratio": 2.0},
    },
    # 变体：AFNO 分块模式（blocks=4 → 16 个块各自独立权重），
    # 在"完全共享权重"与"每频率对独立权重"之间插值，用来找拐点。
    "afno_blocks": {
        "model": "afno",
        "MODEL_NAME": "afno",
        "SAVE_DIR":   _abs("../checkpoints/afno_blocks"),
        "MODEL_EXTRA": {"afno_hidden_ratio": 2.0, "afno_blocks": 4},
    },

    # DeepONet：显式算子分解 branch(几何) ⊗ trunk(坐标)。
    # 代表"另一条算子学习路线"——若它明显更差，说明目标场的空间模式
    # 无法由几何无关的全局基函数张成（局域结构是必需的）。
    "deeponet": {
        "model": "deeponet",
        "MODEL_NAME": "deeponet",
        "SAVE_DIR":   _abs("../checkpoints/deeponet"),
        "MODEL_EXTRA": {"deeponet_basis": 128},
    },
}

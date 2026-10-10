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
    # ============ 频率泛化口径（本项目的核心目标，此前**没有**被度量）============
    # 现有的 val_geom 是**按形状**划分的：训练/验证的 9 个频点与 2 个极化完全
    # 相同（实测确认），所以它只衡量"新形状"，**测不到频率/极化泛化**。
    # 用一个不测目标的指标去选模型，选出的可能是形状拟合更好、频率插值更差的那个。
    #
    # 这两个实验用现成的 TRAIN_FREQS 机制把**频点**留出（训练只见 7 个频点），
    # 于是 val_geom 变成真正的"未见频点"误差。留出的 10.0/11.0 GHz 是等间隔
    # 9 点里的**内插点**（不是外推），符合"宽带内插值"这个真实诉求。
    #
    # 注：留出频点后样本量降到 7/9。若要同时保留全量训练，另建一个
    # "训练用全部频点、评估时单独过滤"的口径——但那要动 build_split，
    # 会改变所有历史 run 的划分指纹，先不做。
    "fno_attention_freqhold": {
        "model": "fno_attention",
        "MODEL_NAME": "fno_attention",
        "SAVE_DIR":   _abs("../checkpoints/fno_attention_freqhold"),
        "TRAIN_FREQS": [8.0, 8.5, 9.0, 9.5, 10.5, 11.5, 12.0],
    },
    "fno_attention_no_attn_freqhold": {
        "model": "fno_attention",
        "MODEL_NAME": "fno_attention",
        "SAVE_DIR":   _abs("../checkpoints/fno_attention_no_attn_freqhold"),
        "TRAIN_FREQS": [8.0, 8.5, 9.0, 9.5, 10.5, 11.5, 12.0],
        "SPECTRAL_ATTENTION_ENABLE": False,
    },
    # FiLM 的频率留出版本：与上面两个同口径（同划分、同留出频点），
    # 用于回答"频率泛化上 FiLM 是否同样不输主模型"。
    "fno_film_freqhold": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/fno_film_freqhold"),
        "TRAIN_FREQS": [8.0, 8.5, 9.0, 9.5, 10.5, 11.5, 12.0],
    },
    # 同上，但**关闭频率插值增强**。实测线性插值标签本身有 11%~19% 的误差
    # （场随频率振荡，等间隔三点插值不成立），30% 的 batch 拿到的是假标签；
    # 但它也可能是模型学会频点间过渡的唯一来源。A/B 定净效应：
    #   若关闭后 freqhold 的 val_geom 更差 -> 增强虽脏但有效，保留
    #   若更好 -> 灌假标签是净伤害，直接关掉
    "fno_film_freqhold_nointerp": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/fno_film_freqhold_nointerp"),
        "TRAIN_FREQS": [8.0, 8.5, 9.0, 9.5, 10.5, 11.5, 12.0],
        "FREQ_INTERP_PROB": 0.0,
    },
    # 当前最优配置（nointerp）+ em_split 头（H 显式依赖 E 预测）。
    # 回答"H 头看到 E 预测"这个结构先验在 FiLM 骨架上是否仍有增益。
    "fno_film_em_split_freqhold_nointerp": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/fno_film_em_freqhold_nointerp"),
        "TRAIN_FREQS": [8.0, 8.5, 9.0, 9.5, 10.5, 11.5, 12.0],
        "FREQ_INTERP_PROB": 0.0,
        "HEAD_MODE": "em_split",
    },
    # ==================== 模式间谱自注意力（注意力×FNO 的干净结合）====================
    # fno_mattn = fno_film 原样 + 模式间自注意力修正（γ·Wv 门控、Wv 零初始化）。
    # 出厂态数值上等于 film（修正恒 0），训练中注意力把「固定线性模式混合器
    # w1/w2」变成逐样本自适应混合。同口径 A/B 回答：模式轴的注意力是否有增益
    # （对照通道轴注意力实测均匀退化、贡献仅 5.3%）。
    "fno_mattn_freqhold_nointerp": {
        "model": "fno_mattn",
        "MODEL_NAME": "fno_mattn",
        "SAVE_DIR":   _abs("../checkpoints/fno_mattn_freqhold_nointerp"),
        "TRAIN_FREQS": [8.0, 8.5, 9.0, 9.5, 10.5, 11.5, 12.0],
        "FREQ_INTERP_PROB": 0.0,
    },
    # 消融：**参数逐位相同**，只把模式间成对交换（QKᵀ）换成逐模式独立变换
    # （ctx = v，to_q/to_k 照建但不参与前向）。
    # 为什么必须有这一条：mattn 比 fno_film 多 84K 参数（+2.9%），val_geom 只
    # 好 2.6% —— 不排除"增益只是来自多出来的参数"。本实验把差异收敛成单一变量
    # （有没有模式混合），是判断模式注意力**机制**是否有效的唯一干净对照。
    "fno_mattnv_freqhold_nointerp": {
        "model": "fno_mattn",
        "MODEL_NAME": "fno_mattn",
        "SAVE_DIR":   _abs("../checkpoints/fno_mattnv_freqhold_nointerp"),
        "TRAIN_FREQS": [8.0, 8.5, 9.0, 9.5, 10.5, 11.5, 12.0],
        "FREQ_INTERP_PROB": 0.0,
        "MODEL_EXTRA": {"mattn_use_attn": False},
    },
    # ============ 单频点对照：多频点共享容量是否伤害单频点拟合 ============
    # 说明：现有模型要在 7 个频点上共享同一套谱权重（w1/w2 与几何无关、
    # 频率只经 FiLM 的逐通道标量进入）。问题：这份容量被摊薄了吗？
    # 做法：只训练**单个频点**（TRAIN_FREQS=[10.5]），其余完全同口径。
    # 关键可比性：build_split 按 (category, shape) 分层，与频点无关，
    # 所以单频点模型的验证形状与 nointerp_long 是**同一批**，val_geom 可直接比。
    # 判据：若单频点在 10.5 GHz 的误差显著低于 nointerp_long 在该频点的
    # 0.2084 → 多频点确实在摊薄容量；若相当或更差 → 容量不是瓶颈。
    "fno_film_single_105": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/fno_film_single_105"),
        "TRAIN_FREQS": [10.5],
        "FREQ_INTERP_PROB": 0.0,
    },
    # ==================== WIME 波先验乘法编码（arXiv 2408.02971 的适配）====================
    # 该工作针对"训练只有稀疏离散波长、要在未见波长泛化"，与本项目同构。
    # 核心：条件嵌入不做成标量，而做成**2D 空间波图样**（谱一致性），
    # 每层与特征图逐元素相乘注入。
    #
    # 适配：原文是斜入射平面波相位斜坡，本项目周期边界+垂直入射，改用
    # Floquet 模式基 + 频率相关倏逝衰减（实测 k_c 随 f 上升、高阶模占比涨 16 倍）。
    #
    # A/B 口径与最优对照 nointerp_long 完全一致（频率留出 + 关插值 + 100 轮）。
    # 三臂对照：
    #   fno_wime(use=True)   → WIME 有没有增益
    #   fno_wime(use=False)  → 关掉后应与 fno_film 逐位一致（证明是单变量对照）
    #   fno_film             → 参考点 0.2265
    "fno_wime_freqhold_nointerp": {
        "model": "fno_wime",
        "MODEL_NAME": "fno_wime",
        "SAVE_DIR":   _abs("../checkpoints/fno_wime_freqhold_nointerp"),
        "TRAIN_FREQS": [8.0, 8.5, 9.0, 9.5, 10.5, 11.5, 12.0],
        "FREQ_INTERP_PROB": 0.0,
    },
    # 消融：关掉 WIME 通路（参数逐位保留，只是波先验不注入）。
    # 作用：排除"多出来的参数"这个混杂变量，与 mattnv 的做法一致。
    "fno_wime_off_freqhold_nointerp": {
        "model": "fno_wime",
        "MODEL_NAME": "fno_wime",
        "SAVE_DIR":   _abs("../checkpoints/fno_wime_off_freqhold_nointerp"),
        "TRAIN_FREQS": [8.0, 8.5, 9.0, 9.5, 10.5, 11.5, 12.0],
        "FREQ_INTERP_PROB": 0.0,
        "MODEL_EXTRA": {"wime_mode": "off"},
    },
    # ==================== 色散坐标谱卷积（论文主候选）====================
    # fno_film 的唯一结构替换：w0(k)·(1+g(ξ))，ξ=k/m0(f) 为无量纲色散坐标
    # （m0 = k0·L/2π 由频率物理决定）。谱形状随频率的移动变成结构自带性质，
    # 8→12 GHz 的插值不再只靠条件化。模式间注意力在模式维做 softmax
    # （对照主模型的通道维）。
    #
    # A/B 口径与最优对照 nointerp_long 完全一致：频率留出 + 关插值 + 100 轮。
    # 对照组 val_geom=0.2265。模型名 fno_disp。
    "fno_disp_freqhold_nointerp": {
        "model": "fno_disp",
        "MODEL_NAME": "fno_disp",
        "SAVE_DIR":   _abs("../checkpoints/fno_disp_freqhold_nointerp"),
        "TRAIN_FREQS": [8.0, 8.5, 9.0, 9.5, 10.5, 11.5, 12.0],
        "FREQ_INTERP_PROB": 0.0,
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

    # ==================== 周期性 / 平移增强对照（2026-09 修复后新增）====================
    # 与 fno_film_freqhold_nointerp 只差下面标出的一项，其余全同（同划分、同损失）。
    # 注意：这几条沿用 freqhold 的数据口径，验证集里仍没有 10/11 GHz——
    # 它们只回答「周期性/增强对已见频率下、未见形状的泛化有没有帮助」，
    # 不回答频率泛化（那要等 (形状,频率) 划分，见 PLAN.md 第 2 阶段）。
    # HFSS 元胞是周期边界：去掉右/下补零（PADDING=0），FFT 卷积严格周期。
    "film_periodic": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/film_periodic"),
        "TRAIN_FREQS": [8.0, 8.5, 9.0, 9.5, 10.5, 11.5, 12.0],
        "FREQ_INTERP_PROB": 0.0,
        "PADDING": 0,
    },
    # 周期平移增强（AUG_SHIFT），PADDING 保持历史值 7。
    "film_aug": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/film_aug"),
        "TRAIN_FREQS": [8.0, 8.5, 9.0, 9.5, 10.5, 11.5, 12.0],
        "FREQ_INTERP_PROB": 0.0,
        "AUG_SHIFT": True,
    },
    # PADDING=0 + 周期平移增强（二者叠加）。
    "film_periodic_aug": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/film_periodic_aug"),
        "TRAIN_FREQS": [8.0, 8.5, 9.0, 9.5, 10.5, 11.5, 12.0],
        "FREQ_INTERP_PROB": 0.0,
        "PADDING": 0,
        "AUG_SHIFT": True,
    },

    # ==================== 真正的频率泛化评估（保留频点进评估集，不进训练/选模）====================
    # 与 film_* 的区别：10/11 GHz 不再被从数据里整个删掉，而是作为两组评估集
    #   unseenF_seenS（训练形状×保留频点）/ unseenF_unseenS（验证形状×保留频点）
    # 在最优轮上计算并写入 summary.json["holdout"]。常规 val 集（验证形状×已见频点）不变，
    # 与旧的 freqhold 实验同形状、同频点，可直接对比。
    "hold_base": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/hold_base"),
        "TRAIN_FREQS": None,              # 全部频点，保留哪些由 HOLDOUT_FREQS 决定
        "HOLDOUT_FREQS": [10.0, 11.0],
        "FREQ_INTERP_PROB": 0.0,
    },
    "hold_per": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/hold_per"),
        "TRAIN_FREQS": None,              # 全部频点，保留哪些由 HOLDOUT_FREQS 决定
        "HOLDOUT_FREQS": [10.0, 11.0],
        "FREQ_INTERP_PROB": 0.0,
        "PADDING": 0,
    },
    "hold_aug": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/hold_aug"),
        "TRAIN_FREQS": None,              # 全部频点，保留哪些由 HOLDOUT_FREQS 决定
        "HOLDOUT_FREQS": [10.0, 11.0],
        "FREQ_INTERP_PROB": 0.0,
        "AUG_SHIFT": True,
    },
    "hold_per_aug": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/hold_per_aug"),
        "TRAIN_FREQS": None,              # 全部频点，保留哪些由 HOLDOUT_FREQS 决定
        "HOLDOUT_FREQS": [10.0, 11.0],
        "FREQ_INTERP_PROB": 0.0,
        "PADDING": 0,
        "AUG_SHIFT": True,
    },

    # ==================== 第二轮：只保留 10 GHz + 频率编码/dropout 因子实验 ====================
    # 因子 A：频率编码带宽（默认 1~16 倍 vs 低带宽 1~2 倍，只 4 个特征）
    # 因子 B：PROJ_DROPOUT（0.05 vs 0；项目自己的待办里也列了"dropout off"）
    "h10_base": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/h10_base"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],          # 只留 10 GHz，训练有 8 个频点
        "FREQ_INTERP_PROB": 0.0,
    },
    "h10_nodrop": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/h10_nodrop"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],          # 只留 10 GHz，训练有 8 个频点
        "FREQ_INTERP_PROB": 0.0,
        "PROJ_DROPOUT": 0.0,
    },
    "h10_low": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/h10_low"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],          # 只留 10 GHz，训练有 8 个频点
        "FREQ_INTERP_PROB": 0.0,
        "MODEL_EXTRA": {"freq_enc_n": 4, "freq_enc_max_log2": 1.0},
    },
    "h10_low_nodrop": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/h10_low_nodrop"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],          # 只留 10 GHz，训练有 8 个频点
        "FREQ_INTERP_PROB": 0.0,
        "PROJ_DROPOUT": 0.0,
        "MODEL_EXTRA": {"freq_enc_n": 4, "freq_enc_max_log2": 1.0},
    },
    "h10_pf": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/h10_pf"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,              # 相位参考系：AC/DC 各自去掉线性相位
    },
    "h10_pf_nodrop": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/h10_pf_nodrop"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "PROJ_DROPOUT": 0.0,
    },
    "h10_pf_wrong": {                     # 对照：相位表乘 0.5（故意错的旋转），验证增益确实来自正确的相位
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/h10_pf_wrong"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "PHASE_FRAME_SCALE": 0.5,
    },
    "h10_all": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/h10_all"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],   # 8 类形状一起训（数据量 x8）
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "LR_WARMUP_EPOCHS": 2,
        "PHASE_FRAME": False,
    },
    "h10_all_pf": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/h10_all_pf"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],   # 8 类形状一起训（数据量 x8）
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "LR_WARMUP_EPOCHS": 2,
        "PHASE_FRAME": True,
    },
    "h10_poly3": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/h10_poly3"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODEL_EXTRA": {"poly_k": 3, "free_body": True},
    },
    "h10_poly4": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/h10_poly4"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODEL_EXTRA": {"poly_k": 4, "free_body": True},
    },
    "h10_poly6": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/h10_poly6"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True},
    },
    "h10_poly4_nopf": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/h10_poly4_nopf"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": False,
        "MODEL_EXTRA": {"poly_k": 4, "free_body": True},
    },
    "h10_poly5": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/h10_poly5"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODEL_EXTRA": {"poly_k": 5, "free_body": True},
    },
    "h10_poly7": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/h10_poly7"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODEL_EXTRA": {"poly_k": 7, "free_body": True},
    },
    "h10_poly8": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/h10_poly8"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODEL_EXTRA": {"poly_k": 8, "free_body": True},
    },
    "h10_poly6_nopf": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/h10_poly6_nopf"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": False,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True},
    },
    "h11_poly6": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/h11_poly6"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [11.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True},
    },
    "h9_poly6": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/h9_poly6"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [9.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True},
    },
    "p6s": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/p6s"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256},
    },
    "p6s_m16": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/p6s_m16"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODES1": 16,
        "MODES2": 16,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256},
    },
    "p6s_m16w96": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/p6s_m16w96"),
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODES1": 16,
        "MODES2": 16,
        "WIDTH": 96,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256},
    },
    "p6s_all": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/p6s_all"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256},
    },
    "A6": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/A6"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256},
    },
    "A7": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/A7"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODEL_EXTRA": {"poly_k": 7, "free_body": True, "poly_head_hidden": 256},
    },
    "A6m16": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/A6m16"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "MODES1": 16,
        "MODES2": 16,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256},
    },
    "A6ema": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/A6ema"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256,
                        "poly_basis_mode": "chebyshev"},
    },
    # ── 极点-留数频率基（本轮新增，见 models/fno2d_pole.py 文件头）────────
    #
    # 与 A6ema **逐项相同**，只换频率基：K=6 切比雪夫 → 4 极点 + 2 阶多项式背景。
    # 输出头通道数一样（6×12 vs (4+2)×12），所以这是"同参数量、换基"的干净对照。
    #
    # 动机来自 diag_pf 的诊断：留出频点 10 GHz 的 geom 0.388 里幅值项占 76%，
    # |pred|/|true|=0.787 —— 多项式在未见频点把谐振峰削平了。有理基的洛伦兹形状
    # 正是谐振响应的本征形式。
    "A6pole": {
        "model": "fno_film_pole",
        "MODEL_NAME": "fno_film_pole",
        "SAVE_DIR":   _abs("../checkpoints/A6pole"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {"pole_p": 4, "pole_k0": 2, "free_body": True,
                        "poly_head_hidden": 256, "pole_delta": True},
    },
    # 同上，但**关掉**硬编码的相位表。有理基的复留数本来就能产生随频率的相位旋转，
    # 这一组回答"那张 8 个实测常数的表还需不需要"——如果不需要，论文里就少一处魔数，
    # 也少一个审稿人必问的问题。
    "A6pole_nopf": {
        "model": "fno_film_pole",
        "MODEL_NAME": "fno_film_pole",
        "SAVE_DIR":   _abs("../checkpoints/A6pole_nopf"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": False,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {"pole_p": 8, "pole_k0": 2, "free_body": True,
                        "poly_head_hidden": 256, "pole_delta": True},
    },
    "A6ema_h1011": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/A6ema_h1011"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0, 11.0],     # 同时保留两个频点（7 个训练频点）
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256},
    },
    # ============ A6ema 组件消融（2x2；EMA 固定）============
    # constbasis 保留 K=6 的全部系数头与参数量，只把 Chebyshev 基改为常数平均；
    # 因而可将频率基贡献与参数量变化区分开。
    "A6ema_nophase": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/A6ema_nophase"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": False,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256,
                        "poly_basis_mode": "chebyshev"},
    },
    "A6ema_constbasis": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/A6ema_constbasis"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256,
                        "poly_basis_mode": "constant"},
    },
    "A6ema_nophase_constbasis": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/A6ema_nophase_constbasis"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": False,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256,
                        "poly_basis_mode": "constant"},
    },
    # EMA 单独对照：其余与 A6ema 完全相同。
    "A6_noema": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/A6_noema"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.0,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256,
                        "poly_basis_mode": "chebyshev"},
    },

    # ============ 同一 holdout / 同一训练配方下的架构对照 ============
    # 所有架构统一使用 PhaseFrame + EMA；实际参数量由 summary.json 记录，
    # 不把参数量不同的结构对比误写成参数匹配。
    "A6ema_fno_baseline": {
        "model": "fno_baseline",
        "MODEL_NAME": "fno_baseline",
        "SAVE_DIR":   _abs("../checkpoints/A6ema_fno_baseline"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
    },
    "A6ema_fno_film": {
        "model": "fno_film",
        "MODEL_NAME": "fno_film",
        "SAVE_DIR":   _abs("../checkpoints/A6ema_fno_film"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
    },
    "A6ema_fno_attention": {
        "model": "fno_attention",
        "MODEL_NAME": "fno_attention",
        "SAVE_DIR":   _abs("../checkpoints/A6ema_fno_attention"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
    },
    "A6ema_unet": {
        "model": "unet",
        "MODEL_NAME": "unet",
        "SAVE_DIR":   _abs("../checkpoints/A6ema_unet"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
    },
    "LC25": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/LC25"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "TRAIN_SHAPE_FRAC": 0.25,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256},
    },
    "LC50": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/LC50"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "TRAIN_SHAPE_FRAC": 0.5,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256},
    },
    "GAP15": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/GAP15"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0, 10.5],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256},
    },
    "GAP20": {
        "model": "fno_film_poly",
        "MODEL_NAME": "fno_film_poly",
        "SAVE_DIR":   _abs("../checkpoints/GAP20"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [9.5, 10.0, 10.5],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {"poly_k": 6, "free_body": True, "poly_head_hidden": 256},
    },
    # ============ 麦克斯韦-Floquet 物理等变主模型（FNO2dMaxwell）及正交消融族 ============
    # 与 A6ema 完全同场景（SCENARIO=be4d101cf198，同划分、同损失 GEN1、同参数量 ~2.20M）。
    "A6maxwell": {
        "model": "fno_maxwell",
        "MODEL_NAME": "fno_maxwell",
        "SAVE_DIR":   _abs("../checkpoints/A6maxwell"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {
            "poly_k": 6, "free_body": True, "poly_head_hidden": 192,
            "raw_frame": True, "pol_equivariant": True,
            "floquet_split": True, "tem_dc": True, "maxwell_head": True,
        },
    },
    # 消融 1：退回旧散射场 log1p 归一化空间 PhaseFrame（验证去缠绕物理总场相位系的核心贡献）
    "A6maxwell_oldpf": {
        "model": "fno_maxwell",
        "MODEL_NAME": "fno_maxwell",
        "SAVE_DIR":   _abs("../checkpoints/A6maxwell_oldpf"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {
            "poly_k": 6, "free_body": True, "poly_head_hidden": 192,
            "raw_frame": False, "pol_equivariant": True,
            "floquet_split": True, "tem_dc": False, "maxwell_head": True,
        },
    },
    # 消融 2：关闭 (0,0) 阶 TEM 平面波阻抗耦合（改为 6 通道独立总场 DC 预测）
    "A6maxwell_notem": {
        "model": "fno_maxwell",
        "MODEL_NAME": "fno_maxwell",
        "SAVE_DIR":   _abs("../checkpoints/A6maxwell_notem"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {
            "poly_k": 6, "free_body": True, "poly_head_hidden": 192,
            "raw_frame": True, "pol_equivariant": True,
            "floquet_split": True, "tem_dc": False, "maxwell_head": True,
        },
    },
    # 消融 3：关闭 Floquet (0,0) 镜面 DC 与高阶衍射 AC 的显式解耦
    "A6maxwell_nofloquet": {
        "model": "fno_maxwell",
        "MODEL_NAME": "fno_maxwell",
        "SAVE_DIR":   _abs("../checkpoints/A6maxwell_nofloquet"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {
            "poly_k": 6, "free_body": True, "poly_head_hidden": 192,
            "raw_frame": True, "pol_equivariant": True,
            "floquet_split": False, "tem_dc": False, "maxwell_head": True,
        },
    },
    # 消融 4：关闭精确 D2 极化反射等变（改为标量极化条件输入）
    "A6maxwell_nopoleq": {
        "model": "fno_maxwell",
        "MODEL_NAME": "fno_maxwell",
        "SAVE_DIR":   _abs("../checkpoints/A6maxwell_nopoleq"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {
            "poly_k": 6, "free_body": True, "poly_head_hidden": 192,
            "raw_frame": True, "pol_equivariant": False,
            "floquet_split": True, "tem_dc": True, "maxwell_head": True,
        },
    },
    # 消融 5：关闭法拉第旋度耦合头与边界梯度提升
    "A6maxwell_nomaxwell": {
        "model": "fno_maxwell",
        "MODEL_NAME": "fno_maxwell",
        "SAVE_DIR":   _abs("../checkpoints/A6maxwell_nomaxwell"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {
            "poly_k": 6, "free_body": True, "poly_head_hidden": 192,
            "raw_frame": True, "pol_equivariant": True,
            "floquet_split": True, "tem_dc": True, "maxwell_head": False,
        },
    },
    # 宽频带缺口泛化测试（对照 GAP15 / GAP20）
    "A6maxwell_GAP15": {
        "model": "fno_maxwell",
        "MODEL_NAME": "fno_maxwell",
        "SAVE_DIR":   _abs("../checkpoints/A6maxwell_GAP15"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0, 10.5],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {
            "poly_k": 6, "free_body": True, "poly_head_hidden": 192,
            "raw_frame": True, "pol_equivariant": True,
            "floquet_split": True, "tem_dc": True, "maxwell_head": True,
        },
    },
    "A6maxwell_GAP20": {
        "model": "fno_maxwell",
        "MODEL_NAME": "fno_maxwell",
        "SAVE_DIR":   _abs("../checkpoints/A6maxwell_GAP20"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [9.5, 10.0, 10.5],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {
            "poly_k": 6, "free_body": True, "poly_head_hidden": 192,
            "raw_frame": True, "pol_equivariant": True,
            "floquet_split": True, "tem_dc": True, "maxwell_head": True,
        },
    },

    # ==================== 麦克斯韦-Floquet D4/V4 物理等变算子 V2（FNO2dMaxwellV2） ====================
    "A6maxwell_v2": {
        "model": "fno_maxwell_v2",
        "MODEL_NAME": "fno_maxwell_v2",
        "SAVE_DIR":   _abs("../checkpoints/A6maxwell_v2"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {
            "poly_k": 8, "free_body": True, "poly_head_hidden": 192,
            "raw_frame": True, "pol_equivariant": True,
            "v4_equivariant": True, "v4_reynolds_eval": True,
            "floquet_split": True, "tem_dc": True, "polar_dc": True,
            "maxwell_head": True,
        },
    },
    # V2 消融 1：关闭 V4 二面体群反射等变与雷诺投影（仅保留 Z2 极化等变 + V2 头）
    "A6maxwell_v2_nov4": {
        "model": "fno_maxwell_v2",
        "MODEL_NAME": "fno_maxwell_v2",
        "SAVE_DIR":   _abs("../checkpoints/A6maxwell_v2_nov4"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {
            "poly_k": 8, "free_body": True, "poly_head_hidden": 192,
            "raw_frame": True, "pol_equivariant": True,
            "v4_equivariant": False, "v4_reynolds_eval": False,
            "floquet_split": True, "tem_dc": True, "polar_dc": True,
            "maxwell_head": True,
        },
    },
    # V2 宽频带缺口泛化测试（GAP15 / GAP20）
    "A6maxwell_v2_GAP15": {
        "model": "fno_maxwell_v2",
        "MODEL_NAME": "fno_maxwell_v2",
        "SAVE_DIR":   _abs("../checkpoints/A6maxwell_v2_GAP15"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [10.0, 10.5],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {
            "poly_k": 8, "free_body": True, "poly_head_hidden": 192,
            "raw_frame": True, "pol_equivariant": True,
            "v4_equivariant": True, "v4_reynolds_eval": True,
            "floquet_split": True, "tem_dc": True, "polar_dc": True,
            "maxwell_head": True,
        },
    },
    "A6maxwell_v2_GAP20": {
        "model": "fno_maxwell_v2",
        "MODEL_NAME": "fno_maxwell_v2",
        "SAVE_DIR":   _abs("../checkpoints/A6maxwell_v2_GAP20"),
        "TRAIN_CATEGORIES": ["circle","circlering","cross","cross_complex","double_rect","open_ring","rect_ring","rectangle"],
        "LR_WARMUP_EPOCHS": 2,
        "TRAIN_FREQS": None,
        "HOLDOUT_FREQS": [9.5, 10.0, 10.5],
        "FREQ_INTERP_PROB": 0.0,
        "PHASE_FRAME": True,
        "EMA_DECAY": 0.995,
        "MODEL_EXTRA": {
            "poly_k": 8, "free_body": True, "poly_head_hidden": 192,
            "raw_frame": True, "pol_equivariant": True,
            "v4_equivariant": True, "v4_reynolds_eval": True,
            "floquet_split": True, "tem_dc": True, "polar_dc": True,
            "maxwell_head": True,
        },
    },
}


# =============================================================================
# 基线代次配方（GEN1）：让"这一代基线用哪套损失"可被机器校验
# =============================================================================
"""基线代次配方。

动机：用户定的流程是「先固定一版合理的损失，**全部模型集体跑一次**得到基线；
之后只迭代主模型，等主模型提升后再把新损失同步给全部模型跑一次」。这条流程
最怕的是**单侧漂移**：

  - 改了 LOSS_WEIGHT_VECTOR 却只重跑了几个模型 → 基线与主模型不再同代，
    对比出来的"提升"可能只是权重不同，而终端上完全看不出来；
  - 反过来，有人改回旧值再跑 → 新老 run 混在一张表里。

`tools/train.py` 在启动时调 `check_recipe()`：只要当前生效的 `LOSS_KEYS`
与下面这组值对不上就**直接报错**，除非显式传 `--allow-recipe-drift`。

**升代次怎么做**（就三步）：
  1. 把下面 `GEN1` 的值改成新的一代；`GEN1_NAME` 同步改（如 "gen2"）；
  2. 全部模型重跑一次；
  3. `python -m tools.report` 会按场景/损失指纹把两代分成两张表，
     跨代不会混在一起。

注意：`LOSS_CH_WEIGHT_POWER` 在 `LOSS_KEYS` 里，所以它也算"食谱的一部分"，
改了同样要求重跑——这是有意的，它确实改变训练目标。
"""

GEN1_NAME = "gen1"

# 与 configs/base.py 的默认值**逐字相同**，此处显式写出来是为了让它成为一个
# 可校验的契约，而不是"碰巧等于默认值"。改损失请改这里并升代次。
GEN1 = {
    "LOSS_SPACE": "physical",
    "USE_LOSS_CURRICULUM": False,
    "DETAIL_LOSS_START_EPOCH": 40,
    "DETAIL_LOSS_RAMP_EPOCHS": 200,
    "LOSS_ENABLE_VECTOR": [1, 0, 1, 0, 0, 0, 0, 1, 0],
    "LOSS_WEIGHT_VECTOR": [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "LOSS_CH_WEIGHT_POWER": 0.5,
    "FIELD_H_GROUP_WEIGHT": 1.5,
    "DC_REL_WEIGHT": 1.0,
    "AC_REL_WEIGHT": 1.0,
    "FIELD_DIR_ANGLE_WEIGHT": 2.0,
    "FIELD_COMPLEX_PHASE_WEIGHT": 2.0,
    "FIELD_TOTAL_MAG_WEIGHT": 4,
    "AUX_COMPLEX_CH_DC_WEIGHT": 1.0,
    "AUX_COMPLEX_CH_AC_WEIGHT": 1.0,
    "AUX_COMPLEX_CH_NORMALIZE_BY_TOTAL": True,
    "AUX_COMPLEX_CH_DC_DENOM_FLOOR_RATIO": 0.1,
    "AUX_COMPLEX_CH_LOG_COMPRESS": True,
    # L5/L6 频域子权重。本代 L5/L6 都没启用（见 LOSS_ENABLE_VECTOR 的下标 4/5），
    # 所以这两个值在当前是**惰性**的——但照样记进来：否则哪天有人单独改它们、
    # 又碰巧启用了 L5/L6，那就成了不受 check_recipe 监管的改动。
    "LOWFREQ_PATCH": 4,
    "HIGHFREQ_CUTOFF": 6,
}

# 当前应当生效的配方（切代次就改这一行）
ACTIVE_RECIPE = GEN1_NAME
RECIPES = {GEN1_NAME: GEN1}


def check_recipe(base_module=None) -> dict:
    """校验当前生效的损失配置属于哪一代；不属于任何一代则抛错。

    Returns:
        {"name": 代次名, "drift": [(键, 期望值, 实际值), ...]}
        `drift` 为空表示完全吻合。
    """
    from configs.scenario import LOSS_KEYS

    if base_module is None:
        from configs import base as base_module

    expected = RECIPES[ACTIVE_RECIPE]
    drift = []
    for k in LOSS_KEYS:
        exp = expected.get(k)
        act = getattr(base_module, k, None)
        if exp is None:
            continue
        # 列表逐元素比，避免 == 对 list 的坑（这里其实安全，但保持一致）
        if isinstance(exp, list):
            same = (isinstance(act, list) and len(act) == len(exp)
                    and all(a == b for a, b in zip(act, exp)))
        else:
            same = (act == exp)
        if not same:
            drift.append((k, exp, act))
    return {"name": ACTIVE_RECIPE, "drift": drift}

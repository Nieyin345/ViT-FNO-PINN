# -*- coding: utf-8 -*-
"""场景 / 损失 的划分与指纹。

## 为什么需要这个

主模型的损失函数**本来就是要反复改的变量**，所以不能把配置整体冻结
（那样等于把研究方法锁死）。但"改了什么"必须能被机器认出来，否则：

  - 换了一版损失，新旧数字混在一张表里，看着可比其实不可比；
  - 改了一个训练条件（比如归一化方式），旧 checkpoint 已经加载不进去了，
    而汇总表里还留着它们的分数，没人会发现。

于是把 `configs/base.py` 的大写配置项分成两桶：

    SCENARIO  —— 数据与训练条件。**必须一致**，两个 run 才可比。
                 改这里 = 换了一代，旧 run 必须重跑。
    LOSS      —— 损失食谱。**允许随时改**，改完仍可与旧 run 比（见下）。
                 记录下来只为了解释"提升是哪一版损失带来的"。

## 为什么损失可以随便改而结果仍可比

因为保存/早停指标 `val_geom` / `val_rel` **不读任何损失配置**：
`Trainer.geom_metric` 调用 `field_geom_loss_eh(pred, true)`，用的是子权重的默认值
1/1/1，而不是 `FIELD_DIR_ANGLE_WEIGHT` 等训练侧旋钮；`rel_l2_error` 调用
`channel_group_balanced_l2(pred, true)`，`h_boost` 走默认 1.0 而非
`FIELD_H_GROUP_WEIGHT`。所以损失怎么改，量出来的尺子不变。

（`tools/smoke.py` 的 `check_metric_independence` 会守住这条不变式。）

## 两桶之外

模型结构（`WIDTH` / `MODES1` / `HEAD_MODE` / `MODEL_EXTRA` ...）两桶都不进——
它就是被对比的那个变量，已由 `ModelConfig.to_dict()` 随 checkpoint 记录。
"""

import hashlib
import json
import os

# =============================================================================
# 一、场景：必须一致
# =============================================================================
# 数据与划分
SCENARIO_KEYS = (
    "DATA_DIR",
    "TRAIN_CATEGORIES",
    "TRAIN_FREQS",
    "DATASET_POLARIZATIONS",
    "TARGET_D_OVER_LAMBDA",
    "SMOOTH_SIGMA",
    "USE_SCATTER_FIELD",
    "BACKGROUND_PARAMS_PATH",
    "VAL_RATIO",
    "SHUFFLE_SEED",
    # 训练条件
    "N_EPOCHS",
    "BATCH_SIZE",
    "LR",
    "LR_WARMUP_EPOCHS",
    "WEIGHT_DECAY",
    "GRAD_CLIP_MAX_NORM",
    "EARLY_STOP_PATIENCE",
    "EARLY_STOP_MIN_DELTA",
    "FREQ_INTERP_PROB",
    # 输出归一化：改这里旧 checkpoint 直接加载不进去，属于换了一代
    "USE_OUTPUT_STANDARDIZATION",
    "OUTPUT_NORM_MODE",
    "OUTPUT_STANDARDIZATION_EPS",
)

# =============================================================================
# 二、损失食谱：允许改，记录下来
# =============================================================================
LOSS_KEYS = (
    "LOSS_SPACE",
    "USE_LOSS_CURRICULUM",
    "DETAIL_LOSS_START_EPOCH",
    "DETAIL_LOSS_RAMP_EPOCHS",
    "LOSS_ENABLE_VECTOR",
    "LOSS_WEIGHT_VECTOR",
    # L1 的 H 组加权
    "FIELD_H_GROUP_WEIGHT",
    # L2/L7 的 DC/AC 子权重
    "DC_REL_WEIGHT",
    "AC_REL_WEIGHT",
    # L3 几何损失子权重
    "FIELD_DIR_ANGLE_WEIGHT",
    "FIELD_COMPLEX_PHASE_WEIGHT",
    "FIELD_TOTAL_MAG_WEIGHT",
    # L4 稳定版通道 DC/AC
    "AUX_COMPLEX_CH_DC_WEIGHT",
    "AUX_COMPLEX_CH_AC_WEIGHT",
    "AUX_COMPLEX_CH_NORMALIZE_BY_TOTAL",
    "AUX_COMPLEX_CH_DC_DENOM_FLOOR_RATIO",
    "AUX_COMPLEX_CH_LOG_COMPRESS",
    # L5/L6 频域
    "LOWFREQ_PATCH",
    "HIGHFREQ_CUTOFF",
)

# 损失项编号 -> 名字。LOSS_*_VECTOR 的下标含义只存在于 configs/base.py 的注释里，
# 这里复制一份用于把向量渲染成人能读的形式（report 输出用）。
LOSS_NAMES = (
    "L1 total_rel       E/H分组总场相对L2",
    "L2 directional     三方向幅值DC/AC",
    "L3 geom            全场几何(方向角/相位/幅值)",
    "L4 ch_dcac_stable  12通道DC/AC(稳定版)",
    "L5 lowfreq         低频频谱约束",
    "L6 highfreq        高频抑制",
    "L7 ch_dcac_raw     12通道DC/AC(原始)",
    "L8 ch_rel6_mean    逐通道分组相对L2",
    "L9 pinn_maxwell    麦克斯韦残差",
)


def _jsonable(v):
    """把配置值转成可稳定序列化的形式。"""
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in sorted(v.items())}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


def snapshot(keys, base_module=None) -> dict:
    """从 `configs.base` 的模块全局里取一组键的当前值。

    必须在 `apply_experiment()` 与 `apply_cli_overrides()` **之后**调用，
    否则拿到的是默认值而不是本次运行真正用的值。
    """
    if base_module is None:
        from configs import base as base_module
    out = {}
    for k in keys:
        if hasattr(base_module, k):
            out[k] = _jsonable(getattr(base_module, k))
    return out


def fingerprint(snap: dict) -> str:
    """把快照压成短哈希（12 位十六进制）。"""
    blob = json.dumps(snap, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def split_fingerprint(train_idx, val_idx) -> str:
    """训练/验证划分本身的指纹。

    这是"同样的训练集、同样的验证"最直接的证据——比哈希配置键更硬，
    因为即使配置没动，数据文件增删改也会让划分变化，而配置哈希看不出来。
    """
    payload = {
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# =============================================================================
# 三、架构指纹：判断"这条旧结果还算不算数"
# =============================================================================
def arch_fingerprint(model) -> str:
    """模型**架构**的指纹：state_dict 的 (键, 形状) 签名。

    这是"要不要重跑"的机械判据。旧 checkpoint 能否被新代码 `strict=True`
    加载进来，只取决于 (键, 形状) 集合是否一致，所以直接哈希它。

    它**只看架构，不看权重值**：同一架构重训，指纹不变（这是对的——
    重训得到的是同一把尺子下的另一个分数，可比）。

    改了模型类（增删层、改通道数）→ 指纹变 → 那是新的一代，旧结果作废。
    只改损失 / 学习率策略 → 指纹不变 → 旧结果仍然有效，不必重跑。
    """
    items = [(k, tuple(v.shape)) for k, v in model.state_dict().items()]
    blob = json.dumps(sorted(items), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def arch_fingerprint_from_cfg(model_name, cfg_dict) -> str:
    """从**存档里记下的** model_name + model_config 重算架构指纹。

    这样 `tools/report.py` 不必持有 `.pth` 文件也能判断某条历史结果是否过期：
    用当前代码重建那个模型，若指纹与存档记录的不一致，说明模型类动过。

    Raises:
        Exception: 存档的配置在当前代码下已经构造不出来了（改了必填参数、
            删了模型、`extra` 语义变了）——这**同样**是"该代已作废"的信号，
            调用方应捕获并标记为过期，而不是让整个报告崩掉。
    """
    from core.config import ModelConfig
    from core.registry import build_model

    return arch_fingerprint(build_model(model_name, ModelConfig.from_dict(cfg_dict)))


def scenario(base_module=None, train_idx=None, val_idx=None) -> dict:
    """完整场景描述（含划分指纹）。"""
    snap = snapshot(SCENARIO_KEYS, base_module)
    sfp = split_fingerprint(train_idx, val_idx) if train_idx is not None else None
    return {
        "keys": snap,
        "split": sfp,
        "fingerprint": fingerprint({"keys": snap, "split": sfp}),
    }


def loss_recipe_from_snapshot(snap: dict) -> dict:
    """给一份**已存下的**损失键快照补上 `_active` 与 `fingerprint`。

    从快照算而不是从当前 config 算，是为了让 `tools/report.py` 能重新解读
    历史 run 的食谱——那些 run 是在别的配置下跑的，当前 config 代表不了它们。

    注意 `fingerprint` 只覆盖配置键，**不含 `_active`**：`_active` 是从
    `LOSS_ENABLE_VECTOR` / `LOSS_WEIGHT_VECTOR` 推出来的展示层，
    把它算进指纹会让"同一份损失"因为展示差异而分成两版。
    """
    snap = dict(snap)
    enable = snap.get("LOSS_ENABLE_VECTOR") or []
    weight = snap.get("LOSS_WEIGHT_VECTOR") or []
    active = []
    for i, (e, w) in enumerate(zip(enable, weight)):
        if int(e) != 0 and float(w) != 0.0:
            name = LOSS_NAMES[i].split()[1] if i < len(LOSS_NAMES) else f"idx{i}"
            active.append(f"{name}={w}")
    snap["_active"] = active
    snap["fingerprint"] = fingerprint({k: v for k, v in snap.items() if k != "_active"})
    return snap


def loss_recipe(base_module=None) -> dict:
    """损失食谱快照 + 人类可读的启用项列表 + 指纹。"""
    return loss_recipe_from_snapshot(snapshot(LOSS_KEYS, base_module))


def _config_only(snap: dict) -> dict:
    """剥掉展示层键（`_active` / `fingerprint`），只留真正的配置项。

    否则差异列表里会出现 "fingerprint: aaa -> bbb" 这种噪声——
    它本来就是由其它项算出来的，不是独立的一项配置。
    """
    return {k: v for k, v in snap.items()
            if not k.startswith("_") and k != "fingerprint"}


def diff(a: dict, b: dict) -> list:
    """两组快照的差异，返回 [(键, a值, b值)]。用于告诉用户"到底改了什么"。"""
    a, b = _config_only(a), _config_only(b)
    out = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out.append((k, a.get(k), b.get(k)))
    return out


def render_loss_recipe(recipe: dict) -> str:
    """把损失食谱渲染成多行可读文本。"""
    lines = [f"  指纹 {recipe.get('fingerprint', '?')}"]
    lines.append(f"  计算空间 {recipe.get('LOSS_SPACE')}"
                 + ("（课程式调度开启）" if recipe.get("USE_LOSS_CURRICULUM") else ""))
    active = recipe.get("_active") or []
    lines.append("  启用项  " + ("、".join(active) if active else "（无）"))
    lines.append("  L3 子权重  方向角={} 相位={} 幅值={}".format(
        recipe.get("FIELD_DIR_ANGLE_WEIGHT"),
        recipe.get("FIELD_COMPLEX_PHASE_WEIGHT"),
        recipe.get("FIELD_TOTAL_MAG_WEIGHT")))
    lines.append("  H 组加权  {}".format(recipe.get("FIELD_H_GROUP_WEIGHT")))
    return "\n".join(lines)

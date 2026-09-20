# -*- coding: utf-8 -*-
"""模型结构配置：全仓库唯一的结构参数来源。

重构背景：原先 6 个模型的 `__init__` 参数名、顺序、默认值互不相同
（`FNO2dBaseline` 的 `out_channels` 甚至排在 `in_channels` 前面；`FNO2dFiLM`
收了 `freq_dim` 却在内部写死 32），且构造时要读全局 config
（`FNO2d` 读 `USE_SINGLE_OUTPUT_HEAD`、`FNO2dV2` 读 `PROJ_DROPOUT`、
`SpectralConv2d` 读 6 个消融开关）。

现在统一为：

    ModelConfig  —— 结构 + 条件化 + 输出头 + 频域层开关
    SpectralConfig —— 频域卷积层的消融开关

所有模型都是 `def __init__(self, cfg: ModelConfig)`，构造过程对全局配置零依赖。
模型专属的新参数走 `cfg.extra` 字典，无需改动本文件。
"""

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, Tuple


# 输出头模式（含义见 models/fno2d.py 等）：
#   single   - 共享特征直接映射到全部输出通道（FNO2d 默认，靠输出归一化平衡 E/H 量级差）
#   em_split - 3 个 E 头 + 3 个 H 头，H 显式依赖 E 的预测（FNO2d 旧模式）
#   grouped  - 每 2 个通道（一个复分量）一个 DirectionHead（FNO2dFiLM）
HEAD_MODES = ("single", "em_split", "grouped")

# 核条件化方式（FNO2dV2）
KERNEL_COND_MODES = ("film", "lora", "none")


@dataclass(frozen=True)
class SpectralConfig:
    """频域卷积层的消融开关（原 config.py 的 SPECTRAL_* 系列）。

    原先这些开关在 `spectral_conv2d.py` 模块级 `import config` 后于 `__init__`
    里读取，导致模型构造与全局配置耦合。现在显式传入。
    """
    attention: bool = True            # 关掉后等价于纯 FNO 频域卷积
    freq_conditioning: bool = True    # 超网络是否调制 Q
    geom_conditioning: bool = True    # 超网络是否调制 K
    gamma_init: float = 0.1           # 注意力残差门控初值
    gamma_fixed: bool = False         # True 则 gamma 不参与训练
    decay: bool = False               # 频域 Butterworth 衰减
    decay_k0: float = 6.0
    decay_order: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "SpectralConfig":
        return _build(cls, d or {})


@dataclass(frozen=True)
class ModelConfig:
    """模型结构配置。

    结构参数默认值对齐 configs/base.py 的默认实验（MODES1/MODES2/WIDTH/N_LAYERS）。
    """
    # ---- 结构 ----
    modes1: int = 8
    modes2: int = 8
    width: int = 64
    n_layers: int = 5
    in_channels: int = 2          # 原始输入通道（SDF + Mask），不含模型内拼的额外通道
    out_channels: int = 12
    freq_dim: int = 32            # 频率/几何嵌入维度
    padding: int = 7

    # ---- 条件化 ----
    cond_channels: int = 2        # 空间广播式条件（baseline/unet）的通道数
    use_pol_cond: bool = True     # FNO2dV2：极化走独立编码分支
    use_sine_input: bool = True   # FNO2dV2：频率以空间正弦嵌入通道注入
    kernel_cond: str = "film"     # FNO2dV2: film | lora | none
    lora_rank: int = 4

    # ---- 输出头 ----
    head: str = "single"          # single | em_split | grouped
    dropout: float = 0.0          # 输出头 dropout（原 PROJ_DROPOUT）

    # ---- 频域层 ----
    spectral: SpectralConfig = field(default_factory=SpectralConfig)

    # ---- 扩展口：新模型的自定义参数放这里，本文件无需改动 ----
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.head not in HEAD_MODES:
            raise ValueError(f"未知 head 模式: {self.head}，可选 {HEAD_MODES}")
        if self.kernel_cond not in KERNEL_COND_MODES:
            raise ValueError(f"未知 kernel_cond: {self.kernel_cond}，可选 {KERNEL_COND_MODES}")
        if self.out_channels % 2 != 0:
            raise ValueError(f"out_channels 必须是偶数（实部+虚部成对），得到 {self.out_channels}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d) -> "ModelConfig":
        return _build(cls, d or {})

    # ---- 便捷访问 ----
    @property
    def n_heads(self) -> int:
        """复分量个数（每 2 个通道一个）。"""
        return self.out_channels // 2

    def replace(self, **kw) -> "ModelConfig":
        """返回替换若干字段后的新配置（frozen，故不原地改）。"""
        d = self.to_dict()
        d.update(kw)
        return ModelConfig.from_dict(d)


def _build(cls, d: Dict[str, Any]):
    """从 dict 构造 dataclass，处理嵌套 SpectralConfig 与未知键。

    未知键不报错：它们被塞进 `extra`，这样旧 checkpoint 或新模型的额外字段
    都能往返而不炸。`extra` 里显式给的键优先。
    """
    d = dict(d)
    known = {f.name for f in fields(cls)}
    nested = {}
    if "spectral" in d:
        sp = d.pop("spectral")
        nested["spectral"] = sp if isinstance(sp, SpectralConfig) else SpectralConfig.from_dict(sp)
    extra = dict(d.pop("extra", {}) or {})
    unknown = {k: v for k, v in d.items() if k not in known}
    extra.update(unknown)
    clean = {k: v for k, v in d.items() if k in known}
    if "extra" in known:
        clean["extra"] = extra
    clean.update(nested)
    return cls(**clean)

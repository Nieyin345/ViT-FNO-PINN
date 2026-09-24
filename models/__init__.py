# -*- coding: utf-8 -*-
"""模型集合：import 本包即触发所有模型的自注册。

**加模型只需两步**：
  1. 新建 models/<你的模型>.py，用 `@register("名字")` 装饰模型类，
     签名必须是 `def __init__(self, cfg: ModelConfig)`，并继承 FNOBase（或 nn.Module）。
  2. 在下面加一行 `from . import <你的模型模块>`。

之后 训练 / 评估 / GUI / S11 / 传播 / 对比实验 六条链路全部自动认得它。
"""

from . import fno2d            # noqa: F401  fno_attention
from . import fno2d_baseline   # noqa: F401  fno_baseline
from . import fno2d_disp       # noqa: F401  fno_disp（色散坐标谱卷积）
from . import fno2d_film       # noqa: F401  fno_film
from . import fno2d_lora       # noqa: F401  fno_lora
from . import fno2d_v2         # noqa: F401  fno_v2
from . import unet2d           # noqa: F401  unet

# ---- 对比基线族（不同的"混合范围 / 混合域 / 算子族"）----
# 局部 vs 全局 × 空间域 vs 频域，构成完整的对照网：
#   mlp     感受野 1×1，无任何混合        —— 参照系
#   cnn     局部空间卷积（全程 64×64）     —— 局部混合
#   vit     空间域全局自注意力             —— 全局混合（空间域）
#   afno    频域逐频率对共享权重 MLP       —— 全局混合（频域，弱于 FNO 的谱卷积）
#   deeponet 显式 branch(几何)⊗trunk(坐标) —— 另一条算子学习路线
from . import mlp2d            # noqa: F401  mlp
from . import cnn2d            # noqa: F401  cnn
from . import vit2d            # noqa: F401  vit
from . import afno2d           # noqa: F401  afno
from . import deeponet2d       # noqa: F401  deeponet

from core.registry import available, build_model, get, register  # noqa: F401

__all__ = ["available", "build_model", "get", "register"]

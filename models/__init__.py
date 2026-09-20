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
from . import fno2d_film       # noqa: F401  fno_film
from . import fno2d_lora       # noqa: F401  fno_lora
from . import fno2d_v2         # noqa: F401  fno_v2
from . import unet2d           # noqa: F401  unet

from core.registry import available, build_model, get, register  # noqa: F401

__all__ = ["available", "build_model", "get", "register"]

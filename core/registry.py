# -*- coding: utf-8 -*-
"""模型注册表：全仓库唯一的"模型名 → 模型类"映射。

重构背景：原先模型构造靠"模型名字符串"在**四**处分叉，且支持的集合互相不一致：
  - tools/train.py 旧 MODEL_REGISTRY        —— 6 个（唯一含 fno_v2）
  - tools/ui_eval.py 旧 _MODEL_REGISTRY     —— 5 个，未知名字静默退回 FNO2d
  - field2field/propagate_fno_prediction.py —— 5 个（缺 fno_v2）
  - tools/calc_s11.py                       —— 1 个（写死 FNO2dLoRA）

现在模型在**自己的文件里**自注册，消费方只调 `build_model(name, cfg)`。
未知名字一律抛错（不再静默降级），并列出现有可选值。
"""

from typing import Dict, List, Type

import torch.nn as nn

from .config import ModelConfig

_REGISTRY: Dict[str, Type[nn.Module]] = {}
_LOAD_ATTEMPTED = False


def _ensure_loaded() -> None:
    """确保各模型模块已被 import（即已完成自注册）。

    模型靠 `models/__init__.py` 逐个 import 来触发 `@register`。消费方若忘了
    `import models`，注册表就是空的，`build_model` 会报出令人困惑的
    "未知模型: 'fno_v2'，可选: []"。这里做一次惰性兜底导入。

    用 `_LOAD_ATTEMPTED` 而非 `if not _REGISTRY` 做守卫：如此即使 models 导入失败
    （或确实一个模型都没注册），也只尝试一次，不会无限递归。
    """
    global _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return
    _LOAD_ATTEMPTED = True
    try:
        import models  # noqa: F401
    except ImportError:
        pass



def register(name: str):
    """类装饰器：把模型注册到全局表，并记下它的注册名。"""

    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(
                f"模型名重复: {name!r} 已注册为 {_REGISTRY[name].__name__}，"
                f"现在又用于 {cls.__name__}"
            )
        _REGISTRY[name] = cls
        cls.model_name = name
        return cls

    return deco


def available() -> List[str]:
    """已注册的模型名（排序）。"""
    _ensure_loaded()
    return sorted(_REGISTRY.keys())


def get(name: str) -> Type[nn.Module]:
    _ensure_loaded()
    if name not in _REGISTRY:
        raise ValueError(f"未知模型: {name!r}，可选: {available()}")
    return _REGISTRY[name]


def build_model(name: str, cfg: ModelConfig) -> nn.Module:
    """按名字构造模型。所有模型的统一签名是 `__init__(cfg: ModelConfig)`。"""
    cls = get(name)
    model = cls(cfg)
    model.model_name = name
    return model


def _clear_for_tests():
    """仅供测试：清空注册表。"""
    _REGISTRY.clear()

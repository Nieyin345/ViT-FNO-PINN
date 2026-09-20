# -*- coding: utf-8 -*-
"""统一模型基座。

    core.config      ModelConfig / SpectralConfig —— 结构配置
    core.base        Cond / FNOBase             —— 统一块协议与骨架
    core.encoders    条件编码器                  —— 全模型共用
    core.spectral    频域卷积层                  —— 构造参数化，去掉 config 全局
    core.registry    模型注册表 + build_model    —— 唯一构造入口
    core.checkpoint  checkpoint 存取             —— 唯一加载路径，strict=True
"""

from .base import Cond, FNOBase
from .config import ModelConfig, SpectralConfig
from .registry import available, build_model, get, register

__all__ = [
    "Cond", "FNOBase",
    "ModelConfig", "SpectralConfig",
    "available", "build_model", "get", "register",
]

# -*- coding: utf-8 -*-
"""可调参数。

    configs.base         全局默认值（数据 / 模型结构 / 训练 / 损失）
    configs.experiments  实验覆盖表（每个实验只写与默认值不同的字段）

**模型结构参数**的规范形式是 `core.config.ModelConfig`，
由 `configs.base.model_config_from_globals()` 从这里的扁平全局值构造出来。
消费方一律调那个函数，不要各自拼 kwargs 字典。
"""

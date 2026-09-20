# -*- coding: utf-8 -*-
"""checkpoint 存取：全仓库唯一的模型加载路径。

重构背景（这是本轮修复的核心症状）：
  原先模型是先用全局 config 构造出来，再把 checkpoint 滤**缩**到这个模型的
  key/shape 集合上，然后 `load_state_dict(..., strict=False)`，只报告
  "匹配 N 层，跳过 M 层"。于是一个架构错的运行**看起来像成功的运行**。

  实测到的具体 bug：fno_v2 的 checkpoint 交给 eval 的 `_infer_model_kwargs` 时，
  它从 `lift.0.weight` 读出 in_channels=6（因为 FNO2dV2 的 lift 内部已拼了 4 个
  正弦通道），而 `FNO2dV2.__init__` 的 in_channels 语义是 2（正弦通道它在内部加），
  于是 57/58 层匹配、`lift` 被静默随机初始化。

现在把构建配置随权重一起存（`model_config`），加载时用它重建模型，`strict=True`。
架构不符直接抛错。
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from data.norm_utils import OutputNormalizer

from .config import ModelConfig
from .registry import build_model

FORMAT_VERSION = 2


@dataclass
class LoadedCheckpoint:
    """一次加载的完整结果。"""
    model: nn.Module
    cfg: ModelConfig
    normalizer: Optional[OutputNormalizer]
    meta: Dict[str, Any]

    @property
    def epoch(self) -> Optional[int]:
        return self.meta.get("epoch")

    @property
    def val_rel(self) -> Optional[float]:
        return self.meta.get("val_rel")

    @property
    def val_geom(self) -> Optional[float]:
        return self.meta.get("val_geom")

    @property
    def model_name(self) -> str:
        return self.meta.get("model_name", getattr(self.model, "model_name", "?"))


def save(path: str, model: nn.Module, cfg: ModelConfig, *,
         optimizer=None, epoch: int = 0, val_loss=None, val_rel=None,
         val_geom=None, normalizer: Optional[OutputNormalizer] = None,
         extra: Optional[Dict[str, Any]] = None) -> str:
    """写出 checkpoint，包含重建模型所需的全部信息。"""
    model_name = getattr(model, "model_name", None)
    if model_name is None:
        raise ValueError(
            f"模型 {type(model).__name__} 没有 model_name 属性——"
            "它没有通过 core.registry.register 注册，无法安全存取。"
        )

    ckpt: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "model_config": cfg.to_dict(),
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_rel": val_rel,
        "val_geom": val_geom,
        "use_output_standardization": normalizer is not None and normalizer.mode != "none",
        "normalizer": normalizer.to_dict() if normalizer is not None else None,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        ckpt.update(extra)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(ckpt, path)
    return path


def read_meta(path: str) -> Dict[str, Any]:
    """只读 checkpoint 的元信息（不建模型），用于列表/筛选。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "model_name": ckpt.get("model_name"),
        "model_config": ckpt.get("model_config"),
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "val_rel": ckpt.get("val_rel"),
        "val_geom": ckpt.get("val_geom"),
        "format_version": ckpt.get("format_version"),
    }


def load(path: str, map_location: str = "cpu", *, eval_mode: bool = True,
         quiet: bool = False) -> LoadedCheckpoint:
    """加载 checkpoint，按其中记录的配置重建模型，并 `strict=True` 载入权重。

    Raises:
        FileNotFoundError: 路径不存在
        KeyError:           checkpoint 缺少 model_name / model_config（格式 v1，需重训）
        RuntimeError:      权重与重建出的架构不匹配（这是**期望行为**，不是 bug）
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}")

    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if "model_config" not in ckpt or "model_name" not in ckpt:
        raise KeyError(
            f"{path} 是旧格式 checkpoint（缺 model_name/model_config），"
            f"无法安全重建模型。本次重构后旧权重已作废，请重新训练。"
            f"（现有键: {sorted(ckpt.keys())}）"
        )

    cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = build_model(ckpt["model_name"], cfg)

    state = _strip_module_prefix(ckpt["model_state_dict"])
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"权重与模型架构不匹配: {path}\n"
            f"  模型: {ckpt['model_name']}\n"
            f"  配置: width={cfg.width} layers={cfg.n_layers} "
            f"modes={cfg.modes1}x{cfg.modes2} out={cfg.out_channels} "
            f"head={cfg.head} kernel_cond={cfg.kernel_cond}\n"
            f"  原始错误: {e}"
        ) from e

    normalizer = OutputNormalizer.from_checkpoint(ckpt)
    if eval_mode:
        model.eval()

    lc = LoadedCheckpoint(model=model, cfg=cfg, normalizer=normalizer, meta=ckpt)
    if not quiet:
        print(f"[Checkpoint] {os.path.basename(path)} → {lc.model_name} "
              f"(epoch={lc.epoch}, val_geom={lc.val_geom}, val_rel={lc.val_rel})")
    return lc


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """去掉 DataParallel 的 `module.` 前缀。"""
    if any(k.startswith("module.") for k in state):
        return {k[len("module."):] if k.startswith("module.") else k: v
                for k, v in state.items()}
    return state


def missing_or_unexpected(path: str, model: nn.Module) -> List[str]:
    """诊断用：列出 checkpoint 与模型的 key 差异（不加载）。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = _strip_module_prefix(ckpt.get("model_state_dict", {}))
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    diffs = []
    for k in sorted(model_keys - ckpt_keys):
        diffs.append(f"  缺 (模型有/ckpt 无): {k}")
    for k in sorted(ckpt_keys - model_keys):
        diffs.append(f"  多 (ckpt 有/模型无): {k}")
    return diffs

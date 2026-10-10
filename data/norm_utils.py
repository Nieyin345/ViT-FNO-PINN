# -*- coding: utf-8 -*-
"""输出归一化工具：zscore 与 log1p 两种模式。

背景：模型输出 12 通道（Ex/Ey/Ez/Hx/Hy/Hz 各实虚部），电场（std ~ 1e2~1e3）与
磁场（Hx ~ 3，Hy/Hz ~ 1）尺度相差约 3 个数量级；且为散射场，通道内有正负值、
大量接近 0 的取值。直接做线性标准化会让小尺度通道的细微结构被大尺度通道淹没。

- zscore: x -> (x - mean) / std             （逐通道线性标准化，旧行为）
- log:    x -> sign(x) * log1p(|x| / scale) （逐通道对数压缩动态范围，保留符号），
          再对变换结果做 z-score（logmean/logstd），使各通道训练目标 ~N(0,1)。

反变换（log）：y -> sign(y) * expm1(|y|) * scale。
- "none": 恒等变换，用于调试/对比。

本模块同时支持 numpy（数据加载路径，x 为 (C,H,W)）与 torch（训练/推理路径，
x 为 (B,C,H,W)），并提供 checkpoint 存取，保证训练、评估、UI 反归一化一致。
"""

import numpy as np
import torch

MODES = ("zscore", "log", "none")


class OutputNormalizer:
    def __init__(self, mode="zscore", mean=None, std=None, scale=None,
                 logmean=None, logstd=None, eps=1e-8):
        if mode not in MODES:
            raise ValueError(f"不支持的归一化模式: {mode}，可选 {MODES}")
        self.mode = mode
        self.eps = float(eps)

        self.mean = self._arr(mean)
        self.std = self._arr(std)
        self.scale = self._arr(scale)
        self.logmean = self._arr(logmean)
        self.logstd = self._arr(logstd)

        if self.mode == "zscore":
            if self.mean is None or self.std is None:
                raise ValueError("zscore 模式需要 mean/std 参数")
            self.std = np.maximum(self.std, self.eps)
        elif self.mode == "log":
            if self.scale is None or self.logmean is None or self.logstd is None:
                raise ValueError("log 模式需要 scale/logmean/logstd 参数")
            self.scale = np.maximum(self.scale, self.eps)
            self.logstd = np.maximum(self.logstd, self.eps)

    @staticmethod
    def _arr(a):
        return None if a is None else np.asarray(a, dtype=np.float32).reshape(-1)

    @property
    def n_channels(self):
        for a in (self.mean, self.std, self.scale, self.logmean, self.logstd):
            if a is not None:
                return len(a)
        return 0

    # ---------------------------------------------------------------- 参数切片
    def _params(self, ch_slice=None):
        def _sl(a):
            return a if (a is None or ch_slice is None) else a[ch_slice]
        return (_sl(self.mean), _sl(self.std), _sl(self.scale),
                _sl(self.logmean), _sl(self.logstd))

    # ---------------------------------------------------------- numpy（数据加载）
    def normalize_np(self, x, ch_slice=None):
        mean, std, scale, logmean, logstd = self._params(ch_slice)
        if self.mode == "zscore":
            return (x - mean[:, None, None]) / std[:, None, None]
        if self.mode == "log":
            y = np.sign(x) * np.log1p(np.abs(x) / scale[:, None, None])
            return (y - logmean[:, None, None]) / logstd[:, None, None]
        return x

    def denormalize_np(self, y, ch_slice=None):
        mean, std, scale, logmean, logstd = self._params(ch_slice)
        if self.mode == "zscore":
            return y * std[:, None, None] + mean[:, None, None]
        if self.mode == "log":
            y_std = y * logstd[:, None, None] + logmean[:, None, None]
            return np.sign(y_std) * np.expm1(np.abs(y_std)) * scale[:, None, None]
        return y

    # ---------------------------------------------------------- torch（训练/推理）
    def tensor_params(self, device, ch_slice=None):
        mean, std, scale, logmean, logstd = self._params(ch_slice)

        def _t(a):
            if a is None:
                return None
            return torch.from_numpy(a).view(1, -1, 1, 1).to(device)

        return {"mean": _t(mean), "std": _t(std), "scale": _t(scale),
                "logmean": _t(logmean), "logstd": _t(logstd)}

    def normalize(self, x, ch_slice=None):
        if torch.is_tensor(x):
            tp = self.tensor_params(x.device, ch_slice)
            if self.mode == "zscore":
                return (x - tp["mean"]) / tp["std"]
            if self.mode == "log":
                y = torch.sign(x) * torch.log1p(torch.abs(x) / tp["scale"])
                return (y - tp["logmean"]) / tp["logstd"]
            return x
        return self.normalize_np(x, ch_slice)

    def denormalize(self, y, ch_slice=None, tp=None):
        if not torch.is_tensor(y):
            return self.denormalize_np(y, ch_slice)
        if ch_slice is None and getattr(y, "_phys_cache", None) is not None:
            return y._phys_cache
        if tp is None:
            tp = self.tensor_params(y.device, ch_slice)
        if self.mode == "zscore":
            return y * tp["std"] + tp["mean"]
        if self.mode == "log":
            y_std = y * tp["logstd"] + tp["logmean"]
            return torch.sign(y_std) * torch.expm1(torch.abs(y_std)) * tp["scale"]
        return y

    # ------------------------------------------------------------ checkpoint 持久化
    def to_dict(self):
        d = {"output_norm_mode": self.mode}
        if self.mode == "zscore":
            d["output_norm_mean"] = self.mean
            d["output_norm_std"] = self.std
        elif self.mode == "log":
            d["output_norm_scale"] = self.scale
            d["output_norm_logmean"] = self.logmean
            d["output_norm_logstd"] = self.logstd
        return d

    @classmethod
    def from_dict(cls, d, eps=1e-8):
        """从 `to_dict()` 产出的扁平 dict 恢复归一化器。

        无 output_norm_mode 时按 zscore 解释（旧 checkpoint / npz 统计文件）。
        参数缺失时返回 None（而不是抛错），调用方据此判定"无归一化"。
        """
        if not isinstance(d, dict):
            return None
        mode = d.get("output_norm_mode", "zscore")
        if mode == "none":
            return cls(mode="none", eps=eps)
        if mode == "log":
            keys = ("output_norm_scale", "output_norm_logmean", "output_norm_logstd")
            if all(k in d for k in keys):
                return cls(mode="log",
                           scale=d["output_norm_scale"],
                           logmean=d["output_norm_logmean"],
                           logstd=d["output_norm_logstd"],
                           eps=eps)
            return None
        if mode == "zscore":
            if "output_norm_mean" in d and "output_norm_std" in d:
                return cls(mode="zscore",
                           mean=d["output_norm_mean"],
                           std=d["output_norm_std"],
                           eps=eps)
            return None
        raise ValueError(f"不支持的归一化模式: {mode}，可选 {MODES}")

    @classmethod
    def from_checkpoint(cls, ckpt, eps=1e-8):
        """从 checkpoint dict 恢复归一化器。

        新格式把扁平 dict 放在 `ckpt["normalizer"]` 下；旧格式直接平铺在顶层，
        这里一并兼容（旧 checkpoint 无 output_norm_mode 时按 zscore 解释）。
        """
        if not isinstance(ckpt, dict):
            return None
        nested = ckpt.get("normalizer")
        if isinstance(nested, dict):
            return cls.from_dict(nested, eps=eps)
        return cls.from_dict(ckpt, eps=eps)


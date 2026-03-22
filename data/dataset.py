"""
电磁场数据集

inputs.npy  shape: (N, 2, 64, 64)  — channel 0: source one-hot, channel 1: permittivity 归一化 [0,1]
outputs.npy shape: (N, 2, 64, 64)  — channel 0: Ez real, channel 1: Ez imag，已归一化到 [-1,1]

PINN 训练只用 inputs（无监督），outputs 仅在 Stage 3 半监督阶段使用。
n_supervised 控制使用多少个样本的真值标签（从前 n 个样本取）。
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


class EMDataset(Dataset):

    def __init__(self, data_dir: str = "data/em_data_meep", n_total: int = -1):
        inputs = np.load(f"{data_dir}/inputs.npy").astype(np.float32)
        full_n = len(inputs)
        group_size = C.GROUP_SIZE

        # 分层截取：每 group_size 个样本（一种形状）各取前 k 个
        if n_total > 0 and n_total < full_n:
            n_groups = full_n // group_size
            k = n_total // n_groups  # 每组取 k 个
            idx = np.concatenate([
                np.arange(g * group_size, g * group_size + k)
                for g in range(n_groups)
            ])
            inputs = inputs[idx]
        elif n_total > 0:
            inputs = inputs[:n_total]
        self.source       = torch.from_numpy(inputs[:, 0:1])  # (N, 1, H, W)
        self.permittivity = torch.from_numpy(inputs[:, 1:2])  # (N, 1, H, W)
        self.N = len(self.source)

        # 加载真值，构造 supervised mask
        outputs = np.load(f"{data_dir}/outputs.npy").astype(np.float32)
        if n_total > 0 and n_total < full_n:
            outputs = outputs[idx]
        elif n_total > 0:
            outputs = outputs[:n_total]
        self.Ez_true = torch.from_numpy(outputs)   # (N, 2, H, W)，real + imag

        # 归一化系数和等效复数源强度
        scales_path = f"{data_dir}/scales.npy"
        s_effs_path = f"{data_dir}/s_effs.npy"
        if os.path.exists(s_effs_path):
            s_effs = np.load(s_effs_path).astype(np.float32)
            if n_total > 0 and n_total < full_n:
                s_effs = s_effs[idx]
            elif n_total > 0:
                s_effs = s_effs[:n_total]
            self.s_eff = torch.from_numpy(s_effs)               # (N, 2)
        else:
            # 旧数据：默认纯虚数源 s_eff = j*2.14（统计均值）
            self.s_eff = torch.zeros(self.N, 2, dtype=torch.float32)
            self.s_eff[:, 1] = 2.14

        self.supervised_mask = torch.ones(self.N, dtype=torch.bool)  # 全部有监督

        print(f"[Dataset] {self.N} 个样本，有监督: {self.supervised_mask.sum().item()} 个")
        print(f"  source       范围: [{self.source.min():.3f}, {self.source.max():.3f}]")
        print(f"  permittivity 范围: [{self.permittivity.min():.3f}, {self.permittivity.max():.3f}]")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {
            "source":       self.source[idx],           # (1, H, W)
            "permittivity": self.permittivity[idx],      # (1, H, W)
            "Ez_true":      self.Ez_true[idx],           # (2, H, W)
            "supervised":   self.supervised_mask[idx],   # bool scalar
            "s_eff":        self.s_eff[idx],             # (2,) 复数源强度 [real, imag]
        }

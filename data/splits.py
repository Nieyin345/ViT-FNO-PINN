"""
公共 split 工具 — 训练和 eval 共用，保证验证集完全一致

支持两种模式：
  - 普通随机 split（get_splits）
  - 分层 split（get_stratified_splits）：每 group_size 个样本按 val_ratio 比例取验证集
    适用于数据集由多种形状各 N 个拼接而成的情况
"""
import numpy as np


def get_splits(n_total: int, val_ratio: float = 0.2, seed: int = 42):
    """随机 split，返回 (train_indices, val_indices)"""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    n_val = int(n_total * val_ratio)
    val_indices   = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()
    return train_indices, val_indices


def get_stratified_splits(n_total: int, group_size: int = 100,
                          val_ratio: float = 0.2, seed: int = 42):
    """
    分层 split：每 group_size 个样本中按 val_ratio 比例取验证集。
    要求 n_total 是 group_size 的整数倍。

    返回 (train_indices, val_indices)
    """
    assert n_total % group_size == 0, \
        f"n_total={n_total} 必须是 group_size={group_size} 的整数倍"
    n_groups = n_total // group_size
    n_val_per_group = max(1, int(group_size * val_ratio))
    rng = np.random.default_rng(seed)

    val_indices   = []
    train_indices = []
    for g in range(n_groups):
        group = np.arange(g * group_size, (g + 1) * group_size)
        perm  = rng.permutation(group_size)
        val_indices.extend(group[perm[:n_val_per_group]].tolist())
        train_indices.extend(group[perm[n_val_per_group:]].tolist())

    return train_indices, val_indices

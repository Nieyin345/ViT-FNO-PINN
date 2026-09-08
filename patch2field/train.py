# -*- coding: utf-8 -*-
"""全频点电磁场训练脚本"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    matplotlib = None
    plt = None
    HAS_MATPLOTLIB = False

from data.dataset import EMMultiFreqDataset
from models.fno2d import FNO2d
from models.fno2d_baseline import FNO2dBaseline
from models.fno2d_film import FNO2dFiLM
from models.unet2d import UNet2d
from models.fno2d_lora import FNO2dLoRA
from models.losses import field_rel_l2_loss, field_geom_loss, freq_band_loss, dc_ac_rel_loss, dc_ac_directional_mean_loss, highfreq_suppress_loss
import config as C

# 模型注册表，新增对比模型在这里加一行
MODEL_REGISTRY = {
    "fno_attention": FNO2d,
    "fno_baseline":  FNO2dBaseline,
    "fno_film":      FNO2dFiLM,
    "unet":          UNet2d,
    "fno_lora":      FNO2dLoRA,
}


def build_split(dataset):
    """按四个类别分别划分，同一形状的不同频点在同一集合"""
    # 按类别分组
    category_groups = defaultdict(lambda: defaultdict(list))
    for i, s in enumerate(dataset.samples):
        category = s["category"]  # circle/normal, circle/revert, cross_base/normal, cross_base/revert
        shape_name = s["name"]    # circle_1, circle_revert_1, cross_base_1, etc.
        category_groups[category][shape_name].append(i)

    rng = torch.Generator().manual_seed(C.SHUFFLE_SEED)
    train_idx, val_idx = [], []

    print("按类别分层划分:")
    print("=" * 50)
    
    # 对每个类别分别划分
    for category in sorted(category_groups.keys()):
        shapes = sorted(category_groups[category].keys())
        n_shapes = len(shapes)
        n_val_shapes = max(1, int(n_shapes * C.VAL_RATIO))
        n_train_shapes = n_shapes - n_val_shapes
        
        # 随机选择验证集形状
        perm = torch.randperm(n_shapes, generator=rng).tolist()
        val_shapes = set(shapes[i] for i in perm[:n_val_shapes])
        
        # 统计数据量
        train_count = val_count = 0
        for shape_name, idxs in category_groups[category].items():
            if shape_name in val_shapes:
                val_idx.extend(idxs)
                val_count += len(idxs)
            else:
                train_idx.extend(idxs)
                train_count += len(idxs)
        
        print(f"{category}:")
        print(f"  总形状: {n_shapes}, 训练形状: {n_train_shapes}, 验证形状: {n_val_shapes}")
        print(f"  训练数据: {train_count}, 验证数据: {val_count}")

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] 设备: {self.device}")
        os.makedirs(getattr(C, "SAVE_DIR", "checkpoints"), exist_ok=True)
        if not HAS_MATPLOTLIB:
            print("[警告] 未检测到 matplotlib，可视化与训练曲线保存将被跳过。")

        target_freq = None
        if C.TRAIN_FREQS is not None and len(C.TRAIN_FREQS) == 1:
            target_freq = f"{float(C.TRAIN_FREQS[0]):.6f}GHz"

        full_dataset = EMMultiFreqDataset(
            data_dir=C.DATA_DIR,
            target_freq=target_freq,
            output_field="all",
            normalize=False,
            target_d_over_lambda=getattr(C, "TARGET_D_OVER_LAMBDA", 0.1),
            smooth_sigma=getattr(C, "SMOOTH_SIGMA", 0.0),
            use_scatter=getattr(C, "USE_SCATTER_FIELD", False),
            bg_params_path=getattr(C, "BACKGROUND_PARAMS_PATH", "data/blank/background_params.npz"),
        )

        # 按类别过滤
        if C.TRAIN_CATEGORIES is not None:
            full_dataset.samples = [
                s for s in full_dataset.samples
                if any(s["category"] == cat or s["category"].startswith(cat + "/")
                       for cat in C.TRAIN_CATEGORIES)
            ]
            print(f"[数据] 过滤类别: {C.TRAIN_CATEGORIES}，剩余 {len(full_dataset.samples)} 条")

        # 按频率过滤
        if C.TRAIN_FREQS is not None:
            full_dataset.samples = [
                s for s in full_dataset.samples
                if s["freq_ghz"] in C.TRAIN_FREQS
            ]
            print(f"[数据] 过滤频率: {C.TRAIN_FREQS}，剩余 {len(full_dataset.samples)} 条")

        train_ds, val_ds = build_split(full_dataset)

        # 过滤后重建 shape_to_indices，确保索引和 samples 对齐
        from collections import defaultdict
        full_dataset.shape_to_indices = defaultdict(list)
        for i, s in enumerate(full_dataset.samples):
            full_dataset.shape_to_indices[(s["category"], s["name"])].append(i)

        self.use_output_norm = bool(getattr(C, "USE_OUTPUT_STANDARDIZATION", False))
        self.output_norm_mode = str(getattr(C, "OUTPUT_NORM_MODE", "zscore")).lower()
        self.output_normalizer = None
        self._denorm_tp = None
        if self.use_output_norm:
            self.output_normalizer = self._compute_output_normalizer(full_dataset, train_ds)
            full_dataset.set_output_normalization(self.output_normalizer)
            full_dataset.normalize = True
            self._denorm_tp = self.output_normalizer.tensor_params(self.device)
            stats_path = getattr(C, "OUTPUT_NORM_STATS_PATH", f"{C.SAVE_DIR}/output_norm_stats.npz")
            try:
                np.savez(stats_path, **self.output_normalizer.to_dict())
                print(f"[输出标准化] 已保存统计量: {stats_path}")
            except Exception as e:
                print(f"[输出标准化] 统计量保存失败: {e}")
            n_ch = self.output_normalizer.n_channels
            ch_names = ["Ex_r", "Ex_i", "Ey_r", "Ey_i", "Ez_r", "Ez_i",
                        "Hx_r", "Hx_i", "Hy_r", "Hy_i", "Hz_r", "Hz_i"][:n_ch]
            if self.output_norm_mode == "log":
                stat_str = ", ".join(
                    f"{n}(scale={sc:.3e},logmu={lm:.3e},logstd={ls:.3e})"
                    for n, sc, lm, ls in zip(ch_names, self.output_normalizer.scale,
                                             self.output_normalizer.logmean,
                                             self.output_normalizer.logstd)
                )
            else:
                stat_str = ", ".join(
                    f"{n}(mu={m:.3e},std={s:.3e})"
                    for n, m, s in zip(ch_names, self.output_normalizer.mean,
                                       self.output_normalizer.std)
                )
            print(f"[输出标准化] 模式={self.output_norm_mode}  通道统计: {stat_str}")

        # L9(PINN) 数据可用性检查：v2_cache 缓存平面没有相邻 z 平面，L9 无法生效
        w_vec = self._get_loss_weight_vector()
        if len(w_vec) > 8 and w_vec[8] != 0.0:
            if not any(s.get("fmt") == "v2" for s in full_dataset.samples):
                print("[警告] L9(PINN 麦克斯韦残差) 已启用，但当前数据为 v2_cache 缓存平面"
                      "（无 out_zminus/out_zplus 相邻平面），该损失不会参与训练。"
                      "如需启用请改用 v2 体数据（field_volume_merged.npy），"
                      "或在 config.py 中将 LOSS_ENABLE_VECTOR[8] 设为 0。")

        print(f"[数据] 训练: {len(train_ds)}, 验证: {len(val_ds)}")

        self.train_loader = DataLoader(
            train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
            num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY, drop_last=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
            num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY
        )

        model_name = getattr(C, "MODEL_NAME", "fno_attention")
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            raise ValueError(f"未知模型: {model_name}，可选: {list(MODEL_REGISTRY.keys())}")
        kwargs = dict(
            modes1=C.MODES1, modes2=C.MODES2, width=C.WIDTH,
            n_layers=C.N_LAYERS, out_channels=getattr(C, "OUT_CHANNELS", 6),
        )
        if model_name == "fno_lora":
            kwargs["lora_rank"] = getattr(C, "LORA_RANK", 4)
        self.model = model_cls(**kwargs).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[模型] 参数量: {n_params:,}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY
        )

        # Warmup + Cosine annealing 学习率调度
        warmup_epochs = getattr(C, "LR_WARMUP_EPOCHS", 20)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, C.N_EPOCHS - warmup_epochs)
            return 0.01 + 0.99 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.history = {"train": [], "val": []}
        self.best_val = float("inf")
        self.patience = 0
        self.only_save_pth = bool(getattr(C, "ONLY_SAVE_PTH", False))
        if self.only_save_pth:
            print("[输出] ONLY_SAVE_PTH=True，已关闭训练曲线保存")

    def _compute_output_normalizer(self, dataset, subset):
        """Compute normalization stats from training subset only.

        - zscore: 原始通道 mean/std
        - log:    scale=原始 std，再对 sign*log1p(|x|/scale) 结果做二次标准化（logmean/logstd）
        """
        from torch.utils.data import Subset
        from data.norm_utils import OutputNormalizer
        indices = subset.indices if isinstance(subset, Subset) else list(range(len(dataset)))

        # 先取一个样本确定实际通道数，不依赖 OUT_CHANNELS 配置
        sample0 = dataset[int(indices[0])]
        n_ch = sample0["output"].shape[0]

        sum_v  = np.zeros(n_ch, dtype=np.float64)
        sum_sq = np.zeros(n_ch, dtype=np.float64)
        count  = 0

        for idx in indices:
            sample = dataset[int(idx)]
            out = sample["output"].numpy().astype(np.float64)  # (n_ch, H, W)
            flat = out.reshape(n_ch, -1)
            sum_v  += flat.sum(axis=1)
            sum_sq += (flat * flat).sum(axis=1)
            count  += flat.shape[1]

        eps  = float(getattr(C, "OUTPUT_STANDARDIZATION_EPS", 1e-8))
        mean = sum_v / max(1, count)
        var  = np.maximum(sum_sq / max(1, count) - mean * mean, eps * eps)
        std  = np.sqrt(var)

        if self.output_norm_mode == "zscore":
            return OutputNormalizer(mode="zscore", mean=mean, std=std, eps=eps)
        if self.output_norm_mode == "log":
            # 二次遍历：log1p 压缩后逐通道标准化
            scale = np.maximum(std, eps)
            ly_sum = np.zeros(n_ch, dtype=np.float64)
            ly_sq  = np.zeros(n_ch, dtype=np.float64)
            for idx in indices:
                sample = dataset[int(idx)]
                out = sample["output"].numpy().astype(np.float64)  # (n_ch, H, W)
                flat = out.reshape(n_ch, -1)
                y = np.sign(flat) * np.log1p(np.abs(flat) / scale[:, None])
                ly_sum += y.sum(axis=1)
                ly_sq  += (y * y).sum(axis=1)
            logmean = ly_sum / max(1, count)
            logvar  = np.maximum(ly_sq / max(1, count) - logmean * logmean, eps * eps)
            logstd  = np.sqrt(logvar)
            return OutputNormalizer(mode="log", scale=scale,
                                    logmean=logmean, logstd=logstd, eps=eps)
        if self.output_norm_mode == "none":
            return OutputNormalizer(mode="none")
        raise ValueError(f"不支持的 OUTPUT_NORM_MODE: {self.output_norm_mode}")

    def _ramp_weight(self, target, epoch, warmup_epochs):
        """Linear warmup for auxiliary terms to avoid early optimization conflicts."""
        target = float(target)
        if target <= 0:
            return 0.0
        warmup_epochs = int(max(0, warmup_epochs))
        if warmup_epochs == 0:
            return target
        ratio = min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))
        return target * ratio

    def _delayed_ramp(self, target, epoch, start_epoch, ramp_epochs):
        """Start from 0 before start_epoch, then linearly ramp to target."""
        target = float(target)
        start_epoch = int(max(1, start_epoch))
        if epoch < start_epoch or target <= 0:
            return 0.0
        shifted_epoch = epoch - start_epoch + 1
        return self._ramp_weight(target, shifted_epoch, ramp_epochs)

    def _get_loss_weight_vector(self):
        """Build effective loss weight vector from config enable/weight vectors."""
        default_enable = [1, 1, 1, 1, 0, 1, 0, 0, 0]
        default_weight = [1.0, 0.4, 0.03, 0.08, 0.0, 1e-3, 0.0, 0.0, 0.0]

        enable_vec = list(getattr(C, "LOSS_ENABLE_VECTOR", default_enable))
        weight_vec = list(getattr(C, "LOSS_WEIGHT_VECTOR", default_weight))

        # 动态取配置向量的最大长度，不再硬编码截断
        n = max(len(enable_vec), len(weight_vec), len(default_enable))
        if len(enable_vec) < n:
            enable_vec = enable_vec + [0] * (n - len(enable_vec))
        if len(weight_vec) < n:
            weight_vec = weight_vec + [0.0] * (n - len(weight_vec))

        out = []
        for e, w in zip(enable_vec, weight_vec):
            on = int(e) != 0
            out.append(float(w) if on else 0.0)
        return out

    def _get_effective_loss_weight_vector(self, epoch):
        """Apply optional curriculum gate to detailed losses (index >=1)."""
        w = self._get_loss_weight_vector()
        if bool(getattr(C, "USE_LOSS_CURRICULUM", False)):  # 默认关闭课程式学习
            detail_start = int(getattr(C, "DETAIL_LOSS_START_EPOCH", 40))
            detail_ramp = int(getattr(C, "DETAIL_LOSS_RAMP_EPOCHS", 200))
            gate = self._delayed_ramp(1.0, epoch, detail_start, detail_ramp)
            for i in range(1, len(w)):
                w[i] *= gate
        return w

    def _denorm_output(self, x):
        if self._denorm_tp is None:
            return x
        return self.output_normalizer.denormalize(x, tp=self._denorm_tp)

    def complex_channel_rel_error(self, pred, true):
        """6通道分别计算DC/AC误差后取平均。"""
        eps = 1e-6
        w_dc = float(getattr(C, "AUX_COMPLEX_CH_DC_WEIGHT", 1.0))
        w_ac = float(getattr(C, "AUX_COMPLEX_CH_AC_WEIGHT", 1.0))

        pred_work = pred
        true_work = true

        # 将6通道按总场模值归一化，使该项更关注“通道分解结构”，减少与总场幅值损失的梯度冲突。
        if bool(getattr(C, "AUX_COMPLEX_CH_NORMALIZE_BY_TOTAL", True)) and pred.shape[1] >= 6:
            # 电场用电场总场归一化，磁场用磁场总场归一化
            true_ex = torch.sqrt(true[:, 0] ** 2 + true[:, 1] ** 2 + eps)
            true_ey = torch.sqrt(true[:, 2] ** 2 + true[:, 3] ** 2 + eps)
            true_ez = torch.sqrt(true[:, 4] ** 2 + true[:, 5] ** 2 + eps)
            true_e_total = torch.sqrt(true_ex ** 2 + true_ey ** 2 + true_ez ** 2 + eps)
            scale_e = true_e_total[:, None, :, :].clamp(min=eps)
            pred_work = pred.clone()
            true_work = true.clone()
            pred_work[:, :6] = pred[:, :6] / scale_e
            true_work[:, :6] = true[:, :6] / scale_e
            # 磁场有 12 通道时单独用磁场总场归一化
            if pred.shape[1] >= 12:
                true_hx = torch.sqrt(true[:, 6] ** 2 + true[:, 7] ** 2 + eps)
                true_hy = torch.sqrt(true[:, 8] ** 2 + true[:, 9] ** 2 + eps)
                true_hz = torch.sqrt(true[:, 10] ** 2 + true[:, 11] ** 2 + eps)
                true_h_total = torch.sqrt(true_hx ** 2 + true_hy ** 2 + true_hz ** 2 + eps)
                scale_h = true_h_total[:, None, :, :].clamp(min=eps)
                pred_work[:, 6:] = pred[:, 6:] / scale_h
                true_work[:, 6:] = true[:, 6:] / scale_h

        # 每个通道各自的DC（空间均值）相对误差
        pred_dc = pred_work.mean(dim=[2, 3])
        true_dc = true_work.mean(dim=[2, 3])
        true_rms = torch.sqrt((true_work ** 2).mean(dim=[2, 3]) + eps)
        denom_floor_ratio = float(getattr(C, "AUX_COMPLEX_CH_DC_DENOM_FLOOR_RATIO", 0.1))
        dc_denom = torch.maximum(true_dc.abs(), denom_floor_ratio * true_rms) + eps
        dc_rel = (torch.abs(pred_dc - true_dc) / dc_denom).mean()

        # 每个通道各自的AC（去均值后）相对L2，再跨通道平均
        pred_ac = pred_work - pred_dc[:, :, None, None]
        true_ac = true_work - true_dc[:, :, None, None]
        B, nC = pred_work.shape[:2]
        diff = (pred_ac - true_ac).reshape(B, nC, -1).norm(dim=2)
        norm = true_ac.reshape(B, nC, -1).norm(dim=2).clamp(min=eps)
        ac_rel = (diff / norm).mean()

        total = w_dc * dc_rel + w_ac * ac_rel

        # 对极端大误差做压缩，防止该辅助项后期主导优化导致塌缩。
        if bool(getattr(C, "AUX_COMPLEX_CH_LOG_COMPRESS", True)):
            total = torch.log1p(total)

        return total

    def channel_mean_rel_l2_error(self, pred, true):
        """12通道分别算相对L2，再对通道取平均。"""
        eps = 1e-6
        b, n_ch = pred.shape[:2]
        diff = (pred - true).reshape(b, n_ch, -1).norm(dim=2)   # (B, n_ch)
        base = true.reshape(b, n_ch, -1).norm(dim=2).clamp(min=eps)  # (B, n_ch)
        per_channel = (diff / base).mean(dim=0)   # (n_ch,) 先对batch平均
        return per_channel.mean()                 # 再对通道平均

    def loss_fn(self, pred, true, epoch=1, batch=None):
        """
        向量化损失组合（顺序见 config.py 的 LOSS_VECTOR_ORDER 说明）：
        0: 总场相对L2
        1: 三方向幅值DC/AC
        2: 全场几何损失
        3: 6通道DC/AC（实虚部）
        4: 低频频谱约束
        5: 高频抑制
        6: 6通道DC/AC（原始）
        7: 6通道逐通道相对L2均值（SaveRel6口径）
        8: 麦克斯韦残差约束 (PINN Loss)
        """
        w = self._get_effective_loss_weight_vector(epoch)
        loss = pred.new_tensor(0.0)

        # 12 通道时，电场相关 loss 只取前 6 通道（Ex/Ey/Ez）
        pred_e = pred[:, :6]
        true_e = true[:, :6]

        # 0) 总场相对L2
        if w[0] != 0.0:
            loss = loss + w[0] * field_rel_l2_loss(pred_e, true_e)

        # 1) 三方向幅值DC/AC
        if w[1] != 0.0:
            v, _ = dc_ac_directional_mean_loss(
                pred_e, true_e,
                w_dc=float(getattr(C, "DC_REL_WEIGHT", 1.0)),
                w_ac=float(getattr(C, "AC_REL_WEIGHT", 1.0)),
            )
            loss = loss + w[1] * v

        # 2) 全场几何（归一化空间计算，避免 E/H 量级差异导致梯度失衡）
        if w[2] != 0.0:
            # 电场 geom loss（前 6 通道，归一化空间）
            v_e, _ = field_geom_loss(
                pred[:, :6], true[:, :6],
                w_dir=float(getattr(C, "FIELD_DIR_ANGLE_WEIGHT", 1.0)),
                w_phase=float(getattr(C, "FIELD_COMPLEX_PHASE_WEIGHT", 1.0)),
                w_mag=float(getattr(C, "FIELD_TOTAL_MAG_WEIGHT", 1.0)),
            )
            v = v_e
            # 磁场 geom loss（后 6 通道，归一化空间，仅 12 通道时生效）
            if pred.shape[1] >= 12:
                v_h, _ = field_geom_loss(
                    pred[:, 6:12], true[:, 6:12],
                    w_dir=float(getattr(C, "FIELD_DIR_ANGLE_WEIGHT", 1.0)),
                    w_phase=float(getattr(C, "FIELD_COMPLEX_PHASE_WEIGHT", 1.0)),
                    w_mag=float(getattr(C, "FIELD_TOTAL_MAG_WEIGHT", 1.0)),
                )
                v = (v_e + v_h) * 0.5
            loss = loss + w[2] * v

        # 3) 通道DC/AC（稳定版，全通道）
        if w[3] != 0.0:
            loss = loss + w[3] * self.complex_channel_rel_error(pred, true)

        # 4/5) 频域约束（全通道）
        if w[4] != 0.0 or w[5] != 0.0:
            k = int(getattr(C, "LOWFREQ_PATCH", 4))
            cutoff = int(getattr(C, "HIGHFREQ_CUTOFF", 12))
            low_loss, hf_loss = freq_band_loss(pred, true, low_h=k, low_w=k, hf_cutoff=cutoff)
            if w[4] != 0.0:
                loss = loss + w[4] * low_loss
            if w[5] != 0.0:
                # 高频抑制：在物理空间计算预测值的高频能量
                pred_phys = self._denorm_output(pred)
                loss = loss + w[5] * highfreq_suppress_loss(pred_phys, cutoff=cutoff)

        # 6) 通道原始DC/AC（全通道）
        if w[6] != 0.0:
            v, _ = dc_ac_rel_loss(
                pred, true,
                w_dc=float(getattr(C, "DC_REL_WEIGHT", 1.0)),
                w_ac=float(getattr(C, "AC_REL_WEIGHT", 1.0)),
            )
            loss = loss + w[6] * v

        # 7) 逐通道相对L2均值（全通道，与 geom_metric 口径对齐）
        if len(w) > 7 and w[7] != 0.0:
            loss = loss + w[7] * self.channel_mean_rel_l2_error(pred, true)

        # 8) PINN 麦克斯韦残差损失 (使用相邻平面的真实场计算精确 z 差分)
        if len(w) > 8 and w[8] != 0.0 and batch is not None and "out_zminus" in batch:
            from models.losses import maxwell_pinn_loss
            # PINN 需要未被标准化的真实物理尺度预测值来代入麦克斯韦方程
            pred_phys = self._denorm_output(pred)
            true_zminus = batch["out_zminus"].to(self.device)
            true_zplus = batch["out_zplus"].to(self.device)
            freq_ghz = batch["freq_ghz"].to(self.device)
            
            pinn_v, _ = maxwell_pinn_loss(pred_phys, true_zminus, true_zplus, freq_ghz)
            loss = loss + w[8] * pinn_v

        if loss.detach().abs().item() == 0.0:
            loss = field_rel_l2_loss(pred_e, true_e)
        return loss

    def rel_l2_error(self, pred, true):
        """物理空间逐通道相对 L2（先反归一化，避免 log 归一化空间失真）。"""
        return self.channel_mean_rel_l2_error(self._denorm_output(pred),
                                              self._denorm_output(true))

    def geom_metric(self, pred, true):
        """物理空间 geom 误差（方向角+相位+幅值），用作保存/早停指标。"""
        pred_p = self._denorm_output(pred)
        true_p = self._denorm_output(true)
        v_e, _ = field_geom_loss(pred_p[:, :6], true_p[:, :6])
        if pred_p.shape[1] >= 12:
            v_h, _ = field_geom_loss(pred_p[:, 6:12], true_p[:, 6:12])
            return (v_e + v_h) * 0.5
        return v_e

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_rel  = 0
        total_save_rel = 0
        use_freq_interp = getattr(C, "FREQ_INTERP_PROB", 0.0) > 0
        interp_prob = getattr(C, "FREQ_INTERP_PROB", 0.0)
        dataset = self.train_loader.dataset
        # 获取底层 dataset（可能被 Subset 包装）
        from torch.utils.data import Subset
        base_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
        has_shape_index = use_freq_interp and hasattr(base_ds, 'shape_to_indices')

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False):
            inp  = batch["input"].to(self.device)
            tgt  = batch["output"].to(self.device)
            freq = batch["freq"].to(self.device)
            # 频率插值增强：随机对 batch 中的样本做频率插值
            if has_shape_index and torch.rand(1).item() < interp_prob:
                import random
                new_tgt  = tgt.clone()
                new_freq = freq.clone()
                for b in range(inp.shape[0]):
                    cat = batch["category"][b]
                    name = batch["name"][b]
                    key = (cat, name)
                    indices = base_ds.shape_to_indices.get(key, [])
                    if len(indices) >= 2:
                        # 随机取另一个频点
                        idx_b = random.choice(indices)
                        s_b = base_ds[idx_b]
                        out_b = torch.from_numpy(s_b["output"].numpy() if hasattr(s_b["output"], 'numpy') else s_b["output"]).to(self.device) if not isinstance(s_b["output"], torch.Tensor) else s_b["output"].to(self.device)
                        freq_b = s_b["freq"].to(self.device) if isinstance(s_b["freq"], torch.Tensor) else torch.tensor(s_b["freq"]).to(self.device)
                        # 线性插值
                        alpha = torch.rand(1).item()
                        new_tgt[b]  = alpha * tgt[b]  + (1 - alpha) * out_b
                        new_freq[b] = alpha * freq[b] + (1 - alpha) * freq_b.view_as(freq[b])
                tgt  = new_tgt
                freq = new_freq
            pred = self.model(inp, freq)
            loss = self.loss_fn(pred, tgt, epoch=epoch, batch=batch)
            
            # 检查NaN/Inf
            if not torch.isfinite(loss):
                print(f"\n[警告] Epoch {epoch}: 损失为 {loss.item()}，跳过此batch")
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            clip_max_norm = float(getattr(C, "GRAD_CLIP_MAX_NORM", 1.0))
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_max_norm)
            self.optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
                total_rel += self.rel_l2_error(pred, tgt).item()
                total_save_rel += self.geom_metric(pred, tgt).item()
        n = len(self.train_loader)
        return total_loss / n, total_rel / n, total_save_rel / n

    def validate(self, epoch=1):
        self.model.eval()
        total_loss = 0
        total_rel  = 0
        total_save_rel = 0
        vis_pred = vis_tgt = None
        with torch.no_grad():
            for batch in self.val_loader:
                inp  = batch["input"].to(self.device)
                tgt  = batch["output"].to(self.device)
                freq = batch["freq"].to(self.device)
                pred = self.model(inp, freq)
                total_loss += self.loss_fn(pred, tgt, epoch=epoch, batch=batch).item()
                total_rel += self.rel_l2_error(pred, tgt).item()
                total_save_rel += self.geom_metric(pred, tgt).item()
                if vis_pred is None:
                    vis_pred, vis_tgt = self._denorm_output(pred).detach(), self._denorm_output(tgt).detach()
        n = len(self.val_loader)
        return total_loss / n, total_rel / n, total_save_rel / n, vis_pred, vis_tgt

    def run(self):
        print(f"\n{'='*50}\n开始训练 {C.N_EPOCHS} epochs\n{'='*50}\n")
        for epoch in range(1, C.N_EPOCHS + 1):
            train_loss, train_rel, train_save_rel = self.train_epoch(epoch)
            self.history["train"].append(train_loss)
            self.scheduler.step()


            val_loss, val_rel, val_save_rel, vis_pred, vis_tgt = self.validate(epoch=epoch)
            self.history["val"].append(val_loss)
            # 按 Geom（方向角+相位+幅值综合误差）保存最优模型
            is_best = val_save_rel < self.best_val - C.EARLY_STOP_MIN_DELTA
            tag = " [BEST]" if is_best else ""
            print(f"Epoch {epoch}/{C.N_EPOCHS}  "
                  f"Loss(train/val): {train_loss:.4f}/{val_loss:.4f}  "
                  f"Geom(train/val): {train_save_rel:.4f}/{val_save_rel:.4f}  "
                  f"RelL2(train/val): {train_rel:.4f}/{val_rel:.4f}{tag}")

            # 始终保存最新权重，便于中断恢复与回看最近状态
            latest_every = int(getattr(C, "SAVE_LATEST_EVERY", 1))
            if latest_every > 0 and (epoch % latest_every == 0):
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_rel": val_rel,
                    "val_geom": val_save_rel,
                    "use_output_standardization": self.use_output_norm,
                }
                if self.use_output_norm:
                    ckpt.update(self.output_normalizer.to_dict())
                torch.save(ckpt, f"{C.SAVE_DIR}/latest.pth")

            if is_best:
                self.best_val = val_save_rel
                self.patience = 0
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_rel": val_rel,
                    "val_geom": val_save_rel,
                    "use_output_standardization": self.use_output_norm,
                }
                if self.use_output_norm:
                    ckpt.update(self.output_normalizer.to_dict())
                torch.save(ckpt, f"{C.SAVE_DIR}/best.pth")
            else:
                self.patience += 1
                if self.patience >= C.EARLY_STOP_PATIENCE:
                    print(f"\n早停 (patience={C.EARLY_STOP_PATIENCE})")
                    break

        if not self.only_save_pth:
            self._save_curve()
        print(f"\n训练完成！最佳保存指标: {self.best_val:.6f}")

    def _save_curve(self):
        if not HAS_MATPLOTLIB:
            print("[跳过] matplotlib 不可用，未保存训练曲线图。")
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.history["train"], label="Train")
        axes[0].set_title("Training Loss"); axes[0].grid(True); axes[0].legend()
        val_x = list(range(len(self.history["val"])))
        axes[1].plot(val_x, self.history["val"], label="Val", color="orange")
        axes[1].set_title("Validation Loss"); axes[1].grid(True); axes[1].legend()
        plt.tight_layout()
        plt.savefig(f"{C.SAVE_DIR}/training_curve.png", dpi=150)
        plt.close()
        print(f"[保存] {C.SAVE_DIR}/training_curve.png")


if __name__ == "__main__":
    Trainer().run()

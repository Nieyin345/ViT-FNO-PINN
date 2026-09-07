"""
电磁场多频点数据集

数据结构 v1（旧）：
  data/em_data/<category>/<sample>/
    patch_plane.npy       (65, 65, 2)  ch0=SDF, ch1=二值掩码
    field_slice/*_<freq>GHz.npy  (65, 65, 8)
      通道：[x, y, ex_r, ex_i, ey_r, ey_i, ez_r, ez_i]

数据结构 v2（新）：
  data/data/<category>/<sample>/
    patch_plane.npy       (64, 64, 2)
    field_volume_merged.npy  (9, 22, 64, 64, 12)
      axis0: 9 个频点 [8.0~12.0 GHz]
      axis1: 22 个 z 位置，2mm~23mm，步长 1mm
      axis2/3: 64×64 空间
      axis4: 12 通道 (Ex_r,Ex_i, Ey_r,Ey_i, Ez_r,Ez_i, Hx_r,Hx_i, Hy_r,Hy_i, Hz_r,Hz_i)
      → 按归一化距离 d/λ 插值后取电场前 6 通道作为输出

输入：(2, H, W)  输出：(6, H, W) = [ex_r, ex_i, ey_r, ey_i, ez_r, ez_i]
附加：freq (1,) 归一化频率 [0,1]
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from scipy.interpolate import interp1d

FREQ_LIST = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
FREQ_MIN, FREQ_MAX = FREQ_LIST[0], FREQ_LIST[-1]

# v2 数据的 z 轴位置（mm），2mm~23mm，步长 1mm
_Z_MM_V2 = np.arange(2, 24, 1, dtype=np.float32)  # shape (22,)
_C_LIGHT_MM = 3e11  # 光速 mm/s

def _wavelength_mm(freq_ghz):
    return _C_LIGHT_MM / (freq_ghz * 1e9)

def _interp_field_at_d_over_lambda(field_volume, freq_idx, target_d_lam, kind="linear"):
    """从 v2 体数据中插值得到指定 d/λ 处的场平面，返回 (H, W, 12)。"""
    lam = _wavelength_mm(FREQ_LIST[freq_idx])
    field_freq = field_volume[freq_idx]  # (n_z, H, W, 12)
    n_z, H, W, C = field_freq.shape
    
    # 动态适应不同的切片数量 (起点2mm，步长1mm)
    z_mm = np.arange(2, 2 + n_z, 1, dtype=np.float32)
    z_norm = z_mm / lam  # (n_z,)
    
    flat = field_freq.reshape(n_z, -1)
    # n_z 较小时（如3层）cubic插值会报错，默认使用 linear
    interp_kind = kind if n_z >= 4 else "linear"
    # 当目标距离超出范围时，通过 fill_value="extrapolate" 允许外推，以增强鲁棒性
    f = interp_kind_func = interp1d(z_norm, flat, axis=0, kind=interp_kind, bounds_error=False, fill_value="extrapolate")
    return f(target_d_lam).reshape(H, W, C).astype(np.float32)

# 形状和类型到 index 的映射（扫描时动态建立）
_SHAPE_INDEX = {}
_TYPE_INDEX = {"normal": 0, "revert": 1}

def get_shape_index(shape_name):
    if shape_name not in _SHAPE_INDEX:
        _SHAPE_INDEX[shape_name] = len(_SHAPE_INDEX)
    return _SHAPE_INDEX[shape_name]

def get_type_index(type_name):
    return _TYPE_INDEX.get(type_name, 0)


class EMMultiFreqDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/em_data",
        target_freq: str = None,
        output_field: str = "all",
        normalize: bool = False,
        max_samples: int = None,
        output_norm_mean=None,
        output_norm_std=None,
        target_d_over_lambda: float = 0.1,  # v2 数据：插值目标归一化距离
        smooth_sigma: float = 0.0,          # 已废弃，保留参数兼容旧调用
        use_scatter: bool = False,          # 是否减去背景场得到散射场
        bg_params_path: str = "data/blank/background_params.npz",
    ):
        self.output_field = output_field.lower()
        self.normalize    = normalize
        self.target_d_over_lambda = target_d_over_lambda
        self.use_scatter  = use_scatter
        self.bg_params_path = bg_params_path
        self.output_norm_mean = None
        self.output_norm_std = None
        if output_norm_mean is not None and output_norm_std is not None:
            self.set_output_normalization(output_norm_mean, output_norm_std)

        freq_targets = FREQ_LIST if target_freq is None else \
                       [float(target_freq.replace("GHz", ""))]
        freq_targets_set = set(freq_targets)

        def _iter_dirs(path):
            try:
                with os.scandir(path) as it:
                    return [e.path for e in it if e.is_dir()]
            except (FileNotFoundError, PermissionError):
                return []

        def _build_freq_to_file(field_slice_dir):
            """一次扫描 field_slice，建立频率到文件的映射，避免重复 glob。"""
            freq_map = {}
            try:
                with os.scandir(field_slice_dir) as it:
                    for e in it:
                        if not e.is_file() or not e.name.endswith("GHz.npy"):
                            continue
                        # 兼容名字前缀，按最后一个 '_' 后的频率解析
                        try:
                            token = e.name.rsplit("_", 1)[-1]
                            f_ghz = float(token.replace("GHz.npy", ""))
                        except ValueError:
                            continue
                        if f_ghz in freq_targets_set:
                            freq_map[f_ghz] = e.path
            except (FileNotFoundError, PermissionError):
                return {}
            return freq_map

        def _append_sample_records(sample_dir, category_name):
            patch_file = os.path.join(sample_dir, "patch_plane.npy")
            if not os.path.isfile(patch_file):
                return

            # v2 格式：优先读缓存的插值平面，没有则读体数据
            cache_file = os.path.join(sample_dir,
                f"field_plane_d{self.target_d_over_lambda:.2f}.npy")
            volume_file = os.path.join(sample_dir, "field_volume_merged.npy")

            if os.path.isfile(cache_file):
                # 缓存文件：(9, 64, 64, 12)，直接按频点索引取
                for fi, freq_ghz in enumerate(freq_targets):
                    if freq_ghz not in freq_targets_set:
                        continue
                    self.samples.append({
                        "patch":    patch_file,
                        "field":    cache_file,
                        "category": category_name,
                        "name":     os.path.basename(sample_dir),
                        "freq_ghz": freq_ghz,
                        "freq_idx": FREQ_LIST.index(freq_ghz),
                        "fmt":      "v2_cache",
                    })
            elif os.path.isfile(volume_file):
                for fi, freq_ghz in enumerate(freq_targets):
                    if freq_ghz not in freq_targets_set:
                        continue
                    self.samples.append({
                        "patch":    patch_file,
                        "field":    volume_file,
                        "category": category_name,
                        "name":     os.path.basename(sample_dir),
                        "freq_ghz": freq_ghz,
                        "freq_idx": FREQ_LIST.index(freq_ghz),
                        "fmt":      "v2",
                    })
            else:
                # v2 不存在，尝试 v1 格式：field_slice/*.npy
                field_slice_dir = os.path.join(sample_dir, "field_slice")
                freq_map = _build_freq_to_file(field_slice_dir)
                for freq_ghz, field_file in freq_map.items():
                    self.samples.append({
                        "patch":    patch_file,
                        "field":    field_file,
                        "category": category_name,
                        "name":     os.path.basename(sample_dir),
                        "freq_ghz": freq_ghz,
                        "fmt":      "v1",
                    })
                return

        self.samples = []
        for cat_dir in sorted(_iter_dirs(data_dir)):
            cat_name = os.path.basename(cat_dir)

            # 检测是否有子类别（两级 vs 三级目录）
            # 如果子目录里没有 patch_plane.npy，说明是子类别目录
            sub_dirs = sorted(_iter_dirs(cat_dir))

            # 判断是否为子类别层：取第一个子目录，看里面有没有 patch_plane.npy
            has_subcat = False
            if sub_dirs:
                first_sub = sub_dirs[0]
                first_sub_patch = os.path.join(first_sub, "patch_plane.npy")
                first_sub_sub = _iter_dirs(first_sub)
                # 如果第一个子目录里没有 patch_plane.npy 但有子目录，说明是子类别
                if not os.path.exists(first_sub_patch) and first_sub_sub:
                    has_subcat = True

            if has_subcat:
                # 三级：data_dir/category/subcategory/sample
                for subcat_dir in sub_dirs:
                    subcat_name = os.path.basename(subcat_dir)
                    full_cat_name = f"{cat_name}/{subcat_name}"
                    for sample_dir in sorted(_iter_dirs(subcat_dir)):
                        _append_sample_records(sample_dir, full_cat_name)
            else:
                # 两级：data_dir/category/sample（原有逻辑）
                for sample_dir in sub_dirs:
                    _append_sample_records(sample_dir, cat_name)

        if max_samples:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(f"未找到任何样本: {data_dir}")

        cat_counts  = Counter(s["category"] for s in self.samples)
        freq_counts = Counter(s["freq_ghz"]  for s in self.samples)
        print(f"[EMMultiFreqDataset] {len(self.samples)} 条记录")
        print(f"  类别: {dict(sorted(cat_counts.items()))}")
        print(f"  频点: {sorted(freq_counts.keys())} GHz")

        # 按 (category, name) 分组，用于频率插值训练
        # shape_to_indices[(cat, name)] = [idx0, idx1, ...]  各频点的索引
        from collections import defaultdict
        self.shape_to_indices = defaultdict(list)
        for i, s in enumerate(self.samples):
            self.shape_to_indices[(s["category"], s["name"])].append(i)

    def set_output_normalization(self, mean, std):
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        std = np.asarray(std, dtype=np.float32).reshape(-1)
        assert mean.size == std.size, "mean/std 通道数不一致"
        std = np.maximum(std, 1e-8)
        self.output_norm_mean = mean
        self.output_norm_std = std

    def _output_norm_slice(self, n_channels=None):
        if self.output_field == "all":
            return slice(0, n_channels) if n_channels else slice(None)
        if self.output_field == "ex":
            return slice(0, 2)
        if self.output_field == "ey":
            return slice(2, 4)
        if self.output_field == "ez":
            return slice(4, 6)
        raise ValueError(f"不支持的 output_field: {self.output_field}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 输入：SDF + Mask
        patch = np.load(s["patch"], mmap_mode="r").astype(np.float32)
        sdf   = patch[:, :, 0] / (np.abs(patch[:, :, 0]).max() + 1e-8)
        mask  = patch[:, :, 1]
        inp = torch.from_numpy(np.stack([sdf, mask], axis=0))

        # 输出：根据格式读取
        if s.get("fmt", "v1") == "v2_cache":
            # 缓存格式：(9, 64, 64, 12)，直接按频点索引取
            planes = np.load(s["field"], mmap_mode="r")           # (9, 64, 64, 12)
            plane  = planes[s["freq_idx"]]         # (64, 64, 12)
            ex_r, ex_i = plane[:, :, 0], plane[:, :, 1]
            ey_r, ey_i = plane[:, :, 2], plane[:, :, 3]
            ez_r, ez_i = plane[:, :, 4], plane[:, :, 5]
            hx_r, hx_i = plane[:, :, 6], plane[:, :, 7]
            hy_r, hy_i = plane[:, :, 8], plane[:, :, 9]
            hz_r, hz_i = plane[:, :, 10], plane[:, :, 11]
            if self.output_field == "all":
                out = np.stack([ex_r, ex_i, ey_r, ey_i, ez_r, ez_i,
                                hx_r, hx_i, hy_r, hy_i, hz_r, hz_i], axis=0)
            elif self.output_field == "ez":
                out = np.stack([ez_r, ez_i], axis=0)
            elif self.output_field == "ex":
                out = np.stack([ex_r, ex_i], axis=0)
            elif self.output_field == "ey":
                out = np.stack([ey_r, ey_i], axis=0)
            else:
                raise ValueError(f"不支持的 output_field: {self.output_field}")
        elif s.get("fmt", "v1") == "v2":
            field_volume = np.load(s["field"], mmap_mode="r")  # 可能为 (9, 22, H, W, 12) 或 (2, 9, 3, H, W, 12)
            
            # 如果是 6D 双极化格式，目前默认取极化 0 的数据
            if field_volume.ndim == 6 and field_volume.shape[0] == 2:
                field_volume = field_volume[0]
                
            def _extract_all_channels(d_lam):
                plane = _interp_field_at_d_over_lambda(
                    field_volume, s["freq_idx"], d_lam
                )  # (H, W, 12)
                ex_r, ex_i = plane[:, :, 0], plane[:, :, 1]
                ey_r, ey_i = plane[:, :, 2], plane[:, :, 3]
                ez_r, ez_i = plane[:, :, 4], plane[:, :, 5]
                hx_r, hx_i = plane[:, :, 6], plane[:, :, 7]
                hy_r, hy_i = plane[:, :, 8], plane[:, :, 9]
                hz_r, hz_i = plane[:, :, 10], plane[:, :, 11]
                return np.stack([ex_r, ex_i, ey_r, ey_i, ez_r, ez_i,
                                 hx_r, hx_i, hy_r, hy_i, hz_r, hz_i], axis=0)

            # 获取中心平面的全部 12 通道
            out = _extract_all_channels(self.target_d_over_lambda)
            
            # 获取 z-1mm 和 z+1mm 的场，用于 PINN 中心差分
            lam_mm = _wavelength_mm(s["freq_ghz"])
            delta_d_lam = 1.0 / lam_mm  # 1mm 的波长偏移
            out_zminus = _extract_all_channels(self.target_d_over_lambda - delta_d_lam)
            out_zplus  = _extract_all_channels(self.target_d_over_lambda + delta_d_lam)
            
            # 兼容以前的单场截取
            if self.output_field == "ez":
                out = out[4:6]
            elif self.output_field == "ex":
                out = out[0:2]
            elif self.output_field == "ey":
                out = out[2:4]
            elif self.output_field != "all":
                raise ValueError(f"不支持的 output_field: {self.output_field}")
        else:
            # v1：从 field_slice 文件读取（只有电场）
            field = np.load(s["field"]).astype(np.float32)
            ex_r, ex_i = field[:, :, 2], field[:, :, 3]
            ey_r, ey_i = field[:, :, 4], field[:, :, 5]
            ez_r, ez_i = field[:, :, 6], field[:, :, 7]
            if self.output_field == "all":
                out = np.stack([ex_r, ex_i, ey_r, ey_i, ez_r, ez_i], axis=0)
            elif self.output_field == "ez":
                out = np.stack([ez_r, ez_i], axis=0)
            elif self.output_field == "ex":
                out = np.stack([ex_r, ex_i], axis=0)
            elif self.output_field == "ey":
                out = np.stack([ey_r, ey_i], axis=0)
            else:
                raise ValueError(f"不支持的 output_field: {self.output_field}")

        # 散射场：减去拟合的理想背景场（仅 v2 格式支持）
        if self.use_scatter and s.get("fmt", "v1").startswith("v2"):
            from data.fit_background import get_background_field
            from data.interpolate_field import get_wavelength_mm
            freq_ghz = s["freq_ghz"]
            z_mm = self.target_d_over_lambda * get_wavelength_mm(freq_ghz)
            bg = get_background_field(freq_ghz, z_mm, self.bg_params_path)  # (12,)
            # out shape: (n_ch, H, W)，bg shape: (12,)，只减对应通道
            n_ch = out.shape[0]
            bg_broadcast = bg[:n_ch, None, None]  # (n_ch, 1, 1)
            out = out - bg_broadcast

        if self.normalize:
            if self.output_norm_mean is None or self.output_norm_std is None:
                raise RuntimeError("normalize=True 但未设置 output_norm_mean/std")
            sl = self._output_norm_slice(n_channels=out.shape[0])
            mean = self.output_norm_mean[sl][:, None, None]
            std = self.output_norm_std[sl][:, None, None]
            out = (out - mean) / std

        freq_norm = (s["freq_ghz"] - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)
        
        # 解析形状和类型
        category_parts = s["category"].split("/")
        shape_name = category_parts[0]  # circle, cross_base, ...
        type_name = category_parts[1] if len(category_parts) > 1 else "normal"  # normal, revert

        res = {
            "input":     inp,
            "output":    torch.from_numpy(out),
            "name":      s["name"],
            "category":  s["category"],
            "freq_ghz":  torch.tensor(s["freq_ghz"], dtype=torch.float32),
            "freq":      torch.tensor([freq_norm], dtype=torch.float32),
            "shape_idx": torch.tensor([get_shape_index(shape_name)], dtype=torch.long),
            "type_idx":  torch.tensor([get_type_index(type_name)], dtype=torch.long),
        }
        
        if 'out_zminus' in locals() and 'out_zplus' in locals():
            res["out_zminus"] = torch.from_numpy(out_zminus)
            res["out_zplus"]  = torch.from_numpy(out_zplus)
            
        return res

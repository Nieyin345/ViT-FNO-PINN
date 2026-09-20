"""
电磁场多频点数据集（兼容单极化 / 双极化 v2 数据）

数据结构：
  data/data/<category>/<sample>/
    patch_plane.npy       (64, 64, 2)
    field_volume_merged.npy
      - 单极化: (9, N_Z, 64, 64, 12)
      - 双极化: (2, 9, N_Z, 64, 64, 12)   [极化0, 极化1]
    field_plane_d0.10.npy
      - 单极化缓存: (9, 64, 64, 12)
      - 双极化缓存: (2, 9, 64, 64, 12)

输入：(2, H, W)  输出：(12, H, W) = Ex/Ey/Ez/Hx/Hy/Hz 各实部+虚部
附加：freq (2,) = [归一化频率, 极化索引]，其中极化作为模型条件与频率并列输入。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter, defaultdict
from scipy.interpolate import interp1d

FREQ_LIST = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
FREQ_MIN, FREQ_MAX = FREQ_LIST[0], FREQ_LIST[-1]

# v2 数据的 z 轴位置（mm，相对贴片），2mm 起、步长 1mm；
# 新双极化数据只导出 3 层时也用同一约定（2/3/4mm）。
_Z_MM_START = 2.0
_Z_MM_STEP = 1.0
_C_LIGHT_MM = 3e11  # 光速 mm/s


def _wavelength_mm(freq_ghz):
    return _C_LIGHT_MM / (freq_ghz * 1e9)


def _z_mm_for_nz(n_z):
    """根据当前体数据层数构造相对贴片的 z 轴，兼容旧 22 层和新 3 层。"""
    return np.arange(_Z_MM_START, _Z_MM_START + n_z, _Z_MM_STEP, dtype=np.float32)


def _interp_field_at_d_over_lambda(field_volume, freq_idx, target_d_lam, kind="linear"):
    """从单极化体数据 (9, N_Z, H, W, 12) 插值得到指定 d/λ 处的场平面。"""
    lam = _wavelength_mm(FREQ_LIST[freq_idx])
    field_freq = field_volume[freq_idx]  # (n_z, H, W, 12)
    n_z, H, W, C = field_freq.shape

    z_mm = _z_mm_for_nz(n_z)
    z_norm = z_mm / lam  # (n_z,)

    flat = field_freq.reshape(n_z, -1)
    # n_z 较小时使用 linear，避免 cubic 插值在边界/短轴上抖动。
    interp_kind = kind if n_z >= 4 else "linear"
    f = interp1d(z_norm, flat, axis=0, kind=interp_kind,
                 bounds_error=False, fill_value="extrapolate")
    return f(target_d_lam).reshape(H, W, C).astype(np.float32)


def _extract_output_channels(plane, output_field):
    """把 (H, W, 12) 平面按配置切出训练输出。"""
    ex_r, ex_i = plane[:, :, 0], plane[:, :, 1]
    ey_r, ey_i = plane[:, :, 2], plane[:, :, 3]
    ez_r, ez_i = plane[:, :, 4], plane[:, :, 5]
    hx_r, hx_i = plane[:, :, 6], plane[:, :, 7]
    hy_r, hy_i = plane[:, :, 8], plane[:, :, 9]
    hz_r, hz_i = plane[:, :, 10], plane[:, :, 11]

    if output_field == "all":
        return np.stack([ex_r, ex_i, ey_r, ey_i, ez_r, ez_i,
                         hx_r, hx_i, hy_r, hy_i, hz_r, hz_i], axis=0)
    if output_field == "ex":
        return np.stack([ex_r, ex_i], axis=0)
    if output_field == "ey":
        return np.stack([ey_r, ey_i], axis=0)
    if output_field == "ez":
        return np.stack([ez_r, ez_i], axis=0)
    raise ValueError(f"不支持的 output_field: {output_field}")


# 形状和类型到 index 的映射（扫描时动态建立）
_SHAPE_INDEX = {}
_TYPE_INDEX = {"normal": 0, "revert": 1}


def get_shape_index(shape_name):
    if shape_name not in _SHAPE_INDEX:
        _SHAPE_INDEX[shape_name] = len(_SHAPE_INDEX)
    return _SHAPE_INDEX[shape_name]


def get_type_index(type_name):
    return _TYPE_INDEX.get(type_name, 0)


def _field_file_polarization_count(path):
    """读取 npy header，返回可用极化数：
    6D=(2,9,...) 或 5D 双极化缓存 (2,9,H,W,C) 返回 2，其余返回 1。"""
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 6:
            return int(arr.shape[0])
        if arr.ndim == 5 and arr.shape[0] == 2 and arr.shape[1] == 9:
            return 2
    except Exception:
        pass
    return 1


class EMMultiFreqDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/data",
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
        polarizations=None,                 # None = 使用每个文件实际可用的极化
    ):
        self.output_field = output_field.lower()
        self.normalize = normalize
        self.target_d_over_lambda = target_d_over_lambda
        self.use_scatter = use_scatter
        self.bg_params_path = bg_params_path
        self.output_normalizer = None
        self.polarizations = self._normalize_polarizations(polarizations)
        if output_norm_mean is not None and output_norm_std is not None:
            from data.norm_utils import OutputNormalizer
            self.set_output_normalization(
                OutputNormalizer(mode="zscore", mean=output_norm_mean, std=output_norm_std)
            )

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

        def _append_polarization_records(base_record, field_path, fmt, n_pols):
            if n_pols is None:
                n_pols = _field_file_polarization_count(field_path)
            selected = self._select_polarizations(n_pols)
            for pol in selected:
                record = dict(base_record)
                record.update({
                    "field": field_path,
                    "polarization": pol,
                    "n_polarizations": n_pols,
                    "fmt": fmt,
                })
                self.samples.append(record)

        def _append_sample_records(sample_dir, category_name):
            patch_file = os.path.join(sample_dir, "patch_plane.npy")
            if not os.path.isfile(patch_file):
                return

            cache_file = os.path.join(sample_dir,
                f"field_plane_d{self.target_d_over_lambda:.2f}.npy")
            volume_file = os.path.join(sample_dir, "field_volume_merged.npy")

            base = {
                "patch": patch_file,
                "category": category_name,
                "name": os.path.basename(sample_dir),
            }

            if os.path.isfile(cache_file):
                # 缓存文件：(9, H, W, 12) 或 (2, 9, H, W, 12)
                n_pols = _field_file_polarization_count(cache_file)
                for freq_ghz in freq_targets:
                    if freq_ghz not in freq_targets_set:
                        continue
                    row = dict(base)
                    row.update({
                        "freq_ghz": freq_ghz,
                        "freq_idx": FREQ_LIST.index(freq_ghz),
                    })
                    _append_polarization_records(row, cache_file, "v2_cache", n_pols)
            elif os.path.isfile(volume_file):
                n_pols = _field_file_polarization_count(volume_file)
                for freq_ghz in freq_targets:
                    if freq_ghz not in freq_targets_set:
                        continue
                    row = dict(base)
                    row.update({
                        "freq_ghz": freq_ghz,
                        "freq_idx": FREQ_LIST.index(freq_ghz),
                    })
                    _append_polarization_records(row, volume_file, "v2", n_pols)
            else:
                # v1 格式：field_slice/*.npy，只有单极化电场。
                field_slice_dir = os.path.join(sample_dir, "field_slice")
                freq_map = _build_freq_to_file(field_slice_dir)
                for freq_ghz, field_file in freq_map.items():
                    row = dict(base)
                    row.update({
                        "field": field_file,
                        "freq_ghz": freq_ghz,
                        "polarization": 0,
                        "n_polarizations": 1,
                        "fmt": "v1",
                    })
                    self.samples.append(row)

        self.samples = []
        for cat_dir in sorted(_iter_dirs(data_dir)):
            cat_name = os.path.basename(cat_dir)

            sub_dirs = sorted(_iter_dirs(cat_dir))

            has_subcat = False
            if sub_dirs:
                first_sub = sub_dirs[0]
                first_sub_patch = os.path.join(first_sub, "patch_plane.npy")
                first_sub_sub = _iter_dirs(first_sub)
                if not os.path.exists(first_sub_patch) and first_sub_sub:
                    has_subcat = True

            if has_subcat:
                for subcat_dir in sub_dirs:
                    subcat_name = os.path.basename(subcat_dir)
                    full_cat_name = f"{cat_name}/{subcat_name}"
                    for sample_dir in sorted(_iter_dirs(subcat_dir)):
                        _append_sample_records(sample_dir, full_cat_name)
            else:
                for sample_dir in sub_dirs:
                    _append_sample_records(sample_dir, cat_name)

        if max_samples:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(f"未找到任何样本: {data_dir}")

        cat_counts = Counter(s["category"] for s in self.samples)
        freq_counts = Counter(s["freq_ghz"] for s in self.samples)
        pol_counts = Counter(s.get("polarization", 0) for s in self.samples)
        print(f"[EMMultiFreqDataset] {len(self.samples)} 条记录")
        print(f"  类别: {dict(sorted(cat_counts.items()))}")
        print(f"  频点: {sorted(freq_counts.keys())} GHz")
        print(f"  极化记录: {dict(sorted(pol_counts.items()))}")

        # 按 (category, name, polarization) 分组，供频率插值训练使用。
        self.shape_to_indices = defaultdict(list)
        for i, s in enumerate(self.samples):
            key = (s["category"], s["name"], s.get("polarization", 0))
            self.shape_to_indices[key].append(i)

    @staticmethod
    def _normalize_polarizations(polarizations):
        if polarizations is None:
            return None
        if isinstance(polarizations, (int, np.integer)):
            return [int(polarizations)]
        return [int(p) for p in polarizations]

    def _select_polarizations(self, n_pols):
        if self.polarizations is None:
            return list(range(n_pols))
        return [p for p in self.polarizations if p < n_pols]

    def set_output_normalization(self, normalizer):
        """设置输出归一化器（data.norm_utils.OutputNormalizer）。"""
        self.output_normalizer = normalizer

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
        pol = int(s.get("polarization", 0))

        # 输入：SDF + Mask
        patch = np.load(s["patch"], mmap_mode="r").astype(np.float32)
        sdf = patch[:, :, 0] / (np.abs(patch[:, :, 0]).max() + 1e-8)
        mask = patch[:, :, 1]
        inp = torch.from_numpy(np.stack([sdf, mask], axis=0))

        # 输出：根据格式读取
        if s.get("fmt", "v1") == "v2_cache":
            planes = np.load(s["field"], mmap_mode="r")
            if planes.ndim == 6:
                plane = planes[pol, s["freq_idx"], :, :, :]
            elif planes.ndim == 5 and planes.shape[0] == 2 and planes.shape[1] == 9:
                # 双极化缓存 (2,9,H,W,C)
                plane = planes[pol, s["freq_idx"], :, :, :]
            else:
                plane = planes[s["freq_idx"]]
            out = _extract_output_channels(plane, self.output_field)

        elif s.get("fmt", "v1") == "v2":
            field_volume = np.load(s["field"], mmap_mode="r")
            if field_volume.ndim == 6:
                if pol >= field_volume.shape[0]:
                    raise ValueError(
                        f"{s['field']} 只有 {field_volume.shape[0]} 个极化，"
                        f"请求极化 {pol}"
                    )
                field_volume = field_volume[pol]
            elif pol != 0:
                raise ValueError(
                    f"{s['field']} 是单极化文件，不能请求极化 {pol}"
                )

            def _extract_all_channels(d_lam):
                plane = _interp_field_at_d_over_lambda(
                    field_volume, s["freq_idx"], d_lam
                )
                return _extract_output_channels(plane, "all")

            out = _extract_all_channels(self.target_d_over_lambda)

            # 获取 z-1mm 和 z+1mm 的场，用于 PINN 中心差分（v2 体数据路径）。
            lam_mm = _wavelength_mm(s["freq_ghz"])
            delta_d_lam = 1.0 / lam_mm
            out_zminus = _extract_all_channels(self.target_d_over_lambda - delta_d_lam)
            out_zplus = _extract_all_channels(self.target_d_over_lambda + delta_d_lam)

            # 兼容旧 output_field 配置。
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

        # 散射场：减去拟合的理想背景场（仅 v2 格式支持）。
        if self.use_scatter and s.get("fmt", "v1").startswith("v2"):
            from data.fit_background import get_background_field
            from data.interpolate_field import get_wavelength_mm
            freq_ghz = s["freq_ghz"]
            z_mm = self.target_d_over_lambda * get_wavelength_mm(freq_ghz)
            bg = get_background_field(
                freq_ghz, z_mm, self.bg_params_path, polarization=pol
            )  # (12,)
            n_ch = out.shape[0]
            bg_broadcast = bg[:n_ch, None, None]  # (n_ch, 1, 1)
            out = out - bg_broadcast

        if self.normalize:
            if self.output_normalizer is None:
                raise RuntimeError("normalize=True 但未设置 output_normalizer")
            sl = self._output_norm_slice(n_channels=out.shape[0])
            out = self.output_normalizer.normalize_np(out, ch_slice=sl)

        freq_norm = (s["freq_ghz"] - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)
        pol_norm = float(pol)

        # 解析形状和类型
        category_parts = s["category"].split("/")
        shape_name = category_parts[0]
        type_name = category_parts[1] if len(category_parts) > 1 else "normal"

        res = {
            "input": inp,
            "output": torch.from_numpy(out),
            "name": s["name"],
            "category": s["category"],
            "freq_ghz": torch.tensor(s["freq_ghz"], dtype=torch.float32),
            "freq": torch.tensor([freq_norm, pol_norm], dtype=torch.float32),
            "pol_idx": torch.tensor([pol], dtype=torch.long),
            "shape_idx": torch.tensor([get_shape_index(shape_name)], dtype=torch.long),
            "type_idx": torch.tensor([get_type_index(type_name)], dtype=torch.long),
        }

        if "out_zminus" in locals() and "out_zplus" in locals():
            res["out_zminus"] = torch.from_numpy(out_zminus)
            res["out_zplus"] = torch.from_numpy(out_zplus)

        return res

# -*- coding: utf-8 -*-
"""相位参考系（PhaseFrame）：把"随频率快速旋转的全局相位"从网络里拿掉。

诊断结论（Meep 数据，rectangle，pol0/1 × 9 频点；见 PLAN.md 第 10 节）：
  * 散射场 = 空间均值(DC) + 空间起伏(AC)，两部分各自"几乎是频率无关的形状 × 随频率线性旋转的相位"
  * AC（所有通道一致）：相邻 0.5 GHz 旋转约 -148°（-155°→-146° 缓变），旋转后相邻频点 AC 插值误差 1.73→0.07
  * DC：共极化分量 (pol0: Ey,Hx / pol1: Ex,Hy) 约 -80°/0.5GHz；交叉极化分量约 -160°/0.5GHz
  * FiLM 只能做逐通道的缩放/平移，无法在 (实部,虚部) 之间做随 f 变化的旋转 → 未见频点时相位错位
做法：网络在"去旋转系"输出 u，包装层 F = DC(u)·e^{jθ_dc(f)} + AC(u)·e^{jθ_ac(f)}（归一化空间 ↔ 物理空间在内部完成）。
线性、可逆、可导；损失/指标仍在原始(旋转)系里算，与旧结果可直接比较。
θ 表只用训练频点估计（10 GHz 处的值由相邻两步插值得到，不用 10 GHz 数据）。
"""
import math
import numpy as np
import torch

F0, DF = 8.0, 0.5                      # 频率网格
# 相邻 0.5 GHz 的 AC 步进相位（度）：8→8.5, ..., 11.5→12（9.5→10.5 两步取 9→9.5 与 10.5→11 的均值）
AC_STEPS = [-155.6, -153.5, -148.6, -148.4, -148.4, -148.25, -147.8, -145.9]
# DC 步进（度/0.5GHz）：[Ex,Ey,Ez,Hx,Hy,Hz]，按极化
DC_STEP = {0: [-160., -80., -160., -80., -160., -160.],
           1: [-80., -160., -160., -160., -80., -160.]}

# ── 物理总场（Raw-Field）去缠绕规范相位表（仅由 8 个训练频点估计，不含 10.0 GHz）──
# 在极化规范系（pol=0 原系，pol=1 经精确麦克斯韦反射对偶变换 T）下：
#   1. (0,0) 阶 Floquet TEM 反射平面波相位步进（度/0.5GHz）：
TEM_PERP_STEPS = [-155.0, -160.1, -158.4, -154.8, -154.8, -156.8, -150.7, -149.7]
TEM_PARA_STEPS = [-143.8, -139.8, -136.9, -135.8, -135.8, -135.2, -135.8, -136.3]
#   2. 6 通道总场 DC 步进表（供关闭 TEM 阻抗耦合的消融实验使用）：
RAW_DC_STEPS_6 = [
    [-160.73, -163.71, -160.91, -149.24, -149.24, -163.88, -161.87, -150.55],
    [-133.59, -133.38, -131.74, -131.98, -131.98, -132.58, -131.87, -131.93],
    [-148.05, -165.59, -163.14, -151.96, -151.96, -150.69, -151.79, -147.64],
    [-135.62, -134.08, -134.72, -133.15, -133.15, -132.96, -134.30, -134.70],
    [-160.78, -163.76, -160.96, -149.29, -149.29, -163.93, -161.92, -150.60],
    [-156.13, -166.52, -177.49, -138.22, -138.22, -136.66, -136.47, -136.78],
]
#   3. 6 通道高阶 Floquet 衍射（零均值 AC）步进表（度/0.5GHz）：
RAW_AC_STEPS_6 = [
    [-157.54, -160.19, -159.34, -145.55, -145.55, -154.86, -154.45, -148.36],
    [-150.34, -155.85, -156.34, -143.39, -143.39, -148.49, -149.69, -146.65],
    [-151.77, -157.20, -157.22, -144.59, -144.59, -151.09, -152.10, -148.79],
    [-151.57, -154.00, -154.13, -141.93, -141.93, -148.02, -150.20, -145.21],
    [-158.54, -162.51, -165.54, -143.55, -143.55, -152.42, -149.62, -147.66],
    [-153.88, -155.93, -156.69, -141.95, -141.95, -148.74, -149.78, -145.15],
]


def _ac_table():
    cum = [0.0]
    for s in AC_STEPS:
        cum.append(cum[-1] + s)
    ref = cum[4]                        # 以 10 GHz 为 0
    return torch.tensor([c - ref for c in cum], dtype=torch.float32)


def _cum_table_1d(steps, ref_idx=4):
    cum = [0.0]
    for s in steps:
        cum.append(cum[-1] + float(s))
    ref = cum[ref_idx]
    return torch.tensor([math.radians(c - ref) for c in cum], dtype=torch.float32)


def _cum_table_2d(steps_2d, ref_idx=4):
    return torch.stack([_cum_table_1d(row, ref_idx=ref_idx) for row in steps_2d], dim=0)


def _build_bg_polar_tables(device):
    """预计算 9 个网格频点下双极化解析背景场的极坐标幅值与去缠绕相位表。"""
    try:
        from configs import base as C
        from data.fit_background import get_background_field
        from data.interpolate_field import get_wavelength_mm
        bg_path = getattr(C, "BACKGROUND_PARAMS_PATH", "data/blank/background_params.npz")
        d_lam = float(getattr(C, "TARGET_D_OVER_LAMBDA", 0.1))
        freqs = [F0 + DF * i for i in range(9)]
        raw = np.zeros((2, 9, 12), dtype=np.float64)
        for pol in (0, 1):
            for fi, f in enumerate(freqs):
                z_mm = d_lam * get_wavelength_mm(f)
                raw[pol, fi] = get_background_field(f, z_mm, bg_path, polarization=pol)
        re = raw[:, :, 0::2]
        im = raw[:, :, 1::2]
        amp = np.hypot(re, im).astype(np.float32)                     # (2, 9, 6)
        phs = np.unwrap(np.arctan2(im, re), axis=1).astype(np.float32)  # (2, 9, 6)
        return (torch.from_numpy(amp).to(device),
                torch.from_numpy(phs).to(device))
    except Exception:
        return (torch.zeros((2, 9, 6), dtype=torch.float32, device=device),
                torch.zeros((2, 9, 6), dtype=torch.float32, device=device))


class PhaseFramedModel(torch.nn.Module):
    """`model` + 相位系的包装:forward 出来的就是**旋转回物理系**的归一化预测。

    存在的理由见 `core/checkpoint.py` 里接它的那段注释 —— 相位系过去只接在训练器
    内部,所有离线评估都在拿去旋转的预测比没去旋转的真值。包成模型之后,
    "从 checkpoint 拿到的模型"与"训练时在算损失的那个函数"是同一个东西。

    注意 `PhaseFrame` 自带设备绑定(相位表在 __init__ 里 .to(device)),所以这里
    重写 `_apply` 跟着模型一起搬设备,否则 `.cuda()` 之后会在表与张量之间炸设备不符。
    """

    def __init__(self, model, phase_frame):
        super().__init__()
        self.model = model
        self.phase_frame = phase_frame
        # 让 model_name / 其它属性查得到(评估脚本会读 model.model_name)
        self.model_name = getattr(model, "model_name", None)

    def forward(self, x, freq=None):
        return self.phase_frame(self.model(x, freq), freq)

    def _apply(self, fn, *a, **kw):
        out = super()._apply(fn, *a, **kw)
        pf = self.phase_frame
        pf.ac = fn(pf.ac)
        pf.dc = fn(pf.dc)
        pf.raw_dc_cum = fn(pf.raw_dc_cum)
        pf.raw_ac_cum = fn(pf.raw_ac_cum)
        pf.tem_perp_cum = fn(pf.tem_perp_cum)
        pf.tem_para_cum = fn(pf.tem_para_cum)
        pf.bg_amp = fn(pf.bg_amp)
        pf.bg_phs = fn(pf.bg_phs)
        return out

    def __getattr__(self, name):
        # nn.Module.__getattr__ 只查 _parameters/_buffers/_modules;其余转给内层模型,
        # 这样 `pole_table()` 这类模型自带的方法透过包装层仍然调得到。
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__["_modules"]["model"], name)


class PhaseFrame:
    def __init__(self, normalizer, freq_min, freq_max, device, scale=1.0):
        self.nm = normalizer
        self.fmin, self.fmax = float(freq_min), float(freq_max)
        self.ac = _ac_table().to(device)
        self.dc = torch.tensor([DC_STEP[0], DC_STEP[1]], dtype=torch.float32, device=device)  # (2,6)
        self.scale = float(scale)
        self.raw_dc_cum = _cum_table_2d(RAW_DC_STEPS_6).to(device)     # (6, 9) rad
        self.raw_ac_cum = _cum_table_2d(RAW_AC_STEPS_6).to(device)     # (6, 9) rad
        self.tem_perp_cum = _cum_table_1d(TEM_PERP_STEPS).to(device)   # (9,) rad
        self.tem_para_cum = _cum_table_1d(TEM_PARA_STEPS).to(device)   # (9,) rad
        self.bg_amp, self.bg_phs = _build_bg_polar_tables(device)      # (2, 9, 6)

    def _angles(self, freq):
        f = freq[:, 0] * (self.fmax - self.fmin) + self.fmin           # GHz (B,)
        x = ((f - F0) / DF).clamp(0, len(self.ac) - 1.000001)
        i0 = x.floor().long(); fr = x - i0
        th_ac = self.ac[i0] * (1 - fr) + self.ac[i0 + 1] * fr           # (B,) 度
        pol = freq[:, 1].round().long().clamp(0, 1)
        th_dc = self.dc[pol] * ((f - 10.0) / DF)[:, None]               # (B,6)
        d2r = math.pi / 180.0 * self.scale
        return th_ac[:, None] * d2r, th_dc * d2r                        # (B,1),(B,6)

    def _interp_1d(self, tbl_1d, i0, fr):
        return (tbl_1d[i0] * (1.0 - fr) + tbl_1d[i0 + 1] * fr) * self.scale

    def _interp_2d(self, tbl_2d, i0, fr):
        v0 = tbl_2d[:, i0].transpose(0, 1)
        v1 = tbl_2d[:, i0 + 1].transpose(0, 1)
        return (v0 * (1.0 - fr[:, None]) + v1 * fr[:, None]) * self.scale

    def _interp_bg(self, pol, i0, fr):
        a0 = self.bg_amp[pol, i0]
        a1 = self.bg_amp[pol, i0 + 1]
        p0 = self.bg_phs[pol, i0]
        p1 = self.bg_phs[pol, i0 + 1]
        w = fr[:, None]
        amp = a0 * (1.0 - w) + a1 * w
        phs = p0 * (1.0 - w) + p1 * w
        re = amp * torch.cos(phs)
        im = amp * torch.sin(phs)
        B = pol.shape[0]
        return torch.stack([re, im], dim=2).reshape(B, 12, 1, 1)

    def apply_raw_canonical(self, u_canon, freq):
        """从极化规范系的去旋转物理预测重构散射场并归一化。"""
        B, C, H, W = u_canon.shape
        if freq is None:
            f_norm = u_canon.new_full((B,), 0.5)
            pol = torch.zeros(B, dtype=torch.long, device=u_canon.device)
        else:
            if freq.dim() == 1:
                freq = freq.unsqueeze(0)
            f_norm = freq[:, 0]
            pol = (freq[:, 1].round().long().clamp(0, 1)
                   if freq.shape[1] > 1
                   else torch.zeros(B, dtype=torch.long, device=u_canon.device))

        f = f_norm * (self.fmax - self.fmin) + self.fmin               # (B,) GHz
        x_idx = ((f - F0) / DF).clamp(0.0, float(self.tem_perp_cum.shape[0] - 1.000001))
        i0 = x_idx.floor().long()
        fr = x_idx - i0.float()

        u_re, u_im = u_canon[:, 0::2], u_canon[:, 1::2]                # (B, 6, H, W)
        u_dc_re = u_re.mean(dim=(2, 3), keepdim=True)
        u_dc_im = u_im.mean(dim=(2, 3), keepdim=True)
        u_ac_re = u_re - u_dc_re
        u_ac_im = u_im - u_dc_im

        # 1. 高阶 Floquet 衍射分量（零均值 AC）相位恢复
        th_ac = self._interp_2d(self.raw_ac_cum, i0, fr)[:, :, None, None]  # (B, 6, 1, 1)
        ca, sa = torch.cos(th_ac), torch.sin(th_ac)
        ac_re = u_ac_re * ca - u_ac_im * sa
        ac_im = u_ac_re * sa + u_ac_im * ca

        # 2. (0,0) 阶 Floquet 镜面分量（DC）相位恢复与麦克斯韦 TEM 波阻抗重构
        if getattr(u_canon, "_tem_dc", True):
            th_perp = self._interp_1d(self.tem_perp_cum, i0, fr)[:, None, None, None]
            th_para = self._interp_1d(self.tem_para_cum, i0, fr)[:, None, None, None]
            c_perp, s_perp = torch.cos(th_perp), torch.sin(th_perp)
            c_para, s_para = torch.cos(th_para), torch.sin(th_para)

            up_re = 0.5 * (u_dc_re[:, 0:1] + 376.50 * u_dc_re[:, 4:5])
            up_im = 0.5 * (u_dc_im[:, 0:1] + 376.50 * u_dc_im[:, 4:5])
            upa_re = 0.5 * (u_dc_re[:, 1:2] - 376.50 * u_dc_re[:, 3:4])
            upa_im = 0.5 * (u_dc_im[:, 1:2] - 376.50 * u_dc_im[:, 3:4])

            e_perp_re = up_re * c_perp - up_im * s_perp
            e_perp_im = up_re * s_perp + up_im * c_perp
            e_para_re = upa_re * c_para - upa_im * s_para
            e_para_im = upa_re * s_para + upa_im * c_para

            ang_inc = ((169.05 - 128.353 * ((f - 10.0) / 0.5)) * (math.pi / 180.0))[:, None, None, None]
            amp_inc = (1824.0 - 0.75 * (f - 10.0))[:, None, None, None]
            e_inc_re = amp_inc * torch.cos(ang_inc)
            e_inc_im = amp_inc * torch.sin(ang_inc)

            ang_z = (0.10 * f * (math.pi / 180.0))[:, None, None, None]
            inv_z_re = torch.cos(ang_z) / 376.50
            inv_z_im = -torch.sin(ang_z) / 376.50

            d3_re = e_inc_re - e_para_re
            d3_im = e_inc_im - e_para_im
            dc3_re = d3_re * inv_z_re - d3_im * inv_z_im
            dc3_im = d3_re * inv_z_im + d3_im * inv_z_re
            dc4_re = e_perp_re * inv_z_re - e_perp_im * inv_z_im
            dc4_im = e_perp_re * inv_z_im + e_perp_im * inv_z_re

            dc2_re = u_dc_re[:, 2:3] * ca[:, 2:3] - u_dc_im[:, 2:3] * sa[:, 2:3]
            dc2_im = u_dc_re[:, 2:3] * sa[:, 2:3] + u_dc_im[:, 2:3] * ca[:, 2:3]
            dc5_re = u_dc_re[:, 5:6] * ca[:, 5:6] - u_dc_im[:, 5:6] * sa[:, 5:6]
            dc5_im = u_dc_re[:, 5:6] * sa[:, 5:6] + u_dc_im[:, 5:6] * ca[:, 5:6]

            dc_re = torch.cat([e_perp_re, e_inc_re + e_para_re, dc2_re, dc3_re, dc4_re, dc5_re], dim=1)
            dc_im = torch.cat([e_perp_im, e_inc_im + e_para_im, dc2_im, dc3_im, dc4_im, dc5_im], dim=1)
        else:
            th_dc = self._interp_2d(self.raw_dc_cum, i0, fr)[:, :, None, None]
            cd, sd = torch.cos(th_dc), torch.sin(th_dc)
            ang_inc = ((169.05 - 128.353 * ((f - 10.0) / 0.5)) * (math.pi / 180.0))[:, None, None, None]
            amp_inc = (1824.0 - 0.75 * (f - 10.0))[:, None, None, None]
            e_inc_re = amp_inc * torch.cos(ang_inc)
            e_inc_im = amp_inc * torch.sin(ang_inc)
            ang_z = (0.10 * f * (math.pi / 180.0))[:, None, None, None]
            inv_z_re = torch.cos(ang_z) / 376.50
            inv_z_im = -torch.sin(ang_z) / 376.50
            p_re = u_dc_re.new_tensor([-14.88, 303.43, 0.0, -303.43 / 376.50, -14.88 / 376.50, 0.0]).view(1, 6, 1, 1)
            p_im = u_dc_im.new_tensor([5.30, -112.24, 0.0, 112.24 / 376.50, 5.30 / 376.50, 0.0]).view(1, 6, 1, 1)
            tot_u_re = u_dc_re + p_re
            tot_u_im = u_dc_im + p_im
            rot_re = tot_u_re * cd - tot_u_im * sd
            rot_im = tot_u_re * sd + tot_u_im * cd
            inc3_re = e_inc_re * inv_z_re - e_inc_im * inv_z_im
            inc3_im = e_inc_re * inv_z_im + e_inc_im * inv_z_re
            z1 = torch.zeros_like(e_inc_re)
            inc_re = torch.cat([z1, e_inc_re, z1, inc3_re, z1, z1], dim=1)
            inc_im = torch.cat([z1, e_inc_im, z1, inc3_im, z1, z1], dim=1)
            dc_re = rot_re + inc_re
            dc_im = rot_im + inc_im

        raw_re = dc_re + ac_re
        raw_im = dc_im + ac_im
        y_raw_canon = torch.stack([raw_re, raw_im], dim=2).reshape(B, C, H, W)

        # 3. 极化 1 的精确麦克斯韦对偶反射变换 T: (x <-> y, E_x <-> E_y, H_x <-> -H_y, H_z -> -H_z)
        if getattr(u_canon, "_pol_equivariant", True):
            yT = y_raw_canon.transpose(-2, -1)
            y_raw_p1 = torch.cat([
                yT[:, 2:4], yT[:, 0:2], yT[:, 4:6],
                -yT[:, 8:10], -yT[:, 6:8], -yT[:, 10:12],
            ], dim=1)
            mask_p1 = (pol == 1).view(B, 1, 1, 1)
            y_raw = torch.where(mask_p1, y_raw_p1, y_raw_canon)
        else:
            y_raw_p1 = torch.cat([
                y_raw_canon[:, 2:4], y_raw_canon[:, 0:2], y_raw_canon[:, 4:6],
                -y_raw_canon[:, 8:10], -y_raw_canon[:, 6:8], -y_raw_canon[:, 10:12],
            ], dim=1)
            mask_p1 = (pol == 1).view(B, 1, 1, 1)
            y_raw = torch.where(mask_p1, y_raw_p1, y_raw_canon)

        # 4. 减去解析背景驻波 B(f, pol) 得到物理散射场，并附带零损耗物理缓存
        bg = self._interp_bg(pol, i0, fr).to(y_raw.dtype)
        y_scat = y_raw - bg
        out_norm = self.nm.normalize(y_scat)
        if torch.is_tensor(out_norm):
            out_norm._phys_cache = y_scat
        return out_norm

    def __call__(self, pred_norm, freq):
        if getattr(pred_norm, "_is_raw_canonical", False):
            return self.apply_raw_canonical(pred_norm, freq)
        if getattr(pred_norm, "_pol_equivariant", False):
            B, C, H, W = pred_norm.shape
            if freq is None:
                freq_c = pred_norm.new_zeros((B, 2))
                freq_c[:, 0] = 0.5
                pol = torch.zeros(B, dtype=torch.long, device=pred_norm.device)
            else:
                if freq.dim() == 1:
                    freq = freq.unsqueeze(0)
                pol = (freq[:, 1].round().long().clamp(0, 1)
                       if freq.shape[1] > 1
                       else torch.zeros(B, dtype=torch.long, device=pred_norm.device))
                freq_c = freq.clone()
                if freq_c.shape[1] > 1:
                    freq_c[:, 1] = 0.0
            u_dc = pred_norm.mean(dim=(-2, -1), keepdim=True)
            u_ac = pred_norm - u_dc
            dc_s = pred_norm.new_tensor([450.0, 500.0, 5800.0, 5300.0, 0.0, 0.0, 11.0, 9.6, 1.2, 1.3, 0.0, 0.0]).view(1, 12, 1, 1)
            dc_p = pred_norm.new_tensor([2.81, 8.19, 1263.22, 1100.20, 0.0, 0.0, 2.49, -1.44, 0.01, 0.02, 0.0, 0.0]).view(1, 12, 1, 1)
            ac_s = pred_norm.new_tensor([545.69, 515.14, 790.28, 682.19, 267.47, 250.32, 1.3746, 1.3183, 1.7848, 1.5641, 0.7997, 0.7731]).view(1, 12, 1, 1)
            x = (u_dc * dc_s + dc_p) + u_ac * ac_s
            z_re, z_im = x[:, 0::2], x[:, 1::2]
            dre, dim = z_re.mean(dim=(2, 3), keepdim=True), z_im.mean(dim=(2, 3), keepdim=True)
            are, aim = z_re - dre, z_im - dim
            th_ac, th_dc = self._angles(freq_c)
            ca, sa = torch.cos(th_ac)[..., None, None], torch.sin(th_ac)[..., None, None]
            cd, sd = torch.cos(th_dc)[..., None, None], torch.sin(th_dc)[..., None, None]
            re = dre * cd - dim * sd + are * ca - aim * sa
            im = dre * sd + dim * cd + are * sa + aim * ca
            out_c = torch.stack([re, im], dim=2).reshape(B, C, H, W)
            outT = out_c.transpose(-2, -1)
            out_p1 = torch.cat([
                outT[:, 2:4], outT[:, 0:2], outT[:, 4:6],
                -outT[:, 8:10], -outT[:, 6:8], -outT[:, 10:12],
            ], dim=1)
            mask_p1 = (pol == 1).view(B, 1, 1, 1)
            out = torch.where(mask_p1, out_p1, out_c)
            out_norm = self.nm.normalize(out)
            if torch.is_tensor(out_norm):
                out_norm._phys_cache = out
            return out_norm
        x = self.nm.denormalize(pred_norm)                              # 物理空间 (B,12,H,W)
        B, C, H, W = x.shape
        z_re, z_im = x[:, 0::2], x[:, 1::2]                             # (B,6,H,W)
        dre, dim = z_re.mean(dim=(2, 3), keepdim=True), z_im.mean(dim=(2, 3), keepdim=True)
        are, aim = z_re - dre, z_im - dim
        th_ac, th_dc = self._angles(freq)
        ca, sa = torch.cos(th_ac)[..., None, None], torch.sin(th_ac)[..., None, None]
        cd, sd = torch.cos(th_dc)[..., None, None], torch.sin(th_dc)[..., None, None]
        re = dre * cd - dim * sd + are * ca - aim * sa
        im = dre * sd + dim * cd + are * sa + aim * ca
        out = torch.stack([re, im], dim=2).reshape(B, C, H, W)          # 交错回 re,im,re,im...
        return self.nm.normalize(out)

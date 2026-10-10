# -*- coding: utf-8 -*-
"""损失函数 - 修改损失函数只需要改这个文件"""

import torch


def field_geom_loss(
    pred,
    true,
    w_dir=1.0,
    w_phase=1.0,
    w_mag=1.0,
    eps=1e-6,
):
    """
    全场几何损失（输出仍为 6 通道实部/虚部）：
    1) 方向角误差：基于 |Ex|,|Ey|,|Ez| 的方向夹角
    2) 复矢量相位角误差：基于复内积相位差
    3) 总场幅值误差：|E| 的相对误差
    """
    # 6 通道 -> 3 个复分量
    pred_c = torch.stack([
        torch.complex(pred[:, 0], pred[:, 1]),
        torch.complex(pred[:, 2], pred[:, 3]),
        torch.complex(pred[:, 4], pred[:, 5]),
    ], dim=1)  # (B,3,H,W), complex
    true_c = torch.stack([
        torch.complex(true[:, 0], true[:, 1]),
        torch.complex(true[:, 2], true[:, 3]),
        torch.complex(true[:, 4], true[:, 5]),
    ], dim=1)

    # 分量幅值与总场幅值
    pred_amp = torch.abs(pred_c)  # (B,3,H,W)
    true_amp = torch.abs(true_c)
    pred_total = torch.sqrt((pred_amp ** 2).sum(dim=1) + eps)  # (B,H,W)
    true_total = torch.sqrt((true_amp ** 2).sum(dim=1) + eps)

    # 对低幅值区域降权（这些点预测意义小）
    w = true_total.detach()
    w = w / (w.mean(dim=[1, 2], keepdim=True) + eps)
    w = w.clamp(min=0.0, max=10.0)
    w_norm = w.sum() + eps

    # 1) 方向角误差：用分量幅值向量方向
    pred_dir = pred_amp / (torch.sqrt((pred_amp ** 2).sum(dim=1, keepdim=True) + eps))
    true_dir = true_amp / (torch.sqrt((true_amp ** 2).sum(dim=1, keepdim=True) + eps))
    cos_dir = (pred_dir * true_dir).sum(dim=1).clamp(-1.0, 1.0)  # (B,H,W)
    dir_loss = ((1.0 - cos_dir) * w).sum() / w_norm

    # 2) 复矢量相位角误差：复内积相位差
    inner = (pred_c * torch.conj(true_c)).sum(dim=1)  # (B,H,W), complex
    cos_phase = (inner.real / (torch.abs(inner) + eps)).clamp(-1.0, 1.0)
    phase_loss = ((1.0 - cos_phase) * w).sum() / w_norm

    # 3) 总场幅值误差（全局归一化，避免小幅值点产生爆炸性相对误差）
    global_scale = true_total.mean(dim=[1, 2], keepdim=True).detach() + eps
    mag_rel = torch.abs(pred_total - true_total) / global_scale
    mag_loss = (mag_rel * w).sum() / w_norm

    total = w_dir * dir_loss + w_phase * phase_loss + w_mag * mag_loss
    return total, {
        "dir_loss": dir_loss.detach(),
        "phase_loss": phase_loss.detach(),
        "mag_loss": mag_loss.detach(),
    }


def _complex_total_amp(field, eps=1e-12):
    """(B, 2M, H, W) 实虚成对 -> (B, M, H, W) 各分量幅值"""
    B, C, H, W = field.shape
    M = C // 2
    return torch.sqrt(field[:, 0::2] ** 2 + field[:, 1::2] ** 2 + eps)


def field_geom_loss_eh(pred, true, **kw):
    """全场几何损失，E/H 分组各自算完再平均（12 通道版）。

    背景：`field_geom_loss` 只读传入张量的前 6 个通道、当作 3 个复分量
    `(0,1),(2,3),(4,5)`。`tools/eval.py` 曾经直接把 12 通道张量喂进去，
    拿到的其实是**只有电场**的 geom，而它自己的 docstring 写着"12 通道整体算的"，
    于是终端上打印的 Geom 与 `tools/train.py` 的 `val_geom` 口径悄悄不一致。

    这里把 E/H 拆分集中到一处，训练与评估共用，避免各处再手写切片。
    低于 12 通道时退回单组计算（兼容旧的 6 通道 E-only 模型）。
    """
    v_e, _ = field_geom_loss(pred[:, :6], true[:, :6], **kw)
    if pred.shape[1] >= 12:
        v_h, _ = field_geom_loss(pred[:, 6:12], true[:, 6:12], **kw)
        return (v_e + v_h) * 0.5
    return v_e


def e_h_total_rel_l2(pred, true, eps=1e-6, h_boost=1.0):
    """
    E/H 分组【总场幅值】相对 L2，再平均（物理空间）。

    核心动机：E 组 rms 与 H 组 rms 相差约 400 倍，直接合并算总场相对 L2
    会让 H 场完全被淹没。改为 E、H 各自算总场相对 L2 再取平均，
    迫使模型对 E 和 H 都达到比例精度 —— 对最终总场最有利。

    h_boost>1 时提高 H 组（弱分量）权重：FNO 存在"傅里叶参数化偏差"，
    天然偏置主导频率(强 E 场)，非主导的 H 场分量会被淹没（SpecB-FNO 依据），
    额外加权 H 组可促使网络真正去拟合弱场结构。
    """
    def _total(field):
        amp = _complex_total_amp(field, eps=eps)            # (B,M,H,W)
        return torch.sqrt((amp ** 2).sum(dim=1) + eps)       # (B,H,W)
    if pred.shape[1] >= 12:
        e = 0.5 * (_rel_l2_2d(_total(pred[:, :6]), _total(true[:, :6]), eps)
                   + h_boost * _rel_l2_2d(_total(pred[:, 6:12]), _total(true[:, 6:12]), eps))
        return e
    return _rel_l2_2d(_total(pred), _total(true), eps)


def _rel_l2_2d(pred, true, eps=1e-6):
    B = pred.shape[0]
    d = (pred - true).reshape(B, -1).norm(dim=1)
    n = true.reshape(B, -1).norm(dim=1).clamp(min=eps)
    return (d / n).mean()


def channel_group_balanced_l2(pred, true, eps=1e-6, h_boost=1.0, ch_weight_power=0.0):
    """逐通道相对 L2：E 组(6) 通道内取均值 + H 组(6) 通道内取均值，再平均。

    物理空间逐通道相对误差是尺度不变的，天然平衡 Ex(200) / Ey(2300) / Ez(350)
    与 H 组内各通道的差异；E、H 两组再取平均实现跨组平衡。

    h_boost>1 时提高 H 组权重：H 组是非主导(弱)分量，易被 FNO 谱偏差淹没，
    提权促使模型精细化拟合 H 组各通道。

    ## ch_weight_power：逐通道等权的隐含后果（训练侧用 0.5，指标侧用 0.0）

    逐通道等权平均**不等于**逐通道同等影响。rel_c = ||Δc|| / ||t_c||，于是
    损失对第 c 通道的梯度范数 ∝ 1/||t_c|| —— **通道越弱，梯度越大**。
    实测（散射场、真实数据、当前 12 通道配置）：

        ||t_c|| 从 Ex 的 109062 到 Hz 的 12.2，跨 4 个数量级
        → Hz_r 单独拿走 43% 的梯度，Hz/Hy 四个通道合计 76%，
          而 rms 最大的 Ey(1506) 只拿 0.01%、Ex(2700) 拿 0.00%

    也就是说：名义上 L1/L3/L8 的权重是 1.0/0.5/1.0，实际梯度贡献是
    0.8% / 1.3% / 97.8%。**L8 内部又把 3/4 的预算给了 4 个最弱的场分量**，
    而 best.pth 的判据 val_geom 对应的 L3 只拿到 1.3%。

    本参数把"组内平均"从等权换成按 ||t_c||^p 加权（权重和为 1）：

        p=0.0 → 与旧行为**逐位相同**（指标的默认值，保证 val_rel 口径不变）
        p=0.5 → 组内份额跨度从数千倍压到几十倍（训练默认）
        p=1.0 → 退化成绝对误差等权，Ey 反过来碾压 Hz

    注意它**只影响训练**：`Trainer.rel_l2_error` 不传该参数，故 val_rel
    仍是旧口径，跨实验可比（由 tools/smoke.py::check_metric_independence 守着）。
    """
    def _grp(field_p, field_t):
        B, nch = field_p.shape[:2]
        d = (field_p - field_t).reshape(B, nch, -1).norm(dim=2)
        n = field_t.reshape(B, nch, -1).norm(dim=2).clamp(min=eps)
        rel = d / n                                   # (B, nch)
        if not ch_weight_power:
            return rel.mean()
        w = n.pow(float(ch_weight_power))
        w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
        return (w * rel).sum(dim=1).mean()
    if pred.shape[1] >= 12:
        return 0.5 * (_grp(pred[:, :6], true[:, :6])
                      + h_boost * _grp(pred[:, 6:12], true[:, 6:12]))
    return _grp(pred, true)


def lowfreq_fft_loss(pred, true, low_h=4, low_w=4):
    """频域低频结构相对误差：比较 rFFT 低频块（默认 4x4）的相对 L2。"""
    pred_ft = torch.fft.rfft2(pred.float(), norm="ortho")
    true_ft = torch.fft.rfft2(true.float(), norm="ortho")

    h_bins = min(low_h, pred_ft.shape[-2])
    w_bins = min(low_w, pred_ft.shape[-1])

    pred_low = pred_ft[..., :h_bins, :w_bins]
    true_low = true_ft[..., :h_bins, :w_bins]

    # (B, C, Hf, Wf) -> (B, C, -1)
    diff_norm = torch.norm((pred_low - true_low).reshape(pred_low.shape[0], pred_low.shape[1], -1), dim=2)
    true_norm = torch.norm(true_low.reshape(true_low.shape[0], true_low.shape[1], -1), dim=2).clamp(min=1e-6)
    return (diff_norm / true_norm).mean()


def highfreq_suppress_loss(pred, cutoff=12):
    """
    高频抑制（按通道独立）：
    - 对每个通道分别计算高频能量占比 high_energy / total_energy
    - 再对 batch 与通道取平均
    这样可避免单个大幅值通道主导惩罚。
    """
    pred_ft = torch.fft.rfft2(pred.float(), norm="ortho")  # (B,C,H,W//2+1)
    Hf, Wf = pred_ft.shape[-2], pred_ft.shape[-1]

    kx = torch.arange(Hf, device=pred.device, dtype=torch.float32)
    kx = torch.minimum(kx, Hf - kx)  # 处理 rfft2 第一维的正负频折叠
    ky = torch.arange(Wf, device=pred.device, dtype=torch.float32)
    kr = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)

    high_mask = kr > float(cutoff)
    if not torch.any(high_mask):
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    spec_power = pred_ft.abs().pow(2)  # (B,C,Hf,Wf)
    # (B,C,N_high)
    high_power = spec_power[..., high_mask]
    # 每通道高频平均能量
    high_energy = high_power.mean(dim=-1)  # (B,C)
    # 每通道总能量用于归一化，减小通道幅值尺度差异的影响
    total_energy = spec_power.mean(dim=[-2, -1]).clamp(min=1e-12)  # (B,C)
    channel_ratio = high_energy / total_energy
    return channel_ratio.mean().to(pred.dtype)


def freq_band_loss(pred, true, low_h=4, low_w=4, hf_cutoff=12):
    """频域组合损失：低频相对误差 + 高频抑制。"""
    low_loss = lowfreq_fft_loss(pred, true, low_h=low_h, low_w=low_w)
    hf_loss = highfreq_suppress_loss(pred, cutoff=hf_cutoff)
    return low_loss, hf_loss

def field_rel_l2_loss(pred, true, freq=None, cat=None,
                      freq_weights=None, cat_weights=None, global_mean_dc=None):
    """
    将 6 通道 (Ex_r, Ex_i, Ey_r, Ey_i, Ez_r, Ez_i)
    先还原为总场模值 |E| = sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2)，
    再统一计算相对 L2。
    这样把三个分量完全合在一起优化，避免分量独立权重带来的偏置。
    """
    B, nC = pred.shape[:2]
    if nC >= 6:
        pred_ex = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-12)
        pred_ey = torch.sqrt(pred[:, 2] ** 2 + pred[:, 3] ** 2 + 1e-12)
        pred_ez = torch.sqrt(pred[:, 4] ** 2 + pred[:, 5] ** 2 + 1e-12)
        true_ex = torch.sqrt(true[:, 0] ** 2 + true[:, 1] ** 2 + 1e-12)
        true_ey = torch.sqrt(true[:, 2] ** 2 + true[:, 3] ** 2 + 1e-12)
        true_ez = torch.sqrt(true[:, 4] ** 2 + true[:, 5] ** 2 + 1e-12)

        pred_total = torch.sqrt(pred_ex ** 2 + pred_ey ** 2 + pred_ez ** 2 + 1e-12)
        true_total = torch.sqrt(true_ex ** 2 + true_ey ** 2 + true_ez ** 2 + 1e-12)

        diff = (pred_total - true_total).reshape(B, -1).norm(dim=1)
        norm = true_total.reshape(B, -1).norm(dim=1).clamp(min=1e-6)
        return (diff / norm).mean()

    # 兜底：非 6 通道输入按整体相对 L2 计算
    diff = (pred - true).reshape(B, -1).norm(dim=1)
    norm = true.reshape(B, -1).norm(dim=1).clamp(min=1e-6)
    return (diff / norm).mean()


def dc_ac_rel_loss(pred, true, w_dc=1.0, w_ac=1.0, eps=1e-6):
    """DC/AC 相对误差约束：比较空间均值(DC)与去均值分量(AC)的相对误差。"""
    # DC: 每样本每通道空间均值
    pred_dc = pred.mean(dim=[2, 3])
    true_dc = true.mean(dim=[2, 3])

    dc_rel = torch.abs(pred_dc - true_dc) / (true_dc.abs() + eps)
    dc_rel = dc_rel.mean()

    # AC: 去掉 DC 后的相对误差
    pred_ac = pred - pred_dc[:, :, None, None]
    true_ac = true - true_dc[:, :, None, None]
    ac_rel = field_rel_l2_loss(pred_ac, true_ac)

    total = w_dc * dc_rel + w_ac * ac_rel
    return total, {
        "dc_rel": dc_rel.detach(),
        "ac_rel": ac_rel.detach(),
    }


def dc_ac_directional_mean_loss(pred, true, w_dc=1.0, w_ac=1.0, eps=1e-6):
    """
    按三个物理方向(Ex/Ey/Ez)的幅值分别计算 DC/AC，再对三方向取均值。

    - 方向幅值: |Ex|, |Ey|, |Ez|
    - DC: 每个方向的空间均值相对误差
    - AC: 每个方向去均值后的相对 L2
    """
    if pred.shape[1] < 6 or true.shape[1] < 6:
        return dc_ac_rel_loss(pred, true, w_dc=w_dc, w_ac=w_ac, eps=eps)

    pred_ex = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + eps)
    pred_ey = torch.sqrt(pred[:, 2] ** 2 + pred[:, 3] ** 2 + eps)
    pred_ez = torch.sqrt(pred[:, 4] ** 2 + pred[:, 5] ** 2 + eps)
    true_ex = torch.sqrt(true[:, 0] ** 2 + true[:, 1] ** 2 + eps)
    true_ey = torch.sqrt(true[:, 2] ** 2 + true[:, 3] ** 2 + eps)
    true_ez = torch.sqrt(true[:, 4] ** 2 + true[:, 5] ** 2 + eps)

    pred_dirs = torch.stack([pred_ex, pred_ey, pred_ez], dim=1)  # (B,3,H,W)
    true_dirs = torch.stack([true_ex, true_ey, true_ez], dim=1)

    # DC: 三方向空间均值的相对误差
    pred_dc = pred_dirs.mean(dim=[2, 3])
    true_dc = true_dirs.mean(dim=[2, 3])
    dc_rel = (torch.abs(pred_dc - true_dc) / (true_dc.abs() + eps)).mean()

    # AC: 三方向去均值后的相对L2，再平均
    pred_ac = pred_dirs - pred_dc[:, :, None, None]
    true_ac = true_dirs - true_dc[:, :, None, None]
    diff = (pred_ac - true_ac).reshape(pred_ac.shape[0], pred_ac.shape[1], -1).norm(dim=2)
    norm = true_ac.reshape(true_ac.shape[0], true_ac.shape[1], -1).norm(dim=2).clamp(min=eps)
    ac_rel = (diff / norm).mean()

    total = w_dc * dc_rel + w_ac * ac_rel
    return total, {
        "dc_rel": dc_rel.detach(),
        "ac_rel": ac_rel.detach(),
    }

def maxwell_pinn_loss(pred, true_zminus, true_zplus, freq_ghz, delta_z_mm=1.0, dx_mm=15.0/64.0, dy_mm=15.0/64.0):  # 元胞 15 mm / 64 点（旧默认 30/64 差 2 倍）
    """
    推断时物理优化 (TTO) 使用的严格麦克斯韦残差损失 (基于中心差分)。
    
    参数:
    pred: 网络预测的目标平面电磁场，shape (B, 12, H, W)，必须是未被标准化(或已反标准化)的真实物理尺度。
    true_zminus: 真实数据集在 z - delta_z 处的电磁场，shape (B, 12, H, W)。
    true_zplus: 真实数据集在 z + delta_z 处的电磁场，shape (B, 12, H, W)。
    freq_ghz: 频率 (B,) 或 scalar，单位 GHz。
    delta_z_mm: 相邻平面距离目标平面的物理距离 (mm)，默认 1.0mm。
    dx_mm, dy_mm: 空间网格分辨率 (mm)。
    
    返回:
    物理约束残差 Loss。
    """
    B, _, H, W = pred.shape
    
    # 物理常数
    mu_0 = 4 * torch.pi * 1e-7
    # omega = 2 * pi * f (注意单位统一，这里长度单位是 mm，所以把频率换算下)
    # 或者直接把 mm 换算成 m 来算
    # 为了防止数值溢出，我们在计算时统一转为国际标准单位 (m, Hz)
    
    # 将长度转换为米
    dx = dx_mm * 1e-3
    dy = dy_mm * 1e-3
    dz = delta_z_mm * 1e-3
    
    # 将频率转换为 Hz
    if isinstance(freq_ghz, torch.Tensor):
        omega = 2 * torch.pi * freq_ghz.view(B, 1, 1) * 1e9  # (B, 1, 1)
    else:
        omega = 2 * torch.pi * freq_ghz * 1e9
        
    # 解析预测场 (目标平面 z)
    E_x_pred = torch.complex(pred[:, 0], pred[:, 1])
    E_y_pred = torch.complex(pred[:, 2], pred[:, 3])
    E_z_pred = torch.complex(pred[:, 4], pred[:, 5])
    H_x_pred = torch.complex(pred[:, 6], pred[:, 7])
    H_y_pred = torch.complex(pred[:, 8], pred[:, 9])
    H_z_pred = torch.complex(pred[:, 10], pred[:, 11])
    
    # 真实场的相邻平面 (只需要计算 dE/dz)
    E_x_minus = torch.complex(true_zminus[:, 0], true_zminus[:, 1])
    E_x_plus  = torch.complex(true_zplus[:, 0], true_zplus[:, 1])
    E_y_minus = torch.complex(true_zminus[:, 2], true_zminus[:, 3])
    E_y_plus  = torch.complex(true_zplus[:, 2], true_zplus[:, 3])
    
    # 核心创新：真实中心差分计算 z 偏导 (精度 O(dz^2))
    # z 轴正向是沿着波传播方向，差分为 (E(z+dz) - E(z-dz)) / (2*dz)
    dEx_dz = (E_x_plus - E_x_minus) / (2 * dz)
    dEy_dz = (E_y_plus - E_y_minus) / (2 * dz)
    
    # 网络内部计算横向偏导 (使用中心差分)
    # torch.gradient 默认在边界使用前向/后向差分，在内部使用中心差分
    dEz_dy, dEz_dx = torch.gradient(E_z_pred, spacing=(dy, dx), dim=(-2, -1))
    dEx_dy, dEx_dx = torch.gradient(E_x_pred, spacing=(dy, dx), dim=(-2, -1))
    dEy_dy, dEy_dx = torch.gradient(E_y_pred, spacing=(dy, dx), dim=(-2, -1))
    
    # 麦克斯韦方程旋度项: rot E = -j * omega * mu * H
    # x 分量: dEz/dy - dEy/dz = -j w mu Hx
    # y 分量: dEx/dz - dEz/dx = -j w mu Hy
    # z 分量: dEy/dx - dEx/dy = -j w mu Hz
    
    j_omega_mu = 1j * omega * mu_0
    
    # 残差 (Residuals)
    res_x = (dEz_dy - dEy_dz) + j_omega_mu * H_x_pred
    res_y = (dEx_dz - dEz_dx) + j_omega_mu * H_y_pred
    res_z = (dEy_dx - dEx_dy) + j_omega_mu * H_z_pred
    
    # 归一化残差：除以 w * mu * |H| 的平均量级，防止绝对数值极小导致浮点下溢
    scale = (omega * mu_0 * torch.abs(H_x_pred).mean() + 1e-12).detach()
    
    loss_x = torch.mean(torch.abs(res_x) ** 2) / (scale ** 2)
    loss_y = torch.mean(torch.abs(res_y) ** 2) / (scale ** 2)
    loss_z = torch.mean(torch.abs(res_z) ** 2) / (scale ** 2)
    
    total_pinn_loss = loss_x + loss_y + loss_z
    return total_pinn_loss, {
        "pinn_x": loss_x.detach(),
        "pinn_y": loss_y.detach(),
        "pinn_z": loss_z.detach(),
    }

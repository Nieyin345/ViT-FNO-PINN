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

def maxwell_pinn_loss(pred, true_zminus, true_zplus, freq_ghz, delta_z_mm=1.0, dx_mm=30.0/64.0, dy_mm=30.0/64.0):
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

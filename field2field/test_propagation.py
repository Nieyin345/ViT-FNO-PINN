# -*- coding: utf-8 -*-
"""PWS 传播的解析解校验（纯 numpy，无需 HFSS 数据，CPU 秒级）。

构造一个**严格满足麦克斯韦方程**的周期场（时间因子 e^{+jωt}，HFSS 约定）：
  - 前向(+z)谐波：Ex/Ey 给定，Ez = −(k·E_t)/kz，H = k×E/(ωμ)，倏逝谐波向上衰减；
  - 可选的后向(−z)传播谐波（用于检验 ep/em 双向分解）。
然后从 z0 面取 (E,H) 喂给 `propagate_em_twoway_from_source`，与解析解在目标 z 比较。

用法（任意目录）：  python field2field/test_propagation.py
退出码 0 = 全部通过。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from propagation import propagate_em_twoway_from_source  # noqa: E402

C0 = 299792458.0
MU0 = 4e-7 * math.pi


def build_field(L_mm, N, f_ghz, z0_mm, seed=0, nmax=4, harm_ratio=0.3,
                dc=(1.0 + 0.5j, 0.3 - 0.2j), backward=False):
    """返回 (at(z_mm)->(E,H), dx_m)。harm_ratio = 非零阶谐波能量 / 零阶能量（量级）。"""
    rng = np.random.default_rng(seed)
    L = L_mm * 1e-3
    f = f_ghz * 1e9
    k = 2 * math.pi * f / C0
    w = 2 * math.pi * f
    dx = L / N
    K = 2 * math.pi * np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(K, K, indexing="ij")
    m = np.abs(np.fft.fftfreq(N, d=1.0 / N)) <= nmax
    M = m[:, None] & m[None, :]
    n_h = max(int(M.sum()) - 1, 1)
    amp = math.sqrt(harm_ratio / n_h) * N * N * math.sqrt(abs(dc[0]) ** 2 + abs(dc[1]) ** 2)

    def rnd():
        return (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * M * amp / math.sqrt(2)

    kt2 = KX ** 2 + KY ** 2
    kzr = np.sqrt(np.maximum(k ** 2 - kt2, 0.0) + 0j)
    prop = kt2 < k ** 2
    kz_f = np.where(prop, kzr, -1j * np.sqrt(np.maximum(kt2 - k ** 2, 0.0)))   # 向上衰减
    kxp, kyp = -KX, -KY                                                        # 物理波矢（FFT 符号）

    waves = []
    sgn_list = [+1, -1] if backward else [+1]
    for s in sgn_list:
        Ex, Ey = rnd(), rnd()
        if s == +1:
            Ex[0, 0] = dc[0] * N * N
            Ey[0, 0] = dc[1] * N * N
        else:                      # 后向波只取传播谐波
            Ex = np.where(prop, Ex, 0)
            Ey = np.where(prop, Ey, 0)
            Ex[0, 0] = 0.4 * dc[0] * N * N
            Ey[0, 0] = -0.2 * dc[1] * N * N
        kz = s * kz_f
        kzs = np.where(np.abs(kz) < 1e-9, 1e-9, kz)
        Ez = -(kxp * Ex + kyp * Ey) / kzs
        Hx = (kyp * Ez - kz * Ey) / (w * MU0)
        Hy = (kz * Ex - kxp * Ez) / (w * MU0)
        Hz = (kxp * Ey - kyp * Ex) / (w * MU0)
        waves.append((kz, (Ex, Ey, Ez), (Hx, Hy, Hz)))

    def at(z_mm):
        dz = (z_mm - z0_mm) * 1e-3
        E = np.zeros((3, N, N), complex)
        H = np.zeros((3, N, N), complex)
        for kz, Ec, Hc in waves:
            ph = np.exp(-1j * kz * dz)
            for i in range(3):
                E[i] += np.fft.ifft2(Ec[i] * ph)
                H[i] += np.fft.ifft2(Hc[i] * ph)
        return E, H

    return at, dx


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def run_case(name, L_mm, f_ghz, backward, tol_e, tol_h, lp_cutoff=None, z_list=(5.0, 9.0, 15.0, 23.0)):
    at, dx = build_field(L_mm, 64, f_ghz, 3.0, backward=backward)
    E0, H0 = at(3.0)
    kw = {} if lp_cutoff is None else {"lp_cutoff": lp_cutoff}
    worst = 0.0
    for z in z_list:
        Et, Ht = at(z)
        Ep, Hp = propagate_em_twoway_from_source(E0, H0, dx, dx, f_ghz * 1e9, (z - 3.0) * 1e-3, **kw)
        eE = [rel(Ep[i], Et[i]) for i in range(3)]
        eH = [rel(Hp[i], Ht[i]) for i in range(3)]
        worst = max(worst, max(eE) / tol_e, max(eH) / tol_h)
        print(f"  {name:<34} z={z:5.1f}  E(x,y,z)={['%.1e' % v for v in eE]}  H(x,y,z)={['%.1e' % v for v in eH]}")
    return worst <= 1.0


def main():
    ok = True
    print("[1] 15 mm 元胞 / 12 GHz：仅零阶传播 + 倏逝谐波（本项目真实情形）")
    ok &= run_case("default (lp=inf)", 15.0, 12.0, False, 2e-2, 2e-2)
    print("[2] 15 mm 元胞 / 8 GHz")
    ok &= run_case("default (lp=inf)", 15.0, 8.0, False, 2e-2, 2e-2)
    print("[3] 60 mm 大元胞：多个传播谐波（检验 Ez/Hz 符号与前向传播）")
    ok &= run_case("default (lp=inf) forward", 60.0, 12.0, False, 5e-2, 5e-2, z_list=(5.0, 9.0, 15.0))
    print("[4] 60 mm 大元胞：前向 + 后向传播波（检验 ep/em 双向分解）")
    ok &= run_case("default (lp=inf) two-way", 60.0, 12.0, True, 5e-2, 5e-2, z_list=(5.0, 9.0, 15.0))

    print("[5] 回归：旧默认 lp_cutoff=0.8 必须明显更差（确认测试有辨别力）")
    at, dx = build_field(15.0, 64, 12.0, 3.0)
    E0, H0 = at(3.0)
    Et, _ = at(5.0)
    Ep_old, _ = propagate_em_twoway_from_source(E0, H0, dx, dx, 12e9, 2e-3, lp_cutoff=0.8)
    Ep_new, _ = propagate_em_twoway_from_source(E0, H0, dx, dx, 12e9, 2e-3)
    e_old = rel(Ep_old, Et)
    e_new = rel(Ep_new, Et)
    print(f"  z=5mm 总 E 相对误差：lp=0.8 → {e_old:.3e}    lp=inf → {e_new:.3e}")
    ok &= (e_new < 1e-2) and (e_old > 10 * e_new)

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

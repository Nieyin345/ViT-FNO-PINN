from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class ComplexField3D:
    x: np.ndarray  # (nx,)
    y: np.ndarray  # (ny,)
    z: np.ndarray  # (nz,)
    data: np.ndarray  # (3, nx, ny, nz)
    name: str


def parse_targets_mm(raw: str) -> List[float]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("--targets-mm must contain at least one value, e.g. 10,15")
    return vals


def load_fld(path: Path, name: str, drop_periodic_endpoint: bool = True) -> ComplexField3D:
    raw = np.loadtxt(path, skiprows=2)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != 9:
        raise ValueError(f"{path} should have 9 numeric columns after header, got {raw.shape[1]}")

    x = raw[:, 0]
    y = raw[:, 1]
    z = raw[:, 2]
    c0 = raw[:, 3] + 1j * raw[:, 4]
    c1 = raw[:, 5] + 1j * raw[:, 6]
    c2 = raw[:, 7] + 1j * raw[:, 8]

    xu = np.unique(x)
    yu = np.unique(y)
    zu = np.unique(z)

    nx, ny, nz = len(xu), len(yu), len(zu)
    expected = nx * ny * nz
    if raw.shape[0] != expected:
        raise ValueError(
            f"{path} is not a complete regular grid: rows={raw.shape[0]}, expected={expected}"
        )

    ix = np.searchsorted(xu, x)
    iy = np.searchsorted(yu, y)
    iz = np.searchsorted(zu, z)

    data = np.empty((3, nx, ny, nz), dtype=np.complex128)
    data[0, ix, iy, iz] = c0
    data[1, ix, iy, iz] = c1
    data[2, ix, iy, iz] = c2

    # HFSS 导出时通常包含周期边界的两端点（如 x=0 和 x=15mm 都在），
    # 对 FFT 来说这两个点是同一个点，需要去掉最后一个，保证周期性。
    if drop_periodic_endpoint and nx > 1 and ny > 1:
        dx = xu[1] - xu[0]
        dy = yu[1] - yu[0]
        x_periodic = abs((xu[-1] - xu[0]) - dx * (nx - 1)) < dx * 1e-6
        y_periodic = abs((yu[-1] - yu[0]) - dy * (ny - 1)) < dy * 1e-6
        if x_periodic:
            xu = xu[:-1]
            data = data[:, :-1, :, :]
        if y_periodic:
            yu = yu[:-1]
            data = data[:, :, :-1, :]

    return ComplexField3D(x=xu, y=yu, z=zu, data=data, name=name)


def nearest_index(values: np.ndarray, target: float, tol: float = 1e-9) -> int:
    idx = int(np.argmin(np.abs(values - target)))
    err = abs(values[idx] - target)
    if err > tol:
        raise ValueError(
            f"Target z={target:.9g} m not found (closest={values[idx]:.9g} m, abs_err={err:.3g})"
        )
    return idx


def ensure_same_grid(a: ComplexField3D, b: ComplexField3D) -> None:
    if not (np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y) and np.array_equal(a.z, b.z)):
        raise ValueError("E/H grids are not identical. Please export on the same grid first.")


def write_fld_plane(path: Path, field_name: str, x: np.ndarray, y: np.ndarray, z0: float, plane: np.ndarray) -> None:
    nx, ny = len(x), len(y)
    dx_mm = (x[1] - x[0]) * 1e3 if nx > 1 else 0.0
    dy_mm = (y[1] - y[0]) * 1e3 if ny > 1 else 0.0

    min_x_mm, max_x_mm = float(x.min() * 1e3), float(x.max() * 1e3)
    min_y_mm, max_y_mm = float(y.min() * 1e3), float(y.max() * 1e3)
    z_mm = float(z0 * 1e3)

    with path.open("w", encoding="ascii", newline="") as f:
        f.write(
            f"Grid Output Min: [{min_x_mm:.6g}mm {min_y_mm:.6g}mm {z_mm:.6g}mm] "
            f"Max: [{max_x_mm:.6g}mm {max_y_mm:.6g}mm {z_mm:.6g}mm] "
            f"Grid Size: [{dx_mm:.6g}mm {dy_mm:.6g}mm 0mm] \n"
        )
        f.write(f'X, Y, Z, Complex Vector data "Predicted<{field_name}>"\n')

        for ix in range(nx):
            xv = float(x[ix])
            for iy in range(ny):
                yv = float(y[iy])
                v0 = plane[0, ix, iy]
                v1 = plane[1, ix, iy]
                v2 = plane[2, ix, iy]
                f.write(
                    f"{xv:.16e} {yv:.16e} {z0:.16e}  "
                    f"{v0.real:.16e} {v0.imag:.16e} "
                    f"{v1.real:.16e} {v1.imag:.16e} "
                    f"{v2.real:.16e} {v2.imag:.16e}\n"
                )

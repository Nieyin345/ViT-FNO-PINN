from __future__ import annotations

from typing import List, Tuple

import numpy as np


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    num = np.linalg.norm((pred - ref).ravel())
    den = np.linalg.norm(ref.ravel())
    if den == 0:
        return float("nan")
    return float(num / den)


def collect_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    field_label: str,
    target_mm: float,
) -> List[Tuple[str, float, str, float]]:
    names = ["x", "y", "z"]
    rows: List[Tuple[str, float, str, float]] = []

    for i, comp_name in enumerate(names):
        rows.append((field_label, target_mm, f"{comp_name}_relL2", rel_l2(pred[i], ref[i])))

    pred_mag = np.sqrt(np.sum(np.abs(pred) ** 2, axis=0))
    ref_mag = np.sqrt(np.sum(np.abs(ref) ** 2, axis=0))
    rows.append((field_label, target_mm, "mag_relL2", rel_l2(pred_mag, ref_mag)))
    return rows

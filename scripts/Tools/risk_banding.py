from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RiskResult:
    band: str
    probability: float
    kpi_summary: dict


def band_from_paths(
    paths: np.ndarray,
    current_level: float,
    er_wait_baseline_min: float = 72.0,
    bed_occupancy_baseline: float = 0.84,
    eta_er: float = 0.8,
    eta_bed: float = 0.6,
    thr_er_min: float = 90.0,
    thr_bed: float = 0.92,
    band_cut_low: float = 0.15,
    band_cut_high: float = 0.40,
) -> RiskResult:
    x = np.asarray(paths, dtype=float)
    if x.ndim != 2 or x.size == 0:
        return RiskResult(band="LOW", probability=0.0, kpi_summary={})

    current = max(1e-6, float(current_level))
    max_demand = x.max(axis=1)
    shock = (max_demand - current) / current

    er_wait = er_wait_baseline_min * (1.0 + eta_er * shock)
    bed_occ = bed_occupancy_baseline + eta_bed * shock

    exceed = (er_wait > thr_er_min) | (bed_occ > thr_bed)
    p = float(exceed.mean())

    if p >= band_cut_high:
        band = "HIGH"
    elif p >= band_cut_low:
        band = "MED"
    else:
        band = "LOW"

    return RiskResult(
        band=band,
        probability=p,
        kpi_summary={
            "er_wait_mean": float(er_wait.mean()),
            "bed_occupancy_mean": float(bed_occ.mean()),
            "p_exceed": p,
        },
    )


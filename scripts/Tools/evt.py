from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.stats import genpareto


def fit_pot(data: np.ndarray, threshold: float, min_excess: int = 5) -> Tuple[float, float, float]:
    """
    Peak-Over-Threshold: 임계치 초과분에 GPD 적합.
    반환 (shape, loc, scale)
    """
    x = np.asarray(data, dtype=float)
    excess = x[x > threshold] - threshold
    if excess.size < max(1, int(min_excess)):
        return (0.1, 0.0, max(1e-6, np.std(excess) if excess.size > 1 else 1.0))
    c, loc, scale = genpareto.fit(excess, floc=0.0)
    return (float(c), float(loc), float(scale))


def replace_tail_with_evt(paths: np.ndarray, gpd_params: Tuple[float, float, float], threshold: float) -> np.ndarray:
    """
    경로의 값 중 threshold를 초과하는 부분을 GPD 샘플로 치환해 상위 꼬리 과소추정 방지.
    """
    c, loc, scale = gpd_params
    x = np.asarray(paths, dtype=float).copy()
    if x.ndim == 1:
        x = x[None, :]
    rng = np.random.default_rng()
    mask = x > threshold
    if not np.any(mask):
        return x
    num = int(mask.sum())
    excess = genpareto.rvs(c, loc=loc, scale=scale, size=num, random_state=rng)
    x[mask] = threshold + np.maximum(0.0, excess)
    return x


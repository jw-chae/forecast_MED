from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from properscoring import crps_ensemble  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency missing
    crps_ensemble = None  # type: ignore


@dataclass(slots=True)
class PredictionBundle:
    """Container for forecast results used by the metric calculators."""

    point: pd.Series
    quantiles: Mapping[float, pd.Series] | None = None
    samples: Optional[pd.DataFrame] = None  # shape: (len(point), n_samples)


MetricFn = Callable[[pd.Series, PredictionBundle], float]


def _ensure_alignment(actual: pd.Series, pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    joined = pd.concat([actual.rename("actual"), pred.rename("pred")], axis=1, join="inner").dropna()
    return joined["actual"], joined["pred"]


def mae(actual: pd.Series, bundle: PredictionBundle) -> float:
    a, p = _ensure_alignment(actual, bundle.point)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: pd.Series, bundle: PredictionBundle) -> float:
    a, p = _ensure_alignment(actual, bundle.point)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: pd.Series, bundle: PredictionBundle) -> float:
    a, p = _ensure_alignment(actual, bundle.point)
    denom = np.maximum(1e-6, np.abs(a))
    return float(np.mean(np.abs((a - p) / denom)))


def sharpness(actual: pd.Series, bundle: PredictionBundle) -> float:
    if not bundle.quantiles:
        return float("nan")
    lower = min(bundle.quantiles.keys())
    upper = max(bundle.quantiles.keys())
    q_lower = bundle.quantiles[lower]
    q_upper = bundle.quantiles[upper]
    _, q_lower = _ensure_alignment(actual, q_lower)
    _, q_upper = _ensure_alignment(actual, q_upper)
    return float(np.mean(q_upper - q_lower))


def coverage(actual: pd.Series, bundle: PredictionBundle, level: float) -> float:
    if not bundle.quantiles:
        return float("nan")
    
    target_lower = (1 - level) / 2
    target_upper = 1 - (1 - level) / 2
    
    # Find closest keys with tolerance
    lower_key = None
    upper_key = None
    
    keys = list(bundle.quantiles.keys())
    
    # Helper to find closest key
    def find_closest(target, keys, tol=1e-4):
        closest = None
        min_diff = float('inf')
        for k in keys:
            diff = abs(k - target)
            if diff < min_diff:
                min_diff = diff
                closest = k
        if min_diff <= tol:
            return closest
        return None

    lower_key = find_closest(target_lower, keys)
    upper_key = find_closest(target_upper, keys)

    if lower_key is None or upper_key is None:
        # print(f"DEBUG: Coverage keys not found. Level: {level}, Target: [{target_lower}, {target_upper}], Available: {keys}")
        return float("nan")
        
    lower = bundle.quantiles[lower_key]
    upper = bundle.quantiles[upper_key]
    
    a, lower = _ensure_alignment(actual, lower)
    _, upper = _ensure_alignment(actual, upper)
    inside = (a >= lower) & (a <= upper)
    return float(np.mean(inside))


def peak_recall(actual: pd.Series, bundle: PredictionBundle, window: int = 2) -> float:
    a = actual.sort_index()
    threshold = a.rolling(window, min_periods=1).max().quantile(0.9)
    peak_idx = a[a >= threshold].index
    if peak_idx.empty:
        return float("nan")
    predicted = bundle.point.sort_index()
    detected = predicted.loc[peak_idx]
    return float(np.mean(detected.index.isin(peak_idx)))


def peak_mae(actual: pd.Series, bundle: PredictionBundle, window: int = 2) -> float:
    a = actual.sort_index()
    threshold = a.rolling(window, min_periods=1).max().quantile(0.9)
    peak_idx = a[a >= threshold].index
    if peak_idx.empty:
        return float("nan")
    p = bundle.point.reindex(peak_idx).dropna()
    a = actual.reindex(p.index)
    return float(np.mean(np.abs(a - p)))


def crps(actual: pd.Series, bundle: PredictionBundle) -> float:
    if bundle.samples is not None and crps_ensemble is not None:
        ens = bundle.samples.loc[actual.index.intersection(bundle.samples.index)].to_numpy().T
        obs = actual.loc[bundle.samples.index].to_numpy()
        return float(np.mean(crps_ensemble(obs, ens)))
    if bundle.quantiles:
        losses = []
        for q, q_pred in bundle.quantiles.items():
            a, p = _ensure_alignment(actual, q_pred)
            errors = a - p
            loss = np.maximum(q * errors, (q - 1) * errors)
            losses.append(np.mean(loss))
        if losses:
            return 2 * float(np.mean(losses))
    # fallback: absolute error
    return mae(actual, bundle)


METRICS_REGISTRY: Dict[str, MetricFn] = {
    "mae": mae,
    "rmse": rmse,
    "mape": mape,
    "crps": crps,
    "sharpness": sharpness,
    "coverage_95": lambda a, b: coverage(a, b, 0.95),
    "coverage_90": lambda a, b: coverage(a, b, 0.90),
    "coverage_80": lambda a, b: coverage(a, b, 0.80),
    "peak_recall_2w": lambda a, b: peak_recall(a, b, window=2),
    "peak_precision_2w": lambda a, b: peak_recall(a, b, window=2),  # proxy
    "peak_mae": peak_mae,
}


def evaluate_metrics(actual: pd.Series, bundle: PredictionBundle, metric_names: Iterable[str]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for name in metric_names:
        if name not in METRICS_REGISTRY:
            raise KeyError(f"Unknown metric '{name}'")
        try:
            results[name] = float(METRICS_REGISTRY[name](actual, bundle))
        except Exception as exc:
            results[name] = float("nan")
            results[f"{name}_error"] = str(exc)
    return results


__all__ = ["PredictionBundle", "evaluate_metrics", "METRICS_REGISTRY"]

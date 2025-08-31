from __future__ import annotations

from typing import Dict, Any
import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(1e-9, (np.abs(y_true) + np.abs(y_pred)) / 2.0)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    from scipy.stats import norm
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(1e-6, np.asarray(sigma, dtype=float))
    z = (y_true - mu) / sigma
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(np.maximum(0.0, crps)))


def interval_coverage(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    q_lo = np.asarray(q_lo, dtype=float)
    q_hi = np.asarray(q_hi, dtype=float)
    ok = (y_true >= q_lo) & (y_true <= q_hi)
    return float(np.mean(ok))


def mae(y_true: np.ndarray, y_pred: np.ndarray, use_median: bool = True) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ae = np.abs(y_true - y_pred)
    return float(np.median(ae) if use_median else np.mean(ae))


def peak_metrics(
    y_true: np.ndarray,
    q50: np.ndarray,
    alpha_top: float = 0.1,
    window_recall: int = 2,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    if y_true.size == 0 or q50.size == 0:
        return {"top_alpha_time_rmse": None, "top_alpha_height_rmse": None, "recall_pm2w": None}
    n_top = max(1, int(np.ceil(alpha_top * len(y_true))))
    top_idx_true = np.argsort(y_true)[-n_top:]
    top_idx_pred = np.argsort(q50)[-n_top:]
    # 시간 RMSE: 정렬 후 인덱스 차이
    top_idx_true_sorted = np.sort(top_idx_true)
    top_idx_pred_sorted = np.sort(top_idx_pred)
    k = min(len(top_idx_true_sorted), len(top_idx_pred_sorted))
    time_rmse = float(np.sqrt(np.mean((top_idx_true_sorted[:k] - top_idx_pred_sorted[:k]) ** 2))) if k > 0 else None
    height_rmse = float(np.sqrt(np.mean((y_true[top_idx_true_sorted[:k]] - q50[top_idx_pred_sorted[:k]]) ** 2))) if k > 0 else None
    # recall@±2주
    recall = 0.0
    for ti in top_idx_true:
        if np.any(np.abs(top_idx_pred - ti) <= window_recall):
            recall += 1.0
    recall = recall / float(n_top)
    return {"top_alpha_time_rmse": time_rmse, "top_alpha_height_rmse": height_rmse, "recall_pm2w": float(recall)}


def kpi_exceed_probs(paths: np.ndarray, bed_thr: float = 0.92, er_thr: float = 90.0) -> Dict[str, float]:
    x = np.asarray(paths, dtype=float)
    if x.size == 0:
        return {"p_bed_gt_0_92": 0.0, "p_er_gt_90": 0.0}
    p_bed = float((x > bed_thr).mean())
    p_er = float((x > er_thr).mean())
    return {"p_bed_gt_0_92": p_bed, "p_er_gt_90": p_er}



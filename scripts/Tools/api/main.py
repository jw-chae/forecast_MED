from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np

# Local imports
from ..adapters import load_his_outpatient_series
from ..scenario_engine import extract_growth_episodes, generate_paths_conditional
from ..evt import fit_pot, replace_tail_with_evt


class SimParams(BaseModel):
    amplitude_quantile: float = 0.9
    amplitude_multiplier: float = 1.8
    ratio_cap_quantile: float = 0.98
    warmup_weeks: int = 1
    use_delta_quantile: bool = True
    delta_quantile: float = 0.05
    news_signal: Optional[List[float]] = None
    quality: float = 0.72
    nb_dispersion_k: Optional[float] = 5.0
    start_value_override: Optional[float] = None
    # 안정화 캡(가드레일)
    r_boost_cap: float = 2.0
    scale_cap: float = 1.6
    x_cap_multiplier: float = 2.0


class RunSimRequest(BaseModel):
    disease: str = Field(..., description="CSV column name of disease")
    train_until: str = Field(..., description="YYYY-MM-DD")
    end: str = Field(..., description="YYYY-MM-DD")
    season_profile: Optional[str] = Field(None, description="preset: 'flu' or 'hfmd'")
    horizon: Optional[int] = None
    params: SimParams = Field(default_factory=SimParams)


class RunSimResponse(BaseModel):
    horizon: int
    sample_size: int
    metrics: Dict[str, Any]
    quantiles: Dict[str, List[float]]
    mean_path: List[float]
    params_used: Dict[str, Any]


app = FastAPI(title="Epi Forecast Simulator API")


def month_to_weight_flu(month: int) -> float:
    if month in (11, 12, 1, 2):
        return 0.7
    if month == 10:
        return 0.4
    if month in (3, 4, 5):
        return 0.08
    return 0.15


def build_news_vector(dates: List[str], profile: Optional[str]) -> np.ndarray:
    if profile is None:
        return np.zeros(len(dates), dtype=float)
    if profile.lower() == "flu":
        weights = [month_to_weight_flu(int(d[5:7])) for d in dates]
        return np.array(weights, dtype=float)
    # default flat
    return np.zeros(len(dates), dtype=float)


@app.post("/run_sim", response_model=RunSimResponse)
def run_sim(req: RunSimRequest):
    # Load series
    base = __file__
    dt_index, series = load_his_outpatient_series(
        str(
            __file__.replace("/api/main.py", "/../../processed_data/his_outpatient_weekly_epi_counts.csv")
        ),
        req.disease,
    )
    dates = [d.strftime("%Y-%m-%d") for d in dt_index]
    # Split
    split_idx = max(i for i, d in enumerate(dates) if d <= req.train_until)
    end_idx = max(i for i, d in enumerate(dates) if d <= req.end)
    hist = series[: split_idx + 1].astype(float)
    target = series[split_idx + 1 : end_idx + 1].astype(float)
    horizon = req.horizon or int(len(target))
    future_dates = dates[split_idx + 1 : split_idx + 1 + horizon]

    # Season vector
    if req.params.news_signal is not None:
        news_vec = np.array(req.params.news_signal, dtype=float)
        if news_vec.size < horizon:
            pad = np.full(horizon - news_vec.size, news_vec[-1] if news_vec.size > 0 else 0.0)
            news_vec = np.concatenate([news_vec, pad])
        news_vec = news_vec[:horizon]
    else:
        news_vec = build_news_vector(future_dates, req.season_profile)

    # Episodes and paths
    episodes = extract_growth_episodes(hist)
    season_start_override = float(np.median(hist[-8:]))
    paths = generate_paths_conditional(
        series=hist,
        horizon=horizon,
        n_paths=3000,
        episodes=episodes,
        news_signal=news_vec,
        quality=req.params.quality,
        recent_baseline_window=8,
        amplitude_quantile=req.params.amplitude_quantile,
        amplitude_multiplier=req.params.amplitude_multiplier,
        ratio_cap_quantile=req.params.ratio_cap_quantile,
        warmup_weeks=req.params.warmup_weeks,
        use_delta_quantile=req.params.use_delta_quantile,
        delta_quantile=req.params.delta_quantile,
        nb_dispersion_k=req.params.nb_dispersion_k,
        start_value_override=req.params.start_value_override or season_start_override,
        r_boost_cap=req.params.r_boost_cap,
        scale_cap=req.params.scale_cap,
        x_cap_multiplier=req.params.x_cap_multiplier,
    )

    # EVT tail polish (optional; keep same threshold approach as elsewhere)
    try:
        u = float(np.quantile(hist, 0.9))
        gpd = fit_pot(hist, threshold=u)
        paths = replace_tail_with_evt(paths, gpd_params=gpd, threshold=u)
    except Exception:
        pass

    # Quantiles and metrics
    q05 = np.quantile(paths, 0.05, axis=0).tolist()
    q50 = np.quantile(paths, 0.50, axis=0).tolist()
    q80 = np.quantile(paths, 0.80, axis=0).tolist()
    q95 = np.quantile(paths, 0.95, axis=0).tolist()
    mean_path = paths.mean(axis=0).tolist()

    metrics: Dict[str, Any] = {"coverage95": None, "mae_median": None}
    if len(target) >= horizon and horizon > 0:
        tar = target[:horizon]
        cov = np.mean((tar >= np.array(q05)) & (tar <= np.array(q95)))
        mae_med = float(np.mean(np.abs(tar - np.array(q50))))
        metrics.update({"coverage95": float(cov), "mae_median": mae_med})

    resp = RunSimResponse(
        horizon=horizon,
        sample_size=int(paths.shape[0]),
        quantiles={"q05": q05, "q50": q50, "q80": q80, "q95": q95},
        mean_path=mean_path,
        metrics=metrics,
        params_used=req.params.dict(),
    )
    return resp



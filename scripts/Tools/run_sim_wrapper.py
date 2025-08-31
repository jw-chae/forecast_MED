from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from Tools.adapters import load_his_outpatient_series
from Tools.scenario_engine import generate_paths_conditional, extract_growth_episodes
from Tools.evt import fit_pot, replace_tail_with_evt
from Tools.metrics import (
    smape,
    crps_gaussian,
    interval_coverage,
    mae,
    peak_metrics,
    kpi_exceed_probs,
)
from Tools.epidemic_models import run_seir_simulation
from Tools.config import APP_CONFIG


@dataclasses.dataclass
class SimConfig:
    disease: str
    train_until: str
    end: str
    horizon: Optional[int] = None
    season_profile: Optional[str] = None
    seed: int = 42


def _season_vector(dates: List[str], profile: Optional[str]) -> np.ndarray:
    import datetime as _dt
    if profile is None:
        return np.zeros(len(dates), dtype=float)
    prof = profile.lower()
    vec: List[float] = []
    for ds in dates:
        m = _dt.datetime.strptime(ds, "%Y-%m-%d").month
        if prof == "flu":
            if m in (11, 12, 1, 2):
                vec.append(0.7)
            elif m == 10:
                vec.append(0.4)
            elif m in (3, 4, 5):
                vec.append(0.08)
            else:
                vec.append(0.15)
        else:
            vec.append(0.0)
    return np.array(vec, dtype=float)


def run_sim(params: Dict[str, Any], config: SimConfig) -> Dict[str, Any]:
    base = Path(__file__).resolve().parents[2]
    csv_path = base / "processed_data" / "his_outpatient_weekly_epi_counts.csv"
    dt_index, series = load_his_outpatient_series(str(csv_path), config.disease)
    dates = [d.strftime("%Y-%m-%d") for d in dt_index]

    split_idx = max(i for i, d in enumerate(dates) if d <= config.train_until)
    end_idx = max(i for i, d in enumerate(dates) if d <= config.end)
    hist = series[: split_idx + 1].astype(float)
    target = series[split_idx + 1 : end_idx + 1].astype(float)
    # Use the smaller of requested horizon and available target length to avoid mismatch
    if config.horizon is not None:
        horizon = int(min(int(config.horizon), len(target)))
    else:
        horizon = int(len(target))
    future_dates = dates[split_idx + 1 : split_idx + 1 + horizon]

    if isinstance(params.get("news_signal"), list):
        news_vec = np.array(params["news_signal"], dtype=float)
        if news_vec.size < horizon:
            pad = np.full(horizon - news_vec.size, news_vec[-1] if news_vec.size > 0 else 0.0)
            news_vec = np.concatenate([news_vec, pad])
        news_vec = news_vec[:horizon]
    elif isinstance(params.get("news_signal"), (int, float)):
        news_vec = np.full(horizon, float(params.get("news_signal")), dtype=float)
    else:
        # try evidence weekly vector (if provided via observation/evidence path)
        try:
            ev_week = params.get("evidence", {}).get("external_signals", {}).get("news_signal_weekly")  # type: ignore
        except Exception:
            ev_week = None
        if isinstance(ev_week, list) and len(ev_week) > 0:
            nv = np.array(ev_week[:horizon], dtype=float)
            if nv.size < horizon:
                nv = np.pad(nv, (0, horizon - nv.size), mode="edge")
            news_vec = nv
        else:
            news_vec = _season_vector(future_dates, params.get("season_profile") or config.season_profile)

    episodes = extract_growth_episodes(hist)
    season_start_override = float(np.median(hist[-8:]))

    n_paths = int(params.get("n_paths", 5000))
    # resolve start value override
    _svo = params.get("start_value_override", None)
    if _svo is None:
        _svo = season_start_override

    seir_curve = None
    if params.get("use_seir_hybrid", False):
        try:
            # SEIR 시뮬레이션 실행
            seir_params = params.copy()
            population = APP_CONFIG.seir.population
            initial_exposed = APP_CONFIG.seir.initial_exposed
            initial_infectious = int(hist[-1]) if len(hist) > 0 else 1
            initial_recovered = APP_CONFIG.seir.initial_recovered
            beta = seir_params.get("seir_beta", 0.5)
            incubation_days = seir_params.get("seir_incubation_days", 4.0)
            infectious_days = APP_CONFIG.seir.infectious_days

            daily_infectious = run_seir_simulation(
                population=population,
                initial_exposed=initial_exposed,
                initial_infectious=initial_infectious,
                initial_recovered=initial_recovered,
                beta=beta,
                incubation_days=incubation_days,
                infectious_days=infectious_days,
                days=horizon * 7,
            )
            # Aggregate daily to weekly, starting from the first Monday
            weekly_infectious = (
                [np.mean(daily_infectious[i:i+7]) for i in range(0, len(daily_infectious), 7)]
            )
            seir_curve = np.array(weekly_infectious)

        except Exception as e:
            print(f"[WARN] SEIR hybrid simulation failed: {e}. Falling back to standard method.")
            seir_curve = None


    paths = generate_paths_conditional(
        series=hist,
        horizon=horizon,
        n_paths=1000,
        episodes=episodes,
        news_signal=float(params.get("news_signal", 0.05)),
        quality=float(params.get("quality", 0.72)),
        recent_baseline_window=int(params.get("recent_baseline_window", 8)),
        amplitude_quantile=float(params.get("amplitude_quantile", 0.9)),
        amplitude_multiplier=float(params.get("amplitude_multiplier", 1.8)),
        ratio_cap_quantile=float(params.get("ratio_cap_quantile", 0.98)),
        warmup_weeks=int(params.get("warmup_weeks", 1)),
        use_delta_quantile=bool(params.get("use_delta_quantile", True)),
        delta_quantile=float(params.get("delta_quantile", 0.05)),
        nb_dispersion_k=(
            None
            if params.get("nb_dispersion_k") is None
            else float(params.get("nb_dispersion_k"))
        ),
        start_value_override=float(params.get("start_value_override", 0.0))
        if params.get("start_value_override") is not None
        else None,
        r_boost_cap=float(params.get("r_boost_cap", 2.0)),
        scale_cap=float(params.get("scale_cap", 1.6)),
        x_cap_multiplier=float(params.get("x_cap_multiplier", 2.0)),
        seir_infection_curve=seir_curve,
    )

    # EVT 보정
    u_quantile = float(params.get("evt_u_quantile", 0.9))
    u = float(np.quantile(hist, u_quantile))
    gpd = fit_pot(hist, threshold=u, min_excess=int(params.get("min_excess", 5)))
    paths = replace_tail_with_evt(paths, gpd_params=gpd, threshold=u)
    # enforce non-negativity for count forecasts
    paths = np.maximum(paths, 0.0)

    # Quantiles
    q05 = np.quantile(paths, 0.05, axis=0)
    q50 = np.quantile(paths, 0.50, axis=0)
    q80 = np.quantile(paths, 0.80, axis=0)
    q95 = np.quantile(paths, 0.95, axis=0)
    # non-negative quantiles
    q05 = np.maximum(q05, 0.0)
    q50 = np.maximum(q50, 0.0)
    q80 = np.maximum(q80, 0.0)
    q95 = np.maximum(q95, 0.0)
    mean_path = paths.mean(axis=0)

    # Metrics
    metrics: Dict[str, Any] = {}
    if horizon > 0:
        tar = target[:horizon]
        # Optional coverage calibration (post-hoc band scaling)
        # Guarded: only apply when explicitly enabled to avoid leakage in backtests
        _allow_cal = bool(params.get("enable_posthoc_calibration", False))
        desired_cov = params.get("calibrate_coverage_to") if _allow_cal else None
        q05_use, q95_use = q05.copy(), q95.copy()
        if isinstance(desired_cov, (int, float)) and 0.0 < float(desired_cov) <= 1.0:
            def cov_with_scale(s: float) -> float:
                lo = q50 - s * (q50 - q05)
                hi = q50 + s * (q95 - q50)
                return float(np.mean((tar >= lo) & (tar <= hi)))
            # binary search scale s
            s_lo, s_hi = 1.0, 10.0
            cov0 = cov_with_scale(1.0)
            target_cov = float(desired_cov)
            if cov0 < target_cov:
                for _ in range(16):
                    s_mid = 0.5 * (s_lo + s_hi)
                    c_mid = cov_with_scale(s_mid)
                    if c_mid < target_cov:
                        s_lo = s_mid
                    else:
                        s_hi = s_mid
                s_star = s_hi
                q05_use = q50 - s_star * (q50 - q05)
                q95_use = q50 + s_star * (q95 - q50)
            # if cov0 already >= target, keep original
        metrics["mae_median"] = float(mae(tar, q50, use_median=True))
        metrics["smape"] = float(smape(tar, q50))
        # coverage95 계산 시, 하한은 q05_use, 상한은 q95_use를 사용하도록 수정합니다.
        # 또한, np.minimum을 통해 하한이 상한보다 커지는 경우를 방지합니다.
        lower_bound = np.minimum(q05_use, q95_use)
        upper_bound = np.maximum(q05_use, q95_use)
        metrics["coverage95"] = float(interval_coverage(tar, lower_bound, upper_bound))
        # Approximate CRPS via Gaussian around median with band width proxy
        sigma = np.maximum(1e-6, (q95_use - q05) / 3.92)
        metrics["crps"] = float(crps_gaussian(tar, q50, sigma))
        metrics.update(peak_metrics(tar, q50, alpha_top=float(params.get("peak_alpha", 0.1)), window_recall=2))
    metrics.update(kpi_exceed_probs(paths))
    # Fail-safe additional KPI: Bed > 0.98
    p_bed_gt_0_98 = float((paths > 0.98).mean())
    metrics["p_bed_gt_0_98"] = p_bed_gt_0_98

    return {
        "horizon": horizon,
        "quantiles": {"q05": q05.tolist(), "q50": q50.tolist(), "q80": q80.tolist(), "q95": q95.tolist()},
        "mean_path": mean_path.tolist(),
        "metrics": metrics,
        "params_used": params,
        "seed": int(config.seed),
        # for visualization
        "dates_hist": dates[: split_idx + 1],
        "values_hist": hist.tolist(),
        "dates_target": dates[split_idx + 1 : split_idx + 1 + horizon],
        "values_target": target[:horizon].tolist(),
        "future_dates": future_dates,
    }



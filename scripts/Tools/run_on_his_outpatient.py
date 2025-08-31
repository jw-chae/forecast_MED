from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from adapters import load_his_outpatient_series
from fusion import precision_weighted_fusion
from scenario_engine import extract_growth_episodes, generate_paths_conditional
from evt import fit_pot, replace_tail_with_evt
from risk_banding import band_from_paths


def main() -> None:
    base = Path(__file__).resolve().parents[2]
    csv_path = base / "processed_data" / "his_outpatient_weekly_epi_counts.csv"

    try:
        _, series = load_his_outpatient_series(str(csv_path), "流行性感冒")
    except Exception:
        _, series = load_his_outpatient_series(str(csv_path), "Other")

    y = series.astype(float)
    yhat_mean = float(np.mean(y[-8:])) if len(y) >= 8 else float(np.mean(y))
    yhat_var = float(np.var(y[-16:])) if len(y) >= 16 else max(1.0, float(np.var(y)))

    fusion_res = precision_weighted_fusion(
        yhat_mean=yhat_mean,
        yhat_var=yhat_var,
        y_obs=float(y[-1]),
        data_quality=0.72,
        manual_bias_mean=0.20,
        manual_bias_sd=0.10,
        news_signal=0.35,
    )

    episodes = extract_growth_episodes(y)
    paths = generate_paths_conditional(
        series=y,
        horizon=8,
        n_paths=5000,
        episodes=episodes,
        news_signal=0.35,
        quality=0.72,
        random_state=123,
    )

    u = float(np.quantile(y, 0.9))
    gpd = fit_pot(y, threshold=u)
    paths_evt = replace_tail_with_evt(paths, gpd_params=gpd, threshold=u)

    risk = band_from_paths(
        paths_evt,
        current_level=float(y[-1]),
        er_wait_baseline_min=72.0,
        bed_occupancy_baseline=0.84,
    )

    out = {
        "fusion": fusion_res.as_dict(),
        "risk": {
            "band": risk.band,
            "probability": risk.probability,
            "kpi_summary": risk.kpi_summary,
        },
    }

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()


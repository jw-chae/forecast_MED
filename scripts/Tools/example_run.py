from __future__ import annotations

import numpy as np
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import sys


def _load_local_module(name: str):
    path = Path(__file__).parent / f"{name}.py"
    spec = spec_from_file_location(name, str(path))
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


fusion = _load_local_module("fusion")
scenario_engine = _load_local_module("scenario_engine")
evt = _load_local_module("evt")
risk_banding = _load_local_module("risk_banding")


def generate_synthetic_series(n: int = 160, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(8.0, 12.0, n)
    noise = rng.normal(0.0, 1.2, size=n)
    series = np.maximum(0.0, base + noise)
    # 두 번의 스파이크 추가
    for center, amp, width in [(90, 25.0, 6), (130, 15.0, 8)]:
        for t in range(n):
            series[t] += amp * np.exp(-((t - center) ** 2) / (2.0 * (width ** 2)))
    return series


def main() -> None:
    y = generate_synthetic_series()
    yhat_mean = float(np.mean(y[-8:]))
    yhat_var = max(1.0, float(np.var(y[-16:])))

    # 관측 품질/편향/뉴스 예시
    fusion_res = fusion.precision_weighted_fusion(
        yhat_mean=yhat_mean,
        yhat_var=yhat_var,
        y_obs=float(y[-1]),
        data_quality=0.72,
        manual_bias_mean=0.20,
        manual_bias_sd=0.10,
        news_signal=0.35,
    )

    episodes = scenario_engine.extract_growth_episodes(y)
    paths = scenario_engine.generate_paths_conditional(
        series=y,
        horizon=8,
        n_paths=2000,
        episodes=episodes,
        news_signal=0.35,
        quality=0.72,
        random_state=7,
    )

    # EVT 꼬리 보정(선택)
    u = float(np.quantile(y, 0.9))
    gpd = evt.fit_pot(y, threshold=u)
    paths_evt = evt.replace_tail_with_evt(paths, gpd_params=gpd, threshold=u)

    risk = risk_banding.band_from_paths(
        paths_evt,
        current_level=float(y[-1]),
        er_wait_baseline_min=72.0,
        bed_occupancy_baseline=0.84,
    )

    result = {
        "fusion": fusion_res.as_dict(),
        "risk": {
            "band": risk.band,
            "probability": risk.probability,
            "kpi_summary": risk.kpi_summary,
        },
    }

    print(result)


if __name__ == "__main__":
    main()



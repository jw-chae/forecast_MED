from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base_model import BaseModel, ModelOutput
from ..metrics import PredictionBundle
from ..core.core.scenario_engine import extract_growth_episodes, generate_paths_conditional
from ..core.core.evt import fit_pot, replace_tail_with_evt


class EpiToolsModel(BaseModel):
    def __init__(self, model_config: Dict[str, object]) -> None:
        super().__init__(model_config)
        self._train_series: Optional[np.ndarray] = None
        self._train_index: Optional[pd.Index] = None

    def fit(self, train_df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None) -> None:
        self._train_series = train_df.iloc[:, 0].astype(float).to_numpy()
        self._train_index = train_df.index

    def forecast(self, horizon: int, *, start_index: Optional[pd.Index] = None) -> ModelOutput:
        if self._train_series is None:
            raise RuntimeError("Model must be fit before forecasting")
        series = self._train_series
        episodes = extract_growth_episodes(series)
        quality = float(self.model_config.get("llm", {}).get("default_quality", 0.72))
        news_signal = float(self.model_config.get("llm", {}).get("default_news", 0.3))
        params = self.model_config.get("scenario_engine", {})
        paths = generate_paths_conditional(
            series=series,
            horizon=horizon,
            n_paths=int(params.get("n_paths", 1000)),
            episodes=episodes,
            news_signal=news_signal,
            quality=quality,
            random_state=int(self.model_config.get("random_seed", 42)),
        )
        evt_cfg = self.model_config.get("evt", {})
        if evt_cfg.get("enabled", True):
            threshold = float(np.quantile(series, evt_cfg.get("pot_quantile", 0.9)))
            gpd = fit_pot(series, threshold=threshold)
            paths = replace_tail_with_evt(paths, gpd_params=gpd, threshold=threshold)
        if start_index is not None:
            forecast_index = start_index
        elif self._train_index is not None:
            freq = pd.infer_freq(self._train_index) or "W-MON"
            last = self._train_index[-1]
            forecast_index = pd.date_range(start=last + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)
        else:
            forecast_index = pd.RangeIndex(horizon)

        quantile_levels = self.model_config.get("forecast", {}).get("quantiles", [0.05, 0.5, 0.95])
        quantiles = {
            float(q): pd.Series(np.quantile(paths, float(q), axis=0), index=forecast_index)
            for q in quantile_levels
        }
        point = quantiles.get(0.5)
        if point is None:
            point = pd.Series(np.mean(paths, axis=0), index=forecast_index)

        samples = pd.DataFrame(paths, columns=forecast_index)
        bundle = PredictionBundle(point=point, quantiles=quantiles, samples=samples)
        return ModelOutput(bundle)

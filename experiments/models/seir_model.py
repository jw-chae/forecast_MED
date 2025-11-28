from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base_model import BaseModel, ModelOutput
from ..metrics import PredictionBundle


def _simulate_seir(
    population: float,
    beta: float,
    sigma: float,
    gamma: float,
    exposed0: float,
    infectious0: float,
    horizon: int,
) -> np.ndarray:
    susceptible = population - exposed0 - infectious0
    exposed = exposed0
    infectious = infectious0
    recovered = population - susceptible - exposed - infectious
    history = []
    for _ in range(horizon):
        new_exposed = beta * susceptible * infectious / population
        new_infectious = sigma * exposed
        new_recovered = gamma * infectious

        susceptible = max(0.0, susceptible - new_exposed)
        exposed = max(0.0, exposed + new_exposed - new_infectious)
        infectious = max(0.0, infectious + new_infectious - new_recovered)
        recovered = max(0.0, recovered + new_recovered)
        history.append(infectious)
    return np.array(history)


class SEIRModel(BaseModel):
    def __init__(self, model_config: Dict[str, object]) -> None:
        super().__init__(model_config)
        self._train_last_value: float = 0.0
        self._train_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None) -> None:
        # 값 컬럼 찾기 (value 또는 첫 번째 숫자 컬럼)
        value_col = None
        if "value" in train_df.columns:
            value_col = "value"
        else:
            # 숫자 타입 컬럼 중 첫 번째 선택
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                value_col = numeric_cols[0]
            else:
                value_col = train_df.columns[0]
        
        series = train_df[value_col]
        self._train_last_value = float(series.iloc[-1])
        self._train_mean = float(series.mean())

    def forecast(self, horizon: int, *, start_index: Optional[pd.Index] = None) -> ModelOutput:
        cfg = self.model_config.get("params", {})
        population = float(cfg.get("population", 1_000_000))
        beta = float(cfg.get("beta", 0.35))
        sigma = float(cfg.get("sigma", 0.15))
        gamma = float(cfg.get("gamma", 0.1))
        exposed0 = float(cfg.get("initial_exposed", self._train_mean))
        infectious0 = float(cfg.get("initial_infectious", self._train_last_value))

        trajectory = _simulate_seir(population, beta, sigma, gamma, exposed0, infectious0, horizon)
        # Scale trajectory to match observed magnitude
        scale = self._train_last_value / (trajectory[0] + 1e-6) if trajectory[0] > 0 else 1.0
        forecast_values = trajectory * scale

        index = start_index if start_index is not None else pd.RangeIndex(horizon)
        point_forecast = pd.Series(forecast_values, index=index)
        quantiles = {
            0.1: point_forecast * 0.85,
            0.5: point_forecast,
            0.9: point_forecast * 1.15,
        }
        return ModelOutput(PredictionBundle(point=point_forecast, quantiles=quantiles, samples=None))

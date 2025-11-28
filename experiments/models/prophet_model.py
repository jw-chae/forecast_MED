from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_model import BaseModel, ModelOutput
from ..metrics import PredictionBundle

try:  # pragma: no cover - optional dependency
    from prophet import Prophet  # type: ignore
except Exception:  # pragma: no cover - fallback to fbprophet / or missing
    try:
        from fbprophet import Prophet  # type: ignore
    except Exception:  # pragma: no cover
        Prophet = None  # type: ignore


class ProphetModel(BaseModel):
    def __init__(self, model_config: Dict[str, object]) -> None:
        super().__init__(model_config)
        self._model: Optional[Any] = None
        self._train_last_value: float = 0.0
        self._train_index: Optional[pd.Index] = None

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
        
        self._train_last_value = float(train_df[value_col].iloc[-1])
        self._train_index = train_df.index
        
        if Prophet is None:
            return
        
        params = dict(self.model_config.get("params", {}))
        model = Prophet(**params)
        
        # Prophet은 'ds'(날짜)와 'y'(값) 두 컬럼만 필요
        df = pd.DataFrame({
            "ds": train_df.index,
            "y": train_df[value_col].values
        })
        
        model.fit(df)
        self._model = model

    def forecast(self, horizon: int, *, start_index: Optional[pd.Index] = None) -> ModelOutput:
        requested_quantiles = self.model_config.get("forecast", {}).get("quantiles", [0.05, 0.5, 0.95])

        if self._model is None:
            # fallback: persistence with heuristic intervals
            index = start_index if start_index is not None else pd.RangeIndex(horizon)
            point_forecast = pd.Series(np.full(horizon, self._train_last_value), index=index)
            quantiles = {}
            for q in requested_quantiles:
                if q == 0.5:
                    quantiles[q] = point_forecast
                else:
                    deviation = 0.2 * point_forecast * (abs(q - 0.5) / 0.45)
                    quantiles[q] = point_forecast + (deviation if q > 0.5 else -deviation)
            return ModelOutput(PredictionBundle(point=point_forecast, quantiles=quantiles))

        future = self._model.make_future_dataframe(periods=horizon, freq="W-MON")
        
        # Use predictive_samples to get samples for arbitrary quantiles
        samples = self._model.predictive_samples(future)
        yhat_samples = samples['yhat'] # shape (n_periods, n_samples)
        yhat_samples = yhat_samples[-horizon:]
        
        forecast_df = self._model.predict(future)
        fc = forecast_df.tail(horizon)
        index = start_index if start_index is not None else pd.to_datetime(fc["ds"])
        point_forecast = pd.Series(fc["yhat"].to_numpy(), index=index)
        
        quantiles = {}
        for q in requested_quantiles:
            if q == 0.5:
                quantiles[q] = point_forecast
            else:
                q_values = np.quantile(yhat_samples, q, axis=1)
                quantiles[q] = pd.Series(q_values, index=index)
        
        output = ModelOutput(PredictionBundle(point=point_forecast, quantiles=quantiles))
        
        # Log scale information
        if self._train_df is not None:
            self._log_scale_info(self._train_df, output, "Prophet")
        
        return output

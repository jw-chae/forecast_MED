from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_model import BaseModel, ModelOutput
from ..metrics import PredictionBundle

try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
except Exception:  # pragma: no cover
    SARIMAX = None  # type: ignore


class ARIMAModel(BaseModel):
    def __init__(self, model_config: Dict[str, object]) -> None:
        super().__init__(model_config)
        self._result: Optional[Any] = None
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
        
        if SARIMAX is None:
            return
        
        params = dict(self.model_config.get("params", {}))
        order = params.pop("order", (1, 0, 0))
        seasonal_order = params.pop("seasonal_order", (0, 0, 0, 0))
        enforce_stationarity = bool(params.pop("enforce_stationarity", False))
        enforce_invertibility = bool(params.pop("enforce_invertibility", False))
        
        model = SARIMAX(
            train_df[value_col],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            **params,
        )
        self._result = model.fit(disp=False)

    def forecast(self, horizon: int, *, start_index: Optional[pd.Index] = None) -> ModelOutput:
        requested_quantiles = self.model_config.get("forecast", {}).get("quantiles", [0.05, 0.5, 0.95])
        
        if self._result is None:
            index = start_index if start_index is not None else pd.RangeIndex(horizon)
            point_forecast = pd.Series(np.full(horizon, self._train_last_value), index=index)
            quantiles = {}
            for q in requested_quantiles:
                if q == 0.5:
                    quantiles[q] = point_forecast
                else:
                    # Heuristic: assume 10% error margin for 90% CI, scaled by quantile
                    # This is just a placeholder fallback
                    deviation = 0.15 * point_forecast * (abs(q - 0.5) / 0.45)
                    quantiles[q] = point_forecast + (deviation if q > 0.5 else -deviation)
            return ModelOutput(PredictionBundle(point=point_forecast, quantiles=quantiles))
            
        forecast_res = self._result.get_forecast(steps=horizon)
        mean = forecast_res.predicted_mean
        index = start_index if start_index is not None else mean.index
        point_forecast = pd.Series(mean.to_numpy(), index=index)
        
        quantiles = {}
        for q in requested_quantiles:
            if q == 0.5:
                quantiles[q] = point_forecast
                continue
                
            # Calculate alpha for conf_int
            # conf_int(alpha) returns (alpha/2, 1-alpha/2) interval
            if q < 0.5:
                alpha = 2 * q
                ci = forecast_res.conf_int(alpha=alpha)
                quantiles[q] = pd.Series(ci.iloc[:, 0].to_numpy(), index=index)
            else:
                alpha = 2 * (1 - q)
                ci = forecast_res.conf_int(alpha=alpha)
                quantiles[q] = pd.Series(ci.iloc[:, 1].to_numpy(), index=index)
                
        return ModelOutput(PredictionBundle(point=point_forecast, quantiles=quantiles))

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_model import BaseModel, ModelOutput
from ..metrics import PredictionBundle

try:  # pragma: no cover - optional dependency
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # type: ignore


class XGBoostModel(BaseModel):
    def __init__(self, model_config: Dict[str, object]) -> None:
        super().__init__(model_config)
        self._model: Optional[Any] = None
        self._train_last_value: float = 0.0
        self._feature_columns: List[str] = []

    def _build_supervised_matrix(self, series: pd.Series, lags: List[int]) -> pd.DataFrame:
        df = pd.DataFrame({"target": series})
        for lag in lags:
            df[f"lag_{lag}"] = series.shift(lag)
        return df.dropna()

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
        
        if xgb is None:
            return
        
        horizon = int(self.model_config.get("forecast", {}).get("horizon", 4))
        lags = self.model_config.get("forecast", {}).get("feature_lags", [1, 2, 3, 4])
        if not isinstance(lags, list):
            lags = [1, 2, 3, 4]
        
        matrix = self._build_supervised_matrix(series, lags)
        y = matrix["target"].values
        X = matrix.drop(columns=["target"]).values
        self._feature_columns = list(matrix.drop(columns=["target"]).columns)
        
        params = dict(self.model_config.get("params", {}))
        params.setdefault("n_jobs", params.get("n_jobs", 1))
        self._model = xgb.XGBRegressor(**params)
        
        try:
            self._model.fit(X, y)
        except KeyboardInterrupt:  # pragma: no cover - graceful degradation
            logging.getLogger(__name__).warning("XGBoost training interrupted; falling back to persistence forecast")
            self._model = None

    def forecast(self, horizon: int, *, start_index: Optional[pd.Index] = None) -> ModelOutput:
        requested_quantiles = self.model_config.get("forecast", {}).get("quantiles", [0.05, 0.5, 0.95])

        if self._model is None or not self._feature_columns:
            index = start_index if start_index is not None else pd.RangeIndex(horizon)
            point_forecast = pd.Series(np.full(horizon, self._train_last_value), index=index)
            quantiles = {}
            for q in requested_quantiles:
                if q == 0.5:
                    quantiles[q] = point_forecast
                else:
                    deviation = 0.15 * point_forecast * (abs(q - 0.5) / 0.45)
                    quantiles[q] = point_forecast + (deviation if q > 0.5 else -deviation)
            return ModelOutput(PredictionBundle(point=point_forecast, quantiles=quantiles))

        lags = [int(col.split("_")[1]) for col in self._feature_columns]
        history = list(self._train_last_value for _ in range(max(lags) + horizon + 1))
        preds = []
        for step in range(horizon):
            features = [history[-lag] for lag in lags]
            pred = float(self._model.predict(np.array(features).reshape(1, -1))[0])
            history.append(pred)
            preds.append(pred)
        index = start_index if start_index is not None else pd.RangeIndex(horizon)
        point_forecast = pd.Series(preds, index=index)
        
        # Heuristic std dev since we don't have residuals
        std = 0.1 * point_forecast.abs().mean()
        
        from scipy.stats import norm
        quantiles = {}
        for q in requested_quantiles:
            if q == 0.5:
                quantiles[q] = point_forecast
            else:
                z_score = norm.ppf(q)
                quantiles[q] = point_forecast + z_score * std
                
        return ModelOutput(PredictionBundle(point=point_forecast, quantiles=quantiles))

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base_model import BaseModel, ModelOutput
from ..metrics import PredictionBundle

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


class _LSTMForecaster(nn.Module):  # type: ignore[misc]
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out, _ = self.lstm(x)
        out = self.head(out[:, -1, :])
        return out


class LSTMModel(BaseModel):
    def __init__(self, model_config: Dict[str, object]) -> None:
        super().__init__(model_config)
        self._model: Optional[_LSTMForecaster] = None
        self._train_series: Optional[pd.Series] = None
        self._device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else None

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
        self._train_series = series
        
        if torch is None:
            return
        
        cfg = self.model_config.get("params", {})
        window = int(cfg.get("input_window", 16))
        hidden = int(cfg.get("hidden_size", 64))
        layers = int(cfg.get("num_layers", 2))
        dropout = float(cfg.get("dropout", 0.2))
        lr = float(cfg.get("learning_rate", 1e-3))
        epochs = int(cfg.get("epochs", 50))
        batch_size = int(cfg.get("batch_size", 32))

        X, y = [], []
        values = series.to_numpy(dtype=np.float32)
        for idx in range(window, len(values)):
            X.append(values[idx - window : idx])
            y.append(values[idx])
        if not X:
            return
        X_arr = np.stack(X)
        y_arr = np.array(y, dtype=np.float32)

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_arr).unsqueeze(-1), torch.from_numpy(y_arr))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = _LSTMForecaster(input_size=1, hidden_size=hidden, num_layers=layers, dropout=dropout).to(self._device)  # type: ignore[arg-type]
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                optimizer.zero_grad()
                pred = model(batch_x).squeeze(-1)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        self._model = model

    def forecast(self, horizon: int, *, start_index: Optional[pd.Index] = None) -> ModelOutput:
        requested_quantiles = self.model_config.get("forecast", {}).get("quantiles", [0.05, 0.5, 0.95])

        if self._train_series is None:
            raise RuntimeError("Model must be fit before forecasting")
        index = start_index if start_index is not None else pd.RangeIndex(horizon)
        if torch is None or self._model is None:
            last_value = float(self._train_series.iloc[-1])
            point_forecast = pd.Series(np.full(horizon, last_value), index=index)
            quantiles = {}
            for q in requested_quantiles:
                if q == 0.5:
                    quantiles[q] = point_forecast
                else:
                    deviation = 0.15 * point_forecast * (abs(q - 0.5) / 0.45)
                    quantiles[q] = point_forecast + (deviation if q > 0.5 else -deviation)
            return ModelOutput(PredictionBundle(point=point_forecast, quantiles=quantiles))

        window = int(self.model_config.get("params", {}).get("input_window", 16))
        history = self._train_series.to_numpy(dtype=np.float32)
        inputs = history[-window:].copy()
        preds = []
        self._model.eval()
        for _ in range(horizon):
            x = torch.from_numpy(inputs[-window:]).reshape(1, window, 1).to(self._device)
            with torch.no_grad():
                pred = self._model(x).item()
            preds.append(pred)
            inputs = np.append(inputs, np.float32(pred)).astype(np.float32, copy=False)
        point_forecast = pd.Series(preds, index=index)
        std = np.std(history[-window:]) if window < len(history) else np.std(history)
        
        from scipy.stats import norm
        quantiles = {}
        for q in requested_quantiles:
            if q == 0.5:
                quantiles[q] = point_forecast
            else:
                z_score = norm.ppf(q)
                quantiles[q] = point_forecast + z_score * std
        return ModelOutput(PredictionBundle(point=point_forecast, quantiles=quantiles))

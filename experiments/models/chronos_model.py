from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .base_model import BaseModel, ModelOutput
from ..metrics import PredictionBundle

try:  # pragma: no cover - optional dependency
    from chronos import ChronosPipeline  # type: ignore
except Exception:  # pragma: no cover
    ChronosPipeline = None  # type: ignore


@dataclass
class _ChronosState:
    pipeline: "ChronosPipeline"  # type: ignore[name-defined]
    quantiles: List[float]


class ChronosModel(BaseModel):
    """Wrapper around the Amazon Chronos foundation models.

    This wrapper expects the `chronos-forecasting` package to be installed.
    Install instructions:

        pip install chronos-forecasting

    Note: Chronos currently relies on recent CUDA-enabled PyTorch builds and can
    require significant GPU memory. Use `device=cpu` to force CPU execution, but
    forecasts will be slower.
    """

    def __init__(self, model_config: Dict[str, object]) -> None:
        super().__init__(model_config)
        if ChronosPipeline is None:
            raise ImportError(
                "ChronosModel을 사용하려면 `chronos-forecasting` 패키지가 필요합니다. "
                "예: pip install chronos-forecasting"
            )

        self.repo_id: str = str(model_config.get("repo_id", "amazon/chronos-t5-small"))
        self.device: str = str(model_config.get("device", "cpu"))
        raw_quantiles: Iterable[float] | None = model_config.get("quantiles")  # type: ignore[assignment]
        self.quantiles: List[float] = (
            sorted({float(q) for q in raw_quantiles}) if raw_quantiles is not None else [0.1, 0.5, 0.9]
        )
        self.num_samples: int = int(model_config.get("num_samples", 20))
        self.pipeline_kwargs: Dict[str, object] = dict(model_config.get("pipeline_kwargs", {}))
        self.normalize: bool = bool(model_config.get("normalize", False))

        self._state: Optional[_ChronosState] = None
        self._train_values: Optional[np.ndarray] = None
        self._train_index: Optional[pd.Index] = None
        self._freq: Optional[str] = None

    # ------------------------------------------------------------------ #
    # BaseModel interface
    # ------------------------------------------------------------------ #

    def fit(self, train_df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None) -> None:
        series = train_df.iloc[:, 0].astype(float).to_numpy()
        if series.size == 0:
            raise ValueError("ChronosModel.fit: 훈련 시계열이 비어 있습니다.")
        self._train_values = series
        self._train_index = train_df.index
        self._freq = self._infer_frequency(train_df.index)

    def forecast(self, horizon: int, *, start_index: Optional[pd.Index] = None) -> ModelOutput:
        if self._train_values is None or self._train_index is None:
            raise RuntimeError("ChronosModel 을 사용하기 전에 fit()을 호출해야 합니다.")

        state = self._ensure_pipeline()
        context = self._train_values.copy()
        prediction_length = horizon

        # ChronosPipeline은 torch.Tensor를 입력으로 받음
        import torch
        context_tensor = torch.tensor(context, dtype=torch.float32)
        
        # predict_quantiles 사용
        quantile_forecasts, sample_forecasts = state.pipeline.predict_quantiles(
            context_tensor,
            prediction_length=prediction_length,
            quantile_levels=state.quantiles,
        )
        
        # 결과 추출: quantile_forecasts shape는 (batch, prediction_length, num_quantiles)
        # 단일 시계열이므로 batch=1
        if quantile_forecasts.dim() == 3:
            quantile_forecasts = quantile_forecasts[0]  # (prediction_length, num_quantiles)
        
        # shape 확인 및 조정
        if quantile_forecasts.shape[0] == len(state.quantiles):
            # (num_quantiles, prediction_length) 형태인 경우 transpose
            quantile_forecasts = quantile_forecasts.T  # (prediction_length, num_quantiles)
        
        # 평균은 median (0.5 quantile) 사용
        median_idx = state.quantiles.index(0.5) if 0.5 in state.quantiles else len(state.quantiles) // 2
        mean_values = quantile_forecasts[:prediction_length, median_idx].cpu().numpy()
        
        # Quantile 추출
        quantile_map: Dict[float, np.ndarray] = {}
        for idx, q in enumerate(state.quantiles):
            if idx < quantile_forecasts.shape[1]:
                quantile_map[float(q)] = quantile_forecasts[:prediction_length, idx].cpu().numpy()

        index = self._build_forecast_index(horizon, start_index)
        point_series = pd.Series(mean_values, index=index)
        quantiles = {float(q): pd.Series(values, index=index) for q, values in quantile_map.items()}
        bundle = PredictionBundle(point=point_series, quantiles=quantiles)
        output = ModelOutput(bundle=bundle)
        
        # Log scale information
        if self._train_values is not None:
            train_df = pd.DataFrame(self._train_values, columns=['value'])
            self._log_scale_info(train_df, output, "Chronos-v1")
        
        return output

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_pipeline(self) -> _ChronosState:
        if self._state is not None:
            return self._state
        # device_map 파라미터 사용
        device_map = self.device if self.device != "cpu" else "cpu"
        pipeline = ChronosPipeline.from_pretrained(
            self.repo_id,
            device_map=device_map,
            **self.pipeline_kwargs
        )  # type: ignore[call-arg]
        self._state = _ChronosState(pipeline=pipeline, quantiles=self.quantiles)
        return self._state

    def _infer_frequency(self, index: pd.Index) -> str:
        if isinstance(index, pd.DatetimeIndex):
            freq = index.freqstr or pd.infer_freq(index)
            if freq:
                return freq
        return "W"

    def _build_forecast_index(self, horizon: int, start_index: Optional[pd.Index]) -> pd.Index:
        if start_index is not None and len(start_index) == horizon:
            return start_index
        if isinstance(self._train_index, pd.DatetimeIndex):
            last_timestamp = self._train_index[-1]
            return pd.date_range(last_timestamp, periods=horizon + 1, freq=self._freq)[1:]
        if isinstance(self._train_index, pd.RangeIndex):
            start = self._train_index[-1] + 1
            return pd.RangeIndex(start, start + horizon)
        return pd.RangeIndex(horizon)



__all__ = ["ChronosModel"]


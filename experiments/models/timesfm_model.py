from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .base_model import BaseModel, ModelOutput
from ..metrics import PredictionBundle

try:  # pragma: no cover - optional dependency
    from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams, freq_map
except Exception:  # pragma: no cover
    TimesFm = None  # type: ignore
    TimesFmCheckpoint = None  # type: ignore
    TimesFmHparams = None  # type: ignore
    freq_map = None  # type: ignore


@dataclass
class _TimesFMState:
    horizon_len: int
    model: TimesFm  # type: ignore[name-defined]
    quantiles: List[float]


class TimesFMModel(BaseModel):
    """Wrapper around Google TimesFM foundation model for experiments."""

    def __init__(self, model_config: Dict[str, object]) -> None:
        super().__init__(model_config)
        if TimesFm is None:
            raise ImportError(
                "timesfm 패키지가 설치되어 있지 않습니다. "
                "pip install timesfm 명령으로 설치한 뒤 다시 시도하세요."
            )

        self.repo_id: str = str(model_config.get("repo_id", "google/timesfm-1.0-200m-pytorch"))
        self.context_len: int = int(model_config.get("context_len", 512))
        self.base_horizon_len: int = int(model_config.get("horizon_len", 64))
        self.input_patch_len: int = int(model_config.get("input_patch_len", 32))
        self.output_patch_len: int = int(model_config.get("output_patch_len", self.base_horizon_len))
        self.backend: str = str(model_config.get("backend", "cpu")).lower()
        self.normalize: bool = bool(model_config.get("normalize", False))
        self.forecast_context_len: Optional[int] = (
            int(model_config["forecast_context_len"]) if model_config.get("forecast_context_len") else None
        )
        self.local_dir: Optional[str] = str(model_config["local_dir"]) if model_config.get("local_dir") else None
        raw_quantiles: Iterable[float] | None = model_config.get("quantiles")  # type: ignore[assignment]
        self.configured_quantiles: Optional[List[float]] = (
            sorted({float(q) for q in raw_quantiles}) if raw_quantiles is not None else None
        )
        self.point_forecast_mode: str = str(model_config.get("point_forecast_mode", "median"))
        self._freq_override: Optional[str] = (
            str(model_config["freq"]) if model_config.get("freq") else None
        )

        self._state: Optional[_TimesFMState] = None
        self._train_values: Optional[np.ndarray] = None
        self._train_index: Optional[pd.Index] = None
        self._freq_code: Optional[int] = None
        self._freq_str: Optional[str] = None

    def fit(self, train_df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None) -> None:
        series = train_df.iloc[:, 0].astype(float).to_numpy()
        if series.size == 0:
            raise ValueError("TimesFMModel.fit: 훈련 시계열이 비어 있습니다.")
        self._train_values = series
        self._train_index = train_df.index
        self._freq_str = self._infer_frequency(train_df.index)
        self._freq_code = self._frequency_to_code(self._freq_str)

    def forecast(self, horizon: int, *, start_index: Optional[pd.Index] = None) -> ModelOutput:
        if self._train_values is None or self._train_index is None or self._freq_code is None:
            raise RuntimeError("TimesFMModel 을 사용하기 전에 fit()을 호출해야 합니다.")

        state = self._ensure_model(horizon)
        ctx_len = self.forecast_context_len or min(self.context_len, self._train_values.shape[0])
        context = self._train_values[-ctx_len:]

        freq_list = [self._freq_code]
        mean_forecast, full_forecast = state.model.forecast(
            inputs=[context],
            freq=freq_list,
            forecast_context_len=ctx_len,
            normalize=self.normalize,
        )

        mean_values = mean_forecast[0][:horizon]
        full_values = full_forecast[0][:horizon]

        index = self._build_forecast_index(horizon, start_index)
        point_series = pd.Series(mean_values, index=index)
        quantiles = self._extract_quantiles(full_values, index, state.quantiles)
        bundle = PredictionBundle(point=point_series, quantiles=quantiles)
        return ModelOutput(bundle=bundle)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self, horizon: int) -> _TimesFMState:
        target_horizon = max(self.base_horizon_len, horizon)
        if self._state is not None and self._state.horizon_len >= target_horizon:
            return self._state

        quantiles = self.configured_quantiles or list(TimesFmHparams().quantiles or [])
        backend = "gpu" if self.backend == "gpu" else "cpu"
        hparams = TimesFmHparams(
            context_len=self.context_len,
            horizon_len=target_horizon,
            input_patch_len=self.input_patch_len,
            output_patch_len=max(self.output_patch_len, target_horizon),
            backend=backend,
            quantiles=quantiles,
            point_forecast_mode=self.point_forecast_mode,  # type: ignore[arg-type]
        )
        checkpoint = TimesFmCheckpoint(
            version="torch",
            huggingface_repo_id=self.repo_id,
            local_dir=self.local_dir,
        )
        model = TimesFm(hparams=hparams, checkpoint=checkpoint)  # type: ignore[call-arg]
        self._state = _TimesFMState(horizon_len=target_horizon, model=model, quantiles=quantiles)
        return self._state

    def _infer_frequency(self, index: pd.Index) -> str:
        if self._freq_override:
            return self._freq_override
        if isinstance(index, pd.DatetimeIndex):
            freq = index.freqstr or pd.infer_freq(index)
            if freq:
                return freq
        return "W"

    def _frequency_to_code(self, freq_str: str) -> int:
        if freq_map is None:
            raise ImportError("timesfm.freq_map 함수를 사용할 수 없습니다.")
        try:
            return int(freq_map(freq_str))
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"주기 문자열 '{freq_str}'을 TimesFM 주파수 코드로 변환하지 못했습니다.") from exc

    def _build_forecast_index(self, horizon: int, start_index: Optional[pd.Index]) -> pd.Index:
        if start_index is not None and len(start_index) == horizon:
            return start_index
        if isinstance(self._train_index, pd.DatetimeIndex):
            last_timestamp = self._train_index[-1]
            freq = self._freq_str or "W"
            return pd.date_range(last_timestamp, periods=horizon + 1, freq=freq)[1:]
        if isinstance(self._train_index, pd.RangeIndex):
            start = self._train_index[-1] + 1
            return pd.RangeIndex(start, start + horizon)
        return pd.RangeIndex(horizon)

    def _extract_quantiles(
        self,
        full_values: np.ndarray,
        index: pd.Index,
        quantiles: List[float],
    ) -> Dict[float, pd.Series]:
        if full_values.ndim != 2 or full_values.shape[1] != (len(quantiles) + 1):
            return {}
        result: Dict[float, pd.Series] = {}
        
        # Map quantile values to their column indices in full_values
        # full_values[:, 0] is the mean.
        # full_values[:, i] corresponds to quantiles[i-1]
        q_to_col = {q: i for i, q in enumerate(quantiles, start=1)}

        for q, col_idx in q_to_col.items():
            result[float(q)] = pd.Series(full_values[:, col_idx], index=index)

        # Heuristic: If 0.025 and 0.975 are missing, but we have 0.1 and 0.9,
        # estimate them assuming a normal distribution.
        # This is to support coverage_95 metric for TimesFM which defaults to 0.1-0.9.
        if 0.1 in q_to_col and 0.9 in q_to_col:
            if 0.025 not in result:
                # Estimate sigma from 0.1 and 0.9 quantiles
                # Q(0.9) - Q(0.1) = 2 * 1.28155 * sigma
                # sigma = (Q(0.9) - Q(0.1)) / 2.5631
                q90 = full_values[:, q_to_col[0.9]]
                q10 = full_values[:, q_to_col[0.1]]
                mean = full_values[:, 0]
                
                sigma = (q90 - q10) / 2.5631
                
                # Q(0.025) = mean - 1.96 * sigma
                q025 = mean - 1.96 * sigma
                result[0.025] = pd.Series(q025, index=index)

            if 0.975 not in result:
                # Re-calculate sigma or reuse if I optimized, but this is fine.
                q90 = full_values[:, q_to_col[0.9]]
                q10 = full_values[:, q_to_col[0.1]]
                mean = full_values[:, 0]
                
                sigma = (q90 - q10) / 2.5631
                
                # Q(0.975) = mean + 1.96 * sigma
                q975 = mean + 1.96 * sigma
                result[0.975] = pd.Series(q975, index=index)

        return result


__all__ = ["TimesFMModel"]

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .base_model import BaseModel, ModelOutput
from ..metrics import PredictionBundle

try:  # pragma: no cover - optional dependency
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # type: ignore
    from gluonts.dataset.common import ListDataset  # type: ignore
except Exception as e:  # pragma: no cover
    # transformers packaging 버전 체크 문제로 인한 실패일 수 있으므로 나중에 재시도
    MoiraiForecast = None  # type: ignore
    MoiraiModule = None  # type: ignore
    ListDataset = None  # type: ignore
    _moirai_import_error = e  # type: ignore


@dataclass
class _MoiraiState:
    module: "MoiraiModule"  # type: ignore[name-defined]


class MoiraiModel(BaseModel):
    """Wrapper around Salesforce Moirai foundation models (via uni2ts).

    Requires `uni2ts` (>=2.0.0) and its dependencies:

        pip install \"uni2ts[forecast]\"  # installs gluonts, torch, etc.
    """

    def __init__(self, model_config: Dict[str, object]) -> None:
        super().__init__(model_config)
        # Import 재시도 (transformers packaging 버전 체크 문제로 인한 실패일 수 있음)
        global MoiraiForecast, MoiraiModule, ListDataset
        if MoiraiForecast is None or MoiraiModule is None or ListDataset is None:
            try:
                # matplotlib libstdc++ 문제 우회
                import os
                os.environ.setdefault('MPLBACKEND', 'Agg')
                
                # transformers 버전 체크 우회를 위한 패치
                import transformers.utils.versions as versions_module
                if hasattr(versions_module, '_compare_versions'):
                    original_compare = versions_module._compare_versions
                    def patched_compare(op, got_ver, want_ver, requirement, pkg, hint):
                        if "packaging" in requirement.lower():
                            return
                        return original_compare(op, got_ver, want_ver, requirement, pkg, hint)
                    versions_module._compare_versions = patched_compare
                
                from uni2ts.model.moirai import MoiraiForecast as MF, MoiraiModule as MM  # type: ignore
                from gluonts.dataset.common import ListDataset as LD  # type: ignore
                MoiraiForecast = MF
                MoiraiModule = MM
                ListDataset = LD
            except Exception as e:
                raise ImportError(
                    f"MoiraiModel을 사용하려면 `uni2ts` 및 `gluonts` 패키지가 필요합니다. "
                    f"예: pip install \"uni2ts[forecast]\"\n"
                    f"원본 오류: {e}"
                ) from e
        
        if MoiraiForecast is None or MoiraiModule is None or ListDataset is None:
            raise ImportError(
                "MoiraiModel을 사용하려면 `uni2ts` 및 `gluonts` 패키지가 필요합니다. "
                "예: pip install \"uni2ts[forecast]\""
            )

        self.repo_id: str = str(model_config.get("repo_id", "Salesforce/moirai-1.0-R-small"))
        self.context_len: Optional[int] = (
            int(model_config["context_len"]) if model_config.get("context_len") else None
        )
        raw_quantiles: Iterable[float] | None = model_config.get("quantiles")  # type: ignore[assignment]
        self.quantiles: List[float] = (
            sorted({float(q) for q in raw_quantiles}) if raw_quantiles is not None else [0.1, 0.5, 0.9]
        )
        self.num_samples: int = int(model_config.get("num_samples", 100))
        self.patch_size: str | int = model_config.get("patch_size", "auto")
        self.batch_size: int = int(model_config.get("batch_size", 32))
        self.device: str = str(model_config.get("device", "cpu"))

        self._state: Optional[_MoiraiState] = None
        self._train_values: Optional[np.ndarray] = None
        self._train_index: Optional[pd.Index] = None
        self._freq: Optional[str] = None

    # ------------------------------------------------------------------ #
    # BaseModel interface
    # ------------------------------------------------------------------ #

    def fit(self, train_df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None) -> None:
        series = train_df.iloc[:, 0].astype(float).to_numpy()
        if series.size == 0:
            raise ValueError("MoiraiModel.fit: 훈련 시계열이 비어 있습니다.")
        self._train_values = series
        self._train_index = train_df.index
        self._freq = self._infer_frequency(train_df.index)

    def forecast(self, horizon: int, *, start_index: Optional[pd.Index] = None) -> ModelOutput:
        if self._train_values is None or self._train_index is None or self._freq is None:
            raise RuntimeError("MoiraiModel 을 사용하기 전에 fit()을 호출해야 합니다.")

        state = self._ensure_module()
        context_len = self.context_len or min(len(self._train_values), 512)
        context = self._train_values[-context_len:]

        model = MoiraiForecast(  # type: ignore[call-arg]
            module=state.module,
            prediction_length=horizon,
            context_length=context_len,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        predictor = model.create_predictor(batch_size=self.batch_size, device=self.device)
        dataset = ListDataset(
            [{"target": context, "start": self._get_start_timestamp(context_len)}],
            freq=self._freq,
        )
        forecasts = list(predictor.predict(dataset))
        if not forecasts:
            raise RuntimeError("Moirai 예측을 생성하지 못했습니다.")
        forecast = forecasts[0]

        mean_values = self._extract_mean(forecast, horizon)
        quantile_map = self._extract_quantiles(forecast, horizon)

        index = self._build_forecast_index(horizon, start_index)
        point_series = pd.Series(mean_values, index=index)
        quantiles = {float(q): pd.Series(values, index=index) for q, values in quantile_map.items()}
        bundle = PredictionBundle(point=point_series, quantiles=quantiles)
        return ModelOutput(bundle=bundle)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_module(self) -> _MoiraiState:
        if self._state is not None:
            return self._state
        module = MoiraiModule.from_pretrained(self.repo_id)  # type: ignore[call-arg]
        self._state = _MoiraiState(module=module)
        return self._state

    def _infer_frequency(self, index: pd.Index) -> str:
        if isinstance(index, pd.DatetimeIndex):
            freq = index.freqstr or pd.infer_freq(index)
            if freq:
                return freq
        return "W"

    def _get_start_timestamp(self, context_len: int):
        if isinstance(self._train_index, pd.DatetimeIndex):
            # context_len이 데이터 길이보다 크면 첫 번째 인덱스 사용
            actual_len = min(context_len, len(self._train_index))
            return self._train_index[-actual_len] if actual_len > 0 else self._train_index[0]
        return pd.Timestamp("2020-01-01")

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

    def _extract_mean(self, forecast, horizon: int) -> np.ndarray:
        if hasattr(forecast, "mean") and forecast.mean is not None:
            return np.asarray(forecast.mean, dtype=float)[:horizon]
        if hasattr(forecast, "samples") and forecast.samples is not None:
            return np.asarray(forecast.samples, dtype=float).mean(axis=0)[:horizon]
        raise ValueError("Moirai 예측 결과에서 평균을 추출할 수 없습니다.")

    def _extract_quantiles(self, forecast, horizon: int) -> Dict[float, np.ndarray]:
        out: Dict[float, np.ndarray] = {}
        for q in self.quantiles:
            if hasattr(forecast, "quantile"):
                try:
                    values = np.asarray(forecast.quantile(q), dtype=float)[:horizon]
                except Exception:
                    values = None
            else:
                values = None
            if values is None and hasattr(forecast, "samples") and forecast.samples is not None:
                values = np.quantile(np.asarray(forecast.samples, dtype=float), q, axis=0)[:horizon]
            if values is not None:
                out[float(q)] = values
        return out


__all__ = ["MoiraiModel"]


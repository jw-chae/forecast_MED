from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .base_model import BaseModel, ModelOutput
from ..metrics import PredictionBundle


class Chronos2Model(BaseModel):
    """Wrapper around the Amazon Chronos-2 foundation model.

    - 단일 시계열(예: 인플루엔자)을 기준으로 rolling forecast를 지원합니다.
    - 내부적으로 Chronos2Pipeline.from_pretrained("amazon/chronos-2")를 사용합니다.
    """

    def __init__(self, model_config: Dict[str, object]) -> None:
        super().__init__(model_config)
        self.repo_id: str = str(model_config.get("repo_id", "amazon/chronos-2"))
        self.device: str = str(model_config.get("device", "cpu"))
        self.local_dir: Optional[str] = (
            str(model_config["local_dir"]) if model_config.get("local_dir") else None
        )
        self.item_id: str = str(model_config.get("item_id", "influenza"))
        self._target_column: str = str(model_config.get("target_column", "target"))

        forecast_cfg = model_config.get("forecast") or {}
        quantile_cfg = forecast_cfg.get("quantiles")
        self._quantile_levels: list[float] | None = self._prepare_quantile_levels(quantile_cfg)

        self._pipeline: Optional[object] = None
        self._train_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # BaseModel interface
    # ------------------------------------------------------------------ #

    def fit(self, train_df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None) -> None:
        if train_df.empty:
            raise ValueError("Chronos2Model.fit: 훈련 시계열이 비어 있습니다.")
        numeric_cols = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c])]
        if not numeric_cols:
            raise ValueError("Chronos2Model.fit: 수치형 타깃 컬럼을 찾지 못했습니다.")
        self._train_df = train_df[numeric_cols[0:1]].astype(float)

    def forecast(self, horizon: int, *, start_index: Optional[pd.Index] = None) -> ModelOutput:
        if self._train_df is None:
            raise RuntimeError("Chronos2Model 을 사용하기 전에 fit()을 호출해야 합니다.")

        pipeline = self._ensure_pipeline()
        train_df = self._train_df
        assert train_df is not None

        context_df = train_df.copy()
        
        # Chronos2가 frequency를 추론할 수 있도록 인덱스에 freq 설정
        if isinstance(context_df.index, pd.DatetimeIndex):
            if context_df.index.freq is None:
                freq = pd.infer_freq(context_df.index) or "W"
                context_df = context_df.asfreq(freq)
        
        context_df = context_df.reset_index()
        context_df.columns = ["timestamp", "target"]
        context_df["item_id"] = self.item_id
        context_df = context_df[["item_id", "timestamp", "target"]]
        context_df["timestamp"] = pd.to_datetime(context_df["timestamp"])

        last_ts = context_df["timestamp"].max()
        quantile_levels = self._quantile_levels or pipeline.quantiles

        forecast_df = pipeline.predict_df(
            context_df,
            prediction_length=horizon,
            quantile_levels=quantile_levels,
            id_column="item_id",
            timestamp_column="timestamp",
            target=self._target_column,
        )

        if not isinstance(forecast_df, pd.DataFrame):
            raise TypeError("Chronos2Pipeline.predict_df()는 pandas.DataFrame을 반환해야 합니다.")

        forecast_one = forecast_df
        if "item_id" in forecast_df.columns:
            forecast_one = forecast_one[forecast_one["item_id"] == self.item_id]
        if "target_name" in forecast_one.columns:
            forecast_one = forecast_one[forecast_one["target_name"] == self._target_column]

        if "timestamp" in forecast_one.columns:
            forecast_one["timestamp"] = pd.to_datetime(forecast_one["timestamp"])
            forecast_one = forecast_one[forecast_one["timestamp"] > last_ts]
            forecast_one = forecast_one.sort_values("timestamp")

        forecast_one = forecast_one.head(horizon)
        if forecast_one.empty:
            raise ValueError("Chronos2Model.forecast: 예측 결과가 비어 있습니다.")

        if start_index is not None and len(start_index) == len(forecast_one):
            index = start_index
        elif "timestamp" in forecast_one.columns:
            index = pd.to_datetime(forecast_one["timestamp"].values)
        else:
            index = pd.RangeIndex(len(forecast_one))

        if "predictions" not in forecast_one.columns:
            raise ValueError("Chronos2Model.forecast: `predictions` 컬럼이 없습니다.")

        point_series = pd.Series(
            forecast_one["predictions"].astype(float).values,
            index=index,
        )

        quantiles: Dict[float, pd.Series] = {}
        skip_cols = {"item_id", "timestamp", "target_name", "predictions"}
        for col in forecast_one.columns:
            if col in skip_cols:
                continue
            try:
                q_level = float(col)
            except ValueError:
                continue
            if 0.0 < q_level < 1.0:
                quantiles[q_level] = pd.Series(
                    forecast_one[col].astype(float).values,
                    index=index,
                )

        output = ModelOutput(bundle=PredictionBundle(point=point_series, quantiles=quantiles or None))
        
        # Log scale information
        if self._train_df is not None:
            self._log_scale_info(self._train_df, output, "Chronos-v2")
        
        return output

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _prepare_quantile_levels(self, quantiles_cfg: list[object] | None) -> list[float] | None:
        if quantiles_cfg is None:
            return None
        if not isinstance(quantiles_cfg, list):
            raise TypeError("Chronos2Model: forecast.quantiles must be a list of numbers.")
        normalized: list[float] = []
        for item in quantiles_cfg:
            try:
                value = float(item)
            except (TypeError, ValueError) as exc:
                raise ValueError("Quantile levels must be numeric") from exc
            if not 0.0 < value < 1.0:
                raise ValueError("Quantile levels must be in the open interval (0, 1)")
            if value not in normalized:
                normalized.append(value)
        if 0.5 not in normalized:
            normalized.append(0.5)
        return normalized

    def _ensure_pipeline(self):
        """Chronos2Pipeline 인스턴스를 lazy-import 방식으로 생성."""
        if self._pipeline is not None:
            return self._pipeline
        try:  # pragma: no cover - optional dependency
            from chronos import Chronos2Pipeline  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Chronos2Model을 사용하려면 `chronos-forecasting` 패키지와 "
                "\"pandas[pyarrow]\" 의존성이 필요합니다. 예:\n"
                "  pip install chronos-forecasting 'pandas[pyarrow]'\n"
                f"원본 오류: {e}"
            ) from e

        model_source = self.local_dir or self.repo_id
        self._pipeline = Chronos2Pipeline.from_pretrained(model_source, device_map=self.device)  # type: ignore[call-arg]
        return self._pipeline


__all__ = ["Chronos2Model"]

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

import pandas as pd

from ..metrics import PredictionBundle


@dataclass(slots=True)
class ModelOutput:
    bundle: PredictionBundle
    artifacts: Dict[str, object] = field(default_factory=dict)


class BaseModel(ABC):
    """Common interface for all experiment models."""

    def __init__(self, model_config: Dict[str, object]) -> None:
        self.model_config = model_config

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None) -> None:
        ...

    @abstractmethod
    def forecast(self, horizon: int, *, start_index: Optional[pd.Index] = None) -> ModelOutput:
        ...

    def rolling_forecast(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        validation_df: Optional[pd.DataFrame] = None,
    ) -> ModelOutput:
        self.fit(train_df, validation_df)
        output = self.forecast(len(test_df), start_index=test_df.index)
        
        # Log scale information for each rolling window (only once per model to avoid spam)
        if not hasattr(self, '_scale_logged'):
            self._log_scale_info(train_df, output)
            self._scale_logged = True
        
        return output

    def supports_quantiles(self) -> bool:
        return True

    def _log_scale_info(self, train_df: pd.DataFrame, forecast_output: ModelOutput, model_name: str = None) -> None:
        """Log scale information for training data and forecasts to detect scaling issues."""
        logger = logging.getLogger("experiments")  # Use main experiments logger
        model_name = model_name or self.__class__.__name__
        
        # Training data scale
        y_train = train_df.iloc[:, 0] if isinstance(train_df, pd.DataFrame) else train_df
        train_min, train_max = y_train.min(), y_train.max()
        
        # Forecast scale
        forecast_mean = forecast_output.bundle.point
        pred_min, pred_max = forecast_mean.min(), forecast_mean.max()
        
        # Quantile ranges (if available)
        quantile_info = ""
        if hasattr(forecast_output.bundle, 'quantiles') and forecast_output.bundle.quantiles:
            lower_q = min(forecast_output.bundle.quantiles.keys())
            upper_q = max(forecast_output.bundle.quantiles.keys())
            q_lower = forecast_output.bundle.quantiles[lower_q]
            q_upper = forecast_output.bundle.quantiles[upper_q]
            quantile_info = f" | Quantile range [{lower_q}]: {q_lower.min():.2f} to [{upper_q}]: {q_upper.max():.2f}"
        
        logger.info(
            f"[{model_name}] Scale check - "
            f"Train: [{train_min:.2f}, {train_max:.2f}] | "
            f"Forecast: [{pred_min:.2f}, {pred_max:.2f}]"
            f"{quantile_info}"
        )
        
        # Warning if forecast scale is way off
        scale_ratio = max(abs(pred_max / train_max) if train_max != 0 else 0,
                         abs(train_min / pred_min) if pred_min != 0 else 0)
        if scale_ratio > 10:
            logger.warning(
                f"[{model_name}] ⚠️  SCALE MISMATCH: Forecast scale differs from training by {scale_ratio:.1f}x"
            )


__all__ = ["BaseModel", "ModelOutput"]

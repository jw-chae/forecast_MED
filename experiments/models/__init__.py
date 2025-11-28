from __future__ import annotations

from typing import Dict, Type

from .base_model import BaseModel
from .prophet_model import ProphetModel
from .arima_model import ARIMAModel
from .xgboost_model import XGBoostModel
from .lstm_model import LSTMModel
from .seir_model import SEIRModel
from .timesfm_model import TimesFMModel

try:
    from .epi_tools_model import EpiToolsModel
except ImportError:
    EpiToolsModel = None

# strategist_hybrid 모델은 선택적(optional)로 처리한다.
try:  # pragma: no cover - optional dependency
    from .strategist_model import StrategistHybridModel  # type: ignore
except Exception:  # pragma: no cover
    StrategistHybridModel = None  # type: ignore

# 선택적(heavy) 의존성이 있는 모델은 import 실패 시 레지스트리에 등록하지 않는다.
try:  # pragma: no cover - optional dependency
    from .chronos_model import ChronosModel  # type: ignore
except Exception:  # pragma: no cover
    ChronosModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .chronos2_model import Chronos2Model  # type: ignore
except Exception:  # pragma: no cover
    Chronos2Model = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .moirai_model import MoiraiModel  # type: ignore
except Exception:  # pragma: no cover
    MoiraiModel = None  # type: ignore


MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "prophet": ProphetModel,
    "arima": ARIMAModel,
    "xgboost": XGBoostModel,
    "lstm": LSTMModel,
    "seir": SEIRModel,
    "timesfm": TimesFMModel,
}

if EpiToolsModel is not None:
    MODEL_REGISTRY["epi_tools"] = EpiToolsModel

if ChronosModel is not None:  # type: ignore[truthy-function]
    MODEL_REGISTRY["chronos"] = ChronosModel  # type: ignore[arg-type]
if Chronos2Model is not None:  # type: ignore[truthy-function]
    MODEL_REGISTRY["chronos2"] = Chronos2Model  # type: ignore[arg-type]
if MoiraiModel is not None:  # type: ignore[truthy-function]
    MODEL_REGISTRY["moirai"] = MoiraiModel  # type: ignore[arg-type]
if 'StrategistHybridModel' in globals() and StrategistHybridModel is not None:  # type: ignore[truthy-function]
    MODEL_REGISTRY["strategist_hybrid"] = StrategistHybridModel  # type: ignore[arg-type]


def get_model_class(model_type: str) -> Type[BaseModel]:
    if model_type not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model type '{model_type}'. Available: {sorted(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type]


__all__ = ["get_model_class", "MODEL_REGISTRY", "BaseModel"]

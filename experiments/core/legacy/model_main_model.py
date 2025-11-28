"""Legacy hybrid simulator used by backup scripts.

NOTE:
- This module is not used by the main LLM rolling forecast
    pipeline (rolling_agent_forecast + SimpleEventInterpreter +
    SimpleForecastGenerator).
- It is kept for compatibility with older tools and experiments
    under scripts/ and experiments copy/.
"""

import numpy as np
from typing import Dict, Any, Optional


class PatternAndSEIR:
    def __init__(self, params: Dict[str, Any]):
        self.params = dict(params or {})

    def fit(self, y_hist: np.ndarray) -> None:
        # Placeholder: in a full implementation we could fit parameters to history
        self._hist = np.asarray(y_hist, dtype=float)

    def forecast(self, horizon: int, seir_curve: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        horizon = int(max(0, horizon))
        if horizon == 0:
            return {
                "q05": np.array([], dtype=float),
                "q50": np.array([], dtype=float),
                "q80": np.array([], dtype=float),
                "q95": np.array([], dtype=float),
                "paths": np.zeros((0, 0), dtype=float),
            }

        last_level = float(self._hist[-1]) if getattr(self, "_hist", None) is not None and len(self._hist) > 0 else 0.0
        trend = float(self.params.get("trend", 0.0))

        base = last_level + trend * np.arange(1, horizon + 1, dtype=float)
        if seir_curve is not None and seir_curve.size >= horizon:
            # simple hybrid: blend pattern with SEIR curve
            alpha = float(self.params.get("seir_blend", 0.3))
            base = (1.0 - alpha) * base + alpha * np.asarray(seir_curve[:horizon], dtype=float)

        noise_scale = float(self.params.get("noise_scale", 0.1))
        n_paths = int(self.params.get("n_paths", 1000))
        rng = np.random.default_rng(int(self.params.get("seed", 42)))
        paths = rng.normal(loc=base, scale=noise_scale * (1.0 + np.abs(base)), size=(n_paths, horizon))
        paths = np.maximum(paths, 0.0)

        q05 = np.quantile(paths, 0.05, axis=0)
        q50 = np.quantile(paths, 0.50, axis=0)
        q80 = np.quantile(paths, 0.80, axis=0)
        q95 = np.quantile(paths, 0.95, axis=0)
        return {"q05": q05, "q50": q50, "q80": q80, "q95": q95, "paths": paths}

"""Simple forecast generator with lag-aware fallback and logging."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .llm_agent import call_llm_json
from .pdf_rag_loader import get_hfmd_rag_loader

logger = logging.getLogger(__name__)


FORECAST_SYSTEM_PROMPT = """You are assisting with weekly infectious-disease forecasting.

LAG POLICY:
- HFMD/手足口病 typically reacts to external events with ~1 week delay
    (impact_lag_weeks = 1). transmission_impact describes expected net effect
    starting in week t+impact_lag_weeks.
- Week 1 forecast should be driven primarily by recent_values and recent trend,
    but interpreted in the context of the full multi‑year hospital time series.
- Weeks >= impact_lag_weeks may incorporate most of transmission_impact.

INPUT JSON SUMMARY (READ IN THIS ORDER):
1) Long‑term hospital history (MANDATORY)
     - full_history: weekly dates + values over multiple years.
     - First, summarize: overall level, seasonality (e.g., typical summer peaks,
         winter lows), and how the *current level* compares to past years.
     - NEVER ignore full_history; treat it as the primary context.

2) Recent 8‑week window
     - recent_values: last 8 weekly hospital cases.
     - recent_trend: {"growth_rate": float, "slope": float}.
     - Use this as a zoom‑in on the latest dynamics, but interpret it *relative
         to* the long‑term pattern from full_history. Short, noisy swings (e.g.,
         15→6→14→20) should not overrule stable seasonal structure.

3) Event interpreter output + external drivers
     - transmission_impact: float in [-1,1] describing expected net effect
         starting in week t+impact_lag_weeks,
     - confidence: float in [0,1] for the above impact,
     - risk_notes: optional strings explaining key drivers (school calendar,
         weather, policy changes, etc.),
     - historical_volatility: float summarizing recent variability.

4) Forecast configuration
     - horizon_weeks: int,
     - impact_lag_weeks: int,
     - mode: "standard" | "advanced".

OUTPUT STRICT JSON ONLY (SIMPLIFIED):
You MUST return a single JSON object with exactly the following top-level keys:
{
    "forecast_mean": [float >= 0],        // length = horizon_weeks, expected mean weekly counts
    "uncertainty_scale": float in [0, 1], // 0 = very low uncertainty, 1 = very high
    "rationale": "English explanation (≤3 sentences), MUST mention how you applied the lag policy"
}

CONSTRAINTS:
- Do NOT output any other top-level keys (no "forecast", no "quantiles").
- Do NOT wrap the JSON in Markdown fences.
- The forecast_mean array MUST have length = horizon_weeks.
- All entries in forecast_mean MUST be ≥ 0.
- uncertainty_scale MUST be a single float in [0, 1].
- Forecast_mean profile should be smooth; avoid implausible spikes that contradict
    lag policy, long‑term seasonality in full_history, or historical volatility.
- Always mention the lag interpretation inside the rationale when applicable,
    and clearly distinguish between effects of past conditions (already visible
    in recent_values) vs. expected effects of *current* conditions via
    transmission_impact.
"""


@dataclass
class ForecastResult:
    q05: List[float]
    q50: List[float]
    q95: List[float]
    uncertainty: List[float]
    prompt_payload: Dict[str, Any]
    raw_response: Optional[str] = None
    parsed_response: Optional[Dict[str, Any]] = None
    llm_metadata: Optional[Dict[str, Any]] = None
    used_fallback: bool = False
    per_week_debug: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q05": self.q05,
            "q50": self.q50,
            "q95": self.q95,
            "uncertainty": self.uncertainty,
            "used_fallback": self.used_fallback,
        }


class SimpleForecastGenerator:
    """Model that turns interpreted events into a probabilistic forecast."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        *,
        temperature: float = 0.6,  # Paper Section 3.1.4: "temperature is set to 0.6"
        thinking_mode: str = "none",
    ) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.thinking_mode = thinking_mode
        self.last_fallback_debug: Optional[List[Dict[str, Any]]] = None

    def generate_forecast(
        self,
        *,
        recent_values: List[float],
        transmission_impact: float,
        confidence: float,
        horizon: int,
        full_history: Optional[Dict[str, Any]] = None,
        risk_notes: Optional[List[str]] = None,
        mode: str = "standard",
        impact_lag_weeks: int = 0,
        use_llm: bool = True,
        allow_fallback: bool = True,
    ) -> ForecastResult:
        volatility = self._compute_volatility(recent_values)
        trend = self._compute_trend(recent_values)
        payload: Dict[str, Any] = {
            "recent_values": recent_values,
            "recent_trend": trend,
            "transmission_impact": transmission_impact,
            "confidence": confidence,
            "historical_volatility": volatility,
            "horizon_weeks": horizon,
            "mode": mode,
            "risk_notes": risk_notes or [],
            "impact_lag_weeks": impact_lag_weeks,
        }
        if full_history is not None:
            payload["full_history"] = full_history

        # Attach HFMD guideline context via PDF RAG when available
        try:
            # Only attempt for HFMD-type diseases; caller typically passes disease name via full_history metadata if needed
            disease_name = None
            if isinstance(full_history, dict):
                disease_name = full_history.get("disease") or full_history.get("disease_name")
            if isinstance(disease_name, str) and "手足口" in disease_name:
                rag = get_hfmd_rag_loader(force_reload=False)
                if rag is not None:
                    query = "HFMD forecasting seasonality peak decline treatment recommendations"
                    guideline_ctx = rag.get_context_for_prompt(query, k=3, max_length=1500)
                    if guideline_ctx:
                        payload["guideline_context"] = guideline_ctx
        except Exception:
            logger.exception("Failed to load HFMD guideline context for forecast generator; continuing without RAG.")

        logger.debug("ForecastGenerator payload summary: %s", self._summarize(payload))

        if not use_llm:
            logger.info("ForecastGenerator running in fallback mode (no LLM).")
            return self._fallback_forecast(payload)

        response, meta = call_llm_json(
            system_prompt=FORECAST_SYSTEM_PROMPT,
            user_payload=payload,
            model=self.model,
            provider=self.provider,
            temperature=self.temperature,
            thinking_mode=self.thinking_mode,
        )
        raw_response = None
        if meta.get("raw_response") is not None:
            try:
                raw_response = json.dumps(meta["raw_response"], ensure_ascii=False)
            except Exception:
                raw_response = str(meta["raw_response"])

        if not response:
            logger.warning("ForecastGenerator received empty response from LLM.")
            if not allow_fallback:
                raise RuntimeError("ForecastGenerator LLM response empty and fallback disabled")
            fallback = self._fallback_forecast(payload)
            fallback.raw_response = raw_response
            fallback.llm_metadata = meta
            return fallback

        return self._parse_response(
            response=response,
            payload=payload,
            raw_response=raw_response,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _summarize(self, payload: Dict[str, Any], max_items: int = 5) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                if len(value) <= max_items:
                    summary[key] = value
                else:
                    summary[key] = value[:max_items] + [f"... (+{len(value) - max_items} more)"]
            elif isinstance(value, dict) and key not in {"full_history"}:
                summary[key] = value
            elif key == "full_history" and isinstance(value, dict):
                summary[key] = {
                    "type": value.get("type", "dict"),
                    "length": len(value.get("values", [])),
                }
            else:
                summary[key] = value
        return summary

    def _parse_response(
        self,
        *,
        response: Dict[str, Any],
        payload: Dict[str, Any],
        raw_response: Optional[str],
        meta: Optional[Dict[str, Any]],
    ) -> ForecastResult:
        logger.debug("Parsing forecast response: %s", response)
        horizon = int(payload["horizon_weeks"])

        # LLM now returns a mean trajectory + a single uncertainty_scale;
        # we turn this into discrete-style quantiles on our side.
        raw_mean = response.get("forecast_mean")
        if raw_mean is None:
            raise ValueError("LLM response missing required 'forecast_mean' field")
        mean_arr = self._coerce_array(raw_mean, horizon)
        mean_arr = self._ensure_non_negative(mean_arr, horizon)

        # Fallback: if LLM forgot forecast_mean, fall back to simple copy of recent last value.
        if not any(mean_arr):
            last_val = float(payload.get("recent_values", [0.0])[-1] or 0.0) if payload.get("recent_values") else 0.0
            mean_arr = [last_val for _ in range(horizon)]

        # Map LLM uncertainty_scale + historical_volatility into a dispersion factor.
        unc = float(response.get("uncertainty_scale", 0.5) or 0.5)
        unc = max(0.0, min(1.0, unc))
        base_vol = float(payload.get("historical_volatility", 0.15) or 0.15)
        dispersion = max(0.25, min(3.0, (0.5 + 2.5 * unc) * (0.5 + 2.0 * base_vol)))

        # Use Poisson-like behavior near dispersion≈1 and Negative-Binomial-style
        # overdispersion when dispersion>1.
        q50: List[float] = []
        q05: List[float] = []
        q95: List[float] = []
        for m in mean_arr:
            m = max(0.0, float(m))
            if dispersion <= 1.0:
                lo, mid, hi = self._poisson_quantiles(m)
            else:
                overdisp = dispersion - 1.0
                lo, mid, hi = self._neg_binom_quantiles(m, overdisp)
            q05.append(lo)
            q50.append(mid)
            q95.append(hi)

        q05 = self._ensure_non_negative(q05, horizon)
        q50 = self._ensure_non_negative(q50, horizon)
        q95 = self._ensure_non_negative(q95, horizon)
        q05, q50, q95 = self._ensure_order(q05, q50, q95)

        # Build a simple per-week uncertainty in [0,1] increasing slightly with horizon.
        uncertainty = self._build_uncertainty(q50, payload["historical_volatility"])

        result = ForecastResult(
            q05=q05,
            q50=q50,
            q95=q95,
            uncertainty=uncertainty,
            prompt_payload=payload,
            raw_response=raw_response,
            parsed_response=response,
            llm_metadata=meta,
            used_fallback=False,
        )
        logger.info(
            "ForecastGenerator result horizon=%d first_week=%.2f",
            horizon,
            q50[0] if q50 else float("nan"),
        )
        return result

    def _fallback_forecast(self, payload: Dict[str, Any]) -> ForecastResult:
        recent_values = payload.get("recent_values", []) or []
        transmission_impact = float(payload.get("transmission_impact", 0.0) or 0.0)
        recent_trend = float(payload.get("recent_trend", {}).get("growth_rate", 0.0))
        historical_volatility = float(payload.get("historical_volatility", 0.15))
        horizon = int(payload.get("horizon_weeks", 1))
        impact_lag_weeks = int(payload.get("impact_lag_weeks", 0))

        forecasts: List[float] = []
        per_week_debug: List[Dict[str, Any]] = []
        current = float(recent_values[-1]) if recent_values else 0.0
        for i in range(horizon):
            if i < impact_lag_weeks:
                effective_growth = recent_trend
                lag_phase = "pre-lag"
            else:
                effective_growth = recent_trend + transmission_impact * 0.5
                lag_phase = "post-lag"
            growth_decay = 0.85 ** i
            raw_next = current * (1.0 + effective_growth * growth_decay)
            raw_next = max(0.0, raw_next)

            max_jump = max(historical_volatility * 2.0 * max(1.0, current), 0.1)
            delta = raw_next - current
            if abs(delta) > max_jump:
                next_val = current + max_jump * (1.0 if delta > 0 else -1.0)
            else:
                next_val = raw_next

            forecasts.append(next_val)
            per_week_debug.append(
                {
                    "week_index": i,
                    "lag_phase": lag_phase,
                    "effective_growth": effective_growth,
                    "growth_decay": growth_decay,
                    "raw_next_val": raw_next,
                    "clamped": next_val,
                }
            )
            current = next_val

        self.last_fallback_debug = per_week_debug

        q50 = forecasts
        spread = max(historical_volatility * 0.5, 0.05)
        q05 = [max(0.0, v - spread * max(1.0, v ** 0.5)) for v in q50]
        q95 = [v + spread * max(1.0, v ** 0.5) for v in q50]
        uncertainty = self._build_uncertainty(q50, historical_volatility)

        result = ForecastResult(
            q05=q05,
            q50=q50,
            q95=q95,
            uncertainty=uncertainty,
            prompt_payload=payload,
            used_fallback=True,
            per_week_debug=per_week_debug,
        )
        logger.debug("Fallback forecast debug: %s", per_week_debug)
        return result

    def _coerce_array(self, values: Optional[Any], horizon: int) -> List[float]:
        if not values:
            return [0.0 for _ in range(horizon)]
        if isinstance(values, list):
            arr = values
        else:
            arr = [values]
        arr = [max(0.0, float(v)) for v in arr]
        if len(arr) < horizon:
            arr += [arr[-1] if arr else 0.0] * (horizon - len(arr))
        return arr[:horizon]

    def _ensure_non_negative(self, arr: List[float], horizon: int) -> List[float]:
        if len(arr) < horizon:
            arr += [0.0] * (horizon - len(arr))
        return [max(0.0, float(v)) for v in arr[:horizon]]

    def _ensure_order(
        self,
        q05: List[float],
        q50: List[float],
        q95: List[float],
    ) -> (List[float], List[float], List[float]):
        ordered_q05, ordered_q50, ordered_q95 = [], [], []
        for lo, mid, hi in zip(q05, q50, q95):
            lo = float(lo)
            mid = max(lo, float(mid))
            hi = max(mid, float(hi))
            ordered_q05.append(lo)
            ordered_q50.append(mid)
            ordered_q95.append(hi)
        return ordered_q05, ordered_q50, ordered_q95

    def _poisson_quantiles(self, mean: float, alpha: float = 0.05) -> (float, float, float):
        """Compute Poisson quantiles using inverse CDF (ppf).

        Paper Section 3.2.3:
        Uses inverse cumulative distribution function for proper discrete distribution.
        Used when dispersion is close to 1 (variance ~ mean).
        """
        from scipy import stats
        
        m = max(0.0, float(mean))
        if m == 0.0:
            return 0.0, 0.0, 0.0
        # Use Poisson inverse CDF for accurate quantiles
        q05 = float(stats.poisson.ppf(0.05, mu=m))
        q50 = float(stats.poisson.ppf(0.50, mu=m))
        q95 = float(stats.poisson.ppf(0.95, mu=m))
        return q05, q50, q95

    def _neg_binom_quantiles(self, mean: float, overdisp: float, alpha: float = 0.05) -> (float, float, float):
        """Compute Negative Binomial quantiles using inverse CDF (ppf).

        Paper Section 3.2.3:
        Given target mean μ and variance σ², NB parameters are:
        n = μ² / (σ² - μ)  for σ² > μ
        p = n / (n + μ)
        
        Variance formula from paper: σ² = (μ · v · (1 + u))²
        Here we receive overdisp which encodes (v · (1 + u)).
        """
        from scipy import stats
        
        m = max(0.0, float(mean))
        if m == 0.0:
            return 0.0, 0.0, 0.0
        
        overdisp = max(0.0, float(overdisp))
        # Paper formula: σ² = (μ · v · (1 + u))² approximated via overdisp
        # Var = m + overdisp * m^2 (NB standard parameterization)
        var = m + overdisp * (m ** 2)
        
        # Fall back to Poisson if underdispersed
        if var <= m:
            return self._poisson_quantiles(m, alpha)
        
        # NB moment matching: n = μ² / (σ² - μ), p = n / (n + μ)
        n = (m ** 2) / (var - m)
        p = n / (n + m)
        
        # Use NB inverse CDF for accurate quantiles
        try:
            q05 = float(stats.nbinom.ppf(0.05, n=n, p=p))
            q50 = float(stats.nbinom.ppf(0.50, n=n, p=p))
            q95 = float(stats.nbinom.ppf(0.95, n=n, p=p))
        except Exception:
            # Fallback to normal approximation if scipy fails
            std = math.sqrt(max(var, 1e-6))
            q05 = max(0.0, m - 1.645 * std)
            q50 = m
            q95 = m + 1.645 * std
        
        return q05, q50, q95

    def _build_uncertainty(self, q50: List[float], volatility: float) -> List[float]:
        base = max(0.1, min(0.9, volatility * 3.0))
        return [min(0.95, base + 0.02 * idx) for idx, _ in enumerate(q50)]

    def _compute_volatility(self, recent_values: List[float]) -> float:
        """Compute volatility as median of recent relative changes, clamped to [0.05, 0.50].
        
        Paper Section 3.2.1:
        v = Clamp(Median(|relative_change|), 0.05, 0.50)
        - Min 0.05: prevents overconfidence (intervals too narrow)
        - Max 0.50: prevents uninformative forecasts (intervals too wide)
        """
        if len(recent_values) < 2:
            return 0.15
        arr = np.array(recent_values, dtype=float)
        deltas = np.abs(np.diff(arr))
        baseline = np.maximum(1.0, arr[:-1])
        pct = deltas / baseline
        raw_vol = float(np.median(pct) if pct.size else 0.15)
        # Clamp to [0.05, 0.50] as per paper
        return max(0.05, min(0.50, raw_vol))

    def _compute_trend(self, recent_values: List[float]) -> Dict[str, float]:
        """Compute growth rate and slope from recent values.
        
        Paper Section 3.1.3:
        g_t = (y_t - y_{t-4}) / max(1.0, y_{t-4})
        
        Fallback for < 4 weeks: use first available value as baseline.
        """
        if len(recent_values) < 2:
            return {"growth_rate": 0.0, "slope": 0.0}
        arr = np.array(recent_values, dtype=float)
        if len(arr) >= 4:
            # Paper formula: 4-week difference
            growth_rate = (arr[-1] - arr[-4]) / max(1.0, arr[-4])
        else:
            # Fallback: use available history (not in paper, implementation detail)
            growth_rate = (arr[-1] - arr[0]) / max(1.0, arr[0])
        slope = float(arr[-1] - arr[-2])
        return {"growth_rate": float(growth_rate), "slope": slope}

"""Simple HFMD-aware event interpreter with logging and lag awareness."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_agent import call_llm_json
from .pdf_rag_loader import get_hfmd_rag_loader

logger = logging.getLogger(__name__)


INTERPRETER_SYSTEM_PROMPT = """You are an infectious-disease analyst translating qualitative context
into HFMD transmission signals.

IMPORTANT LAG POLICY:
- HFMD (Hand-Foot-Mouth Disease, 手足口病) typically shows a 1-week delay between
  behavioral/environmental shifts and reported cases (incubation + reporting).
- The transmission_impact you emit must describe the expected net effect starting
  next week (t+1) and the few weeks after, not primarily the current week.

INPUT JSON FIELDS:
- disease, date, horizon_weeks, impact_lag_weeks (usually 1 for HFMD).
- recent_values (ordered old→new) and derived weekly trend statistics.
- external_data: school calendars, weather summaries, news, government bulletins.
- recent qualitative notes / risk flags if available.

OUTPUT STRICT JSON (no markdown):
{
  "transmission_impact": float in [-1, 1],
  "confidence": float in [0, 1],
  "event_summary": "short natural-language summary",
  "risk_notes": ["zero or more short bullet strings"],
  "lag_rationale": "optional additional note about lag/lead timing"
}

GUIDANCE:
1. Treat school status as the strongest driver, followed by temperature/humidity,
   then other news. Weather alone without schools rarely drives large shifts.
2. transmission_impact > 0 implies conditions that are likely to increase cases
   starting next week; < 0 implies headwinds. Reserve |impact| > 0.6 for
   strongly aligned signals.
3. Mention lag explicitly in your summary when possible (e.g., "school reopening
   may lift cases from next week onward").
4. Be concise; do not restate the full payload.
"""


@dataclass
class EventInterpretation:
    """Structured event interpretation result."""

    transmission_impact: float
    confidence: float
    event_summary: str
    risk_notes: List[str]
    prompt_payload: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[str] = None
    parsed_response: Optional[Dict[str, Any]] = None
    llm_metadata: Optional[Dict[str, Any]] = None
    lag_rationale: Optional[str] = None
    used_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transmission_impact": self.transmission_impact,
            "confidence": self.confidence,
            "event_summary": self.event_summary,
            "risk_notes": self.risk_notes,
            "lag_rationale": self.lag_rationale,
            "used_fallback": self.used_fallback,
        }


class SimpleEventInterpreter:
    """Model that maps evidence packs into a transmission impact score."""

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

    def interpret(
        self,
        *,
        disease: str,
        train_until: str,
        recent_values: List[float],
        external_data: Dict[str, Any],
        horizon: int,
        full_history: Optional[Dict[str, Any]] = None,
        impact_lag_weeks: int = 0,
        use_llm: bool = True,
        allow_fallback: bool = True,
    ) -> EventInterpretation:
        payload = self._build_payload(
            disease=disease,
            train_until=train_until,
            recent_values=recent_values,
            external_data=external_data,
            horizon=horizon,
            full_history=full_history,
            impact_lag_weeks=impact_lag_weeks,
        )

        # Attach HFMD guideline context via PDF RAG when available
        # Paper Section 3.1.2: Dynamic query construction based on contextual signals
        if isinstance(disease, str) and "手足口" in disease:
            try:
                rag = get_hfmd_rag_loader(force_reload=False)
                if rag is not None:
                    query = self._build_dynamic_rag_query(train_until, external_data)
                    guideline_ctx = rag.get_context_for_prompt(query, k=2, max_length=1200)  # Paper: k=2, 1200 chars
                    if guideline_ctx:
                        payload["guideline_context"] = guideline_ctx
            except Exception:
                logger.exception("Failed to load HFMD guideline context; continuing without RAG.")
        logger.debug("EventInterpreter payload summary: %s", self._summarize(payload))

        if not use_llm:
            logger.info("EventInterpreter running in fallback mode (no LLM).")
            return self._fallback_interpretation(payload)

        response, meta = call_llm_json(
            system_prompt=INTERPRETER_SYSTEM_PROMPT,
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
            logger.warning("EventInterpreter received empty response from LLM.")
            if not allow_fallback:
                raise RuntimeError("EventInterpreter LLM response empty and fallback disabled")
            fallback = self._fallback_interpretation(payload)
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
    def _build_payload(
        self,
        *,
        disease: str,
        train_until: str,
        recent_values: List[float],
        external_data: Dict[str, Any],
        horizon: int,
        full_history: Optional[Dict[str, Any]],
        impact_lag_weeks: int,
    ) -> Dict[str, Any]:
        trend = self._compute_trend(recent_values)
        payload: Dict[str, Any] = {
            "disease": disease,
            "date": train_until,
            "recent_values": recent_values,
            "recent_trend": trend,
            "external_data": external_data or {},
            "horizon_weeks": horizon,
            "impact_lag_weeks": impact_lag_weeks,
        }
        if full_history:
            payload["full_history"] = full_history
        return payload

    def _summarize(self, payload: Dict[str, Any], max_items: int = 5) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                if len(value) <= max_items:
                    summary[key] = value
                else:
                    summary[key] = value[:max_items] + [f"... (+{len(value) - max_items} more)"]
            elif isinstance(value, dict) and key not in {"external_data", "full_history"}:
                summary[key] = value
            elif key in {"external_data", "full_history"}:
                summary[key] = {
                    "type": type(value).__name__,
                    "keys": list(value.keys())[:max_items] if isinstance(value, dict) else None,
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
    ) -> EventInterpretation:
        logger.debug("Parsing interpreter response: %s", response)
        try:
            impact = float(response.get("transmission_impact", 0.0))
        except Exception:
            logger.exception("Failed to parse transmission_impact; defaulting to 0.0")
            impact = 0.0
        impact = max(-1.0, min(1.0, impact))

        try:
            confidence = float(response.get("confidence", 0.5))
        except Exception:
            logger.exception("Failed to parse confidence; defaulting to 0.5")
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        summary = str(response.get("event_summary") or "No summary provided")
        lag_rationale = response.get("lag_rationale")
        risk_notes = response.get("risk_notes") or []
        if not isinstance(risk_notes, list):
            risk_notes = [str(risk_notes)]

        parsed = EventInterpretation(
            transmission_impact=impact,
            confidence=confidence,
            event_summary=summary,
            risk_notes=[str(note) for note in risk_notes],
            lag_rationale=str(lag_rationale) if lag_rationale else None,
            prompt_payload=payload,
            raw_response=raw_response,
            parsed_response=response,
            llm_metadata=meta,
            used_fallback=False,
        )
        logger.info(
            "EventInterpreter result impact=%.3f confidence=%.2f notes=%s",
            parsed.transmission_impact,
            parsed.confidence,
            parsed.risk_notes,
        )
        return parsed

    def _fallback_interpretation(self, payload: Dict[str, Any]) -> EventInterpretation:
        recent_values = payload.get("recent_values", []) or []
        external_data = payload.get("external_data", {}) or {}
        impact_lag_weeks = payload.get("impact_lag_weeks", 0)
        trend = self._compute_trend(recent_values)
        growth = trend.get("pct_change", 0.0)
        weeks_growing = trend.get("weeks_growing", 0)
        impact = max(-0.5, min(0.5, growth * 0.6))

        school_status = (external_data.get("events_calendar") or {}).get("school_status")
        if isinstance(school_status, str):
            status_lower = school_status.lower()
            if "close" in status_lower or "vacation" in status_lower:
                impact -= 0.2
            elif "open" in status_lower or "session" in status_lower:
                impact += 0.1

        weather = external_data.get("weather") or {}
        avg_temp = weather.get("avg_temp") or weather.get("temperature_c")
        if avg_temp is None:
            avg_temp = weather.get("monthly_avg_temp")
            if isinstance(avg_temp, list) and avg_temp:
                avg_temp = sum(avg_temp[-3:]) / min(3, len(avg_temp))
        if isinstance(avg_temp, (int, float)):
            if avg_temp >= 25:
                impact += 0.15
            elif avg_temp <= 15:
                impact -= 0.15

        if weeks_growing >= 3 and impact > 0:
            impact += 0.1
        impact = max(-1.0, min(1.0, impact))

        confidence = 0.5 + min(0.3, abs(impact) * 0.4)
        risk_notes: List[str] = []
        if impact > 0.4:
            risk_notes.append("further growth possible")
        elif impact < -0.4:
            risk_notes.append("decline expected")

        summary = "Trend-based interpretation"
        if school_status:
            summary += f"; schools {school_status}"
        if avg_temp is not None:
            summary += f"; avg temp {avg_temp:.1f}C"
        summary += f"; lag={impact_lag_weeks} week"

        interpretation = EventInterpretation(
            transmission_impact=impact,
            confidence=confidence,
            event_summary=summary,
            risk_notes=risk_notes,
            prompt_payload=payload,
            raw_response=None,
            parsed_response=None,
            llm_metadata=None,
            lag_rationale="Deterministic fallback assumes impact applies next week.",
            used_fallback=True,
        )
        logger.debug("Fallback interpretation: %s", interpretation)
        return interpretation

    def _build_dynamic_rag_query(self, train_until: str, external_data: Dict[str, Any]) -> str:
        """Build dynamic RAG query based on contextual signals.
        
        Paper Section 3.1.2:
        - Seasonal factors: "HFMD peak season spring summer" during May-July
        - Environmental factors: "weather temperature impact transmission" when weather data present
        - Social factors: "school in_session outbreak children" when schools in session
        """
        from datetime import datetime
        
        query_parts = ["HFMD"]
        
        # Seasonal factors
        try:
            month = datetime.strptime(train_until, "%Y-%m-%d").month
            if month in (5, 6, 7):
                query_parts.append("peak season spring summer transmission increase")
            elif month in (1, 2, 12):
                query_parts.append("winter low transmission cold weather")
            elif month in (8, 9):
                query_parts.append("autumn school reopening outbreak risk")
            elif month in (3, 4):
                query_parts.append("spring warming transmission onset")
            else:
                query_parts.append("seasonal pattern")
        except Exception:
            query_parts.append("seasonal pattern")
        
        # Environmental factors
        weather = external_data.get("weather_summary_last_7d") or external_data.get("weather") or {}
        if weather:
            query_parts.append("weather temperature humidity impact transmission")
            avg_temp = weather.get("tavg_mean") or weather.get("avg_temp")
            if isinstance(avg_temp, (int, float)):
                if avg_temp >= 25:
                    query_parts.append("warm weather favorable conditions")
                elif avg_temp <= 15:
                    query_parts.append("cold weather reduced transmission")
        
        # Social factors - school status
        school_info = external_data.get("school_calendar") or external_data.get("events_calendar") or {}
        school_status = school_info.get("school_status", "")
        if isinstance(school_status, str):
            status_lower = school_status.lower()
            if "session" in status_lower or "open" in status_lower:
                query_parts.append("school in_session outbreak children kindergarten")
            elif "vacation" in status_lower or "break" in status_lower:
                query_parts.append("school vacation reduced contact transmission")
        
        # Holiday factors
        holidays = school_info.get("holidays") or school_info.get("public_holidays") or []
        if holidays:
            query_parts.append("holiday gathering family transmission")
        
        return " ".join(query_parts)

    def _compute_trend(self, recent_values: List[float]) -> Dict[str, float]:
        if len(recent_values) < 2:
            return {"pct_change": 0.0, "weeks_growing": 0}
        prev = recent_values[:-1]
        latest = recent_values[-1]
        baseline = prev[-1] if prev else 1.0
        pct_change = (latest - baseline) / max(1.0, baseline)
        weeks_growing = 0
        for i in range(len(recent_values) - 1, 0, -1):
            if recent_values[i] > recent_values[i - 1]:
                weeks_growing += 1
            else:
                break
        return {"pct_change": float(pct_change), "weeks_growing": weeks_growing}


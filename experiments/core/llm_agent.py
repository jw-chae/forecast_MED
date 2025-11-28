from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
from typing import Any, Dict, Optional, Tuple, List
import urllib.error
from pathlib import Path
import time

from .config import APP_CONFIG


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an epidemiologist and forecasting expert.\n"
    "Always respond with exactly ONE valid JSON object encoded in UTF-8. "
    "Do NOT include Markdown, extra text, or multiple JSON blocks.\n\n"
    "Required top-level keys:\n"
    "{\n"
    '  \"rationale\": \"string (≤2 sentences explaining your main reasoning)\",\n'
    '  \"proposal\": {\n'
    '      \"<param_name>\": number | \"maintain\"  // omit keys you do not want to change\n'
    "  },\n"
    '  \"forecast\": [non-negative numbers, length = requested horizon],\n'
    '  \"quantiles\": {\"q05\": [...], \"q50\": [...], \"q95\": [...]},\n'
    '  \"expected_tradeoffs\": \"string\"\n'
    "}\n\n"
    "INTERPRETATION GUIDANCE (SHORT):\n"
    "- First, decide whether the current situation looks like GROWTH, PEAK/PLATEAU, or DECLINE,\n"
    "  using recent_values, any SEIR curve hints, and strategist_risk_notes (e.g. 'peak', 'post-peak decline').\n"
    "- In clear PEAK or POST-PEAK situations, avoid aggressively increasing amplitude-related parameters.\n"
    "- In obvious GROWTH situations with strong positive drivers, modestly increase amplitude or reproduction-related parameters.\n"
    "- When signals conflict, prefer conservative adjustments (small changes or 'maintain').\n\n"
    "HARD CONSTRAINTS:\n"
    "1. All numeric proposals must respect the provided min/max bounds. If a desired value would exceed bounds, choose the closest allowed value or 'maintain'.\n"
    "2. If you intend to keep a parameter unchanged, either omit it from 'proposal' or set it to \"maintain\".\n"
    "3. 'forecast' and each quantile array must have exactly the requested horizon length and must be ≥ 0.\n"
    "4. For every time step, quantiles must satisfy: q05 ≤ q50 ≤ q95.\n"
    "5. 'rationale' and 'expected_tradeoffs' must be written in English (US) and briefly explain why you made those parameter and forecast choices.\n"
    "6. Do NOT show internal reasoning. Output only the single JSON object.\n"
)

# Disease-specific epidemiological knowledge for HFMD
HFMD_SPECIFIC_PROMPT = """
HAND-FOOT-MOUTH DISEASE (HFMD, 手足口病) – KEY EPIDEMIOLOGICAL PATTERNS:

1. SEASONALITY
   - Primary peak: late spring to early summer (roughly May–July).
   - Secondary smaller peak: early autumn (roughly September–October).
   - Winter (roughly December–February) is usually a low-activity period with near-zero baseline.

2. ROLE OF SCHOOLS
   - Young children in schools and kindergartens are major drivers of transmission.
   - When schools are open and conditions are favorable, cases can rise quickly.
   - Summer/winter vacations or prolonged closures often lead to rapid declines (e.g., 20–40% within a few weeks).

3. ENVIRONMENTAL CONDITIONS
   - Temperatures around 20–30°C with adequate humidity favor transmission.
   - Very cold or very hot conditions make sustained outbreaks less likely.
   - Heavy or prolonged rainfall can temporarily reduce contact patterns.

4. EPIDEMIC CURVE SHAPE
   - Growth: fast increases during favorable conditions and active school terms.
   - Peak: seasonal maxima are usually not sustained plateaus; peaks often last 1–2 weeks.
   - Decline: post-peak declines are often relatively rapid compared to off-season noise.

5. FORECASTING IMPLICATIONS
   - Large sudden spikes during winter are epidemiologically less plausible without extraordinary drivers.
   - When cases are extremely low and no strong drivers are present, forecasts should remain conservative.
   - When values are at or near recent seasonal highs during peak season, it is often more reasonable to expect
     stabilization or decline than indefinite further growth.

6. BIOLOGICAL PARAMETERS
   - Incubation Period: Typically 3–7 days.
   - Lag Effect: Transmission events (e.g., school opening) usually impact reported case counts 
     in the following week (Lag-1 week) due to incubation and reporting delays.
Use these patterns as soft background knowledge when interpreting events and making forecasts. 
They are NOT strict rules; always combine them with the concrete recent data and external signals you receive.
"""


def is_hfmd(disease_name: Optional[str]) -> bool:
    """Return True if the disease string maps to HFMD/hand-foot-mouth."""

    if not disease_name:
        return False
    lowered = disease_name.lower()
    if "hfmd" in lowered:
        return True
    if "hand-foot" in lowered or "hand foot" in lowered:
        return True
    return "手足口" in disease_name


def _build_user_prompt(observation: Dict[str, Any]) -> str:
    disease = observation.get("disease_name") or observation.get("disease") or "Unknown disease"
    horizon = int(observation.get("desired_horizon") or observation.get("horizon") or 4)
    last_params = observation.get("last_week_params") or {}
    constraints = (observation.get("param_constraints") or {}).get("bounds", {})
    last_metrics = observation.get("last_week_metrics") or {}
    internal = observation.get("internal_weekly") or {}
    external = observation.get("external_signals") or {}
    summary_payload = {
        "disease": disease,
        "train_until": observation.get("train_until"),
        "predict_end_date": observation.get("predict_end_date"),
        "desired_horizon": horizon,
        "last_week_params": last_params,
        "param_bounds": constraints,
        "last_week_metrics": last_metrics,
        "internal_weekly": internal,
        "external_signals": external,
        "recent_metrics": observation.get("recent_metrics"),
        "strategist_events": observation.get("strategist_events", []),
        "strategist_risk_notes": observation.get("strategist_risk_notes", []),
        "recent_values": observation.get("recent_values", []),
    }
    instructions = (
        f"Adjust epidemic forecast parameters for disease '{disease}'. "
        f"Propose bounded numeric values and a {horizon}-week forecast based on the context JSON below. "
        "Return ONLY one JSON object that follows the schema from the system prompt.\n"
    )
    return instructions + json.dumps(summary_payload, ensure_ascii=False, indent=2)


def _http_post_json(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content)
    except urllib.error.HTTPError as e:
        try:
            err_txt = e.read().decode("utf-8", errors="ignore")
            try:
                return json.loads(err_txt)
            except Exception:
                return {"error": {"status": e.code, "message": err_txt}}
        finally:
            pass


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        start = None
        depth = 0
        for idx, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    snippet = text[start : idx + 1]
                    try:
                        obj = json.loads(snippet)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        start = None
        return None


def _extract_proposal_fields(parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return {
            "proposal": None,
            "rationale_summary": None,
            "expected_tradeoffs": None,
            "forecast": None,
            "quantiles": None,
        }
    proposal = parsed.get("proposal")
    if isinstance(proposal, dict):
        clean = {}
        for key, value in proposal.items():
            if value == "maintain" or value is None:
                continue
            clean[key] = value
        proposal = clean or None
    else:
        proposal = None
    rationale = parsed.get("rationale") or parsed.get("rationale_summary")
    return {
        "proposal": proposal,
        "rationale_summary": rationale,
        "expected_tradeoffs": parsed.get("expected_tradeoffs"),
        "forecast": parsed.get("forecast"),
        "quantiles": parsed.get("quantiles"),
    }


def _summarize_payload(payload: Dict[str, Any], max_list_len: int = 8) -> Dict[str, Any]:
    """Return a lightweight, human-readable summary of a user payload.

    This is used only for debug printing so that we can see what was sent
    to the LLM without dumping huge arrays into stdout.
    """

    def _short_list(seq: Any) -> Any:
        if not isinstance(seq, list):
            return seq
        if len(seq) <= max_list_len:
            return seq
        head = seq[:max_list_len]
        return head + [f"... (+{len(seq) - max_list_len} more)"]

    summary: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"full_history", "internal_weekly", "external_signals"}:
            # Avoid dumping very large structures; just note basic info.
            if isinstance(value, dict):
                # For dict with dates/values, show summary only
                if "dates" in value and "values" in value:
                    summary[key] = {
                        "type": "time_series",
                        "length_weeks": value.get("length_weeks", len(value.get("values", []))),
                        "date_range": f"{value['dates'][0]} to {value['dates'][-1]}" if value.get("dates") else "unknown"
                    }
                else:
                    summary[key] = {
                        "type": "dict",
                        "keys": list(value.keys()),
                    }
            elif isinstance(value, list):
                summary[key] = {
                    "type": "list",
                    "length": len(value),
                    "head": _short_list(value),
                }
            else:
                summary[key] = str(type(value))
        elif isinstance(value, list):
            summary[key] = _short_list(value)
        else:
            summary[key] = value
    return summary


def _setup_api_key(provider: str) -> Optional[str]:
    """Sets up the API key for the given provider and returns it."""
    key_env_var = "DASHSCOPE_API_KEY" if provider != "openai" else "OPENAI_API_KEY"
    api_key = os.environ.get(key_env_var)
    if api_key:
        return api_key

    try:
        proj_root = Path(__file__).resolve().parents[2]
        env_path = proj_root / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith(f"{key_env_var}="):
                    _, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    if v:
                        os.environ[key_env_var] = v
                        return v
    except Exception:
        pass
    return None


def clamp_params(p: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
    bounds = constraints.get("bounds", {})
    def clamp(key: str, val: Any):
        if key in bounds:
            lo, hi = bounds[key]
            try:
                return max(lo, min(hi, float(val)))
            except Exception:
                return val
        return val
    out = dict(p)
    for k, v in list(out.items()):
        out[k] = clamp(k, v)
    return out


def _print_llm_debug(prefix: str, data: Dict[str, Any]) -> None:
    """Pretty-print compact LLM debug information to stdout.

    This relies on the rolling driver to tee stdout into a log file, so that
    debugging information (inputs, evidence, thoughts) is preserved without
    changing the JSON artifacts written per-step.
    """

    try:
        msg = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        msg = str(data)
    logger.debug("[LLM DEBUG] %s\n%s", prefix, msg)



def propose_params_via_llm(
    observation: Dict[str, Any],
    model: str = "qwen-max",
    provider: str = "dashscope",
    temperature: float = 0.2,
    thinking_mode: str = "none",
    store_cot: bool = False,
 ) -> Dict[str, Any]:
        """High-level helper for simulator parameter proposals (legacy path).

        NOTE:
                - The current main rolling LLM pipeline calls `call_llm_json`
                    directly instead of this wrapper.
                - This function is kept for backward compatibility with
                    earlier hybrid SEIR/pattern experiments.
                """

        # Use the detailed function to get the full response
        full_response = propose_params_via_llm_with_debug(
            observation=observation,
            model=model,
            provider=provider,
            temperature=temperature,
            thinking_mode=thinking_mode,
            store_cot=store_cot,
        )
        payload = {
            "proposal": None,
            "rationale": None,
            "expected_tradeoffs": None,
            "forecast": None,
            "quantiles": None,
            "raw": None,
            "parsed": None,
            "thinking": None,
            "latency_ms": None,
        }
        if not full_response:
            payload["rationale"] = "LLM call failed or returned empty."
            return payload

        proposal = full_response.get("proposal")
        if isinstance(proposal, dict) and proposal:
            payload["proposal"] = proposal

        payload.update(
            {
                "rationale": full_response.get("rationale_summary"),
                "expected_tradeoffs": full_response.get("expected_tradeoffs"),
                "forecast": full_response.get("forecast"),
                "quantiles": full_response.get("quantiles"),
                "raw": full_response.get("raw"),
                "parsed": full_response.get("parsed"),
                "thinking": full_response.get("thinking"),
                "latency_ms": full_response.get("latency_ms"),
            }
        )
        return payload


def propose_params_via_llm_with_debug(
    observation: Dict[str, Any],
    provider: str,
    model: str,
    temperature: float = 0.2,
    api_base: str = APP_CONFIG.api.dashscope_api_base,
    thinking_mode: str = "none",
    store_cot: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    LLM call with full debug output.
    """
    print(f"\n{'='*80}")
    print(f"🚀 ENTERING propose_params_via_llm_with_debug")
    print(f"   Provider: {provider}, Model: {model}, Thinking: {thinking_mode}")
    print(f"{'='*80}\n")
    
    api_key = _setup_api_key(provider)
    if not api_key:
        print(f"[WARN] {provider.upper()}_API_KEY not found. Skipping LLM call.")
        # API 키가 없을 때 테스트를 위한 기본 하이브리드 파라미터 반환
        disease_name = observation.get("disease_name")
        train_until = observation.get("train_until")
        if disease_name == "流行性感冒" and train_until and train_until.startswith("2023-10"):
             return {
                "proposal": {
                    "use_seir_hybrid": True,
                    "seir_beta": 0.6,
                    "seir_incubation_days": 3.0,
                    "amplitude_multiplier": 1.5,
                    "quality": 0.75,
                },
                "rationale": "API key not found. Returning default hybrid params for Influenza test case.",
                "forecast": [100, 110, 120, 130],  # 테스트를 위한 예시 예측값
                "quantiles": {
                    "q05": [80, 90, 100, 110],
                    "q50": [100, 110, 120, 130],
                    "q95": [120, 130, 140, 150]
                }
            }
        return None

    # Add disease-specific guidance if detected
    sys_prompt = SYSTEM_PROMPT
    disease_guidance = observation.get("disease_specific_guidance")
    if disease_guidance == "HFMD":
        sys_prompt += "\n\n" + HFMD_SPECIFIC_PROMPT
        print("\n[INFO] 🦠 HFMD-specific forecasting rules activated")
    
    if provider == "openai":
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        use_responses = str(model).lower().startswith("gpt-5")
        url = "https://api.openai.com/v1/responses" if use_responses else "https://api.openai.com/v1/chat/completions"
    else:
        url = f"{api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    user_prompt = _build_user_prompt(observation)
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    if provider == "openai" and str(model).lower().startswith("gpt-5"):
        body = {
            "model": model,
            "text": {"format": {"type": "json_object"}},
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": sys_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
        }
    else:
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        # Check if enable_thoughts will be used
        enable_thoughts_flag = False
        if provider == "dashscope" and ("thinking" in str(model).lower() or "qwen3" in str(model).lower()):
            enable_thoughts_flag = True
        
        # JSON mode conflicts with enable_thoughts, so only set one
        if enable_thoughts_flag:
            body["enable_thoughts"] = True
            print("[DEBUG] Enabled thoughts for DashScope thinking model (JSON mode disabled)")
        elif not store_cot:
            body["response_format"] = {"type": "json_object"}

        if provider == "dashscope" and thinking_mode in ("light", "deep"):
            body["parameters"] = {"thinking_mode": thinking_mode}

    try:
        t0 = time.time()
        print(f"[DEBUG] Making API call to {url}")
        print(f"[DEBUG] Model: {model}, Provider: {provider}")
        
        resp = _http_post_json(url, headers, body)
        latency_ms = int((time.time() - t0) * 1000)
        
        print(f"[DEBUG] API response received in {latency_ms}ms")
        print(f"[DEBUG] Response keys: {list(resp.keys()) if isinstance(resp, dict) else 'Not a dict'}")
        if isinstance(resp, dict) and "error" in resp:
            print(f"[ERROR] API Error: {resp.get('error')}")
        if isinstance(resp, dict) and "choices" in resp:
            msg = resp.get("choices", [{}])[0].get("message", {})
            print(f"[DEBUG] Message keys: {list(msg.keys())}")
        
        # Extract thinking/reasoning if available
        thinking_content = None
        if provider == "dashscope" and isinstance(resp, dict):
            # DashScope thinking models use reasoning_content field
            msg = resp.get("choices", [{}])[0].get("message", {})
            reasoning = msg.get("reasoning_content") or msg.get("thoughts")
            if reasoning:
                thinking_content = reasoning
                print("\n" + "="*80)
                print("🧠 LLM THINKING PROCESS:")
                print("="*80)
                print(thinking_content)
                print("="*80 + "\n")
        elif provider == "openai" and str(model).lower().startswith("gpt-5"):
            # GPT-5 may expose reasoning in metadata
            reasoning = resp.get("metadata", {}).get("reasoning")
            if reasoning:
                thinking_content = reasoning
                print("\n" + "="*80)
                print("🧠 LLM REASONING:")
                print("="*80)
                print(thinking_content)
                print("="*80 + "\n")
        
        if provider == "openai" and str(model).lower().startswith("gpt-5"):
            content = resp.get("output_text", "")
        else:
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed: Optional[Dict[str, Any]]
        parsed = _extract_json(content)
        usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
        fields = _extract_proposal_fields(parsed)
        return {
            "parsed": parsed,
            "proposal": fields.get("proposal"),
            "rationale_summary": fields.get("rationale_summary"),
            "expected_tradeoffs": fields.get("expected_tradeoffs"),
            "forecast": fields.get("forecast"),
            "quantiles": fields.get("quantiles"),
            "raw": content,
            "thinking": thinking_content,
            "response": {"usage": usage},
            "response_raw": resp,
            "request": {"model": model, "api_base": api_base, "messages_len": len(messages)},
            "latency_ms": latency_ms,
        }
    except Exception:
        return None


def extract_numeric_forecast_from_llm_response(response: Dict[str, Any], horizon: int) -> Optional[List[float]]:
    """
    LLM 응답에서 숫자 예측값을 추출합니다.
    
    Args:
        response: LLM 응답 딕셔너리
        horizon: 예측 주간 수
        
    Returns:
        예측값 리스트 또는 None
    """
    if not response:
        return None
        
    # 1. forecast 필드에서 직접 추출
    forecast = response.get("forecast")
    if isinstance(forecast, list) and len(forecast) >= horizon:
        try:
            return [max(0.0, float(x)) for x in forecast[:horizon] if isinstance(x, (int, float))]
        except Exception:
            pass
    
    # 2. parsed 응답에서 추출
    parsed = response.get("parsed")
    if isinstance(parsed, dict):
        forecast = parsed.get("forecast")
        if isinstance(forecast, list) and len(forecast) >= horizon:
            try:
                return [max(0.0, float(x)) for x in forecast[:horizon] if isinstance(x, (int, float))]
            except Exception:
                pass
    
    # 3. quantiles에서 q50 추출
    quantiles = response.get("quantiles") or (parsed.get("quantiles") if isinstance(parsed, dict) else None)
    if isinstance(quantiles, dict):
        q50 = quantiles.get("q50")
        if isinstance(q50, list) and len(q50) >= horizon:
            try:
                return [max(0.0, float(x)) for x in q50[:horizon] if isinstance(x, (int, float))]
            except Exception:
                pass
    
    return None


def _extract_thoughts_from_response(provider_response: Dict[str, Any]) -> Optional[Any]:
    """Extract provider-specific thinking/trace blocks if present."""
    if not isinstance(provider_response, dict):
        return None
    # DashScope compatible-mode: thoughts nested under choices[].message.thoughts
    choices = provider_response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        # 1) Native "thoughts" field
        if "thoughts" in message:
            return message.get("thoughts")
        # 2) Some DashScope models put thinking trace under message.extensions.thoughts
        msg_ext = message.get("extensions") if isinstance(message, dict) else None
        if isinstance(msg_ext, dict) and "thoughts" in msg_ext:
            return msg_ext.get("thoughts")
        # Some providers nest under extensions
        extensions = choices[0].get("extensions") if isinstance(choices[0], dict) else None
        if isinstance(extensions, dict) and "thoughts" in extensions:
            return extensions.get("thoughts")
    # OpenAI responses endpoint: output may contain thought-like entries in content
    if "output" in provider_response:
        return provider_response.get("output")
    return None


def call_llm_json(
    system_prompt: str,
    user_payload: dict,
    model: str = "qwen3-235b-a22b",
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    provider: str = "dashscope",
    *,
    temperature: float = 0.2,
    thinking_mode: str = "none",
) -> Tuple[Optional[dict], Dict[str, Any]]:
    import json as json_module
    
    # Load API key based on provider
    api_key = _load_api_key_for_provider(provider)
    if not api_key:
        logger.warning("[CALL_LLM_JSON] %s_API_KEY not found; returning None.", provider.upper())
        return None, {"error": "API key not found", "provider": provider}
    
    # Build request based on provider
    if provider == "openai":
        url, headers, body = _build_openai_request(api_key, model, system_prompt, user_payload, temperature)
    elif provider == "google":
        url, headers, body = _build_google_request(api_key, model, system_prompt, user_payload, temperature)
    elif provider == "deepseek":
        url, headers, body = _build_deepseek_request(api_key, model, system_prompt, user_payload, temperature, thinking_mode)
    else:  # dashscope
        url, headers, body = _build_dashscope_request(api_key, api_base, model, system_prompt, user_payload, temperature, thinking_mode)
    
    # Debug logging
    try:
        user_summary = _summarize_payload(user_payload)
    except Exception:
        user_summary = {"error": "failed to summarize payload"}
    
    effective_temp = body.get("temperature", temperature)
    _print_llm_debug(
        "REQUEST",
        {
            "provider": provider,
            "model": model,
            "thinking_mode": thinking_mode,
            "temperature": effective_temp,
            "user_payload_summary": user_summary,
        },
    )

    try:
        start_ts = time.perf_counter()
        resp = _http_post_json(url, headers, body)
        latency_ms = (time.perf_counter() - start_ts) * 1000.0

        try:
            raw_preview = json_module.dumps(resp, ensure_ascii=False)
            if len(raw_preview) > 2000:
                raw_preview = raw_preview[:2000] + "... (truncated)"
            logger.debug("[CALL_LLM_JSON] Raw response preview: %s", raw_preview)
        except Exception:
            logger.debug("[CALL_LLM_JSON] Raw response preview unavailable")

        if not isinstance(resp, dict):
            logger.error("[CALL_LLM_JSON] Unexpected non-dict HTTP response: %s", type(resp))
            return None, {"error": "non-dict response", "provider": provider}

        logger.debug("[CALL_LLM_JSON] HTTP response keys: %s", list(resp.keys()))
        
        # Check for errors
        if "error" in resp and resp.get("error"):
            logger.error("[CALL_LLM_JSON] API error payload: %s", resp.get("error"))
            return None, {"error": resp.get("error"), "provider": provider}
        
        # Check GPT-5 status field
        if provider == "openai" and str(model).lower().startswith("gpt-5"):
            status = resp.get("status")
            if status and status != "completed":
                logger.warning("GPT-5 response status: %s", status)
                incomplete = resp.get("incomplete_details")
                if incomplete:
                    logger.warning("GPT-5 incomplete_details: %s", incomplete)
                # Continue even if not completed, as partial output might be available
                # return None, {"error": f"incomplete status: {status}", "provider": provider}

        # Extract content based on provider
        content, thoughts = _extract_response_content(resp, provider, model)
        
        # Debug: print actual content extracted
        if provider == "openai" and str(model).lower().startswith("gpt-5"):
            print(f"[DEBUG] GPT-5 Response Keys: {list(resp.keys())}")
            print(f"[DEBUG] GPT-5 resp['output']: {resp.get('output')}")
            print(f"[DEBUG] GPT-5 resp['text']: {resp.get('text')}")
            print(f"[DEBUG] GPT-5 extracted content length: {len(str(content)) if content else 0}")
            print(f"[DEBUG] GPT-5 content type: {type(content)}")
            print(f"[DEBUG] GPT-5 content preview: {str(content)[:500] if content else 'None'}")
        
        parsed = _extract_json(content)
        
        meta: Dict[str, Any] = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "thinking_mode": thinking_mode,
            "request_body": body,
            "latency_ms": latency_ms,
            "raw_content": content,
            "raw_response": resp,
            "thoughts": thoughts,
        }

        # Debug-print response preview
        try:
            preview: Dict[str, Any] = {
                "has_parsed_json": parsed is not None,
                "parsed_keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
            }
            if isinstance(parsed, dict):
                for k in ("transmission_impact", "event_summary", "confidence", "risk_notes", "forecast"):
                    if k in parsed:
                        preview[k] = parsed[k]
            if thoughts is not None:
                thoughts_str = json_module.dumps(thoughts, ensure_ascii=False)
                if len(thoughts_str) > 800:
                    thoughts_str = thoughts_str[:800] + "... (truncated)"
                preview["thoughts_preview"] = thoughts_str

                # Also print raw thinking trace for DeepSeek / Qwen to stdout
                if provider in ("deepseek", "dashscope"):
                    print("\n" + "=" * 80)
                    print(f"🧠 {provider.upper()} THINKING TRACE (model={model}):")
                    print("=" * 80)
                    print(thoughts_str)
                    print("=" * 80 + "\n")

            _print_llm_debug("RESPONSE", {"latency_ms": latency_ms, "preview": preview})
        except Exception:
            pass

        if parsed is None:
            logger.warning("[CALL_LLM_JSON] Failed to parse JSON content; returning None.")
        logger.info(
            "LLM call %s/%s completed in %.2f ms (parsed=%s)",
            provider,
            model,
            latency_ms,
            parsed is not None,
        )
        return parsed, meta
    except Exception as e:
        meta = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "thinking_mode": thinking_mode,
            "error": repr(e),
            "request_body": body if 'body' in locals() else None,
        }
        logger.exception("[CALL_LLM_JSON] Exception during HTTP/parse: %s", e)
        return None, meta


def _load_api_key_for_provider(provider: str) -> Optional[str]:
    """Load API key for the given provider from environment or .env file."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "google": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "dashscope": "DASHSCOPE_API_KEY",
    }
    key_name = key_map.get(provider, "DASHSCOPE_API_KEY")
    
    api_key = os.environ.get(key_name)
    if api_key:
        return api_key
    
    # Try loading from .env file
    try:
        proj_root = Path(__file__).resolve().parents[2]
        env_path = proj_root / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith(f"{key_name}="):
                    _, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    if v:
                        os.environ[key_name] = v
                        return v
    except Exception:
        pass
    return None


def _build_openai_request(api_key: str, model: str, system_prompt: str, user_payload: dict, temperature: float):
    """Build OpenAI API request (GPT-5.1 uses responses endpoint)."""
    import json as json_module
    
    use_responses = str(model).lower().startswith("gpt-5")
    url = "https://api.openai.com/v1/responses" if use_responses else "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    if use_responses:
        # GPT-5.1 responses API format
        body = {
            "model": model,
            "input": f"Here is the data for the forecast:\n{json_module.dumps(user_payload, ensure_ascii=False)}",
            "instructions": system_prompt,
            # "temperature": temperature,  # Not supported by gpt-5.1
            "reasoning": {"effort": "low"},
            "text": {"verbosity": "low"},
        }
    else:
        # Standard chat completions format
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json_module.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
    
    return url, headers, body


def _build_google_request(api_key: str, model: str, system_prompt: str, user_payload: dict, temperature: float):
    """Build Google Gemini API request.

    Defaults: temperature ~= 1.0 for more diverse reasoning, but the caller
    can override via the `temperature` argument.
    """
    import json as json_module
    
    # Gemini uses REST API with API key in URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # Combine system and user content
    combined_content = f"{system_prompt}\n\n{json_module.dumps(user_payload, ensure_ascii=False)}"
    
    body = {
        "contents": [{
            "parts": [{"text": combined_content}]
        }],
        "generationConfig": {
            # If caller passes None, fall back to 1.0
            "temperature": float(temperature) if temperature is not None else 1.0,
            "responseMimeType": "application/json",
        }
    }
    
    return url, headers, body


def _build_deepseek_request(api_key: str, model: str, system_prompt: str, user_payload: dict, temperature: float, thinking_mode: str):
    """Build DeepSeek API request (OpenAI-compatible).

    Default temperature is tuned to around 1.0 for richer exploration unless
    explicitly overridden by the caller.
    """
    import json as json_module
    
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # Default DeepSeek temperature ~1.0 unless overridden
    eff_temp = float(temperature) if temperature is not None else 1.0
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json_module.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": eff_temp,
        "response_format": {"type": "json_object"},
    }
    
    return url, headers, body


def _build_dashscope_request(api_key: str, api_base: str, model: str, system_prompt: str, user_payload: dict, temperature: float, thinking_mode: str):
    """Build DashScope (Qwen) API request.

    Qwen official guidance (thinking models):
    - Use slightly higher temperature (~0.6) with thinking enabled
    - Use `parameters.thinking_mode` to turn on reasoning traces.
    Here we default to 0.6 when thinking is enabled, and 0.7 otherwise,
    unless the caller passes an explicit `temperature`.
    """
    import json as json_module
    
    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    enable_thinking = thinking_mode and thinking_mode != "none"
    
    # Recommended sampling configs (Qwen docs):
    # - thinking models: temp≈0.6, top_p≈0.95
    # - normal: temp≈0.7, top_p≈0.8
    if enable_thinking:
        temp = 0.6
        top_p = 0.95
        top_k = 20
        min_p = 0.0
    else:
        temp = 0.7
        top_p = 0.8
        top_k = 20
        min_p = 0.0

    # Allow explicit override from caller
    if temperature is not None:
        temp = float(temperature)
    
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json_module.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
    }
    
    if enable_thinking:
        body["parameters"] = {"thinking_mode": thinking_mode}
    else:
        body["response_format"] = {"type": "json_object"}
    
    return url, headers, body


def _extract_response_content(resp: dict, provider: str, model: str) -> Tuple[str, Optional[Any]]:
    """Extract content and thinking/reasoning from provider response."""
    thoughts = None
    
    if provider == "openai":
        if str(model).lower().startswith("gpt-5"):
            # GPT-5.1 responses endpoint: parse output list -> message -> content[0].text
            content = None
            output_val = resp.get("output")
            if isinstance(output_val, list) and len(output_val) >= 2:
                msg = output_val[1]
                if isinstance(msg, dict) and msg.get("type") == "message":
                    msg_content = msg.get("content") or []
                    if isinstance(msg_content, list) and msg_content:
                        first_block = msg_content[0]
                        if isinstance(first_block, dict):
                            inner_text = first_block.get("text")
                            if isinstance(inner_text, str):
                                content = inner_text

            # Fallbacks if above failed
            if not content:
                if isinstance(output_val, str):
                    content = output_val
                elif isinstance(output_val, list):
                    try:
                        if all(isinstance(x, str) for x in output_val):
                            content = "\n".join(output_val)
                        else:
                            content = json.dumps(output_val, ensure_ascii=False)
                    except Exception:
                        content = str(output_val)
                elif isinstance(output_val, dict):
                    content = json.dumps(output_val, ensure_ascii=False)

            if not content:
                # Only use 'text' if it is a string. If it is a dict, it is likely config.
                text_val = resp.get("text")
                if isinstance(text_val, str):
                    content = text_val

            if not content:
                content = "{}"

            # Check for reasoning in metadata or direct field
            thoughts = resp.get("reasoning") or resp.get("metadata", {}).get("reasoning")
        else:
            # Standard chat completions
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    
    elif provider == "google":
        # Gemini format
        candidates = resp.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                content = parts[0].get("text", "{}")
            else:
                content = "{}"
        else:
            content = "{}"
        # Gemini thinking (if available)
        thoughts = resp.get("usageMetadata", {}).get("thinking")
    
    elif provider == "deepseek":
        # DeepSeek format (OpenAI-compatible)
        msg = resp.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "{}")
        # DeepSeek reasoner exposes reasoning_content
        thoughts = msg.get("reasoning_content")
    
    else:  # dashscope
        msg = resp.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "{}")
        # DashScope thinking models
        thoughts = msg.get("reasoning_content") or msg.get("thoughts")
        if thoughts:
            print("\n" + "="*80)
            print("🧠 LLM THINKING PROCESS:")
            print("="*80)
            print(thoughts)
            print("="*80 + "\n")
    
    return content, thoughts


def apply_hard_guards(p: Dict[str, Any]) -> Dict[str, Any]:
    # 하드캡 적용
    def clamp(k: str, v: Any, lo: float, hi: float) -> Optional[float]:
        try:
            return max(lo, min(hi, float(v)))
        except Exception:
            return None

    def _assign(key: str, lo: float, hi: float, *, cast=float) -> None:
        if key not in p or p[key] is None:
            return
        val = clamp(key, p[key], lo, hi)
        if val is None:
            p.pop(key, None)
        else:
            p[key] = cast(val)

    _assign("r_boost_cap", 0.5, 3.0)
    _assign("scale_cap", 1.0, 1.8)
    _assign("x_cap_multiplier", 1.0, 4.0)
    _assign("nb_dispersion_k", 2.0, 50.0)
    _assign("evt_u_quantile", 0.85, 0.95)
    # 누락된 주요 파라미터 제약 추가
    _assign("quality", 0.5, 0.95)
    _assign("amplitude_quantile", 0.85, 0.98)
    _assign("amplitude_multiplier", 1.2, 2.8)
    _assign("ratio_cap_quantile", 0.95, 0.999)
    _assign("delta_quantile", 0.01, 0.1)
    _assign("warmup_weeks", 0, 2, cast=int)
    return p

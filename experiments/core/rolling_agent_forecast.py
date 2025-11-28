"""Rolling LLM forecasting pipeline with logging and JSON artifacts."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .llm_agent import is_hfmd
from .simple_event_interpreter import SimpleEventInterpreter
from .simple_forecast_generator import SimpleForecastGenerator

LOGGER_NAME = "experiments.core.rolling_agent_forecast"
logger = logging.getLogger(LOGGER_NAME)
DEFAULT_WEEKLY_CSV = Path(__file__).resolve().parents[2] / "processed_data" / "his_outpatient_weekly_epi_counts.csv"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def dt(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def pick_step_dates(dates: Sequence[str], start_date: str, end_date: str, n_steps: int) -> List[str]:
    in_range = [d for d in dates if start_date <= d <= end_date]
    if not in_range:
        return []
    # n_steps <= 0 이면 기간 내 모든 주를 사용
    if n_steps <= 0 or n_steps >= len(in_range):
        return in_range
    idxs = np.linspace(0, len(in_range) - 1, num=n_steps, dtype=int)
    return [in_range[i] for i in idxs]


def load_series(csv_path: str, disease: str) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    if "diagnosis_time" in df.columns:
        dates = pd.to_datetime(df["diagnosis_time"]).dt.strftime("%Y-%m-%d").tolist()
    elif "week_end_date" in df.columns:
        dates = pd.to_datetime(df["week_end_date"]).dt.strftime("%Y-%m-%d").tolist()
    elif "INSPECTION_DATE" in df.columns:
        dates = pd.to_datetime(df["INSPECTION_DATE"]).dt.strftime("%Y-%m-%d").tolist()
    else:
        raise ValueError("CSV must contain diagnosis_time, week_end_date, or INSPECTION_DATE column")

    if disease in df.columns:
        series = df[disease].astype(float).values
    elif "admissions_total" in df.columns:
        series = df["admissions_total"].astype(float).values
    else:
        raise KeyError(f"Column for disease '{disease}' not found in CSV")
    return dates, series


def summarize_payload(payload: Dict[str, Any], max_items: int = 5) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            if len(value) <= max_items:
                summary[key] = value
            else:
                summary[key] = value[:max_items] + [f"... (+{len(value) - max_items} more)"]
        elif isinstance(value, dict):
            summary[key] = {
                "type": value.get("type", "dict"),
                "keys": list(value.keys())[:max_items],
            }
        else:
            summary[key] = value
    return summary


def ensure_logging(log_path: Path) -> None:
    logger_root = logging.getLogger()
    logger_root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    for handler in list(logger_root.handlers):
        logger_root.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger_root.addHandler(file_handler)
    logger_root.addHandler(stream_handler)


def build_run_id(args: argparse.Namespace, timestamp: str) -> str:
    base = f"{args.batch}_{args.model}_{args.start}_{args.end}_{timestamp}"
    if args.run_name:
        return f"{base}_{args.run_name}"
    return base


def compute_metrics_so_far(actuals: List[float], preds: List[float]) -> Dict[str, Optional[float]]:
    if not actuals or not preds:
        return {"mae_so_far": None, "rmse_so_far": None}
    k = min(len(actuals), len(preds))
    arr_a = np.array(actuals[:k], dtype=float)
    arr_p = np.array(preds[:k], dtype=float)
    mae = float(np.mean(np.abs(arr_a - arr_p)))
    rmse = float(np.sqrt(np.mean((arr_a - arr_p) ** 2)))
    return {"mae_so_far": mae, "rmse_so_far": rmse}


def _load_weather_for_hfmd(is_hongkong: bool = False) -> pd.DataFrame:
    """Load weather data for HFMD forecasting.
    
    Args:
        is_hongkong: If True, load Hong Kong weather data instead of Hangzhou.
    """
    if is_hongkong:
        # Hong Kong weather data
        base = Path(__file__).resolve().parents[2] / "experiments" / "data_for_model" / "手足口病" / "data_HK" / "weather_2010_2025.csv"
    else:
        # Hangzhou (default) weather data
        base = Path(__file__).resolve().parents[2] / "experiments" / "data_for_model" / "手足口病" / "weather_2019_2024.csv"
    
    if not base.exists():
        return pd.DataFrame()
    df = pd.read_csv(base)
    if "date" not in df.columns:
        # 유저 데이터 포맷이 확정되지 않았을 때를 위한 가드
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def _load_monthly_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "report_period" not in df.columns or "hfmd_cases" not in df.columns:
        return pd.DataFrame()
    df["report_period"] = pd.to_datetime(df["report_period"], format="%Y-%m")
    return df.sort_values("report_period")


def _load_events_calendar() -> List[Dict[str, Any]]:
    """Load events calendar from JSON file.
    
    Paper Section 3.1.1:
    Each epidemiological week is labeled with school status (in_session, summer_break, winter_break)
    and public holidays (Spring Festival, National Day, etc.)
    """
    import json
    path = Path(__file__).resolve().parents[2] / "experiments" / "data_for_model" / "手足口病" / "events_calendar.json"
    if not path.exists():
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


_HFMD_WEATHER_CACHE: Optional[pd.DataFrame] = None
_HFMD_WEATHER_HK_CACHE: Optional[pd.DataFrame] = None  # Hong Kong weather cache
_HFMD_STATS_ZJ_CACHE: Optional[pd.DataFrame] = None
_HFMD_STATS_NAT_CACHE: Optional[pd.DataFrame] = None
_HFMD_EVENTS_CACHE: Optional[List[Dict[str, Any]]] = None


def _get_school_status_and_holidays(target_date: datetime, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Determine school status and active holidays for a given date.
    
    Returns:
        {
            "school_status": "in_session" | "summer_break" | "winter_break",
            "holidays": [list of active holiday names],
            "holiday_details": [list of holiday dicts]
        }
    """
    result: Dict[str, Any] = {
        "school_status": "in_session",  # default
        "holidays": [],
        "holiday_details": [],
    }
    
    for event in events:
        event_type = event.get("type", "")
        event_name = event.get("event_name", "")
        
        # Parse dates
        date_start_str = event.get("date_start") or event.get("date")
        date_end_str = event.get("date_end") or date_start_str
        
        if not date_start_str:
            continue
            
        try:
            date_start = datetime.strptime(date_start_str, "%Y-%m-%d")
            date_end = datetime.strptime(date_end_str, "%Y-%m-%d") if date_end_str else date_start
        except ValueError:
            continue
        
        # Check if target_date is within this event's range
        if date_start <= target_date <= date_end:
            if event_type == "School Vacation":
                if "Summer" in event_name:
                    result["school_status"] = "summer_break"
                elif "Winter" in event_name:
                    result["school_status"] = "winter_break"
                else:
                    result["school_status"] = "vacation"
            elif event_type == "School Season":
                result["school_status"] = "in_session"
            elif event_type in ("Public Holiday", "Other Holiday"):
                result["holidays"].append(event_name)
                result["holiday_details"].append(event)
    
    return result


def build_hfmd_external_data(train_until: str, csv_path: str = "") -> Dict[str, Any]:
    """Build external_data payload for HFMD: weather + monthly stats + school calendar.

    Paper Section 3.1.1:
    - weather_2019_2024.csv (Hangzhou) or weather_2010_2025.csv (Hong Kong): 최근 7일 daily + 요약
    - hfmd_statistics_zhejiang/national.csv: 해당 월/직전 월 요약
    - events_calendar.json: 학교 일정 + 공휴일 (Winter/Summer vacation, Spring Festival, etc.)
    
    Args:
        train_until: The date string up to which we have training data.
        csv_path: The path to the CSV file, used to detect Hong Kong vs Hangzhou data.
    """
    global _HFMD_WEATHER_CACHE, _HFMD_WEATHER_HK_CACHE, _HFMD_STATS_ZJ_CACHE, _HFMD_STATS_NAT_CACHE, _HFMD_EVENTS_CACHE

    target_date = dt(train_until)
    
    # Detect if this is Hong Kong data
    csv_path_str = str(csv_path).lower()
    is_hongkong = "hk" in csv_path_str or "hongkong" in csv_path_str or "hong_kong" in csv_path_str

    # Load appropriate weather cache
    if is_hongkong:
        if _HFMD_WEATHER_HK_CACHE is None:
            _HFMD_WEATHER_HK_CACHE = _load_weather_for_hfmd(is_hongkong=True)
        weather_cache = _HFMD_WEATHER_HK_CACHE
    else:
        if _HFMD_WEATHER_CACHE is None:
            _HFMD_WEATHER_CACHE = _load_weather_for_hfmd(is_hongkong=False)
        weather_cache = _HFMD_WEATHER_CACHE

    if _HFMD_STATS_ZJ_CACHE is None:
        path_zj = Path(__file__).resolve().parents[2] / "experiments" / "data_for_model" / "手足口病" / "hfmd_statistics_zhejiang.csv"
        _HFMD_STATS_ZJ_CACHE = _load_monthly_stats(path_zj)
    if _HFMD_STATS_NAT_CACHE is None:
        path_nat = Path(__file__).resolve().parents[2] / "experiments" / "data_for_model" / "手足口病" / "hfmd_statistics_national.csv"
        _HFMD_STATS_NAT_CACHE = _load_monthly_stats(path_nat)
    if _HFMD_EVENTS_CACHE is None:
        _HFMD_EVENTS_CACHE = _load_events_calendar()

    external: Dict[str, Any] = {}

    # 최근 7일 날씨
    if weather_cache is not None and not weather_cache.empty:
        start_date = target_date - timedelta(days=7)
        mask = (weather_cache["date"] >= start_date) & (weather_cache["date"] < target_date)
        recent = weather_cache.loc[mask].copy()
        if not recent.empty:
            daily_records = []
            keep_cols = [c for c in recent.columns if c != "date"]
            for _, row in recent.iterrows():
                rec = {"date": row["date"].strftime("%Y-%m-%d")}
                for c in keep_cols:
                    val = row[c]
                    if pd.isna(val):
                        continue
                    # 숫자는 float로, 그 외는 문자열로 보존
                    if isinstance(val, (int, float, np.number)):
                        rec[c] = float(val)
                    else:
                        rec[c] = str(val)
                daily_records.append(rec)

            external["weather_daily_last_7d"] = daily_records

            # 간단 요약
            summary: Dict[str, Any] = {}
            for c in keep_cols:
                col = recent[c]
                numeric = pd.to_numeric(col, errors="coerce")
                numeric = numeric.dropna()
                if numeric.empty:
                    continue
                summary[f"{c}_mean"] = float(numeric.mean())
                summary[f"{c}_min"] = float(numeric.min())
                summary[f"{c}_max"] = float(numeric.max())
            external["weather_summary_last_7d"] = summary

    # 월간 HFMD 통계 (지역)
    # Month key for train_until; we only want to use statistics from months
    # strictly BEFORE this month to avoid peeking into the future.
    month_key = pd.Period(target_date, freq="M").to_timestamp()
    def _monthly_block(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        if df is None or df.empty:
            return {}
        # 해당 월과 직전 월
        current = df[df["report_period"] == month_key]
        prev = df[df["report_period"] == (month_key - pd.offsets.MonthBegin(1))]
        block: Dict[str, Any] = {}
        if not current.empty:
            row = current.iloc[0]
            block["month"] = row["report_period"].strftime("%Y-%m")
            block["hfmd_cases"] = int(row["hfmd_cases"]) if not pd.isna(row["hfmd_cases"]) else None
        if not prev.empty() if callable(getattr(prev, "empty", None)) else prev.empty is False:
            # 방어적 코딩: prev가 DataFrame인 경우만 처리
            pass
        return block

    if _HFMD_STATS_ZJ_CACHE is not None and not _HFMD_STATS_ZJ_CACHE.empty:
        # Only use information from months strictly before the current month
        cur = _HFMD_STATS_ZJ_CACHE[_HFMD_STATS_ZJ_CACHE["report_period"] == (month_key - pd.offsets.MonthBegin(1))]
        prev = _HFMD_STATS_ZJ_CACHE[_HFMD_STATS_ZJ_CACHE["report_period"] == (month_key - pd.offsets.MonthBegin(2))]
        block_zj: Dict[str, Any] = {}
        if not cur.empty:
            row = cur.iloc[0]
            block_zj["month"] = row["report_period"].strftime("%Y-%m")
            block_zj["hfmd_cases"] = int(row["hfmd_cases"]) if not pd.isna(row["hfmd_cases"]) else None
        if not prev.empty:
            row_p = prev.iloc[0]
            cur_cases = block_zj.get("hfmd_cases")
            prev_cases = int(row_p["hfmd_cases"]) if not pd.isna(row_p["hfmd_cases"]) else None
            if cur_cases is not None and prev_cases not in (None, 0):
                change = (cur_cases - prev_cases) / prev_cases
                block_zj["change_vs_prev"] = float(change)
        if block_zj:
            external["hfmd_monthly_stats_local"] = block_zj

    if _HFMD_STATS_NAT_CACHE is not None and not _HFMD_STATS_NAT_CACHE.empty:
        # Same logic for national stats: only months before the current month
        cur = _HFMD_STATS_NAT_CACHE[_HFMD_STATS_NAT_CACHE["report_period"] == (month_key - pd.offsets.MonthBegin(1))]
        prev = _HFMD_STATS_NAT_CACHE[_HFMD_STATS_NAT_CACHE["report_period"] == (month_key - pd.offsets.MonthBegin(2))]
        block_nat: Dict[str, Any] = {}
        if not cur.empty:
            row = cur.iloc[0]
            block_nat["month"] = row["report_period"].strftime("%Y-%m")
            block_nat["hfmd_cases"] = int(row["hfmd_cases"]) if not pd.isna(row["hfmd_cases"]) else None
        if not prev.empty:
            row_p = prev.iloc[0]
            cur_cases = block_nat.get("hfmd_cases")
            prev_cases = int(row_p["hfmd_cases"]) if not pd.isna(row_p["hfmd_cases"]) else None
            if cur_cases is not None and prev_cases not in (None, 0):
                change = (cur_cases - prev_cases) / prev_cases
                block_nat["change_vs_prev"] = float(change)
        if block_nat:
            external["hfmd_monthly_stats_national"] = block_nat

    # School calendar from events_calendar.json (Paper Section 3.1.1)
    # Replaces hardcoded 7-8월 summer_break logic with proper calendar lookup
    if _HFMD_EVENTS_CACHE:
        calendar_info = _get_school_status_and_holidays(target_date, _HFMD_EVENTS_CACHE)
        external["school_calendar"] = {
            "school_status": calendar_info["school_status"],
            "holidays": calendar_info["holidays"],
        }
        if calendar_info["holiday_details"]:
            external["school_calendar"]["holiday_details"] = calendar_info["holiday_details"]
    else:
        # Fallback: simple rule-based (for dates outside calendar coverage)
        month = target_date.month
        if month in (7, 8):
            school_status = "summer_break"
        elif month in (1, 2):
            school_status = "winter_break"
        else:
            school_status = "in_session"
        external["school_calendar"] = {"school_status": school_status, "holidays": []}

    return external


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_forecast_pipeline(
    args: argparse.Namespace,
    run_id: str,
    *,
    json_path: Path,
) -> Dict[str, Any]:
    csv_path = args.csv_path.strip() or str(DEFAULT_WEEKLY_CSV)
    disease = args.disease
    dates, series = load_series(csv_path, disease)
    logger.info("Loaded %d weeks from %s", len(dates), csv_path)

    step_dates = pick_step_dates(dates, args.start, args.end, args.n_steps)
    if not step_dates:
        logger.warning("No step dates found between %s and %s", args.start, args.end)

    use_llm = not args.no_llm
    # Enable thinking mode for Qwen; others use default
    default_thinking = "light" if args.provider == "dashscope" else "none"
    event_interpreter = SimpleEventInterpreter(
        model=args.model,
        provider=args.provider,
        temperature=args.temperature,
        thinking_mode=default_thinking,
    )
    forecast_generator = SimpleForecastGenerator(
        model=args.model,
        provider=args.provider,
        temperature=args.temperature,
        thinking_mode=default_thinking,
    )

    impact_lag_weeks = 1 if is_hfmd(disease) else 0
    logger.info(
        "Pipeline configuration: model=%s/%s, mode=%s, horizon=%d, impact_lag=%d, use_llm=%s",
        args.provider,
        args.model,
        args.forecast_mode,
        args.horizon,
        impact_lag_weeks,
        use_llm,
    )

    steps_output: List[Dict[str, Any]] = []
    all_actuals: List[float] = []
    all_preds: List[float] = []

    for idx, train_date in enumerate(step_dates):
        logger.info("Step %d/%d - train_until=%s", idx + 1, len(step_dates), train_date)
        try:
            train_idx = dates.index(train_date)
        except ValueError:
            logger.error("Train date %s not found in dataset", train_date)
            continue

        history_values = series[: train_idx + 1]
        history_dates = dates[: train_idx + 1]
        if history_values.size == 0:
            logger.warning("No history available for %s; skipping", train_date)
            continue

        horizon = min(args.horizon, max(0, len(series) - train_idx - 1))
        if horizon <= 0:
            logger.info("No forward horizon available for %s; skipping", train_date)
            continue

        history_window = min(len(history_values), 156)
        hist_values = history_values[-history_window:]
        hist_dates = history_dates[-history_window:]
        full_history = {
            "type": "weekly_series",
            "dates": hist_dates,
            "values": hist_values.tolist(),
            "length_weeks": len(hist_values),
            "date_range": {
                "start": hist_dates[0],
                "end": hist_dates[-1],
            },
        }
        recent_values = hist_values[-8:].tolist()
        history_range = {"start": hist_dates[0], "end": hist_dates[-1]}

        if is_hfmd(disease):
            external_data: Dict[str, Any] = build_hfmd_external_data(train_date, csv_path=str(csv_path))
        else:
            external_data = {}

        # advanced 모드에서는 LLM 실패 시 fallback을 사용하지 않도록 강제
        allow_fallback = not (args.forecast_mode == "advanced" and use_llm)
        
        # skip_model1: Event Interpreter(Model1) 없이 Model2만 LLM으로 실행
        skip_model1 = getattr(args, 'skip_model1', False)
        
        if skip_model1:
            # Model1 skip - neutral values 사용
            from experiments.core.simple_event_interpreter import EventInterpretation
            event_result = EventInterpretation(
                transmission_impact=0.0,
                confidence=0.5,
                event_summary="(Model1 skipped - baseline mode)",
                risk_notes=[],
                prompt_payload={},
                raw_response=None,
                used_fallback=False,
            )
            logger.info("  Model1 (EventInterpreter) skipped, using neutral values")
        else:
            event_result = event_interpreter.interpret(
                disease=disease,
                train_until=train_date,
                recent_values=recent_values,
                external_data=external_data,
                horizon=horizon,
                full_history=full_history,
                impact_lag_weeks=impact_lag_weeks,
                use_llm=use_llm,
                allow_fallback=allow_fallback,
            )

        forecast_result = forecast_generator.generate_forecast(
            recent_values=recent_values,
            transmission_impact=event_result.transmission_impact,
            confidence=event_result.confidence,
            horizon=horizon,
            full_history=full_history,
            risk_notes=event_result.risk_notes,
            mode=args.forecast_mode,
            impact_lag_weeks=impact_lag_weeks,
            use_llm=use_llm,
            allow_fallback=allow_fallback,
        )

        future_slice = series[train_idx + 1 : train_idx + 1 + horizon]
        future_dates = dates[train_idx + 1 : train_idx + 1 + horizon]
        actual_values = future_slice.tolist()

        for actual, pred in zip(actual_values, forecast_result.q50):
            all_actuals.append(actual)
            all_preds.append(pred)

        metrics = compute_metrics_so_far(all_actuals, all_preds)

        steps_output.append(
            {
                "step_index": idx,
                "current_target_date": train_date,
                "history_range": history_range,
                "event_interpreter": {
                    "prompt_summary": summarize_payload(event_result.prompt_payload),
                    "response_raw": event_result.raw_response or ("fallback" if event_result.used_fallback else None),
                    "parsed_events": event_result.to_dict(),
                },
                "predictor": {
                    "prompt_summary": summarize_payload(forecast_result.prompt_payload),
                    "response_raw": forecast_result.raw_response or ("fallback" if forecast_result.used_fallback else None),
                    "parsed_forecast": forecast_result.to_dict(),
                    "predicted_value": forecast_result.q50,
                },
                "metrics": metrics,
                "actual_values": {
                    "dates": future_dates,
                    "values": actual_values,
                },
            }
        )

        logger.info(
            "Step %d summary: impact=%.2f, first_pred=%.2f, first_actual=%s",
            idx + 1,
            event_result.transmission_impact,
            forecast_result.q50[0] if forecast_result.q50 else float("nan"),
            actual_values[0] if actual_values else None,
        )

    meta = {
        "run_id": run_id,
        "disease": disease,
        "dataset": Path(csv_path).name,
        "start_date": args.start,
        "end_date": args.end,
        "horizon": args.horizon,
        "n_steps": len(steps_output),
        "requested_n_steps": args.n_steps,
        "model": args.model,
        "provider": args.provider,
        "forecast_mode": args.forecast_mode,
        "batch": args.batch,
        "temperature": args.temperature,
        "impact_lag_weeks": impact_lag_weeks,
        "no_llm": args.no_llm,
        "skip_model1": getattr(args, 'skip_model1', False),
        "csv_path": csv_path,
    }

    result = {
        "run_id": run_id,
        "meta": meta,
        "steps": steps_output,
    }

    if args.save_json:
        json_text = json.dumps(result, ensure_ascii=False, indent=2)
        json_path.write_text(json_text, encoding="utf-8")
        logger.info("Structured JSON saved to %s", json_path)

        # For the advanced Hangzhou DeepSeek experiments, also store a copy
        # under experiments/results/advanced_hangzhou_deepseek for easier access.
        if args.batch == "advanced_hangzhou_deepseek":
            alt_dir = Path(__file__).resolve().parents[1] / "results" / "advanced_hangzhou_deepseek"
            alt_dir.mkdir(parents=True, exist_ok=True)
            alt_path = alt_dir / f"{run_id}.json"
            alt_path.write_text(json_text, encoding="utf-8")
            logger.info("Mirrored JSON saved to %s", alt_path)

        # Optional postprocess: create plots, summary, predictions.csv
        try:
            from experiments.postprocess_results_json import run_postprocess

            logger.info("Running postprocess_results_json on %s", json_path)
            run_postprocess(str(json_path))
        except Exception as e:  # pragma: no cover - best-effort helper
            logger.error("Postprocessing failed for %s: %s", json_path, e)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lag-aware rolling LLM forecasting pipeline")
    parser.add_argument("--disease", required=True)
    parser.add_argument("--start", required=True, help="train_until start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="train_until end date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=4)
    # n_steps<=0 이면 [start, end] 구간의 모든 주를 step으로 사용
    parser.add_argument("--n_steps", type=int, default=0)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--forecast_mode", default="standard", choices=["standard", "advanced"])
    parser.add_argument("--batch", required=True, help="Experiment batch name")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--csv_path", default="", help="Optional CSV override")
    parser.add_argument("--no_llm", action="store_true", help="Disable all LLM calls")
    parser.add_argument("--skip_model1", action="store_true", help="Skip Event Interpreter (Model1), run only Forecast Generator (Model2) with LLM")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--json_dir", default="results_json")
    parser.add_argument("--run_name", default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    run_id = build_run_id(args, timestamp)

    log_dir = Path(args.log_dir) / args.batch
    json_dir = Path(args.json_dir) / args.batch
    log_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{run_id}.log"
    json_path = json_dir / f"{run_id}.json"

    ensure_logging(log_path)
    logger.info("Starting run %s", run_id)
    logger.info("CLI args: %s", vars(args))

    result = run_forecast_pipeline(args, run_id, json_path=json_path)
    logger.info("Completed run %s with %d steps", run_id, len(result.get("steps", [])))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

@lru_cache(maxsize=1)
def load_disease_metadata() -> List[Dict[str, Any]]:
    """질병 메타데이터를 로드하고 캐시합니다."""
    metadata_path = Path(__file__).parent / "disease_metadata.json"
    if not metadata_path.exists():
        return []
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

@lru_cache(maxsize=1)
def load_calendar_events() -> List[Dict[str, Any]]:
    """캘린더 이벤트를 로드하고 캐시합니다."""
    calendar_path = Path(__file__).parent / "events_calendar.json"
    if not calendar_path.exists():
        return []
    with open(calendar_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_calendar_signals(start_date: str, weeks: int) -> Dict[str, List[int]]:
    """주어진 기간 동안의 주별 공휴일 및 학기 시즌 플래그를 생성합니다."""
    events = load_calendar_events()
    start_dt = datetime.fromisoformat(start_date)
    
    is_holiday = [0] * weeks
    is_school_season = [0] * weeks

    for i in range(weeks):
        week_start = start_dt + timedelta(days=i * 7)
        week_end = week_start + timedelta(days=6)
        
        for event in events:
            event_type = event.get("type")
            if "date" in event:
                event_dt = datetime.fromisoformat(event["date"])
                if week_start <= event_dt <= week_end:
                    if event_type == "Public Holiday":
                        is_holiday[i] = 1
            elif "date_start" in event and "date_end" in event:
                event_start = datetime.fromisoformat(event["date_start"])
                event_end = datetime.fromisoformat(event["date_end"])
                # 주 기간이 이벤트 기간과 겹치는지 확인
                if max(week_start, event_start) <= min(week_end, event_end):
                    if event_type == "School Season":
                        is_school_season[i] = 1
                    elif event_type == "Public Holiday":
                        is_holiday[i] = 1 # 기간으로 된 공휴일 (춘절 등)

    return {"is_public_holiday": is_holiday, "is_school_season": is_school_season}


def build_observation(
    disease: str,
    train_until: str,
    end_date: str,
    last_params: Dict[str, Any],
    last_metrics: Dict[str, Any],
    constraints: Dict[str, Any],
    evidence: Optional[Dict[str, Any]] = None,
    recent_metrics_window: Optional[List[Dict[str, Any]]] = None,
    last_llm_pred_q50: Optional[float] = None,
    last_llm_abs_err: Optional[float] = None,
) -> Dict[str, Any]:
    """LLM 에이전트를 위한 observation을 구성합니다."""
    
    all_metadata = load_disease_metadata()
    disease_meta = next((m for m in all_metadata if m["disease_name_zh"] == disease), None)

    # 캘린더 이벤트 신호 추가
    if evidence is None:
        evidence = {}
    if "external_signals" not in evidence:
        evidence["external_signals"] = {}
    
    train_until_dt = datetime.fromisoformat(train_until)
    end_date_dt = datetime.fromisoformat(end_date)
    # horizon을 주 단위로 계산
    horizon_weeks = (end_date_dt - train_until_dt).days // 7
    
    calendar_signals = get_calendar_signals(train_until, horizon_weeks + 1) #เผื่อ 1주
    evidence["external_signals"]["calendar_events"] = calendar_signals


    obs = {
        "disease_name": disease,
        "disease_metadata": disease_meta,
        "train_until": train_until,
        "predict_end_date": end_date,
        "external_signals": evidence.get("external_signals", {}),
        "last_week_params": last_params,
        "last_week_metrics": last_metrics,
        "recent_metrics": recent_metrics_window,
        "param_constraints": constraints,
    }
    
    if last_llm_pred_q50 is not None:
        obs["last_llm_pred_q50"] = last_llm_pred_q50
    if last_llm_abs_err is not None:
        obs["last_llm_abs_err"] = last_llm_abs_err
        
    return obs

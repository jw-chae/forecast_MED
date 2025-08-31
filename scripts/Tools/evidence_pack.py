from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import date as _date, datetime as _dt, timedelta as _td
import csv
import time

# Optional dependency: meteostat (weather). Guard at import time to avoid module-level crash.
try:  # pragma: no cover - optional dep
    from meteostat import Point, Daily  # type: ignore
    _HAS_METEOSTAT = True
except Exception:  # pragma: no cover - optional dep missing
    Point = Daily = None  # type: ignore
    _HAS_METEOSTAT = False

import numpy as np

from Tools.config import APP_CONFIG
from Tools.web_sources import (
    bing_news_search as bing_search,
    serpapi_google_web_search as serp_search,
    fetch_web_signals,  # proper package import
    _http_get,  # Import helper for crawling
    _extract_disease_count_from_html,  # Import helper for parsing
)


def build_evidence_pack_with_weather(
    base: Dict[str, Any], asof: str, location: str = "hangzhou"
) -> Dict[str, Any]:
    """날씨 데이터를 가져와서 evidence pack에 추가합니다."""
    out = base.copy()
    # If meteostat is not installed, skip gracefully
    if not _HAS_METEOSTAT:
        return out
    
    try:
        if location in APP_CONFIG.locations.locations:
            lat, lon = APP_CONFIG.locations.locations[location]
            point = Point(lat, lon)
        else:
            print(f"[WARN] Location '{location}' not found in config. Skipping weather.")
            return out

        end = _dt.fromisoformat(asof)
        start = end - _td(weeks=8)

        data = Daily(point, start, end)
        data = data.fetch()

        if not data.empty:
            # 주간 집계
            weekly_mean_temp = data["tavg"].resample("W-MON").mean().round(2).tolist()
            weekly_total_precip = data["prcp"].resample("W-MON").sum().round(1).tolist()
            
            # 8주 전체 평균/합계
            last_8w_mean_temp = round(float(data["tavg"].mean()), 2)
            last_8w_total_precip = round(float(data["prcp"].sum()), 1)

            weather_signals = {
                "location": location,
                "weekly_mean_temp": weekly_mean_temp,
                "weekly_total_precip": weekly_total_precip,
                "last_8w_mean_temp": last_8w_mean_temp,
                "last_8w_total_precip": last_8w_total_precip,
            }

            if "external_signals" not in base:
                base["external_signals"] = {}
            base["external_signals"]["weather"] = weather_signals

    except Exception as e:
        print(f"[WARN] Failed to fetch or process weather data: {e}")

    return base


def load_evidence_pack(path_or_dir: str) -> Dict[str, Any]:
    """Load one or more JSON evidence files and merge into a single dict.

    - If a directory is provided, load all *.json files and shallow-merge.
    - Later files override earlier keys.
    """
    p = Path(path_or_dir)
    out: Dict[str, Any] = {}
    files: List[Path]
    if p.is_dir():
        files = sorted(p.glob("*.json"))
    else:
        files = [p]
    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                out.update(obj)
        except Exception:
            continue
    return out


def _get_in(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def map_evidence_to_param_hints(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic mapping from evidence pack to simulator parameter hints.

    Returns subset of params: news_signal, quality, nb_dispersion_k (optional),
    and possibly caps when strong signals exist.
    """
    hints: Dict[str, Any] = {}
    # news/search signals
    search_snr = _get_in(evidence, ["external_signals", "search_snr"], None)
    news_chg = _get_in(evidence, ["external_signals", "news_hits_change_4w"], None)
    school_in = _get_in(evidence, ["external_signals", "school_calendar", "in_session"], None)
    temp_mean = _get_in(evidence, ["external_signals", "weather_weekly", "temp_mean"], None)
    hum_mean = _get_in(evidence, ["external_signals", "weather_weekly", "humidity_mean"], None)

    news_sig = 0.0
    if isinstance(search_snr, (int, float)):
        news_sig += max(0.0, min(1.0, float(search_snr) / 3.0))
    if isinstance(news_chg, (int, float)):
        news_sig += max(0.0, min(1.0, float(news_chg) / 5.0))
    if isinstance(school_in, bool) and school_in:
        news_sig += 0.1
    if isinstance(temp_mean, (int, float)) and isinstance(hum_mean, (int, float)):
        if temp_mean >= 28.0 and hum_mean >= 0.75:
            news_sig += 0.1
    if news_sig > 0:
        hints["news_signal"] = float(min(1.0, news_sig))

    # internal signals to adjust dispersion/quality
    pr = _get_in(evidence, ["internal_weekly", "positivity_rate"], None)
    bed = _get_in(evidence, ["internal_weekly", "bed_util"], None)
    erw = _get_in(evidence, ["internal_weekly", "er_wait_min"], None)
    if isinstance(pr, list) and pr:
        last_pr = float(pr[-1])
        if last_pr >= 0.30:
            hints["nb_dispersion_k"] = 8.0
    # quality: conservative lowering if system stress high
    q = 0.72
    stressed = False
    if isinstance(bed, list) and bed and float(bed[-1]) >= 0.90:
        q -= 0.07
        stressed = True
    if isinstance(erw, list) and erw and float(erw[-1]) >= 90.0:
        q -= 0.05
        stressed = True
    if stressed:
        hints["quality"] = float(max(0.5, min(0.95, q)))

    return hints


def build_evidence_pack_with_web(
    disease_zh: str,
    regions: List[str],
    base: Optional[Dict[str, Any]] = None,
    asof: Optional[str] = None,
    gov_only: bool = False,
    site_whitelist: Optional[List[str]] = None,
    region_keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fetch web signals (with as-of cutoff) and merge into an evidence pack structure.
    base: optional existing evidence to be updated.
    asof: YYYY-MM-DD 컷오프(과거 백테스트 시 필수 권장)
    """
    pack = dict(base or {})
    web = fetch_web_signals(
        disease_zh,
        regions,
        asof=asof,
        gov_only=gov_only,
        site_whitelist=site_whitelist,
        region_keywords=region_keywords,
    )
    ext = pack.get("external_signals", {})
    ext.update(web.get("signals", {}))
    pack["external_signals"] = ext
    pack.setdefault("provenance", []).append({
        "source": "web_search",
        "date": time.strftime("%Y-%m-%d"),
        "url_id": "api:news",
        "asof": (asof or web.get("asof"))
    })
    return pack


# Offline: build evidence pack signals from a previously crawled monthly file
def build_evidence_pack_from_gov_monthly_file(
    file_path: str,
    base: Optional[Dict[str, Any]] = None,
    asof: Optional[str] = None,
    weeks: int = 8,
    future_weeks: int = 0,
    future_decay: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Read monthly national totals CSV and derive lightweight weekly signals.

    Expected CSV columns: month,region,cases_total,deaths_total,source,title,url
    - Distributes each month value uniformly to its 4 ending weeks and aggregates
      relative to the as-of cutoff.
    - If future_weeks>0, also generate a decaying weekly vector signal under
      external_signals.news_signal_weekly sized future_weeks.
    """
    pack = dict(base or {})

    # parse as-of date
    try:
        asof_date = _dt.strptime(asof, "%Y-%m-%d").date() if isinstance(asof, str) and asof else _date.today()
    except Exception:
        asof_date = _date.today()

    # load rows from either CSV or JSONL
    rows: List[Dict[str, Any]] = []
    try:
        if file_path.lower().endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        else: # Assume CSV
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
    except Exception:
        return pack

    # helper: map month to 4 week indices (0=last week as-of, positive=weeks ago)
    def month_weeks(mon_str: str) -> List[int]:
        try:
            y, m = map(int, mon_str.split("-"))
            if m == 12:
                end = _dt(y, m, 31)
            else:
                end = _dt(y, m + 1, 1) - _td(days=1)
        except Exception:
            return []
        ws: List[int] = []
        asof_dt = _dt(asof_date.year, asof_date.month, asof_date.day)
        for k in range(4):
            d = end - _td(days=7 * (3 - k))
            w = int((asof_dt.date() - d.date()).days // 7)
            ws.append(w)
        return ws

    # aggregate weekly counts from months
    w_counts: Dict[int, float] = {}
    provenance: List[Dict[str, Any]] = pack.get("provenance", [])
    for r in rows:
        mon = (r.get("month") or "").strip()
        region = (r.get("region") or "").strip()
        if not mon:
            continue
        # prefer national rows; if region blank, accept
        if region and ("全国" not in region):
            continue
        try:
            cases = float(r.get("cases_total") or 0)
        except Exception:
            cases = 0.0
        if cases <= 0:
            continue
        ws = month_weeks(mon)
        if not ws:
            continue
        val = cases / max(1, len(ws))
        for w in ws:
            if 0 <= w < weeks:
                w_counts[w] = w_counts.get(w, 0.0) + val
        provenance.append({
            "source": "gov_monthly_csv",
            "date": mon + "-01",
            "url_id": r.get("url") or "",
            "title": r.get("title") or "",
        })

    # derive simple signals compatible with web ones
    last = float(w_counts.get(0, 0.0))
    prev_avg = sum(w_counts.get(k, 0.0) for k in (1, 2, 3)) / 3.0 if any(w_counts.get(k, 0.0) for k in (1, 2, 3)) else 0.0
    news_hits_change_4w = float((last - prev_avg) / max(1.0, prev_avg)) if prev_avg > 0 else float(last > 0)
    search_snr = float(min(3.0, last / 500.0))

    ext = pack.get("external_signals", {})
    if news_hits_change_4w is not None:
        ext["news_hits_change_4w"] = news_hits_change_4w
    if search_snr is not None:
        ext["search_snr"] = search_snr

    # Add a structured summary of the most recent report before 'asof'
    latest_report = None
    for r in sorted(rows, key=lambda x: x.get("month", ""), reverse=True):
        mon = r.get("month", "").strip()
        if not mon:
            continue
        try:
            report_date = _dt.strptime(mon, "%Y-%m").date().replace(day=28) # Approximate end of month
            if report_date < asof_date:
                latest_report = r
                break
        except Exception:
            continue
    
    if latest_report:
        try:
            # --- New logic: Read the pre-crawled HTML file ---
            disease_specific_cases = None
            month = latest_report.get("month", "")
            disease_zh = base.get("context_meta", {}).get("disease", "手足口病")
            
            # Construct path to the pre-crawled file
            crawled_dir = Path(__file__).resolve().parents[2] / "reports" / "crawled_html"
            safe_month = month.replace("-", "_")
            html_file = crawled_dir / f"{safe_month}_report.html"

            if html_file.exists():
                try:
                    raw_html = html_file.read_text(encoding="utf-8")
                    if raw_html:
                        disease_specific_cases = _extract_disease_count_from_html(raw_html, disease_zh)
                except Exception as e:
                    print(f"[WARN] Failed to read or parse local file {html_file}: {e}")
            else:
                print(f"[INFO] Pre-crawled file not found for month {month}: {html_file}")


            ext["gov_report_summary"] = {
                "month": latest_report.get("month"),
                "disease_name_zh": disease_zh,
                "cases_total_all_diseases": int(float(latest_report.get("cases_total", 0))),
                "cases_specific_disease": disease_specific_cases, # This can be None
                "title": latest_report.get("title"),
                "source_url": latest_report.get("url"), # Keep original URL for reference
            }
        except (ValueError, TypeError):
            pass # Ignore if casting fails


    # optional: future weekly vector (decaying)
    if isinstance(future_weeks, int) and future_weeks > 0:
        base_news = max(0.05, min(0.7, 0.05 + 0.25 * news_hits_change_4w + 0.3 * (search_snr / 3.0)))
        if not future_decay:
            future_decay = [1.0, 0.85, 0.7, 0.55, 0.45, 0.4, 0.35, 0.3][:future_weeks]
            # pad if needed
            if len(future_decay) < future_weeks:
                future_decay = future_decay + [future_decay[-1]] * (future_weeks - len(future_decay))
        news_vec = [round(base_news * float(w), 4) for w in future_decay[:future_weeks]]
        ext["news_signal_weekly"] = news_vec

    pack["external_signals"] = ext
    pack["provenance"] = provenance
    return pack



# Backward-compatible alias for older import name used in callers
def build_evidence_pack_from_gov_monthly_csv(
    csv_path: str,
    base: Optional[Dict[str, Any]] = None,
    asof: Optional[str] = None,
    weeks: int = 8,
    future_weeks: int = 0,
    future_decay: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compatibility shim: call the new *_file implementation with a CSV path.

    rolling_agent_forecast.py expects build_evidence_pack_from_gov_monthly_csv.
    """
    return build_evidence_pack_from_gov_monthly_file(
        file_path=csv_path,
        base=base,
        asof=asof,
        weeks=weeks,
        future_weeks=future_weeks,
        future_decay=future_decay,
    )

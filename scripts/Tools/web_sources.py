from __future__ import annotations

import os
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, quote
from pathlib import Path
import urllib.request


def _http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 60, retries: int = 3, delay: int = 5) -> Dict[str, Any]:
    # Add a default User-Agent to avoid some gov sites blocking requests
    base_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
    if headers:
        base_headers.update(headers)
    req = urllib.request.Request(url, headers=base_headers, method="GET")
    
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()
                # try json first
                try:
                    return json.loads(raw_bytes.decode("utf-8"))
                except Exception:
                    pass
                # fallback encodings for Chinese gov sites
                for enc in ("utf-8", "gb18030", "gbk"):
                    try:
                        text = raw_bytes.decode(enc)
                        break
                    except Exception:
                        text = None
                if text is None:
                    text = raw_bytes.decode("utf-8", errors="ignore")
                try:
                    return json.loads(text)
                except Exception:
                    return {"raw": text}
        except Exception as e:
            if i < retries - 1:
                print(f"[WARN] http_get failed ({e}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise e # Raise the last exception

    return {} # Should not be reached


def _parse_date(dt: str) -> Optional[str]:
    # best-effort date parsing: return YYYY-MM-DD
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(dt, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def _load_env_key(var: str) -> Optional[str]:
    val = os.environ.get(var)
    if val:
        return val
    # try project .env
    try:
        proj = Path(__file__).resolve().parents[2] / ".env"
        if proj.exists():
            for line in proj.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith(var + "="):
                    _, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    if v:
                        os.environ[var] = v
                        return v
    except Exception:
        pass
    return None


def bing_news_search(query: str, count: int = 20, market: str = "zh-CN") -> List[Dict[str, Any]]:
    key = _load_env_key("BING_API_KEY")
    if not key:
        return []
    params = {
        "q": query,
        "mkt": market,
        "count": count,
        "freshness": "Month",
        "sortBy": "Date",
        "originalImg": False,
        "textDecorations": False,
    }
    url = f"https://api.bing.microsoft.com/v7.0/news/search?{urlencode(params)}"
    print(f"[BING] query=\"{query}\" url={url}")
    data = _http_get(url, headers={"Ocp-Apim-Subscription-Key": key})
    _persist_raw('bing', query, data)
    items: List[Dict[str, Any]] = []
    for v in (data.get("value") or []):
        items.append({
            "title": v.get("name"),
            "url": v.get("url"),
            "snippet": v.get("description"),
            "date": _parse_date(v.get("datePublished")) or "",
            "provider": (v.get("provider") or [{}])[0].get("name"),
        })
    print(f"[BING] items={len(items)}")
    return items


def serpapi_news_search(query: str, num: int = 20, hl: str = "zh-CN") -> List[Dict[str, Any]]:
    key = _load_env_key("SERPAPI_KEY")
    if not key:
        return []
    params = {
        "engine": "google_news",
        "q": query,
        "hl": hl,
        "num": num,
        "api_key": key,
    }
    url = f"https://serpapi.com/search.json?{urlencode(params)}"
    # Avoid printing api_key
    print(f"[SERPAPI gnews] query=\"{query}\"")
    data = _http_get(url)
    _persist_raw('serpapi_gnews', query, data)
    items: List[Dict[str, Any]] = []
    for v in (data.get("news_results") or []):
        items.append({
            "title": v.get("title"),
            "url": v.get("link"),
            "snippet": v.get("snippet"),
            "date": _parse_date(v.get("date")) or "",
            "provider": (v.get("source") or ""),
        })
    print(f"[SERPAPI gnews] items={len(items)}")
    return items


def serpapi_google_news_search(query: str, num: int = 20, hl: str = "zh-CN", gl: str = "cn", location: str = "China") -> List[Dict[str, Any]]:
    key = _load_env_key("SERPAPI_KEY")
    if not key:
        return []
    params = {
        "engine": "google",
        "q": query,
        "hl": hl,
        "gl": gl,
        "location": location,
        "tbm": "nws",
        "num": num,
        "api_key": key,
    }
    url = f"https://serpapi.com/search.json?{urlencode(params)}"
    # Avoid printing api_key
    print(f"[SERPAPI google tbm=nws] query=\"{query}\"")
    data = _http_get(url)
    _persist_raw('serpapi_google_nws', query, data)
    items: List[Dict[str, Any]] = []
    for v in (data.get("news_results") or []):
        items.append({
            "title": v.get("title"),
            "url": v.get("link"),
            "snippet": v.get("snippet"),
            "date": _parse_date(v.get("date")) or "",
            "provider": (v.get("source") or {}).get("name") if isinstance(v.get("source"), dict) else v.get("source"),
        })
    print(f"[SERPAPI google tbm=nws] items={len(items)}")
    return items


def _persist_raw(engine: str, query: str, payload: Dict[str, Any]) -> None:
    try:
        base = Path(__file__).resolve().parents[2] / "reports" / "evidence" / "search_logs"
        base.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        q = quote(query)[:80]
        path = base / f"{engine}_{ts}_{q}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SAVE] {path}")
    except Exception as e:
        print(f"[WARN] persist failed: {e}")


def serpapi_google_web_search(query: str, num: int = 10, hl: str = "zh-CN", gl: str = "cn", location: str = "China") -> List[Dict[str, Any]]:
    """General Google web search via SerpAPI (not news-only)."""
    key = _load_env_key("SERPAPI_KEY")
    if not key:
        return []
    params = {
        "engine": "google",
        "q": query,
        "hl": hl,
        "gl": gl,
        "location": location,
        "num": num,
        "api_key": key,
    }
    url = f"https://serpapi.com/search.json?{urlencode(params)}"
    print(f"[SERPAPI google web] query=\"{query}\"")
    data = _http_get(url)
    _persist_raw('serpapi_google_web', query, data)
    items: List[Dict[str, Any]] = []
    for v in (data.get("organic_results") or []):
        items.append({
            "title": v.get("title"),
            "url": v.get("link"),
            "snippet": v.get("snippet"),
            "date": "",
            "provider": "google",
        })
    print(f"[SERPAPI google web] items={len(items)}")
    return items


def _extract_month_from_text(text: str) -> Optional[str]:
    """Extract YYYY-MM from Chinese text like '2023年1月'."""
    import re
    m = re.search(r"(20\d{2})年\s*(1[0-2]|0?[1-9])月", text)
    if not m:
        return None
    y, mm = int(m.group(1)), int(m.group(2))
    return f"{y:04d}-{mm:02d}"


def _extract_disease_count_from_html(html: str, disease_zh: str) -> Optional[int]:
    """Heuristic HTML parse to get the integer following the disease label in tables/lists."""
    import re
    # Common patterns: <td>手足口病</td><td>4613</td> or '手足口病 4613'
    patterns = [
        rf"{disease_zh}\s*</t[dh][^>]*>\s*<t[dh][^>]*>\s*(\d{{1,7}})\s*</t[dh]>",
        rf"{disease_zh}[^\n\r\d]*(\d{{1,7}})"
    ]
    for pat in patterns:
        m = re.search(pat, html, flags=re.I | re.S)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None


def fetch_official_stats_signals(
    disease_zh: str,
    asof: Optional[str],
    weeks: int = 8,
    site_whitelist: Optional[List[str]] = None,
    region_keywords: Optional[List[str]] = None,
    return_pages: bool = False,
) -> Dict[str, Any]:
    """Fetch official monthly stats pages and convert to weekly signals.

    - Searches around as-of month for pages like '法定传染病 疫情 通报 2023年X月' on government sites.
    - Parses the disease row count and spreads it uniformly across 4 weeks of that month.
    """
    from datetime import date
    try:
        asof_date = date.fromisoformat(asof) if asof else date.today()
    except Exception:
        asof_date = date.today()
    # Build 3 month candidates: as-of and previous two months
    months: List[str] = []
    y, m = asof_date.year, asof_date.month
    for k in range(3):
        mm = m - k
        yy = y
        while mm <= 0:
            mm += 12
            yy -= 1
        months.append(f"{yy}年{mm}月")
    site_whitelist = site_whitelist or ["nhc.gov.cn", "wjw.zj.gov.cn", "zjwjw.gov.cn", "*.gov.cn"]
    queries: List[str] = []
    for mon in months:
        base = f"法定传染病 疫情 通报 {mon}"
        for site in site_whitelist:
            queries.append(f"site:{site} {base}")
    print(f"[OFFICIAL] total queries={len(queries)} asof={asof}")
    pages: List[Dict[str, Any]] = []
    for q in queries:
        items = serpapi_google_web_search(q, num=10) or []
        for it in items[:5]:
            url = it.get("url")
            if not url:
                continue
            try:
                page = _http_get(url)
                raw = page.get("raw") if isinstance(page, dict) else None
                if not raw:
                    continue
                # region gating: must mention at least one region keyword if provided
                if region_keywords:
                    title_plus = (it.get("title") or "") + "\n" + raw
                    if not any(k in title_plus for k in region_keywords):
                        continue
                mon = _extract_month_from_text((it.get("title") or "") + "\n" + raw)
                if not mon:
                    continue
                cnt = _extract_disease_count_from_html(raw, disease_zh)
                if cnt is None:
                    continue
                pages.append({"url": url, "month": mon, "count": int(cnt)})
            except Exception:
                continue
    # Map month counts to weeks relative to as-of
    from datetime import datetime as _dt
    from datetime import timedelta as _td
    def month_weeks(mon_str: str) -> List[int]:
        try:
            y, m = map(int, mon_str.split("-"))
            if m == 12:
                end = _dt(y, m, 31)
            else:
                end = _dt(y, m + 1, 1) - _td(days=1)
        except Exception:
            return []
        asof_dt = _dt(asof_date.year, asof_date.month, asof_date.day)
        ws: List[int] = []
        for k in range(4):
            d = end - _td(days=7 * (3 - k))
            w = int((asof_dt.date() - d.date()).days // 7)
            ws.append(w)
        return ws
    w_counts: Dict[int, float] = {}
    provenance: List[Dict[str, Any]] = []
    for p in pages:
        ws = month_weeks(p["month"])
        if not ws:
            continue
        val = float(p["count"]) / max(1, len(ws))
        for w in ws:
            if 0 <= w < weeks:
                w_counts[w] = w_counts.get(w, 0.0) + val
        provenance.append({"source": "gov_monthly", "date": p["month"] + "-01", "title": "official monthly", "url_id": p["url"]})
    last = float(w_counts.get(0, 0.0))
    prev_avg = sum(w_counts.get(k, 0.0) for k in (1, 2, 3)) / 3.0 if any(w_counts.get(k, 0.0) for k in (1, 2, 3)) else 0.0
    news_hits_change_4w = float((last - prev_avg) / max(1.0, prev_avg)) if prev_avg > 0 else float(last > 0)
    # scale weekly count to a small SNR proxy; tuned conservatively
    search_snr = float(min(3.0, last / 500.0))
    out = {"signals": {"news_hits_change_4w": news_hits_change_4w, "search_snr": search_snr}, "provenance": provenance}
    if return_pages:
        out["pages"] = pages
    return out


def fetch_web_signals(disease_zh: str, regions: List[str], weeks: int = 8, asof: Optional[str] = None, gov_only: bool = False, site_whitelist: Optional[List[str]] = None, region_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """Fetch recent news items and compute naive signals with an as-of cutoff.

    - disease_zh: 질병명(중문) 예: "手足口病", "流行性感冒"
    - regions: ["中国 全国", "浙江省"] 등
    - weeks: 최근 N주 데이터를 요약 (as-of 기준)
    - asof: YYYY-MM-DD 형태의 기준일. 지정 시 해당 날짜를 '현재'로 간주하여 집계.
            미지정 시 UTC today 사용.
    """
    base_q = [
        "传染 病例",
        "疫情",
        "病例 通报",
        "疫情 公报",
        "防疫 公告",
    ]
    queries: List[str] = []
    for r in regions:
        for t in base_q:
            queries.append(f"{r} {disease_zh} {t}")
    items: List[Dict[str, Any]] = []
    prov: List[Dict[str, Any]] = []
    print(f"[FETCH] total queries={len(queries)} regions={regions} disease={disease_zh}")
    if not gov_only:
        for q in queries:
            got = bing_news_search(q, count=25) or serpapi_google_news_search(q, num=25) or serpapi_news_search(q, num=25)
            items.extend(got)
            for it in got[:5]:
                print(f"  - {it.get('date')} | {it.get('provider')} | {it.get('title')}")
    # official sources (averaged in later)
    off = fetch_official_stats_signals(disease_zh, asof=asof, weeks=weeks, site_whitelist=site_whitelist, region_keywords=region_keywords)
    if isinstance(off.get("provenance"), list):
        prov.extend(off.get("provenance"))
    # time window counts (use as-of cutoff)
    _asof = None
    try:
        _asof = (datetime.strptime(asof, "%Y-%m-%d").date() if isinstance(asof, str) and asof else datetime.utcnow().date())
    except Exception:
        _asof = datetime.utcnow().date()

    def week_of(date_str: str) -> int:
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            return -1
        # ignore future-dated items beyond as-of
        if d > _asof:
            return -1
        return int((_asof - d).days // 7)
    # drop items without date and those after as-of
    recent = [x for x in items if x.get("date") and week_of(x.get("date")) >= 0]
    w_counts = {}
    for it in recent:
        w = week_of(it["date"])
        if 0 <= w < weeks:
            w_counts[w] = w_counts.get(w, 0) + 1
    # derive signals: last week vs previous 3-week avg
    last = w_counts.get(0, 0)
    prev_avg = sum(w_counts.get(k, 0) for k in (1,2,3)) / 3.0 if any(w_counts.get(k) for k in (1,2,3)) else 0.0
    news_hits_change_4w = float((last - prev_avg) / max(1.0, prev_avg)) if prev_avg > 0 else float(last > 0)
    # crude search_snr proportional to last-week hits
    search_snr = float(min(3.0, last / 5.0))
    # combine with official signals via average (if present)
    if isinstance(off.get("signals"), dict):
        s = off["signals"]
        news_hits_change_4w = float((news_hits_change_4w + s.get("news_hits_change_4w", news_hits_change_4w)) / 2.0)
        search_snr = float(min(3.0, max(0.0, (search_snr + s.get("search_snr", search_snr)) / 2.0)))
    return {
        "raw_items": recent[:100],
        "signals": {
            "news_hits_change_4w": news_hits_change_4w,
            "search_snr": search_snr,
        },
        "provenance": prov,
        "asof": _asof.strftime("%Y-%m-%d"),
    }



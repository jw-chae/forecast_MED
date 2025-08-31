#!/usr/bin/env python3
from __future__ import annotations

import re
import csv
import sys
import time
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Iterable, Tuple, Any
from pathlib import Path
from urllib.parse import urljoin

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


@dataclass
class MonthlyStat:
    region: str
    month: str  # YYYY-MM
    cases_total: Optional[int]
    deaths_total: Optional[int]
    url: str
    title: str
    source: str  # nhc | ndcpa | zj


def http_get_text(url: str, timeout: int = 25, max_retries: int = 3, sleep_seconds: float = 1.0) -> Optional[str]:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            if requests is None:
                import urllib.request  # lazy
                req = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read()
                # best-effort decode for CN gov sites
                for enc in ("utf-8", "gb18030", "gbk"):
                    try:
                        return raw.decode(enc)
                    except Exception:
                        continue
                return raw.decode("utf-8", errors="ignore")
            else:
                r = requests.get(url, headers=headers, timeout=timeout)
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.encoding = r.apparent_encoding or "utf-8"
                return r.text
        except Exception as e:  # pragma: no cover
            last_exc = e
            time.sleep(sleep_seconds * (1.5 ** attempt))
    print(f"[WARN] GET failed url={url} err={last_exc}")
    return None


def ensure_soup(html: str) -> Optional[Any]:
    if BeautifulSoup is None:
        return None
    return BeautifulSoup(html, "html.parser")


MONTH_RE = re.compile(r"(20\d{2})年\s*(1[0-2]|0?[1-9])月")


def extract_month(text: str) -> Optional[str]:
    m = MONTH_RE.search(text or "")
    if not m:
        return None
    y = int(m.group(1))
    mm = int(m.group(2))
    return f"{y:04d}-{mm:02d}"


def extract_totals(text: str) -> Tuple[Optional[int], Optional[int]]:
    # Common patterns in CN official monthly reports
    # 1) "共报告法定传染病1015490例，死亡2313人"
    # 2) "报告发病727645例、死亡2465人"
    # 3) sometimes punctuation variants: 、 ， , ;
    norm = (text or "").replace("\xa0", " ").replace(" ", "")
    patterns = [
        r"共报告法定传染病(\d{1,9})例[，,、;]?死亡(\d{1,7})人",
        r"报告发病(\d{1,9})例[，,、;]?死亡(\d{1,7})人",
        r"共报告(\d{1,9})例[，,、;]?死亡(\d{1,7})人",
    ]
    for pat in patterns:
        m = re.search(pat, norm)
        if m:
            try:
                return int(m.group(1)), int(m.group(2))
            except Exception:
                pass
    # try to get only cases if deaths not explicitly shown
    only_cases = re.search(r"共报告法定传染病(\d{1,9})例", norm) or re.search(r"报告发病(\d{1,9})例", norm)
    c = int(only_cases.group(1)) if only_cases else None
    return c, None


def in_month_range(mon: str, start_mon: str, end_mon: str) -> bool:
    def key(m: str) -> int:
        y, mm = m.split("-")
        return int(y) * 12 + int(mm)
    return key(start_mon) <= key(mon) <= key(end_mon)


def crawl_nhc_list_urls(max_pages: int = 60) -> List[Tuple[str, str]]:
    base = "https://www.nhc.gov.cn/wjw/yqbb/"
    pages: List[str] = [urljoin(base, "list.shtml")] + [urljoin(base, f"list_{i}.shtml") for i in range(1, max_pages + 1)]
    out: List[Tuple[str, str]] = []  # (title, url)
    for idx, purl in enumerate(pages):
        html = http_get_text(purl)
        if not html:
            continue
        soup = ensure_soup(html)
        if not soup:
            # fallback regex
            for m in re.finditer(r"<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>", html, flags=re.I | re.S):
                title = re.sub(r"<.*?>", "", m.group(2))
                url = urljoin(purl, m.group(1))
                if "法定传染病" in title and ("全国" in title or "疫情概况" in title):
                    out.append((title.strip(), url))
            continue
        for a in soup.select("a"):
            title = (a.get_text() or "").strip()
            if not title:
                continue
            if ("法定传染病" in title) and ("全国" in title or "疫情概况" in title):
                href = a.get("href")
                if not href:
                    continue
                url = urljoin(purl, href)
                out.append((title, url))
        # be polite
        time.sleep(0.2)
    # dedupe
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for t, u in out:
        if u in seen:
            continue
        seen.add(u)
        deduped.append((t, u))
    return deduped


def crawl_ndcpa_list_urls() -> List[Tuple[str, str]]:
    # Two list endpoints observed in search logs:
    # - common list: /jbkzzx/c100016/common/list.html
    # - second list: /jbkzzx/c100016/second/list.html
    bases = [
        "https://www.ndcpa.gov.cn/jbkzzx/c100016/common/list.html",
        "https://www.ndcpa.gov.cn/jbkzzx/c100016/second/list.html",
    ]
    out: List[Tuple[str, str]] = []
    for purl in bases:
        html = http_get_text(purl)
        if not html:
            continue
        soup = ensure_soup(html)
        if soup:
            for a in soup.select("a"):
                title = (a.get_text() or "").strip()
                if not title:
                    continue
                if ("法定传染病" in title) and ("全国" in title or "疫情概况" in title):
                    href = a.get("href")
                    if not href:
                        continue
                    url = urljoin(purl, href)
                    out.append((title, url))
        else:
            for m in re.finditer(r"<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>", html, flags=re.I | re.S):
                title = re.sub(r"<.*?>", "", m.group(2))
                url = urljoin(purl, m.group(1))
                if "法定传染病" in title and ("全国" in title or "疫情概况" in title):
                    out.append((title.strip(), url))
        time.sleep(0.2)
    # Some pages paginate via JS or API; attempt to discover additional pages via content links
    seed = list(out)
    for _, u in seed[:80]:
        html = http_get_text(u)
        if not html:
            continue
        for m in re.finditer(r"href=\"(/jbkzzx/c100016/common/content/[^\"]+)\"", html):
            cu = urljoin(u, m.group(1))
            if cu not in {uu for _, uu in out}:
                out.append(("", cu))
        time.sleep(0.15)
    # dedupe by url
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for t, u in out:
        if u in seen:
            continue
        seen.add(u)
        deduped.append((t, u))
    return deduped


def seed_urls_from_search_logs(log_dir: Optional[str] = None) -> List[Tuple[str, str]]:
    # Load previously saved SerpAPI/Bing logs and extract candidate URLs
    if log_dir is None:
        # default: ../../reports/evidence/search_logs relative to this file
        log_dir = str(Path(__file__).resolve().parents[2] / "reports" / "evidence" / "search_logs")
    out: List[Tuple[str, str]] = []
    p = Path(log_dir)
    if not p.exists():
        return out
    for jf in sorted(p.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        # organic_results (google web)
        for v in (data.get("organic_results") or []):
            title = (v.get("title") or "").strip()
            url = v.get("link") or v.get("url")
            if not url or not title:
                continue
            if ("法定传染病" in title) and ("疫情" in title or "疫情概况" in title):
                if any(host in url for host in ("ndcpa.gov.cn", "nhc.gov.cn", "wsjkw.zj.gov.cn", "wjw.zj.gov.cn")):
                    out.append((title, url))
        # news_results (google news)
        for v in (data.get("news_results") or []):
            title = (v.get("title") or "").strip()
            url = v.get("link") or v.get("url")
            if not url or not title:
                continue
            if ("法定传染病" in title) and ("疫情" in title or "疫情概况" in title):
                if any(host in url for host in ("ndcpa.gov.cn", "nhc.gov.cn", "wsjkw.zj.gov.cn", "wjw.zj.gov.cn")):
                    out.append((title, url))
    # dedupe by url
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for t, u in out:
        if u in seen:
            continue
        seen.add(u)
        deduped.append((t, u))
    return deduped


def parse_article(url: str, title_hint: str = "") -> Optional[MonthlyStat]:
    html = http_get_text(url)
    if not html:
        return None
    soup = ensure_soup(html)
    title = title_hint
    main_text = ""
    if soup:
        # try common content containers
        if not title:
            tnode = soup.select_one("title") or soup.select_one("h1")
            if tnode:
                title = (tnode.get_text() or "").strip()
        for sel in (".article-con", ".article-content", ".content", "#content", ".TRS_Editor", "article", "body"):
            node = soup.select_one(sel)
            if node:
                main_text = node.get_text("\n").strip()
                if main_text and len(main_text) > 50:
                    break
        if not main_text:
            main_text = soup.get_text("\n").strip()
    else:
        title = title or (re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S).group(1).strip() if re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S) else "")
        main_text = re.sub(r"<[^>]+>", "\n", html)
    mon = extract_month(title + "\n" + main_text)
    if not mon:
        return None
    cases, deaths = extract_totals(main_text)
    region = "中国 全国"
    src = "ndcpa" if "ndcpa.gov.cn" in url else ("nhc" if "nhc.gov.cn" in url else ("zj" if "zj.gov.cn" in url else "unknown"))
    return MonthlyStat(region=region, month=mon, cases_total=cases, deaths_total=deaths, url=url, title=title or "", source=src)


def crawl_monthly(start_month: str, end_month: str, max_pages: int = 60) -> List[MonthlyStat]:
    items: List[MonthlyStat] = []
    # 1) NHC (older, through ~2022)
    print("[NHC] scanning list pages ...")
    for title, url in crawl_nhc_list_urls(max_pages=max_pages):
        mon = extract_month(title)
        if not mon or not in_month_range(mon, start_month, end_month):
            continue
        st = parse_article(url, title_hint=title)
        if st:
            items.append(st)
        time.sleep(0.2)
    # 2) NDCPC (newer, 2023+)
    print("[NDCPC] scanning list pages ...")
    for title, url in crawl_ndcpa_list_urls():
        # if title missing, still parse and filter by extracted month later
        st = parse_article(url, title_hint=title)
        if not st:
            continue
        if in_month_range(st.month, start_month, end_month):
            items.append(st)
        time.sleep(0.15)
    # 3) Seed from prior search logs
    print("[SEED] scanning saved search logs ...")
    for title, url in seed_urls_from_search_logs():
        st = parse_article(url, title_hint=title)
        if not st:
            continue
        if in_month_range(st.month, start_month, end_month):
            items.append(st)
        time.sleep(0.05)
    # dedupe by (source, month)
    keyset = set()
    deduped: List[MonthlyStat] = []
    for it in sorted(items, key=lambda x: (x.month, x.source, x.url)):
        k = (it.source, it.month)
        if k in keyset:
            continue
        keyset.add(k)
        deduped.append(it)
    # sanity filter: keep at most one national row per month (prefer ndcpa over nhc for 2023+)
    month_kept: Dict[str, MonthlyStat] = {}
    for it in deduped:
        if it.month not in month_kept:
            month_kept[it.month] = it
            continue
        # prefer newer source ndcpa
        if it.source == "ndcpa" and month_kept[it.month].source != "ndcpa":
            month_kept[it.month] = it
    final = [month_kept[m] for m in sorted(month_kept.keys())]
    print(f"[DONE] collected {len(final)} monthly rows in range {start_month}..{end_month}")
    return final


def save_csv(rows: Iterable[MonthlyStat], out_csv: str, out_jsonl: Optional[str] = None) -> None:
    import os
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "region", "cases_total", "deaths_total", "source", "title", "url"])
        for r in rows:
            w.writerow([r.month, r.region, r.cases_total if r.cases_total is not None else "", r.deaths_total if r.deaths_total is not None else "", r.source, r.title, r.url])
    if out_jsonl:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"[SAVE] {out_csv}{' and ' + out_jsonl if out_jsonl else ''}")


def cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Crawl CN official monthly infectious disease statistics (2019-01 .. 2025-07)")
    p.add_argument("--start", default="2019-01", help="start month YYYY-MM (inclusive)")
    p.add_argument("--end", default="2025-07", help="end month YYYY-MM (inclusive)")
    p.add_argument("--out", default="../../reports/evidence/gov_reports/monthly_stats_2019-01_to_2025-07.csv", help="output CSV path")
    p.add_argument("--jsonl", default="../../reports/evidence/gov_reports/monthly_stats_2019-01_to_2025-07.jsonl", help="optional JSONL output path")
    p.add_argument("--max-pages", type=int, default=60, help="max NHC list pages to scan")
    args = p.parse_args(argv)

    rows = crawl_monthly(args.start, args.end, args.max_pages)
    save_csv(rows, args.out, args.jsonl)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli())



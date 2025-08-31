from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from run_sim_wrapper import run_sim, SimConfig


def load_last_log(disease: str, logs_dir: Path) -> Path | None:
    cand = sorted(logs_dir.glob(f"llm_analyst_{disease}_*.jsonl"))
    return cand[-1] if cand else None


def load_proposals(log_path: Path) -> List[Dict[str, Any]]:
    props: List[Dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        p = rec.get("proposal")
        it = rec.get("iter")
        if isinstance(p, dict) and it is not None:
            props.append({"iter": it, "params": p})
    props.sort(key=lambda x: x["iter"])  # ensure order
    return props


def build_html(disease: str, snapshots: List[Dict[str, Any]]) -> str:
    import json as _json
    plotly_cdn = "https://cdn.plot.ly/plotly-2.27.1.min.js"
    payload = _json.dumps({"disease": disease, "snaps": snapshots}, ensure_ascii=False)
    return f"""
<!DOCTYPE html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\"/>
  <title>Trajectory Comparison · {disease}</title>
  <script src=\"{plotly_cdn}\"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
    #chart {{ width: 100%; height: 640px; }}
    .pill {{ display:inline-block; padding: 4px 10px; border-radius: 999px; background:#f3f4f6; margin-right: 8px; }}
  </style>
</head>
<body>
  <h2>실측 vs 예측 궤적 비교 · {disease}</h2>
  <div id=\"chart\"></div>
  <script>
    const D = {payload};
    const snaps = D.snaps;
    if (!snaps.length) {{ document.getElementById('chart').textContent='no data'; }}
    const s0 = snaps[snaps.length - 1];
    const histX = s0.dates_hist, histY = s0.values_hist;
    const tarX = s0.dates_target, tarY = s0.values_target;
    const traces = [];
    traces.push({{ x: histX, y: histY, name:'학습기간', mode:'lines', line:{{color:'#999'}} }});
    traces.push({{ x: tarX, y: tarY, name:'실측(홀드아웃)', mode:'lines+markers', line:{{color:'#1f77b4'}}, marker:{{size:4}} }});
    for (const s of snaps) {{
      traces.push({{ x: s.future_dates, y: s.quantiles.q50, name:`iter ${'{'}s.iter{'}'} · q50`, mode:'lines', line:{{dash:'dash'}} }});
      traces.push({{ x: s.future_dates, y: s.quantiles.q95, name:`iter ${'{'}s.iter{'}'} · q95`, mode:'lines', line:{{color:'rgba(214,39,40,0.6)'}} }});
      traces.push({{ x: s.future_dates, y: s.quantiles.q05, name:`iter ${'{'}s.iter{'}'} · q05`, mode:'lines', line:{{color:'rgba(214,39,40,0.6)', dash:'dot'}} }});
    }}
    const layout = {{ hovermode:'x unified', xaxis:{{title:'week'}}, yaxis:{{title:'count'}}, legend:{{orientation:'h'}}, margin:{{t:20,r:20,b:80,l:60}} }};
    Plotly.newPlot('chart', traces, layout, {{displaylogo:false}});
  </script>
</body>
</html>
"""


def main():
    base = Path(__file__).resolve().parents[2]
    logs_dir = base / "reports" / "agent_logs"
    out_dir = logs_dir
    for disease in ("手足口病", "流行性感冒"):
        log_path = load_last_log(disease, logs_dir)
        if not log_path:
            continue
        props = load_proposals(log_path)
        # 시뮬레이션 스냅샷 생성(최근 5개)
        snaps: List[Dict[str, Any]] = []
        for rec in props[-5:]:
            params = dict(rec["params"])  # copy
            cfg = SimConfig(disease=disease, train_until="2022-12-31", end="2024-12-31", season_profile="flu")
            sim = run_sim(params, cfg)
            snaps.append({
                "iter": rec["iter"],
                "quantiles": sim["quantiles"],
                "future_dates": sim["future_dates"],
                "dates_hist": sim["dates_hist"],
                "values_hist": sim["values_hist"],
                "dates_target": sim["dates_target"],
                "values_target": sim["values_target"],
            })
        html = build_html(disease, snaps)
        out_file = out_dir / f"compare_trajectory_{disease}.html"
        out_file.write_text(html, encoding="utf-8")
        print(str(out_file))


if __name__ == "__main__":
    main()



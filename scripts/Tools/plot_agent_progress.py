from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any


def load_runs(log_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for fp in sorted(log_dir.glob("llm_analyst_*.jsonl")):
        disease = fp.name.split("_")[1]
        run_id = fp.stem.split("_")[-1]
        series: List[Dict[str, Any]] = []
        for line in fp.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            it = rec.get("iter")
            m = rec.get("metrics") or {}
            cov = m.get("coverage95")
            crps = m.get("crps")
            recall = m.get("recall_pm2w") or m.get("recall@pm2w") or m.get("recall_pm2w".replace("@",""))
            if it is None:
                continue
            series.append({"iter": it, "coverage95": cov, "crps": crps, "recall_pm2w": recall})
        if series:
            series = sorted(series, key=lambda r: r["iter"])  # ensure order
            items.append({"file": fp.name, "disease": disease, "run_id": run_id, "series": series})
    return items


def build_html(data: List[Dict[str, Any]]) -> str:
    plotly_cdn = "https://cdn.plot.ly/plotly-2.27.1.min.js"
    payload = json.dumps(data, ensure_ascii=False)
    html = f"""
<!DOCTYPE html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\"/>
  <title>Agent Progress · Coverage/CRPS/Peak Recall</title>
  <script src=\"{plotly_cdn}\"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
    .row {{ display:flex; gap:24px; flex-wrap: wrap; }}
    .card {{ flex:1 1 520px; border:1px solid #eee; border-radius:12px; padding:12px; }}
    .pill {{ display:inline-block; padding: 4px 10px; border-radius: 999px; background:#f3f4f6; margin-right: 8px; }}
  </style>
 </head>
<body>
  <h2>LLM Agent Progress</h2>
  <div class=\"pill\" id=\"info\"></div>
  <div class=\"row\">
    <div class=\"card\"><div id=\"cov\" style=\"height:380px\"></div></div>
    <div class=\"card\"><div id=\"crps\" style=\"height:380px\"></div></div>
    <div class=\"card\"><div id=\"recall\" style=\"height:380px\"></div></div>
  </div>
  <script>
   const DATA = {payload};
   document.getElementById('info').textContent = 'runs: ' + DATA.length;
   function seriesToTrace(s, metric, color) {{
      const x = s.series.map(r => r.iter);
      const y = s.series.map(r => (r[metric] == null ? null : r[metric]));
      return {{ x, y, mode:'lines+markers', name: s.disease + ' · ' + s.run_id, line:{{color}}, marker:{{size:6}} }};
   }}
   const colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b'];
   const covTraces = DATA.map((s,i)=> seriesToTrace(s,'coverage95', colors[i%colors.length]));
   const crpsTraces = DATA.map((s,i)=> seriesToTrace(s,'crps', colors[i%colors.length]));
   const recTraces = DATA.map((s,i)=> seriesToTrace(s,'recall_pm2w', colors[i%colors.length]));
   Plotly.newPlot('cov', covTraces, {{ title:'coverage95 by iteration', xaxis:{{title:'iter'}}, yaxis:{{title:'coverage95', range:[0,1]}}, legend:{{orientation:'h'}} }}, {{displaylogo:false}});
   Plotly.newPlot('crps', crpsTraces, {{ title:'CRPS by iteration', xaxis:{{title:'iter'}}, yaxis:{{title:'crps'}}, legend:{{orientation:'h'}} }}, {{displaylogo:false}});
   Plotly.newPlot('recall', recTraces, {{ title:'Peak recall@±2w by iteration', xaxis:{{title:'iter'}}, yaxis:{{title:'recall', range:[0,1]}}, legend:{{orientation:'h'}} }}, {{displaylogo:false}});
  </script>
</body>
</html>
"""
    return html


def main():
    base = Path(__file__).resolve().parents[2]
    log_dir = base / "reports" / "agent_logs"
    out_file = log_dir / "agent_progress.html"
    items = load_runs(log_dir)
    html = build_html(items)
    out_file.write_text(html, encoding="utf-8")
    print(str(out_file))


if __name__ == "__main__":
    main()



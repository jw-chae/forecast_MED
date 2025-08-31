from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


PARAM_KEYS = [
    "amplitude_quantile",
    "amplitude_multiplier",
    "ratio_cap_quantile",
    "warmup_weeks",
    "use_delta_quantile",
    "delta_quantile",
    "quality",
    "nb_dispersion_k",
    "start_value_override",
    "r_boost_cap",
    "scale_cap",
    "x_cap_multiplier",
    "evt_u_quantile",
]

METRIC_KEYS = [
    "crps",
    "mae_median",
    "coverage95",
    "recall_pm2w",
]


def read_runs(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for r in data:
            if isinstance(r, dict) and isinstance(r.get("metrics"), dict):
                runs.append(r)
    return runs


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--dir", default=str(Path(__file__).resolve().parents[2] / "reports" / "bo_runs"))
    args = parser.parse_args()

    d = Path(args.dir)
    files = sorted(d.glob("bo_stub_*_2022-12-31_2024-12-31.json"))
    # header
    header = ["질병", "rank"] + PARAM_KEYS + METRIC_KEYS
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for f in files:
        parts = f.name.split("_")
        label = parts[2] if len(parts) > 2 else parts[1]
        runs = read_runs(f)
        runs.sort(key=lambda r: r.get("metrics", {}).get("crps", 1e9) + r.get("metrics", {}).get("mae_median", 1e9))
        for i, r in enumerate(runs[: max(0, min(args.top, len(runs)))], start=1):
            p = r.get("params", {})
            m = r.get("metrics", {})
            row = [label, str(i)]
            for k in PARAM_KEYS:
                row.append(fmt(p.get(k, "-")))
            for k in METRIC_KEYS:
                row.append(fmt(m.get(k, "-")))
            print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()



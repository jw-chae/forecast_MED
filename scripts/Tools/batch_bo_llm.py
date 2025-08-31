from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

from run_sim_wrapper import run_sim, SimConfig


def sample_params(bounds: Dict[str, Tuple[float, float]], rng: np.random.Generator) -> Dict[str, Any]:
    p: Dict[str, Any] = {}
    for k, (lo, hi) in bounds.items():
        if k in {"warmup_weeks"}:
            p[k] = int(rng.integers(int(lo), int(hi) + 1))
        else:
            p[k] = float(rng.uniform(lo, hi))
    # fixed toggles
    p["use_delta_quantile"] = True
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diseases", nargs="*", default=["流行性感冒", "手足口病"])
    parser.add_argument("--train_until", default="2022-12-31")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--season_profile", default="flu")
    parser.add_argument("--n_init", type=int, default=30)
    parser.add_argument("--n_refine", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    out_dir = base / "reports" / "bo_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds: Dict[str, Tuple[float, float]] = {
        "amplitude_quantile": (0.85, 0.98),
        "amplitude_multiplier": (1.2, 2.8),
        "ratio_cap_quantile": (0.95, 0.999),
        "warmup_weeks": (0, 2),
        "delta_quantile": (0.01, 0.1),
        "quality": (0.5, 0.95),
        "nb_dispersion_k": (2.0, 50.0),
        "r_boost_cap": (1.2, 3.0),
        "scale_cap": (1.2, 1.8),
        "x_cap_multiplier": (1.5, 4.0),
        "evt_u_quantile": (0.85, 0.95),
    }

    rng = np.random.default_rng(args.seed)

    for dis in args.diseases:
        config = SimConfig(disease=dis, train_until=args.train_until, end=args.end, season_profile=args.season_profile)
        runs: List[Dict[str, Any]] = []
        # initial exploration
        for _ in range(args.n_init):
            p = sample_params(bounds, rng)
            res = run_sim(p, config)
            runs.append({"params": p, "metrics": res["metrics"]})
        # select top by (CRPS + MAE)
        runs.sort(key=lambda r: (r["metrics"].get("crps", 1e9) + r["metrics"].get("mae_median", 1e9)))
        topk = runs[: max(1, min(6, len(runs)//5))]
        # local refinements around top
        refined: List[Dict[str, Any]] = []
        for base_run in topk:
            bp = base_run["params"]
            for _ in range(args.n_refine):
                p = dict(bp)
                for k, (lo, hi) in bounds.items():
                    if k == "warmup_weeks":
                        continue
                    step = 0.05 * (hi - lo)
                    p[k] = float(np.clip(p[k] + rng.normal(0.0, step), lo, hi))
                res = run_sim(p, config)
                refined.append({"params": p, "metrics": res["metrics"]})
        all_runs = runs + refined
        out_file = out_dir / f"bo_stub_{dis}_{args.train_until}_{args.end}.json"
        out_file.write_text(json.dumps(all_runs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(out_file))


if __name__ == "__main__":
    main()



from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from experiments.metrics import PredictionBundle, evaluate_metrics, METRICS_REGISTRY
from experiments.visualization import create_all_visualizations


def load_rolling_from_results_json(path: Path) -> Tuple[Dict[str, Any], pd.Series, PredictionBundle, pd.DataFrame]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    steps = data.get("steps", [])

    # actual / prediction series
    dates = []
    actuals = []
    preds = []
    q05_list = []
    q50_list = []
    q95_list = []

    for step in steps:
        actual_block = step.get("actual_values", {})
        pred_block = step.get("predictor", {}).get("parsed_forecast", {})
        point_block = step.get("predictor", {}).get("predicted_value", [])

        adates = actual_block.get("dates", [])
        avals = actual_block.get("values", [])
        if not adates or not avals:
            continue
        # horizon=1 가정 → 하나씩
        date = adates[0]
        actual = avals[0]

        # point prediction
        if point_block:
            pred = point_block[0]
        else:
            # fallback: 중앙값
            pred = pred_block.get("q50", [np.nan])[0]

        q05 = pred_block.get("q05", [np.nan])[0]
        q50 = pred_block.get("q50", [np.nan])[0]
        q95 = pred_block.get("q95", [np.nan])[0]

        dates.append(pd.to_datetime(date))
        actuals.append(actual)
        preds.append(pred)
        q05_list.append(q05)
        q50_list.append(q50)
        q95_list.append(q95)

    if not dates:
        raise ValueError(f"No steps with actual_values found in {path}")

    idx = pd.to_datetime(dates)
    actual_series = pd.Series(actuals, index=idx, name="actual")
    pred_series = pd.Series(preds, index=idx, name="prediction")

    quantiles = {
        0.05: pd.Series(q05_list, index=idx),
        0.50: pd.Series(q50_list, index=idx),
        0.95: pd.Series(q95_list, index=idx),
    }

    bundle = PredictionBundle(point=pred_series, quantiles=quantiles)

    # rolling-style predictions_df (as_of = current_target_date, one row per step)
    rows = []
    for i, step in enumerate(steps):
        actual_block = step.get("actual_values", {})
        adates = actual_block.get("dates", [])
        avals = actual_block.get("values", [])
        if not adates or not avals:
            continue
        target_date = pd.to_datetime(adates[0])
        rows.append(
            {
                "as_of": pd.to_datetime(step.get("current_target_date")),
                "target_date": target_date,
                "actual": avals[0],
                "prediction": preds[len(rows)],
            }
        )

    predictions_df = pd.DataFrame(rows)

    return meta, actual_series, bundle, predictions_df


def build_summary(meta: Dict[str, Any], metrics: Dict[str, float], run_dir: Path) -> Dict[str, Any]:
    return {
        "disease": meta.get("disease"),
        "start": meta.get("start_date"),
        "end": meta.get("end_date"),
        "n_steps": meta.get("n_steps"),
        "horizon": meta.get("horizon"),
        "model": f"{meta.get('provider')}/{meta.get('model')}",
        "metrics": metrics,
        "run_dir": str(run_dir),
    }


def run_postprocess(json_path: str, out_dir: str | None = None) -> None:
    """Programmatic entrypoint used by rolling_agent_forecast.

    This mirrors the CLI behavior but takes explicit arguments.
    """
    json_path_obj = Path(json_path).resolve()
    meta, actual, bundle, predictions_df = load_rolling_from_results_json(json_path_obj)

    metric_names = [
        "mae",
        "rmse",
        "mape",
        "crps",
        "coverage_90",
        "coverage_95",
    ]
    metrics = evaluate_metrics(actual, bundle, metric_names)

    if out_dir is not None:
        run_dir = Path(out_dir).resolve()
    else:
        batch = meta.get("batch", "batch")
        provider = meta.get("provider", "model")
        start = meta.get("start_date")
        end = meta.get("end_date")
        root = json_path_obj.parents[2] / "experiments" / "results"
        run_dir = root / f"{batch}_{provider}" / f"simplified______{start}_{end}_v1"

    run_dir.mkdir(parents=True, exist_ok=True)

    pred_csv_path = run_dir / "predictions.csv"
    predictions_df.to_csv(pred_csv_path, index=False)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    create_all_visualizations(actual, bundle, predictions_df, metrics, meta.get("batch", "experiment"), plots_dir)

    summary = build_summary(meta, metrics, run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    args_json = {
        "disease": meta.get("disease"),
        "target": "weekly_lis",
        "start": meta.get("start_date"),
        "end": meta.get("end_date"),
        "n_steps": meta.get("requested_n_steps", 0),
        "horizon": meta.get("horizon", 1),
        "csv_path": meta.get("csv_path", ""),
        "evidence": None,
        "use_web": False,
        "gov_monthly_csv": "",
        "model": meta.get("model"),
        "provider": meta.get("provider"),
        "thinking_mode": "none",
        "temperature": meta.get("temperature"),
        "forecast_mode": meta.get("forecast_mode"),
        "llm_numbers": False,
        "strategist_pipeline": False,
        "no_llm": meta.get("no_llm", False),
        "save_json": True,
        "batch": meta.get("batch"),
    }
    (run_dir / "args.json").write_text(json.dumps(args_json, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Postprocess rolling results_json into plots + summary.")
    parser.add_argument("json_path", type=str, help="Path to results_json file")
    parser.add_argument("--out_dir", type=str, default=None, help="Experiment-style output directory (if omitted, infer from batch/model)")
    args = parser.parse_args()
    run_postprocess(args.json_path, args.out_dir)

    # CLI용 메시지
    print("Postprocess completed.")


if __name__ == "__main__":
    main()

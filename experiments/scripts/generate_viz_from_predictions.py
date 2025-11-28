"""
Generate visualizations and metrics from existing predictions.csv files.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.metrics import PredictionBundle, evaluate_metrics
from experiments.visualization import create_all_visualizations


def process_experiment_dir(experiment_dir: Path, experiment_label: str = None):
    """Process a single experiment directory."""
    predictions_file = experiment_dir / "predictions.csv"
    
    if not predictions_file.exists():
        print(f"❌ No predictions.csv found in {experiment_dir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Processing: {experiment_dir.name}")
    print(f"{'='*60}")
    
    # Load predictions
    predictions_df = pd.read_csv(predictions_file)
    predictions_df["as_of"] = pd.to_datetime(predictions_df["as_of"])
    predictions_df["date"] = pd.to_datetime(predictions_df["date"])
    
    # Get latest predictions for each date
    predictions_sorted = predictions_df.sort_values(["date", "as_of"])
    latest = predictions_sorted.groupby("date", as_index=False).tail(1).set_index("date").sort_index()
    
    print(f"Total predictions: {len(latest)}")
    print(f"Date range: {latest.index.min()} to {latest.index.max()}")
    
    # Create prediction bundle
    actual_series = latest["actual"].astype(float)
    point_series = latest["prediction"].astype(float)
    quantiles_bundle = {
        0.05: latest["q05"].astype(float),
        0.5: point_series,
        0.95: latest["q95"].astype(float),
    }
    bundle = PredictionBundle(point=point_series, quantiles=quantiles_bundle)
    
    # Calculate metrics
    metrics_summary = evaluate_metrics(
        actual_series,
        bundle,
        ["mae", "rmse", "mape", "crps", "coverage_95"],
    )
    
    print("\n📊 Metrics:")
    for metric, value in metrics_summary.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save metrics
    metrics_file = experiment_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n✅ Saved metrics to: {metrics_file}")
    
    # Read args.json to get model info
    args_file = experiment_dir / "args.json"
    if args_file.exists():
        with open(args_file, "r") as f:
            args = json.load(f)
            model_label = f"{args.get('provider', 'unknown')}/{args.get('model', 'unknown')}"
            disease = args.get('disease', 'HFMD')
    else:
        model_label = "unknown"
        disease = "HFMD"
    
    # Use custom label or default
    if experiment_label is None:
        if disease == "手足口病":
            disease = "HFMD"
        experiment_label = f"{disease} ({model_label})"
    
    # Create visualizations
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    try:
        create_all_visualizations(
            actual=actual_series,
            bundle=bundle,
            predictions_df=predictions_df,
            metrics=metrics_summary,
            experiment_name=experiment_label,
            output_dir=plots_dir,
        )
        print(f"✅ Saved visualizations to: {plots_dir}")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create summary.json
    summary = {
        "disease": disease,
        "n_steps": len(latest),
        "model": model_label,
        "metrics": metrics_summary,
        "run_dir": str(experiment_dir),
    }
    summary_file = experiment_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary to: {summary_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations and metrics from existing predictions.csv"
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Specific experiment directory to process",
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Process all experiments in a batch directory (e.g., baseline_hangzhou_qwen)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Process all experiment directories matching pattern (e.g., baseline_*)",
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Custom experiment label for visualization",
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parents[2]
    results_dir = base_dir / "experiments" / "results"
    
    experiment_dirs = []
    
    if args.dir:
        # Single directory
        exp_dir = Path(args.dir)
        if not exp_dir.is_absolute():
            exp_dir = results_dir / exp_dir
        experiment_dirs.append(exp_dir)
    
    elif args.batch:
        # All version directories in a batch
        batch_dir = results_dir / args.batch
        if batch_dir.exists():
            experiment_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]
        else:
            print(f"❌ Batch directory not found: {batch_dir}")
            return
    
    elif args.pattern:
        # All matching batch directories
        import glob
        pattern_path = str(results_dir / args.pattern)
        for batch_path in glob.glob(pattern_path):
            batch_dir = Path(batch_path)
            if batch_dir.is_dir():
                # Add all version directories in this batch
                experiment_dirs.extend([d for d in batch_dir.iterdir() if d.is_dir()])
    
    else:
        print("❌ Please specify --dir, --batch, or --pattern")
        parser.print_help()
        return
    
    if not experiment_dirs:
        print("❌ No experiment directories found")
        return
    
    print(f"\n🔍 Found {len(experiment_dirs)} experiment(s) to process")
    
    success_count = 0
    for exp_dir in sorted(experiment_dirs):
        if process_experiment_dir(exp_dir, args.label):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully processed {success_count}/{len(experiment_dirs)} experiments")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

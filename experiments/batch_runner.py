from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .run_experiment import ExperimentRunner
from .utils import load_yaml


LOGGER = logging.getLogger("experiments.batch_runner")


@dataclass(slots=True)
class JobSpec:
    name: str
    config_path: Path
    overrides: List[str] = field(default_factory=list)
    n_runs: int = 1
    seeds: List[int] = field(default_factory=list)
    batch_name: Optional[str] = None
    enabled: bool = True


class BatchRunner:
    """Manifest 기반으로 여러 실험을 일괄 실행하는 도우미."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        debug: bool = False,
        verbose: bool = False,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        self.manifest_dir = self.manifest_path.parent
        self.debug = debug
        self.verbose = verbose

        manifest = load_yaml(self.manifest_path)
        self.batch_settings: Dict[str, object] = manifest.get("batch", {}) or {}
        self.global_overrides: List[str] = list(manifest.get("global_overrides", []) or [])
        self.stop_on_error = bool(self.batch_settings.get("stop_on_error", False))
        self.default_batch = self.batch_settings.get("default_batch_name") or self.batch_settings.get("name")

        summary_root = Path(self.batch_settings.get("summary_dir", "experiments/results/batch_runs"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_stem = self.manifest_path.stem
        self.summary_dir = summary_root / f"{manifest_stem}_{timestamp}"
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.summary_dir / "batch_summary.json"

        experiments_cfg = manifest.get("experiments")
        if not experiments_cfg:
            raise ValueError("Manifest must include at least one experiment under 'experiments'.")
        self.jobs: List[JobSpec] = self._parse_jobs(experiments_cfg)

    def _parse_jobs(self, experiments_cfg: Iterable[Dict[str, object]]) -> List[JobSpec]:
        jobs: List[JobSpec] = []
        for exp_cfg in experiments_cfg:
            if not isinstance(exp_cfg, dict):
                raise ValueError("Each experiment entry must be a mapping.")
            enabled = exp_cfg.get("enabled", True)
            name = str(exp_cfg.get("name") or Path(str(exp_cfg.get("config", ""))).stem)
            config_raw = exp_cfg.get("config")
            if not config_raw:
                raise ValueError(f"Experiment '{name}' is missing the 'config' field.")
            config_path = self._resolve_path(str(config_raw))
            overrides = list(exp_cfg.get("overrides", []) or [])
            n_runs = int(exp_cfg.get("n_runs") or exp_cfg.get("runs") or 1)
            seeds_raw = exp_cfg.get("seeds") or []
            seeds = [int(s) for s in seeds_raw]
            batch_name = exp_cfg.get("batch") or self.default_batch
            jobs.append(
                JobSpec(
                    name=name,
                    config_path=config_path,
                    overrides=overrides,
                    n_runs=n_runs,
                    seeds=seeds,
                    batch_name=batch_name,
                    enabled=bool(enabled),
                )
            )
        return jobs

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (self.manifest_dir / path).resolve()
        return path

    def run(
        self,
        *,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
        dry_run: bool = False,
    ) -> List[Dict[str, object]]:
        include_set = {name for name in include} if include else None
        exclude_set = {name for name in exclude} if exclude else set()

        selected_jobs = [
            job
            for job in self.jobs
            if job.enabled and (not include_set or job.name in include_set) and job.name not in exclude_set
        ]
        if not selected_jobs:
            LOGGER.warning("No experiments selected for execution. Check include/exclude filters.")
            return []

        LOGGER.info("Executing %d experiments (dry_run=%s)", len(selected_jobs), dry_run)
        sys.stdout.flush()  # 즉시 출력
        results: List[Dict[str, object]] = []
        for job in selected_jobs:
            LOGGER.info("-> %s | config=%s | runs=%d", job.name, job.config_path, job.n_runs)
            sys.stdout.flush()  # 즉시 출력
            if dry_run:
                continue

            for run_idx in range(job.n_runs):
                run_label = f"{job.name} (run {run_idx + 1}/{job.n_runs})"
                overrides = list(self.global_overrides) + list(job.overrides)
                seed_value = job.seeds[run_idx] if run_idx < len(job.seeds) else None
                if seed_value is not None:
                    overrides.append(f"experiment.random_seed={seed_value}")

                LOGGER.info("   ▶ %s | overrides=%s", run_label, overrides if self.verbose else len(overrides))
                sys.stdout.flush()  # 즉시 출력
                try:
                    runner = ExperimentRunner(
                        config_path=job.config_path,
                        overrides=overrides,
                        debug=self.debug,
                        batch_name=job.batch_name,
                    )
                    runner.setup()
                    summary = runner.run()
                    run_record = {
                        "experiment": job.name,
                        "run_index": run_idx + 1,
                        "seed": seed_value,
                        "config": str(job.config_path),
                        "batch_name": job.batch_name,
                        "overrides": overrides,
                        "status": "success",
                        "run_dir": summary.get("run_dir"),
                        "metrics": summary.get("metrics"),
                    }
                    results.append(run_record)
                    LOGGER.info("   ✅ %s finished (run_dir=%s)", run_label, summary.get("run_dir"))
                    sys.stdout.flush()  # 즉시 출력
                except Exception as exc:  # pragma: no cover - defensive path
                    run_record = {
                        "experiment": job.name,
                        "run_index": run_idx + 1,
                        "seed": seed_value,
                        "config": str(job.config_path),
                        "batch_name": job.batch_name,
                        "overrides": overrides,
                        "status": "failed",
                        "error": str(exc),
                    }
                    results.append(run_record)
                    LOGGER.exception("   ❌ %s failed: %s", run_label, exc)
                    sys.stdout.flush()  # 즉시 출력
                    if self.stop_on_error:
                        LOGGER.error("Stop on error is enabled. Halting batch execution.")
                        self._write_summary(results)
                        raise

        if not dry_run:
            self._write_summary(results)
        return results

    def _write_summary(self, results: List[Dict[str, object]]) -> None:
        payload = {
            "manifest": str(self.manifest_path),
            "created_at": datetime.now().isoformat(),
            "results": results,
        }
        with self.summary_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
        LOGGER.info("Batch summary saved to %s", self.summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple experiments from a manifest.")
    parser.add_argument("--manifest", required=True, help="YAML manifest that lists experiments to run.")
    parser.add_argument("--include", nargs="*", help="Subset of experiment names to run.")
    parser.add_argument("--exclude", nargs="*", help="Experiment names to skip.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing experiments.")
    parser.add_argument("--debug", action="store_true", help="Forward debug flag to ExperimentRunner.")
    parser.add_argument("--verbose", action="store_true", help="Show override details per run.")
    return parser.parse_args()


def main() -> None:
    # PYTHONUNBUFFERED 환경변수 설정 (버퍼링 비활성화)
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    logging.basicConfig(
        level=logging.INFO, 
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )
    
    args = parse_args()
    runner = BatchRunner(
        manifest_path=Path(args.manifest),
        debug=args.debug,
        verbose=args.verbose,
    )
    try:
        runner.run(include=args.include, exclude=args.exclude, dry_run=args.dry_run)
    except Exception as exc:  # pragma: no cover - CLI surface
        LOGGER.error("Batch execution failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()


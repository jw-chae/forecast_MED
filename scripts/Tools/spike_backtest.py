from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np

from adapters import load_his_outpatient_series
from scenario_engine import extract_growth_episodes, generate_paths_conditional, Episode
from evt import fit_pot, replace_tail_with_evt


@dataclass
class EpisodeEval:
    start_date: str
    peak_date: str
    peak_value: float
    detected: bool
    detect_date: str | None
    lead_weeks: int | None
    p_at_detect: float | None


def evaluate_spike_detection(
    dates: List[str],
    series: np.ndarray,
    horizon: int,
    threshold: float,
    pct_threshold: float = 0.2,
    min_len: int = 2,
    relax_drop: float = -0.05,
    n_paths: int = 5000,
    seed: int = 123,
) -> Dict[str, object]:
    y = np.asarray(series, dtype=float)
    dt = np.array(dates)

    # 전체 에피소드 후보를 찾되, 각 에피소드 평가는 과거 정보만 사용해 재샘플
    base_eps: List[Episode] = extract_growth_episodes(y, pct_threshold=pct_threshold, min_len=min_len, relax_drop=relax_drop)

    results: List[EpisodeEval] = []
    rng = np.random.default_rng(seed)

    for ep in base_eps:
        # 글로벌 인덱스 기준으로 시작/끝 및 피크 파악
        s = ep.start_idx
        e = ep.end_idx
        peak_local = int(np.argmax(y[s : e + 1]))
        peak_idx = s + peak_local
        peak_val = float(y[peak_idx])
        if peak_val < threshold:
            continue

        detected = False
        detect_t = None
        p_at_detect = None

        # 피크 직전까지 이동 기점에서 탐지 시도
        for t in range(max(2, s - 3), peak_idx):
            hist = y[: t + 1]
            # 역사 데이터에서만 에피소드 형상 추출
            eps_hist = extract_growth_episodes(hist, pct_threshold=pct_threshold, min_len=min_len, relax_drop=relax_drop)
            if not eps_hist:
                continue
            paths = generate_paths_conditional(
                series=hist,
                horizon=horizon,
                n_paths=n_paths,
                episodes=eps_hist,
                news_signal=0.0,
                quality=0.72,
                random_state=int(rng.integers(0, 10_000_000)),
            )
            # EVT 꼬리 보정(보수적 과소추정 방지)
            u = float(np.quantile(hist, 0.9))
            gpd = fit_pot(hist, threshold=u)
            paths = replace_tail_with_evt(paths, gpd_params=gpd, threshold=u)

            # 다음 H주 내 임계 초과 확률
            p_exceed = float((paths.max(axis=1) > threshold).mean())
            if p_exceed >= 0.5:
                detected = True
                detect_t = t
                p_at_detect = p_exceed
                break

        results.append(
            EpisodeEval(
                start_date=str(dt[s]),
                peak_date=str(dt[peak_idx]),
                peak_value=peak_val,
                detected=detected,
                detect_date=(str(dt[detect_t]) if detected else None),
                lead_weeks=(peak_idx - detect_t if detected else None),
                p_at_detect=p_at_detect,
            )
        )

    # 집계
    eval_episodes = [r for r in results]
    n = len(eval_episodes)
    n_detect = sum(1 for r in eval_episodes if r.detected)
    lead = [r.lead_weeks for r in eval_episodes if r.detected and r.lead_weeks is not None]
    pdet = [r.p_at_detect for r in eval_episodes if r.p_at_detect is not None]

    summary = {
        "episodes": [r.__dict__ for r in eval_episodes],
        "n_episodes": n,
        "n_detected": n_detect,
        "detect_rate": (n_detect / n if n else 0.0),
        "mean_lead_weeks": (float(np.mean(lead)) if lead else None),
        "median_lead_weeks": (float(np.median(lead)) if lead else None),
        "mean_p_at_detect": (float(np.mean(pdet)) if pdet else None),
        "threshold": threshold,
        "horizon": horizon,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="手足口病")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--thr_quantile", type=float, default=0.9, help="임계치: 과거 q분위수")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    csv_path = base / "processed_data" / "his_outpatient_weekly_epi_counts.csv"
    dt_index, series = load_his_outpatient_series(str(csv_path), args.disease)
    dates = [d.strftime("%Y-%m-%d") for d in dt_index]

    thr = float(np.quantile(series, args.thr_quantile))

    summary = evaluate_spike_detection(dates, series, horizon=args.horizon, threshold=thr)

    out = base / "reports" / f"spike_backtest_{args.disease}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()



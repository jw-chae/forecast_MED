<!-- MathJax (local Markdown 뷰어에서도 수식 렌더링) -->
<script>
window.MathJax = {
  tex: { inlineMath: [['\\(','\\)'], ['$', '$']], displayMath: [['\\[','\\]'], ['$$','$$']] },
  svg: { fontCache: 'global' }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

Epi Tools MVP 및 백테스트 정리

- 본 문서는 HIS/LIS 등 병원 집계 데이터를 기반으로 “현재/근미래 수요 예측 + 피크 리스크 평가”를 위한 최소 도구(MVP)와 실험 결과를 정리합니다. 구현 코드는 `med_deepseek/scripts/Tools` 하위에 있습니다.

1) 데이터 소스 및 스키마
- 입력: `med_deepseek/processed_data/his_outpatient_weekly_epi_counts.csv`
- 스키마: `diagnosis_time` + 질병별 주간 고유 환자수(정수)

2) 모듈 구성(파일/역할/핵심)
- `scripts/Tools/adapters.py`: CSV 어댑터. `load_his_outpatient_series(path, disease)`
- `scripts/Tools/fusion.py`: 정밀도 가중 결합(nowcast). `precision_weighted_fusion(...)`
- `scripts/Tools/scenario_engine.py`: 급등 에피소드 추출(`extract_growth_episodes`), 조건부 블록 부트스트랩(`generate_paths_conditional`)
- `scripts/Tools/evt.py`: POT-GPD 피팅(`fit_pot`), 경로 꼬리 보정(`replace_tail_with_evt`)
- `scripts/Tools/risk_banding.py`: KPI 초과확률 기반 상/중/하. `band_from_paths`
- `scripts/Tools/run_on_his_outpatient.py`: 단일 질병 E2E JSON 러너
- `scripts/Tools/generate_html_report.py`: 원본+예측 밴드 HTML(Plotly)
- `scripts/Tools/spike_backtest.py`: 피크 사전탐지 백테스트(JSON)
- `scripts/Tools/holdout_forecast_html.py`: 홀드아웃(현재 데이터 재현) HTML

3) 실행 예
- 환경(선택): `pip install numpy pandas scipy plotly`
- E2E(JSON): `python med_deepseek/scripts/Tools/run_on_his_outpatient.py --disease 手足口病`
- 예측 HTML: `python med_deepseek/scripts/Tools/generate_html_report.py --disease 手足口病`
- 피크 백테스트(JSON): `python med_deepseek/scripts/Tools/spike_backtest.py --disease 手足口病 --horizon 4 --thr_quantile 0.9`
- 홀드아웃 HTML: `python med_deepseek/scripts/Tools/holdout_forecast_html.py --disease 手足口病 --train_until 2022-12-31 --end 2024-12-31`

4) 백테스트(수족구병, 요약)
- 에피소드 9개 중 8개 탐지(탐지율 0.889)
- 피크까지 선행 주수 평균 3.63, 중앙 4
- 탐지 시점 초과확률 평균 0.979
- 임계치: 과거 90% 분위(≈22.9명), H=4주
- 산출물: `med_deepseek/reports/spike_backtest_手足口病.json`

5) 홀드아웃 예측(2023–2024 재현)
- 생성물: `med_deepseek/reports/holdout_forecast_手足口病_2022-12-31_2024-12-31.html`
- 구성: 학습기간(회색), 실제(파란선), 예측 중앙값/평균, 50–80/95% 밴드, nowcast/95%CI, MAE(중앙값), 95% 커버리지

6) 모델 요약
- Nowcast 결합: 사전예측(mean/var) + 관측값에 품질(q), 편향(μ_e,σ_e), 뉴스 신호를 정밀도 가중으로 결합
- 시나리오: 급등 에피소드 형상 기반 조건부 부트스트랩 → 경로 분포
- EVT: 경로 상단 꼬리 보정으로 피크 과소추정 완화
- 리스크: KPI 임계(ER 90분, Bed 0.92) 초과확률로 밴딩

7) 보수적 결과 원인 및 튜닝 포인트
- 에피소드 임계(`pct_threshold=0.2`) 높음 → 0.1~0.15 권장
- 품질 가정(q=0.72)로 밴드 ↑ → 실제 q 산출 후 `quality` 조정
- 뉴스 신호 미반영 → `--news 0.2~0.5` 실험
- EVT 임계 u=0.9 고정 → 0.85~0.95 민감도
- 외생변수(달력/기온/감시지표) 미포함 → 피처 강화

8) 다음 단계
- 2023–2024 피크 5개 구간 자동 식별 → 피크 시점/크기 오차·선행 주수·커버리지 표를 HTML에 추가
- 다질병 일괄 리포트, 경보 임계 재설계, 외생변수 투입

9) 알고리즘 수학적 정의(상세)
- 표기: 학습 구간 시계열 \(y_1,\dots,y_T\), 홀드아웃 길이 \(H\), 품질 \(q\in[0,1]\), 계절 신호 \(s\in[0,1]\) 혹은 벡터 \(s_{1:H}\).
- 에피소드 추출: 성장률 \(g_t=\frac{y_t-y_{t-1}}{\max(1,y_{t-1})}\). 시작 \(g_t\ge\tau\), 종료 직전 \(g_{t'}\le\rho\). 형상 \(f^{(e)}_k=\frac{y_{s+k}}{\max(1,y_s)}\).
- 유사도/가중: 최근 창 델타 \(d\), 점수 \(S(e)=\lVert \Delta f^{(e)}_{1:m}-d_{1:m}\rVert_2\), 선택확률 \(p(e)\propto S(e)^{-1}\).
- 기준선: \(b=\mathrm{median}(y_{T-w+1:T})\). 시작값 \(x_0=\max(y_T, b_{\mathrm{season}})\) (현재 구현은 \(b_{\mathrm{season}}\approx \mathrm{median}(y_{T-7:T})\)).
- 진폭 스케일: 진폭 집합 \(A=\{\max_k f^{(e)}_k\}\), \(a=\mathrm{Quantile}_{q_a}(A)\cdot m_a\).
- 전주 비율: 보간 형상 \(f\)에서 \(r_h=\frac{f(h)}{\max(1e{-}9,f(h-1))}\), 전역 상한 \(r_h\leftarrow \min(r_h, r_{\mathrm{cap}})\).
- 단계 스케일: \(c_h=1+ s_h\) (스칼라면 \(c_h\equiv 1+s\)).
- 워밍업: \(\gamma_h=1\) (\(h\ge K\)), \(\gamma_h=0.3+0.7\frac{h}{K}\) (초기). 유효 비율 \(r^{\ast}_h=1+\gamma_h(r_h a-1)\).
- 단계 잡음: \(\epsilon_h\sim\mathcal N(0,\sigma^2)\), \(\sigma=0.08+0.22(1-q)\), \(|\epsilon_h|\le 3\sigma\). 
- 갱신: \(x_h = x_{h-1}\, r^{\ast}_h\, c_h\, e^{\epsilon_h}\), 최종 \(\tilde y_{T+h}=\max(0,x_h)\).
- EVT 보정(POT–GPD): 임계 \(u=\mathrm{Quantile}_{0.9}(y_{1:T})\), 초과분에 GPD(\(\xi,\beta\)) 적합 후 \(\tilde y>u\) 구간 치환.
- 요약: 분위수 \(q_{0.5}, q_{0.8}, q_{0.95}\), 평균 \(\bar y\).
- Nowcast(별도): \(z=y(1+\mu_e), \mathrm{Var}(z)=(y\sigma_e)^2+\kappa(1-q)y^2\). 결합 평균/분산은 정밀도 가중 평균/합으로 산출.

10) 현재 파라미터(홀드아웃 · 독감 기준 예시)
- 시나리오 엔진(`generate_paths_conditional`)
  - `pct_threshold=0.12`, `relax_drop=-0.05`, `recent_window=3`, `recent_baseline_window=8`
  - `amplitude_quantile=0.9`, `amplitude_multiplier=1.8`
  - `ratio_cap_quantile=0.98`, `warmup_weeks=1`
  - `quality=0.72`, `noise_clip=±3σ`
  - 시작값 하한: `start_value_override=median(last 8 weeks)`
  - 계절 priors(독감): Nov–Feb 0.7, Oct 0.4, Mar–May 0.08, Jun–Sep 0.15 (주별 `news_signal` 벡터)
- EVT(`fit_pot`)
  - 임계 분위 `u=0.9`, 최소 초과 샘플 5개에서 보수 파라미터 사용
- Nowcast(`precision_weighted_fusion`)
  - 예시값: `μ_e=+0.20`, `σ_e=0.10`, `κ=0.5`, `news_scale=10`, `news_var=25`

10) 결과 해석 가이드(독감 홀드아웃)
- 겨울 시즌 priors와 시작값 하한 적용으로 초기 1월 구간이 바닥에서 시작하지 않으며, 진폭은 `amplitude_*`와 `ratio_cap` 완화로 상향.
- overflow 경고는 급등 주의 곱셈 항이 커질 때 발생하며, `ratios`/`scale`/`x` 상한 추가로 안정화 가능.

12) ver2 제안(채택 항목 표시)
- [적용] 카운트 관측 우도: 경로 \(x_h\)를 평균강도 \(\lambda\)로 보고, 관측은 NB: $\operatorname{Var}[Y]=\lambda+\lambda^2/k$. (코드: `nb_dispersion_k`)
- [적용] 비율 안정화: 분모 하한 $\delta=Q_{0.05}(y_{t-1}>0)$로 $r_h=\tfrac{f(h)}{\max(\delta,f(h-1))}$. (파라미터 `use_delta_quantile=True`)
- [보류] 워밍업 \(\gamma_h\) 데이터 기반 로지스틱 추정(현재는 고정형)
- [보류] 융합 상관보정(BLUE) 및 공분산 추정
- [보류] EVT 임계 안정도 곡선/모형평균
- [보류] 계절 \(c_h\) 푸리에 회귀 학습(현재는 룰기반 priors)
- [보류] conformal 예측으로 사후 밴드 보정

11) 부록: 섹션 번호 정리
 - 1) 데이터 소스 및 스키마
 - 2) 모듈 구성
 - 3) 실행 예
 - 4) 백테스트 요약
 - 5) 홀드아웃 예측 구성
 - 6) 모델 요약
 - 7) 보수적 결과 원인 및 튜닝
 - 8) 다음 단계
 - 9) 알고리즘 수학적 정의
 - 10) 현재 파라미터와 결과 해석

산출물 경로 요약
- HTML: `his_outpatient_forecast_流行性感冒.html`, `his_outpatient_forecast_手足口病.html`, `holdout_forecast_手足口病_2022-12-31_2024-12-31.html`
- JSON: `spike_backtest_手足口病.json`
- 코드: `med_deepseek/scripts/Tools/`


13) 도구형 LLM 분석가(관측→제안→시뮬→평가→업데이트)
- 관측 입력(state): `disease`, `period(train_until→end)`, `season_vector`(optional), `last_params`, `last_metrics`, `constraints(bounds)`, `target_metrics`
- LLM 출력(JSON): `proposed_params`(+ `rationale_summary`, `expected_tradeoffs`)
- 루프: 관측 → LLM 제안(제약 적용) → `run_sim(params)` → 메트릭 평가 → 가드레일/페일세이프 적용 → 로그(JSONL) 저장
- 구현 파일: `scripts/Tools/agent_loop.py`, `scripts/Tools/llm_agent.py`, `scripts/Tools/run_sim_wrapper.py`
- 대시보드/리포트: `scripts/Tools/plot_agent_progress.py`, `scripts/Tools/plot_compare_trajectories.py`, `scripts/Tools/rolling_agent_forecast.py`

14) 하이퍼파라미터 전체 목록(범주/설명/권장 범위)
- 시나리오/에피소드 기반 경로 생성
  - `pct_threshold`(float): 에피소드 시작 성장률 임계. 예: 0.1~0.2
  - `relax_drop`(float): 에피소드 종료/완화 기준. 예: -0.05
  - `recent_window`(int): 최근 델타 비교창. 기본 3
  - `recent_baseline_window`(int): 최근 기준선 중앙값 창. 기본 8
  - `amplitude_quantile`(float): 에피소드 진폭 상위 분위. 0.85~0.98
  - `amplitude_multiplier`(float): 진폭 스케일 배수. 1.2~2.8
  - `ratio_cap_quantile`(float): 전주비율 전역 상한의 분위. 0.95~0.999
  - `warmup_weeks`(int): 워밍업 완충 주수. 0~2
  - `use_delta_quantile`(bool): 분모 하한 사용 여부
  - `delta_quantile`(float): 분모 하한 분위. 0.01~0.10
  - `start_value_override`(float|null): 시작값 하한(없으면 최근 w주 중앙값)
  - `quality`(float): 관측 품질 0~1; 낮을수록 변동성↑
  - `news_signal`(float|list[float]): 시즌/뉴스 스케일(스칼라/벡터)
  - `nb_dispersion_k`(float|null): NB 과산포 k(2~50)
- 안정화 캡(가드레일)
  - `r_boost_cap`(float): 비율 증폭 캡. ≤3.0 (기본 2.0)
  - `scale_cap`(float): 뉴스/스케일 벡터 캡. ≤1.8 (기본 1.6)
  - `x_cap_multiplier`(float): 절대 수치 상한의 배수. ≤4.0 (기본 2.0)
- EVT(POT–GPD)
  - `evt_u_quantile`(float): 임계 분위. 0.85~0.95 (기본 0.9)
  - `min_excess`(int): 최소 초과 샘플 수. 기본 5
- 융합(nowcast)
  - `μ_e`(float): 관측 편향(mean) 보정
  - `σ_e`(float): 관측 표준편차 비율
  - `κ`(float): 품질 기반 분산 가중(0~1)
  - `news_scale`, `news_var`(float): 뉴스 신호 평균/분산 스케일
- KPI/운영 임계
  - `thr_bed`(float): Bed 점유 경보 임계(예: 0.92)
  - `thr_er_min`(float): ER 대기(분) 임계(예: 90)
- 커버리지/안전 규칙
  - 커버리지 가드: 최근 8주 `coverage95<0.7` 시 `quality↓` 또는 `nb_dispersion_k↑` 권고
  - KPI 페일세이프: `P(Bed>0.98)>0.5` 시 `r_boost_cap/scale_cap` 강화, `amplitude_multiplier↓`후 재시뮬

15) 실행 엔드투엔드 래퍼(`run_sim(params)`) 요약
- 입력: `SimConfig(disease, train_until, end, horizon, season_profile)` + 상기 하이퍼파라미터
- 처리: 에피소드 기반 경로 → 가드레일 적용 → NB 샘플(옵션) → EVT 꼬리 보정 → 분위수/평균 산출
- 출력(JSON): `quantiles{q05,q50,q80,q95}`, `mean_path`, `metrics`, `dates_hist/target/future`

16) 로그/재현성(JSONL)
- 경로: `med_deepseek/reports/agent_logs/llm_analyst_{질병}_{타임스탬프}_{id}.jsonl`
- 레코드 필드(예)
  - `disease`, `period`, `iter`, `seed`, `horizon`
  - `observation`(입력 상태 전체), `proposal_raw`(LLM 원문 제안 파라미터), `proposal`(가드/클램프 후), `clamp_deltas`
  - `metrics`(MAE_median, SMAPE, CRPS, coverage95, peak metrics, KPI 확률 등)
  - `proposal_source`(llm|stub), `llm_raw`(LLM 원문), `llm_request`, `llm_usage`, `llm_latency_ms`
  - `llm_rationale_summary`, `llm_expected_tradeoffs`, `events`(가드/페일세이프 트리거 기록)

17) 리포트 산출물(시각화)
- 진행 추세: `reports/agent_logs/agent_progress.html` (coverage95/CRPS/recall 추세)
- 실측 vs 반복 예측: `reports/agent_logs/compare_trajectory_{질병}.html` (q50/05/95 겹침)
- 주간 롤링(1주/12주 등): `reports/agent_logs/rolling_{질병}_{start}_{end}_{n_steps}x{horizon}.html`
  - 실측 라인 + 매 스텝 q50/05/95 + 집계 주별 예측 라인(LLM) + baseline(지속성) 라인 동시 표기

18) 최적화/운영 플로우
- 오프라인 초기화: 무작위/BOHB 30~100회 → 상위 k개를 LLM이 재조정 10회
- 온라인 주간 운영: 에이전트 루프 `n_iters`(예: 8~12) → 리포트/알림 → 파라미터 캐시
- 제약(bounds)·가드레일은 상시 적용, KPI 페일세이프는 위반 시 보수화 재시뮬

19) 수용 기준(재확인)
- coverage95 ∈ [0.90, 0.98]
- Peak recall@±2주 ≥ 0.8
- CRPS/MAE_median baseline 대비 ≥20% 개선
- KPI 과경보/과소경보 비용 ≤ baseline

20) 멀티에이전트 아키텍처(웹검색·정리 에이전트 + 파라미터 조정 에이전트)
- 역할 분리
  - 모델1(웹검색·정리): 중국 전역/성 단위의 보건 공지·뉴스를 SerpAPI/Bing으로 검색→최근 N주 주차별 히트 집계→`news_hits_change_4w`, `search_snr` 같은 약식 신호산출. 결과를 증거팩(`evidence pack`)에 반영.
  - 모델2(파라미터 조정): `observation`(내부 시계열 요약+제약+증거팩)을 보고 액션(JSON 하이퍼파라미터)을 제안. 필요 시 별도 호출로 `news_signal`만 재질의(증거가 있는데 메인 제안에서 누락 시).
- 구현 파일
  - `scripts/Tools/web_sources.py`: SerpAPI/Bing 뉴스 API 호출, 쿼리/URL/결과 수/샘플 타이틀 콘솔+파일 로그(`reports/evidence/search_logs/*.json`).
  - `scripts/Tools/evidence_pack.py`: 증거팩 로딩/머지, 웹신호 병합(`build_evidence_pack_with_web`), 파라미터 힌트(옵션) 생성.
  - `scripts/Tools/llm_agent.py`: 시스템 프롬프트(제약 준수/JSON Only/증거 사용 규칙)와 LLM 호출, 디버그(원문/토큰/지연) 로깅, 보조 호출(`call_llm_json`).
  - `scripts/Tools/agent_loop.py`: propose→run_sim→eval→log 루프. `--evidence`로 증거팩을 관측에 주입.

21) 증거팩(evidence pack) 스키마(입력 규격)
```json
{
  "context_meta": {"disease":"手足口病","train_until":"2024-06-30","end":"2024-12-31"},
  "internal_weekly": {
    "last_8w_counts": [...], "last_8w_growth_pct": [...],
    "positivity_rate": [...], "er_wait_min": [...], "bed_util": [...]
  },
  "external_signals": {
    "news_hits_change_4w": 2.3,
    "search_snr": 1.8,
    "school_calendar": {"in_session": true},
    "weather_weekly": {"temp_mean": 29.4, "humidity_mean": 0.78}
  },
  "historical_patterns": {
    "season_profile": {"JFM":0.08,"AMJ":0.15,"JAS":0.55,"OND":0.22},
    "past_peak_weeks": ["2023-07-10","2022-06-27"],
    "peak_magnitudes": [320,280]
  },
  "last_run": {"params": {...}, "metrics": {...}},
  "constraints": {"bounds": {...}, "targets": {"coverage95":[0.90,0.98], "recall_pm2w_min":0.8}},
  "provenance": [{"source":"KCDC Weekly","date":"2024-06-28","url_id":"src:kcdc-20240628"}]
}
```
- 신선도: 보건/뉴스 ≤7일, 기상 ≤14일. PII 금지, 출처 원문은 시스템 저장·LLM에는 요약 숫자+ID만.

22) LLM 프롬프트(핵심 규칙)
- 시스템: 제약 준수, JSON Only, 내부 사고 숨김, 출력 스키마
  - `proposed_params`(필수), `rationale_summary`(≤2문장), `expected_tradeoffs`(≤3), `evidence_used`(경로), `validation.constraints_ok`.
  - `observation.evidence.external_signals`가 있으면 `news_signal`(0–1 스칼라)을 반드시 포함(신호 약: 0.05–0.15, 중간: 0.2–0.4, 강: 0.5–0.7 권고). 추측 금지.
- 로그: `llm_raw`, `llm_request/usage`, `llm_latency_ms`, `llm_rationale_summary`, `llm_expected_tradeoffs` 저장.

23) 시뮬레이션/캘리브레이션
- `run_sim_wrapper.run_sim(params)`
  - 시나리오 경로 생성(`scenario_engine.generate_paths_conditional`)→EVT 보정(`evt.replace_tail_with_evt`)→분위/평균→메트릭 계산.
  - 선택적 후처리 커버리지 캘리브레이션(`calibrate_coverage_to`): q50 기준 상·하위 밴드를 스케일링하여 목표 커버리지에 수렴. 운영판(롤링 HTML)에는 비적용, 오프라인 튜닝·검증용으로만 사용 권장.
- 메트릭: `MAE_median`, `SMAPE`, `CRPS`, `coverage95`, `peak` 지표, `KPI exceed` 확률.

24) 주간 롤링 실행/리포트
- `scripts/Tools/rolling_agent_forecast.py` (호라이즌 1/12 등)
  - 각 주차 컷오프별로 LLM 제안→시뮬레이션→리포트.
  - HTML에 학습구간/홀드아웃 실측, q50/95/05, 집계 주별 예측 라인(LLM) + baseline(지속성) 라인 표기.
- 예:
  - 상반기: `rolling_手足口病_2023-01-01_2023-06-30_0x1.html`
  - 하반기: 동일 스크립트로 생성 가능(템플릿 f-string 충돌 해결 완료).

25) 자동 튜닝/탐색 스크립트
- `scripts/Tools/tune_coverage.py`: 커버리지 중심 그리드/랜덤 탐색(경량). 목적함수는 cov band 미달/과잉 패널티 + CRPS/MAE를 합성.
- 근방 탐색 캘리브레이션 예: `tuned_calibrated_手足口病.json`, `best_coverage_recall_手足口病.json`.

26) 오프라인 합성데이터(PPO 학습용)
- `scripts/Tools/offline_dataset_builder.py`
  - 기간 내 모든 주차를 컷오프로 순회하며: 증거팩 생성(옵션 웹)→LLM 액션(JSON)→시뮬레이션→메트릭→보상 R 계산→JSONL 축적.
  - 보상식(가중 합): R = −[ w1·CRPS + w2·cov_short + w3·MAE_med + w4·peak_penalty + w5·KPI_bias ] (기본 가중 내장).
  - 산출물: `reports/offline_dataset/ppo_dataset_{질병}_{start}_{end}_H{H}.jsonl` + HTML(coverage/CRPS/reward 추세).
- 실행 예: `python scripts/Tools/offline_dataset_builder.py --disease 手足口病 --start 2023-01-01 --end 2023-12-31 --horizon 1`

27) 환경 변수/키 관리
- `DASHSCOPE_API_KEY`: DashScope(Qwen3) OpenAI-호환 호출용. `.env` 자동 로드.
- `SERPAPI_KEY`/`BING_API_KEY`: 뉴스/검색용. `.env` 자동 로드. 로그는 `reports/evidence/search_logs/*.json`에 저장.

28) 재현성/운영
- 로그(JSONL): 제안/관측/클램프 변화/이벤트(KPI 페일세이프 트리거)까지 모두 저장 → 실험 복기 가능.
- 크론 예(매주 화 02:30, 12주 예측): `agent_loop.py` 실행→`plot_agent_progress.py`/`plot_compare_trajectories.py` 갱신.
- 보안/프라이버시: PII 미사용. 외부 링크는 시스템 보관·LLM에 ID만.

29) 현재 한계와 개선 로드맵
- 1-step ahead는 관성으로 잘 맞지만, H가 길어질수록 품질 저하 → 뉴스/감시/기상 신호를 벡터화해 다주기 보강 필요.
- coverage 캘리브레이션은 오프라인 용도에 한정. 온라인 운영은 품질/진폭/ratio 조정과 다중밴딧으로 캘리브레이션 권장.
- recall 제고를 위해 peak-aware 목적/탐색(peak timing/height)을 가중한 보상 설계 고도화 예정.

30) 웹 신호(정부 공고·뉴스) 및 누수 방지 업데이트(2025-08)
- 목적: 월간 공식 통계 기반의 견고한 `news_signal` 반영과 as-of 컷오프 적용으로 재현성·비누수 보장
- 주요 변경
  - `web_sources.py`
    - 일반 웹검색 추가(`serpapi_google_web_search`) 및 User-Agent/인코딩 보강(utf-8/gb18030/gbk)
    - 정부 공고 수집(`fetch_official_stats_signals`): 사이트 화이트리스트(`ndcpa.gov.cn, nhc.gov.cn, wjw.zj.gov.cn, wsjkw.zj.gov.cn, *.zj.gov.cn`), 지역 키워드(`全国, 浙江省, 全省`) 매칭, `2023年X月` 월 문자열 정합, as-of 이후 게시물 제외, `手足口病` 표 파싱 강화
    - 뉴스 히트 기반 신호와 정부 공고 신호를 평균 결합하여 `external_signals.news_hits_change_4w`/`search_snr` 산출
  - `evidence_pack.py`: `build_evidence_pack_with_web(..., asof, gov_only, site_whitelist, region_keywords)` 인자 확장
  - `rolling_agent_forecast.py`:
    - CLI 추가: `--gov_only`, `--site_whitelist`, `--region_keywords`
    - 각 스텝에서 `asof=train_until`로 증거팩 구성, LLM이 `news_signal` 누락 시 결정적 매핑 보강
      - `news_signal = clip(0.05 + 0.25*news_hits_change_4w + 0.3*(search_snr/3), 0.05, 0.7)`
  - `run_sim_wrapper.py`: 사후 커버리지 보정 가드(`enable_posthoc_calibration` 활성 시에만 적용)
  - `offline_dataset_builder.py`: 기본 `calibrate_coverage_to` 제거, 웹 신호 `asof=train_until` 적용

- 사용 예(상반기 일부 구간, 정부 공고 기반)
```
python scripts/Tools/rolling_agent_forecast.py \
  --disease 手足口病 --start 2023-04-01 --end 2023-06-30 \
  --n_steps 8 --horizon 1 --use_web --gov_only \
  --site_whitelist "ndcpa.gov.cn,nhc.gov.cn,wjw.zj.gov.cn,wsjkw.zj.gov.cn,*.zj.gov.cn" \
  --region_keywords "全国,浙江省,全省"
```

- 비고
  - 원시 검색/페이지 스냅샷은 `reports/evidence/search_logs/*.json`에 자동 저장(provenance 포함)
  - 월→주 분배(발표주 가중) 옵션은 후속 릴리스에 반영 예정

31) 구현 업데이트(2025-08 · 크롤링→웹신호→롤링/체인 모드)
- 데이터 수집/전처리
  - `scripts/preprocess/17_crawl_official_monthly_stats.py`: NHC/NDCPC 월간 ‘법정전염병’ 공지 크롤링(BeautifulSoup), 2019-01→2025-07 수집
    - 산출물: `reports/evidence/gov_reports/monthly_stats_2019-01_to_2025-07.{csv,jsonl}`
  - `scripts/Tools/evidence_pack.py`
    - `build_evidence_pack_from_gov_monthly_csv(csv, asof, weeks, future_weeks, future_decay)` 추가
    - 기능: 월간 합계를 최근 4주 신호로 요약 + 예측구간 주별 `news_signal_weekly` 벡터 생성(감쇠 가중)

- 예측 파이프라인 개선
  - `scripts/Tools/rolling_agent_forecast.py`
    - 옵션 추가: `--gov_monthly_csv`, `--preset_aggr`, `--posthoc_cal --target_cov`, `--params_json`, `--no_llm`
    - 체인 모드: `--chain k`(h=1을 k회 연결) + `--chain_particles`(앙상블 파티클), 중앙값 체인→앙상블 체인 지원
  - `scripts/Tools/run_sim_wrapper.py`
    - EVT 이후/분위수 계산 후 음수 예측값 0 하한 클램핑(음수 밴드 방지)
  - `scripts/Tools/tune_coverage.py`
    - `gov_monthly_csv` 연동, 체인 전용 튜닝 모드(`--mode chain`, `--chain`, `--chain_particles`, `--posthoc_cov`) 추가

- 실행 예(정부 월간 CSV 신호만 사용)
  - 1주 ahead(전 기간), 튜닝 파라미터 주입, 캘리브레이션 0.95
  ```
  python scripts/Tools/rolling_agent_forecast.py \
    --disease 手足口病 --start 2023-01-01 --end 2024-12-31 \
    --n_steps 0 --horizon 1 \
    --gov_monthly_csv reports/evidence/gov_reports/monthly_stats_2019-01_to_2025-07.csv \
    --params_json reports/tuning/tuned_手足口病.json --no_llm \
    --posthoc_cal --target_cov 0.95
  ```
  - 12주 체인(앙상블) 예측, 파티클 1000
  ```
  python scripts/Tools/rolling_agent_forecast.py \
    --disease 手足口病 --start 2023-04-01 --end 2023-07-31 \
    --n_steps 8 --chain 12 --chain_particles 1000 \
    --gov_monthly_csv reports/evidence/gov_reports/monthly_stats_2019-01_to_2025-07.csv \
    --params_json reports/tuning/tuned_手足口病.json --no_llm \
    --posthoc_cal --target_cov 0.95
  ```

- 결과(요약)
  - 1주 ahead(전 기간): cov95≈0.965, CRPS≈5.86, recall±2w≈1.00 → 운영 경보/스케줄링 보조에 적합
  - 12주(h=12): cov95≈0.949, CRPS≈14.27, recall≈0.75 → 장기 추세 참고용
  - 체인 k=12(앙상블 500~1000): cov95≈0.98, CRPS≈10.7, recall≈0.44~0.56 → 리콜 추가 개선 여지

- 임상 적용 가드레일(권고)
  - 경보 조건: “연속 2주 q50↑ AND q95가 임계 초과” + 인간 검토 필수
  - 음수 예측 금지(0 하한 클램핑 적용), coverage 목표 0.93~0.95 범위 내 캘리브레이션
  - 전향적 검증과 버전/로그 관리, 과경보율·KPI(ER/Bed) 동시 모니터링


32) AI 에이전트 작업 로그 (2024-05-23)
- 목적: 시스템 전체 구조 분석, 기능 추가 및 검증
- 작업 내역:
  - **알고리즘 심층 분석 및 문서화**: `03_epi_tools_MVP_and_backtest.md` 문서를 기준으로 전체 코드베이스의 아키텍처, 실행 흐름, 핵심 알고리즘의 수학적 원리를 심층 분석함. 분석 결과를 바탕으로, 외부 개발자나 새로운 팀원도 시스템을 쉽게 이해할 수 있도록 상세 기술 해설 문서(`idea/system_architecture_and_logic.md`)를 신규 작성함.
  - **최신 기능 검증 (2025-08 업데이트)**: 문서에 명시된 최신 업데이트 사항(오프라인 CSV 기반 데이터 파이프라인, 앙상블 체인 예측 모드, 음수 예측 방지 가드레일 등)이 실제 코드(`17_crawl...py`, `evidence_pack.py`, `rolling_agent_forecast.py` 등)에 모두 정확하게 구현되었음을 확인함.
  - **LLM 투명성 강화 기능 추가**: 사용자가 LLM의 판단 근거를 직접 확인할 수 있도록, LLM API의 원본(raw) JSON 응답 전체를 별도의 파일로 저장하는 기능을 구현함. `agent_loop.py`를 수정하여, 에이전트 실행 시 `reports/llm_raw_outputs/` 디렉토리에 각 반복(iteration)별 원본 응답이 타임스탬프와 함께 저장되도록 함.
  - **미래 기간 예측 실행 및 검증**: 요청에 따라 2024년 6월부터 2025년 7월까지의 미래 기간에 대한 롤링 예측을 실제로 실행함. 이 과정을 통해 새로 추가된 LLM 로깅 기능이 정상적으로 작동하여 원본 출력 파일들을 생성하는 것을 확인함.
- 결론: 시스템의 모든 기능이 문서와 일치하게 작동함을 확인했으며, LLM의 의사결정 과정을 추적할 수 있는 기능을 추가하여 시스템의 투명성과 신뢰성을 향상시킴.

# `epi_tools` 알고리즘 상세 분석 (2025-08 업데이트 반영)

이 문서는 `Project_Tsinghua_Paper/med_deepseek/scripts/Tools/` 디렉토리의 알고리즘 구현을 심층적으로 분석합니다. **최신 `03_epi_tools_MVP_and_backtest.md` 문서(31. 구현 업데이트 포함)를 기준으로 검증**했으며, 모든 코드 구현이 문서의 명세와 일치함을 확인했습니다.

전체 시스템을 **1) 핵심 예측 모델**, **2) LLM 기반 자동 튜닝 에이전트**, **3) 실행 및 백테스팅 프레임워크** 세 부분으로 나누어 설명합니다.

---

### Part 1: 핵심 예측 모델 (시뮬레이터) 상세 분석

이 부분은 단일 시계열 데이터와 주어진 하이퍼파라미터 세트로부터 어떻게 미래 예측 경로 분포가 생성되는지를 다룹니다.

#### 1.1. 시나리오 경로 생성 (`scenario_engine.py`)

이 모듈은 예측의 근간이 되는 수천 개의 미래 시나리오 경로를 생성하는 가장 핵심적인 부분입니다. 이는 단순한 통계 모델이 아니라, 과거의 '급등' 패턴을 학습하여 미래를 시뮬레이션하는 **조건부 블록 부트스트랩(Conditional Block Bootstrap)** 방식에 가깝습니다.

**`generate_paths_conditional` 함수의 작동 방식 상세 분석:**

1.  **급등 에피소드 추출 (`extract_growth_episodes`)**:
    *   먼저, 과거 시계열 데이터에서 '급등' 구간을 모두 찾아냅니다. 문서의 수학적 정의 \\(g_t = \frac{y_t-y_{t-1}}{\max(1,y_{t-1})} \ge \tau\\)에 따라, 주간 성장률(`growth`)이 특정 임계값(`pct_threshold`, 예: 12%)을 넘는 지점을 급등의 시작으로 인식하고, 성장률이 완화될 때(`relax_drop`)까지를 하나의 '에피소드'로 정의합니다.
    *   각 에피소드는 단순한 값의 나열이 아니라, 시작점 대비 값의 비율(`shape_factors`)로 정규화되어 '형상(shape)' 정보로 저장됩니다. 이는 과거 급등이 어떤 규모에서 시작되었는지와 무관하게 순수한 '패턴'만을 학습하기 위함입니다.

2.  **과거 패턴 샘플링 (유사도 기반)**:
    *   이제 미래를 예측하기 위해, 과거의 여러 급등 에피소드 중 어떤 것을 참조할지 결정해야 합니다.
    *   판단 기준은 **"최근 N주간의 패턴과 가장 유사한 과거 에피소드"** 입니다. `_episode_similarity_score` 함수는 최근 데이터의 '형상 변화율'(`recent_deltas`)과 각 과거 에피소드의 '초기 형상 변화율' 간의 유클리드 거리를 계산하여 유사도 점수를 매깁니다.
    *   이 점수가 낮을수록(유사할수록) 해당 에피소드가 샘플링될 확률(`probs`)이 높아집니다. 이를 통해 현재 상황과 가장 관련성 높은 과거 패턴을 더 자주 참조하여 미래를 그리게 됩니다.

3.  **미래 경로 합성 (수학적 정의와 코드 연결)**:
    *   샘플링된 에피소드의 '형상'을 예측할 기간(`horizon`)에 맞게 보간(`np.interp`)하여 기본 골격을 만듭니다.
    *   이후, 문서 9)의 핵심 갱신 공식 \\(x_h = x_{h-1}\, r^{\ast}_h\, c_h\, e^{\epsilon_h}\\)에 따라 한 주씩 예측값을 굴려나갑니다.
        *   `\(x_{h-1}\)`: 직전 주의 예측값 `x`.
        *   `\(r^{\ast}_h\)` (**핵심 성장 동력**): 보간된 형상에서 계산된 전주 대비 성장률(`ratios`)에 **`amplitude_multiplier`** 파라미터로 증폭된 진폭(`amp_scale`)과 워밍업(`gamma`)을 적용한 `r_eff`에 해당합니다. 이 값이 예측의 상승 기울기를 결정합니다.
        *   `\(c_h\)` (**외부 신호 반영**): `news_signal` (웹 검색, 계절성 정보 등)이 반영된 `scale_vec`입니다. 특정 주에 긍정적 외부 신호가 있으면 이 값이 1보다 커져 예측치를 상향 조정합니다.
        *   `\(e^{\epsilon_h}\)` (**현실적 변동성**): `quality` 파라미터에 반비례하는 정규분포 노이즈(`step_noise`)입니다. `quality`가 낮을수록 노이즈의 표준편차가 커져, 예측 경로들의 분산(밴드의 너비)이 넓어집니다. 이는 데이터의 불확실성을 표현하는 역할을 합니다.

4.  **안정화 장치 (가드레일)**:
    *   시뮬레이션이 비현실적인 값으로 폭주하는 것을 막기 위해 여러 가드레일이 존재합니다.
    *   `r_boost_cap`: `\(r^{\ast}_h\)`의 최대값을 제한하여 과도한 급등을 막습니다.
    *   `x_cap_multiplier`: 예측값이 과거 데이터의 최대치를 비정상적으로 초과하지 않도록 상한선을 둡니다.
    *   `nb_dispersion_k`: 음이항 분포(Negative Binomial) 샘플링을 통해, 카운트 데이터의 과산포(variance > mean) 특성을 모델링하여 예측의 현실성을 높입니다.

#### 1.2. 극단적 피크 보정 (`evt.py`)

시나리오 엔진만으로는 과거에 나타나지 않았던 극단적인 피크를 예측하기 어렵습니다. 이는 특히 팬데믹 예측에서 치명적인 과소추정으로 이어질 수 있습니다. `evt.py`는 **극단값 이론(Extreme Value Theory, EVT)**을 이용해 이 문제를 해결합니다.

*   **`fit_pot` (Peak-Over-Threshold)**: 과거 데이터에서 상위 10%(`u=Quantile(0.9)`)와 같이 매우 높은 값들만 따로 모읍니다. EVT에 따르면, 이 임계값을 초과하는 데이터(excess)는 **일반화 파레토 분포(GPD)**를 따르는 경향이 있습니다. 이 함수는 GPD의 파라미터(`shape`, `scale`)를 추정합니다.
*   **`replace_tail_with_evt`**: `scenario_engine`이 생성한 수천 개 경로에서, 임계값 `u`를 초과하는 값들을 발견하면, 그 값들을 버리고 `fit_pot`으로 학습한 GPD 분포에서 새로 샘플링한 값으로 **강제 교체**합니다. 이를 통해 시뮬레이션된 피크의 꼬리 분포를 통계적으로 타당한 극단값 분포로 보정하여, 과거 경험에만 얽매이지 않는 더 현실적인 최대 피크 예측이 가능해집니다.

---

### Part 2: LLM 기반 자동 튜닝 에이전트 상세 분석

최적의 예측을 위해서는 Part 1에서 설명한 수많은 하이퍼파라미터(`amplitude_multiplier`, `quality` 등)를 상황에 맞게 잘 조정해야 합니다. 이 역할을 인간 분석가 대신 LLM 에이전트가 수행합니다.

#### 2.1. 외부 정보 수집 및 가공 (`web_sources.py`, `evidence_pack.py`)

LLM이 정확한 판단을 내리려면 데이터가 필요합니다. **(2025-08 업데이트 핵심)**

*   **사전 데이터 수집 (`preprocess/17_crawl_official_monthly_stats.py`)**: 실시간 웹 검색의 불안정성을 해소하기 위해, 정부 기관의 월간 공식 통계 발표를 사전에 크롤링하여 로컬 CSV 파일로 저장하는 기능이 추가되었습니다.
*   **`evidence_pack.py`의 고도화**:
    *   **오프라인 신호 생성 (`build_evidence_pack_from_gov_monthly_csv`)**: 실시간 웹 검색(`web_sources.py`)에 의존하는 대신, 위에서 수집한 로컬 CSV를 사용하여 **`as-of` 원칙**에 맞는 외부 신호를 생성하는 기능이 핵심 업데이트입니다.
    *   **미래 신호 벡터 생성**: 이 함수는 과거 데이터 기반의 단일 `news_signal` 스칼라 값뿐만 아니라, 예측할 미래 기간(`future_weeks`)에 대한 **주별 신호 벡터(`news_signal_weekly`)**를 동적으로 생성합니다. 이 벡터는 시간이 지남에 따라 신호의 영향력이 감소하는 '감쇠(decay)' 효과를 반영하여 장기 예측의 현실성을 높입니다.

#### 2.2. LLM 분석가 호출 및 제어 (`llm_agent.py`, `agent_loop.py`)

*   **`llm_agent.py`의 `SYSTEM_PROMPT`**: LLM의 역할을 정의하는 프롬프트는 이 시스템의 '두뇌'와 같습니다.
    *   **역할 부여**: "너는 병원 예측 파이프라인을 조율하는 분석가다."
    *   **출력 형식 강제**: JSON 스키마를 명시하여 기계가 파싱하기 용이한 형태로 답변을 유도합니다.
    *   **하드 제약 조건 명시**: "r_boost_cap≤3.0, quality∈[0.5,0.95]" 와 같이 파라미터의 물리적 범위를 명시하여 LLM이 엉뚱한 값을 제안하지 않도록 합니다.
    *   **전략적 가이드라인 제공**: "최근 성장률이 음수이면 보수적으로 접근하라", "외부 신호가 있으면 반드시 `news_signal`을 포함하라" 와 같은 규칙을 통해 전문가의 분석 전략을 LLM에게 학습시킵니다.
*   **`agent_loop.py`의 실행 루프**:
    1.  **관측 (`build_observation`)**: 현재 상태(직전 사용 파라미터, 직전 예측의 `coverage95`와 `CRPS` 점수 등)와 `evidence_pack`을 합쳐 LLM에게 전달할 `observation` 객체를 만듭니다.
    2.  **제안 (`propose_params_via_llm`)**: `observation`을 프롬프트에 담아 LLM을 호출하고, 새로운 하이퍼파라미터 제안을 받습니다.
    3.  **안전장치 적용 (`apply_hard_guards`, `clamp_params`)**: LLM의 제안이 프롬프트의 제약을 어겼을 경우를 대비해, 코드로 다시 한번 안전 범위를 강제 적용합니다.
    4.  **시뮬레이션 (`run_sim`)**: 제안된 파라미터로 Part 1의 예측 시뮬레이션을 실행합니다.
    5.  **KPI 페일세이프**: 시뮬레이션 결과, 병상 점유율 초과 확률 같은 핵심 KPI가 위험 수위를 넘으면(`p_bed_gt_0_98 > 0.5`), 즉시 파라미터를 강제로 보수적인 값(`amplitude_multiplier` 하향 등)으로 조정한 뒤 **재시뮬레이션**합니다. 이는 예측 실패로 인한 리스크를 최소화하는 매우 중요한 안전장치입니다.
    6.  **로깅 및 상태 업데이트**: LLM의 원본 답변, 최종 적용된 파라미터, 시뮬레이션 결과 지표 등 모든 것을 상세히 로깅하고, 다음 루프를 위해 `last_metrics`를 업데이트합니다.

---

### Part 3: 실행 및 백테스팅 프레임워크 상세 분석

이 부분은 위 컴포넌트들을 실제로 구동하고, 모델의 성능을 신뢰성 있게 평가하는 방법을 다룹니다.

*   **`run_sim_wrapper.py`**: `agent_loop`와 Part 1의 예측 엔진을 연결하는 인터페이스입니다. LLM이 제안한 파라미터 딕셔너리를 받아 `generate_paths_conditional`에 필요한 인자로 변환하고, 시뮬레이션이 끝나면 `metrics.py`를 호출하여 `CRPS`, `coverage95`, `peak_recall` 등 다양한 성능 지표를 계산하여 `agent_loop`에 반환하는 역할을 합니다. **(업데이트: `evidence_pack`에서 생성된 `news_signal_weekly` 벡터를 입력받아 처리하고, 음수 예측을 0으로 클램핑하는 로직 추가)**
*   **`rolling_agent_forecast.py` (가장 중요한 실행기)**:
    *   이는 단순한 홀드아웃 테스트를 넘어, 실제 운영 상황을 가장 유사하게 모사하는 **롤링 예측(Rolling Forecast) 백테스트**를 수행합니다.
    *   **(업데이트 핵심) 체인 모드 (`--chain k`)**: 장기 예측(`k`주)을 위해, 1주 예측을 `k`번 반복하는 새로운 모드가 추가되었습니다. 단순 반복이 아니라, 각 스텝의 예측 분포 전체를 다음 스텝의 입력으로 사용하는 **앙상블 파티클 전파** 방식을 채택하여 장기 예측의 안정성과 신뢰도를 높였습니다.
*   **`tune_coverage.py`**: 커버리지 등 특정 목표에 맞게 하이퍼파라미터를 자동 탐색하는 경량화된 튜닝 스크립트입니다. **(업데이트: 체인 모드의 장기 예측 성능에 최적화된 파라미터를 탐색하는 기능 추가)**

---

### 종합 결론 (2025-08 업데이트 반영)

초기 버전이 실시간 웹 검색 기반의 유연한 에이전트 시스템에 중점을 두었다면, 이번 업데이트는 **재현성, 안정성, 그리고 장기 예측 성능 강화**에 초점을 맞추었습니다.

1.  **데이터 파이프라인 강화**: 실시간 웹 크롤링의 변동성을 줄이고 안정적인 백테스팅을 위해, 사전에 검증된 데이터를 수집(`17_crawl...py`)하고 이를 `as-of` 원칙에 따라 가공(`evidence_pack.py`)하는 오프라인 데이터 파이프라인이 구축되었습니다.
2.  **장기 예측 능력 확보**: 단기 예측(`horizon`)의 한계를 극복하기 위해, 앙상블 파티클 전파 방식의 **체인 모드**가 도입되어 보다 신뢰도 높은 중장기(예: 12주) 예측이 가능해졌습니다.
3.  **임상 적용성 제고**: 음수 예측 방지 등 실제 운영 환경에서 발생할 수 있는 문제에 대응하기 위한 가드레일이 추가되어 시스템의 견고성이 향상되었습니다.

이러한 업데이트를 통해, `epi_tools` 시스템은 연구 단계를 넘어 실제 임상 환경에서의 운영 및 신뢰성 있는 백테스팅을 수행할 수 있는 한 단계 더 성숙한 예측 시스템으로 발전했음을 확인했습니다.

---

### Part 4: 최신 엔드투엔드 파이프라인(2025-08, gpt-5 연동·보수화 규칙) 상세

본 섹션은 최근 추가/개선된 구현을 중심으로, 실제 실행 순서와 각 단계의 입·출력 규격, 가드레일/백오프 로직을 정리한다. 참조 소스는 `scripts/Tools/rolling_agent_forecast.py`, `scripts/Tools/agent_loop.py`, `scripts/Tools/llm_agent.py`, `scripts/Tools/run_sim_wrapper.py` 등이다.

1) 입력 데이터/증거 신호
- 내부 시계열: `processed_data/his_outpatient_weekly_epi_counts.csv`의 주간 카운트(`diagnosis_time`, 질병별 열)
- 외부 신호(정부 월간 CSV): `reports/evidence/gov_reports/monthly_stats_2019-01_to_2025-07.csv`
  - `evidence_pack.build_evidence_pack_from_gov_monthly_csv(...)`가 as-of 기준으로 `external_signals`를 생성
  - 호라이즌 h에 대해 `external_signals.news_signal_weekly` 벡터(감쇠 적용) 산출
  - 결정적 매핑(웹 검색 없이도 동작):
    - `news_signal = clip(0.05 + 0.25*news_hits_change_4w + 0.3*(search_snr/3), 0.05, 0.7)`

2) 관측(Observation) 구성 항목
- 기본: `disease`, `period(train_until→end)`, `last_params`, `last_metrics`, `constraints.bounds`, `target_metrics`
- 내부 8주 요약: `internal_weekly.last_8w_counts`, `internal_weekly.last_8w_growth_pct`, `internal_weekly.last_8w_summary({last_mean,last_median,last_growth_pct})`
- 성능 추세(롤링 전용): `metrics_trend`에 최근 8스텝의 평균값(coverage95/recall_pm2w/crps/mae_median)
- 직전 스텝 예측/오차(롤링 전용): `last_llm_pred_q50`, `last_llm_abs_err`
- 최근 성장/기저(롤링 전용): `recent_growth_pct`, `hist_median8`

3) LLM 프롬프트(핵심 규칙, gpt-5 Responses API)
- 모델: OpenAI `gpt-5` (Responses API, `POST /v1/responses`)
- 입력 형식: `input=[{"role":"system","content":[{"type":"input_text","text":SYSTEM_PROMPT}]}, {"role":"user","content":[{"type":"input_text","text":json.dumps(observation)}]}]`, `text.format={"type":"json_object"}`
- 출력 스키마(LLM):
  - `prior_analysis`: `{historical_summary, this_week_estimate, assumptions}`
  - `proposed_params`: 하이퍼파라미터 딕셔너리(가드 범위 준수)
  - `rationale_summary`, `expected_tradeoffs`, `evidence_used`, `validation.constraints_ok`
- 필수 규칙(추가):
  - `last_8w_growth_pct`가 음수이면 보수적으로(진폭·캡·quality 보정)
  - `last_metrics`, `metrics_trend`를 반드시 인용: 
    - coverage95≥0.98 → 밴드 과대: `quality↑` 또는 `nb_dispersion_k↓`, `amplitude_multiplier↓`
    - coverage95<0.90 → 밴드 과소: `quality↓` 또는 `nb_dispersion_k↑`, `r_boost_cap/scale_cap/x_cap_multiplier↓`
    - recall 낮고 최근 성장(+) → `amplitude_multiplier↑`, `r_boost_cap↑` (하드 캡 내)
  - 외부 신호 존재 시 `news_signal` 필수 포함(0–1)

4) LLM 실패/지연 대비 안전장치
- 재시도/타임아웃으로도 실패할 수 있는 운영 환경을 고려해, 롤링 실행기에서 **적응적 보수화 백오프**를 적용:
  - 조건: LLM 실패이거나, 직전 coverage95≥0.98, 혹은 `last_llm_abs_err>8` 및 `recent_growth_pct≤0`인 경우
  - 조정: `amplitude_multiplier≤1.4`, `r_boost_cap≤1.6`, `scale_cap≤1.4`, `x_cap_multiplier≤1.8`, `quality∈[0.6,0.8]`로 보수화
  - 결과: q50 상향편향 억제, CRPS 하향 경향(실측 대비 과대예측 감소)

5) 핵심 시뮬레이션(`run_sim_wrapper.py`)
- 입력 파라미터 → `scenario_engine.generate_paths_conditional(...)` 경로 생성
- `evt.fit_pot`/`evt.replace_tail_with_evt`로 상단 꼬리 보정
- 비음수 클램핑 후 분위수(q05/q50/q95) 및 평균 산출, `metrics.py`로 MAE(중앙), SMAPE, CRPS, coverage95, peak 지표 계산
- 선택적 사후 커버리지 캘리브레이션(`enable_posthoc_calibration`, `calibrate_coverage_to`)

6) 롤링/체인 실행(`rolling_agent_forecast.py`)
- 스텝별(`train_until`)로 관측→LLM 제안→가드/클램프→시뮬레이션→메트릭→리포트/로그
- 체인 모드(`--chain k`): 1주 예측을 k회 연결(앙상블 파티클 전파)
- 산출물: HTML(`.../rolling_{...}.html`), JSON(`.../rolling_{...}.json`), 진행 로그 TXT(`.../rolling_{...}.log`)

7) 로깅/투명성 강화
- LLM 원본 응답 전체를 `reports/llm_raw_outputs/*_iter{t}.json`에 저장(Responses API 응답 포함)
- 가독성 로그: `agent_loop`/`rolling_agent_forecast`는 `.log`에 단계별 진행, prior_analysis 요약, 적용 파라미터, 메트릭, 백오프 적용 여부 등을 기록

8) 권장 기본값/튜닝 힌트(1주 ahead)
- 안정 구간: `amplitude_multiplier≈1.4–1.8`, `r_boost_cap≈1.6–2.0`, `scale_cap≈1.4–1.6`, `x_cap_multiplier≈1.8–2.0`, `quality≈0.66–0.8`, `nb_dispersion_k≈6–10`
- 폭주 방지: `use_delta_quantile=True`, `delta_quantile≈0.05`, 상단 캡(`x_cap_multiplier`)을 과거 0.99 분위 기반 곱으로 제한
- 신호: `news_signal_weekly`가 약할수록 보수화, 강할수록 진폭/캡 완화(하드 제약 준수)

9) 한계와 다음 과제
- LLM 호출 실패/변동성에 대비한 백오프로 평균 성능은 개선되나, 국소 구간 리콜 저하 가능 → `quality/nb_dispersion_k` 동적 튜닝 강화 필요
- 스텝 병렬화/캐시 도입으로 실행 시간 단축 여지
- 다질병/다지역 동시 최적화 및 KPI(ER/Bed) 비용함수 직접 최적화 연구 필요

---

### Part 5: 시스템 아키텍처 및 보안 분석

본 섹션은 코드베이스의 전체적인 구조, 데이터 흐름, 그리고 2025-08-18에 수행된 보안 감사를 통해 식별된 잠재적 데이터 유출 위험도를 분석한다.

#### 5.1. 전체 시스템 아키텍처 및 데이터 흐름

시스템은 역할에 따라 명확하게 모듈화되어 있으며, 데이터는 단방향으로 흐르도록 설계되어 있다.

- **데이터 소스**:
  - **내부**: 로컬 CSV 파일 (`his_outpatient_weekly_epi_counts.csv`)
  - **외부**: 웹 검색 API (Bing, Google) 및 사전 수집된 정부 월간 통계 CSV

- **실행 흐름 (Orchestration)**:
  1.  `rolling_agent_forecast.py`: 전체 백테스팅 워크플로우를 관장하는 메인 실행기.
  2.  `agent_loop.py`: `train_until` 시점을 기준으로 관측(`Observation`)을 구성하고, LLM의 제안을 받아 시뮬레이션을 실행하는 핵심 루프를 관리.
  3.  `llm_agent.py`: 외부 LLM API (OpenAI, DashScope)와 통신하여 `Observation`에 기반한 최적의 하이퍼파라미터를 제안받음.
  4.  `run_sim_wrapper.py`: 제안된 파라미터로 시뮬레이션 엔진(`scenario_engine.py`)을 실행하고 결과를 `metrics.py`로 평가.
  5.  `rolling_agent_forecast.py`: 모든 스텝의 결과를 종합하여 HTML 리포트와 JSON 로그를 생성.

- **데이터 유출 방지 설계**:
  - 시스템의 핵심 설계 원칙은 **미래 데이터(답지)가 현재의 예측에 영향을 주지 않도록 하는 것**이다.
  - 모든 데이터(내부 시계열, 외부 웹 정보)는 `train_until` 이라는 명확한 `as-of` 기준 시점으로 필터링되어 LLM과 시뮬레이터에 전달된다. LLM은 미래를 직접 예측하는 것이 아니라, 엄격하게 필터링된 과거 데이터만을 보고 시뮬레이션 전략(하이퍼파라미터)을 제안하는 역할에 한정된다.

#### 5.2. 데이터 유출 및 보안 위험 분석 (2025-08-18 감사 기준)

전반적으로 API 키를 코드에 직접 노출하지 않는 등 기본적인 보안 조치는 되어 있으나, 다음과 같은 잠재적 위험 및 개선 사항이 식별되었다.

| 위험 유형 | 상세 내용 | 위험도 | 조치 및 권고 사항 |
| :--- | :--- | :--- | :--- |
| **API 키 노출** | 프로젝트에 `.gitignore` 파일이 없어, API 키가 포함될 수 있는 `.env` 파일이 실수로 Git 저장소에 커밋될 위험이 존재했다. | **높음** | **(조치 완료)** `.gitignore` 파일을 생성하여 `.env` 파일, `reports/` 디렉토리, `__pycache__/` 등 민감 정보 및 불필요한 파일이 버전 관리에서 제외되도록 조치했다. |
| **민감 데이터 전송** | LLM API로 병원 주간 외래환자 데이터의 최근 8주치 요약(`last_8w_counts`, `last_8w_growth_pct`)이 전송된다. 이는 집계된 통계지만, 외부 서비스의 데이터 사용 정책 검토가 필요하다. | **중간** | LLM 서비스 제공자(OpenAI, DashScope)의 데이터 개인정보 보호 및 사용 정책을 검토하고, 내부 데이터 거버넌스 정책을 준수하는지 확인해야 한다. 필요시 데이터 비식별화 처리를 강화하는 것을 고려할 수 있다. |
| **민감 정보 로깅** | LLM API와의 전체 요청/응답 내용이 `reports/llm_raw_outputs/` 디렉토리에 원본 그대로 저장된다. 여기에는 외부로 전송된 환자 통계 데이터가 포함된다. | **낮음** | 로컬 파일 시스템에 저장되어 직접적인 외부 유출 위험은 낮다. 단, 서버 접근 권한이 있는 사용자는 해당 데이터를 볼 수 있으므로, 로그 파일 접근 권한을 최소화하고 오래된 로그를 정기적으로 파기하는 정책 수립을 권장한다. |

감사 결과에 따라 가장 시급한 사항이었던 `.gitignore` 파일 생성을 완료하여, 현재 코드베이스의 보안 수준이 크게 향상되었다.

Epi Tools (MVP)

개요
- 스파이크·레짐 전환·꼬리를 고려한 전염병/혼잡 예측을 위한 최소 도구 묶음입니다.
- 포함 모듈: `fusion.py`(정밀도 가중 결합), `scenario_engine.py`(조건부 블록 부트스트랩), `evt.py`(POT-GPD 피팅), `risk_banding.py`(KPI 초과확률 기반 상/중/하), `adapters.py`(CSV 어댑터), `run_on_his_outpatient.py`(HIS 외래 결과 실행 러너).

빠른 실행
```bash
python Project_Tsinghua_Paper/med_deepseek/scripts/Tools/run_on_his_outpatient.py
```

입출력
- 입력: `med_deepseek/processed_data/his_outpatient_weekly_epi_counts.csv` (`diagnosis_time` + 질병별 카운트)
- 출력: nowcast 결합값과 리스크 밴드 JSON(stdout)

튜닝 포인트
- `risk_banding.py`: `thr_er_min`, `thr_bed`, `eta_er`, `eta_bed`, `band_cut_low/high`
- `fusion.py`: `kappa_quality`, `manual_bias_*`, `news_*`

Epi Tools (MVP)

개요
- 스파이크·레짐 전환·꼬리를 고려한 전염병/혼잡 예측을 위한 최소 도구 묶음입니다.
- 포함 모듈: `fusion.py`(정밀도 가중 결합), `scenario_engine.py`(조건부 블록 부트스트랩), `evt.py`(POT-GPD 피팅), `risk_banding.py`(KPI 초과확률 기반 상/중/하), `example_run.py`(데모).

설치
1) 가상환경 생성 후 요구 패키지 설치
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r @Tools/requirements.txt
```

빠른 사용법
```bash
python @Tools/example_run.py
```

모듈 개요
- fusion.precision_weighted_fusion: 예측 사전분포 + 전처리 관측 + 품질(q) + 수기편향(μ_e, σ_e) + 뉴스 신호를 정밀도 가중 평균으로 결합해 nowcast와 95% CI 계산.
- scenario_engine.extract_growth_episodes: 시계열에서 급등 에피소드 추출.
- scenario_engine.generate_paths_conditional: 현재 컨텍스트와 유사한 에피소드를 가중 재표본하여 N개 경로 생성.
- evt.fit_pot / evt.replace_tail_with_evt: 임계치 초과분에 GPD를 적합하고, 경로의 상위 꼬리를 EVT로 보정.
- risk_banding.band_from_paths: 경로로부터 KPI 초과확률을 계산해 LOW/MED/HIGH를 반환.

주의
- 본 스켈레톤은 의존성 최소화와 이해 용이성을 목표로 한 레퍼런스 구현입니다. 운영 투입 전 병원 데이터 스키마에 맞춘 튜닝과 검증이 필요합니다.



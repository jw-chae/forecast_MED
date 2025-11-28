# 실험 설정(YAML) 안내

이 문서는 `experiments/configs/` 디렉터리에 포함된 실험 정의를 요약합니다.

## 디렉터리 구성

- `baselines/`: Prophet, ARIMA, XGBoost, LSTM, SEIR 등 비교용 기준 모델 설정.
- `ablations/`: LLM 미사용, EVT 미사용 등 주요 컴포넌트를 제거한 설정.
- `main/`: 논문의 핵심 파이프라인(Epi Tools + LLM + EVT) 설정.

## 공통 필드 설명

| 섹션 | 설명 |
|------|------|
| `experiment` | 실험명, 설명, 태그, 시드 등 메타데이터 |
| `data` | 데이터 소스 경로, 질병명, 학습/평가 기간, 롤링 파라미터 |
| `model` | 모델 타입, 하이퍼파라미터, LLM 설정, EVT 설정 |
| `evaluation` | 계산할 메트릭, 통계 검정, 비교 대상 실험 리스트 |
| `logging` | 로그 레벨, 콘솔 출력, 플롯 저장 여부 |
| `resources` | CPU/GPU 사용, 메모리 제한, 타임아웃 등 실행 리소스 |

## 주요 설정 파일 요약

### `main/epi_tools_full.yaml`
- 목적: LLM 조정 + EVT 보정이 포함된 전체 파이프라인 평가.
- 특징: 롤링 예측 사용(`enabled: true`), LLM 프롬프트/응답 저장, 다양한 예측 플롯 저장.
- 메트릭: CRPS, MAE, MAPE, 95/90% 커버리지, 피크 관련 지표.
- 참고: 데이터셋의 질병 컬럼이 중국어(`流行性感冒`)로 되어 있어 `data_loader`에서 별칭을 통해 자동 매핑됩니다.

### Baseline 예시
- `baselines/prophet.yaml`: Prophet 기본 설정, 시즈널리티와 휴일 효과 옵션 포함.
- `baselines/xgboost.yaml`: 시계열 특징 엔지니어링 + XGBoost 회귀 기반 예측.

### Ablation 예시
- `ablations/no_llm.yaml`: LLM 보조 없이 규칙 기반 파라미터만 사용하도록 설정.
- `ablations/no_evt.yaml`: EVT 후처리를 비활성화하고 원본 시나리오 엔진 결과만 사용.

## 설정 오버라이드

CLI 실행 시 `--override` 옵션으로 일부 값을 손쉽게 변경할 수 있습니다.

```bash
python -m experiments.run_experiment \
  --config experiments/configs/main/epi_tools_full.yaml \
  --override "data.train_end=2024-03-31" \
  --override "model.forecast.horizon=8"
```

- 중첩 키는 점 표기법을 사용합니다.
- 여러 오버라이드는 `--override` 옵션을 반복하여 전달합니다.

## 새로운 실험 추가 절차

1. 기존 YAML을 복사하여 새 파일을 생성합니다.
2. `experiment.name`을 고유하게 지정합니다.
3. 데이터 경로가 존재하는지 확인하고, 필요 시 `config_validator` 로직을 업데이트합니다.
4. 메트릭, 통계 검정, 비교 대상 등 분석 요구사항을 명시합니다.
5. README 또는 노션 등에 해당 YAML의 목적과 변경점 요약을 남깁니다.

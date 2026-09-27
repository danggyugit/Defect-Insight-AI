# Defect-Insight-AI

공정 제조업 범용(디스플레이/반도체/2차전지/식품 등) 불량 영향인자 자동 탐색 플랫폼.
통계분석 + ML + Explainable AI(SHAP) + Rule Discovery → Streamlit dashboard + 자동 report.

**전체 명세: [docs/PROJECT_PROMPT.md](docs/PROJECT_PROMPT.md)** — 기능 추가/변경 시 항상 이 명세 기준으로 판단.

## 실행

```bash
source .venv/bin/activate                 # Python 3.13 venv
python -m src.data.generator              # synthetic 데이터 생성 → data/raw/*.parquet
python -m src.data.trace_generator        # FDC trace 생성 (딥러닝용, main 데이터 이후 실행)
streamlit run app.py                      # dashboard (기본 포트 8501)
pytest                                    # 테스트
```

## 아키텍처

- `src/data/` generator(합성 데이터) · trace_generator(FDC 파형) · loader · validator
- `src/statistics/` 유의차 검정 · effect size · correlation/MI
- `src/ml/` trainer(LR/RF/XGB/LGBM) · evaluation · feature importance
- `src/explainability/` SHAP 분석
- `src/multivariate/` interaction · rule discovery (Decision Tree 기반)
- `src/dl/` FDC trace 1D-CNN (PyTorch) — tabular이 아닌 **파형 데이터 전용**.
  tabular 분석에 딥러닝을 섞지 말 것 (tree 모델이 최적이라는 것이 설계 전제)
- `src/report/` template 기반 HTML report 생성 (LLM 아님)
- `src/utils/config.py` 모든 상수/경로/seed — **하드코딩 금지, 여기로 모을 것**
- `views/` Streamlit 페이지 (home + 01_overview ~ 10_deeplearning), `app.py`가
  st.navigation으로 한글 그룹 메뉴 구성 — **st.set_page_config는 app.py에서만 호출**
- 페이지 공통 패턴: `page_header()`(제목+분석 여정+쉬운 설명 expander) →
  `insight_box()`(핵심 발견 자동 요약) → `show_disclaimer()`.
  쉬운 설명 문구는 `src/utils/explanations.py`에서만 관리.
  정상/불량 색상은 st_helpers의 COLOR_NORMAL/COLOR_DEFECT 고정
- `models/` 학습 모델 joblib 캐시, `reports/` 생성된 report, `data/` parquet

## 핵심 규칙 (절대 위반 금지)

1. **Leakage**: ML feature에서 식별자(LOT_ID 등), TIMESTAMP, DEFECT_FLAG,
   DEFECT_TYPE, YIELD, QUALITY_SCORE 반드시 제외.
2. **Split**: 기본 `GroupShuffleSplit(groups=LOT_ID)`. random split 금지.
3. **Imbalance**: PR-AUC를 주지표로. class_weight/scale_pos_weight 적용.
4. **다중검정**: 유의차 분석에는 항상 Benjamini-Hochberg 보정 p-value 병기.
5. **문구 규칙**: "원인이다" 단정 금지. "통계적으로 유의한 차이 확인 →
   공정 검증 권고" 형태로. 상관≠인과 disclaimer 유지.
6. **Seed**: 모든 랜덤 연산은 config의 RANDOM_SEED=42 사용.
7. **성능**: feature 전수 계산 금지 — variance→significance/MI→Top-N 3단계 filtering 후
   SHAP/interaction은 상위 10~20개만.
8. **Streamlit**: 재학습/재계산은 `st.cache_data`/`st.cache_resource`로 방지,
   모델은 models/에 joblib 저장 후 재사용.

## Synthetic 데이터의 정답 (검증용)

Generator에 심어진 mechanism — 분석 파이프라인이 이를 검출해야 정상:

- TEMP_004↑, PRESSURE_012↑ → MURA (+두 변수 interaction 시 급증)
- SPEED_007 범위 이탈(U자형) → SCRATCH
- TIME_003+TEMP_004 조합 → PARTICLE
- 특정 EQUIPMENT baseline shift → OPEN_SHORT
- 시간 drift → 후기 PARTICLE 증가
- confounder 변수(불량 무관, 유의 변수와 상관) → false positive 테스트용

FDC trace (딥러닝용, `fdc_traces.npz`): 샘플당 3센서×128step 파형.
DEFECT_TYPE별 파형 이상 — MURA=drift, PARTICLE=spike, SCRATCH=진동,
OPEN_SHORT=level shift. 요약 통계로는 안 잡히는 신호라 1D-CNN이 의미를 가짐
(실측: CNN PR-AUC 0.78 vs tabular LGBM 0.55). CNN도 LOT GroupSplit + pos_weight 필수.

기본 규모: 30,000 samples / ~1,200 LOTs / 400 features / defect rate 6% / 2026-01~08.

## 컨벤션

- Python 3.13, type hint + docstring 필수, 로깅은 `logging` (print 금지)
- 시각화는 Plotly 우선 (SHAP 내장 plot만 matplotlib)
- 데이터 저장은 parquet (pyarrow)
- 커밋/푸시는 사용자 요청 시에만

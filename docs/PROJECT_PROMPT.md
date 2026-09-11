# Manufacturing Defect Insight AI — 최종 프로젝트 명세 (v1.0)

> 2026-09-09 확정. 초안 프롬프트에 대한 검토 결과(샘플 수 정의, 폴더명 통일,
> leakage 컬럼 명문화, GroupSplit 기본화, template 기반 report 명시 등)를 반영한 최종본.

---

## 1. 프로젝트 목표

첨단 제조산업(디스플레이, 반도체, 2차전지 등)에서 제품/LOT/Panel/Wafer/Cell 등의
불량이 발생했을 때, MES·FDC·SPC·설비·품질검사·공정이력 데이터를 기반으로:

1. 불량과 단일 변수 간 통계적으로 유의한 차이 분석
2. 불량과 변수 간 선형/비선형 관계 탐색
3. 여러 공정 변수 간 다중 상관관계 및 interaction 분석
4. Machine Learning 기반 불량 영향인자 ranking
5. SHAP 등 Explainable AI로 모델 판단 근거 설명
6. 주요 불량 발생 조건 자동 탐색 (Rule Discovery)
7. 분석 결과를 사람이 이해하기 쉬운 리포트로 자동 생성

하는 **"Manufacturing Defect Insight AI"** 플랫폼을 구축한다.

핵심 목적: 단순 상관분석이 아니라 *"불량 발생 시 수백~수천 개의 공정 변수 중
어떤 변수가 유의한지, 어떤 변수 조합이 불량과 관련되는지, 어떤 조건에서
불량 확률이 증가하는지 자동으로 탐색"*하는 것.

---

## 2. 개발 원칙

- 실제 제조 현장에서 사용할 수 있는 구조를 고려한다.
- 통계분석 + ML + Explainable AI를 결합한다.
- 단순 correlation coefficient만으로 원인을 단정하지 않는다.
- **상관관계와 인과관계를 명확히 구분한다.**
- p-value뿐 아니라 effect size와 sample size를 함께 고려한다.
- 데이터 leakage를 방지한다 (아래 §11 leakage 규칙 참조).
- 정상/불량 class imbalance를 고려한다.
- 모델 성능과 설명 가능성을 동시에 고려한다.
- 결과를 제조 엔지니어가 이해할 수 있는 형태로 표현한다.
- 모든 분석 결과에 "근거 데이터"와 "분석 방법"을 함께 표시한다.
- MVP 단계에서는 과도한 복잡성을 피하고 확장 가능한 구조를 만든다.

---

## 3. 기술 스택

- **Python 3.13** (venv), pandas / numpy / pyarrow (parquet)
- **Statistics**: scipy, statsmodels
- **ML**: scikit-learn, XGBoost, LightGBM
- **XAI**: SHAP
- **Visualization**: Plotly (기본), matplotlib (SHAP plot 용)
- **Dashboard**: Streamlit (multipage)
- **Report**: HTML report 우선. PDF는 브라우저 인쇄 기반(optional) — weasyprint 등
  시스템 의존성 있는 라이브러리는 MVP에 넣지 않는다.
- requirements.txt, .env.example(향후 MES DB 연결용 주석 placeholder), README.md 제공
- VS Code / Claude Code에서 실행 가능해야 함

---

## 4. 시스템 아키텍처

```
Data Input → Data Validation → Preprocessing → Feature Engineering
→ Statistical Analysis → ML Analysis → Explainable AI (SHAP)
→ Multivariate / Interaction Analysis → Rule Discovery
→ Risk / Condition Analysis → Automated Report → Streamlit Dashboard
```

---

## 5. 데이터 규모 및 스키마 (확정)

Synthetic Manufacturing Dataset으로 MVP를 완성한다. 규모는 config로 조절 가능:

| 항목 | 기본값 | 비고 |
|---|---|---|
| n_samples | **30,000** | 통계 검정력 + ML 학습 확보 |
| n_lots | ~1,200 | LOT당 평균 25 samples |
| n_features | **400** | 옵션으로 1,000+ 스케일 테스트 가능 |
| defect_rate | **6%** | 3~10% 범위에서 config 조절 |
| 기간 | 2026-01-01 ~ 2026-08-31 | timestamp config 조절 가능 |
| random_seed | 42 고정 | reproducibility |

### 컬럼 스키마

- **식별자**: LOT_ID, PANEL_ID, PRODUCT_ID, PROCESS_ID, EQUIPMENT_ID, TIMESTAMP
- **품질(target/meta)**: DEFECT_FLAG(0/1), DEFECT_TYPE(NONE/MURA/PARTICLE/SCRATCH/OPEN_SHORT), QUALITY_SCORE
- **공정 변수**: TEMP_*, PRESSURE_*, SPEED_*, CURRENT_*, VOLTAGE_*, TIME_*, FLOW_*
- **설비 변수**: EQUIP_PARAM_*
- **검사 변수**: INSPECTION_*

---

## 6. Synthetic Data Generator

단순 random data가 아니라 실제 제조공정을 모사한다:

- 정상 데이터: 정규/skewed 분포, 일부 변수 간 correlation, 설비별 offset, 시간 drift
- **불량 mechanism (DEFECT_TYPE과 연결)**:

| Case | Mechanism | DEFECT_TYPE |
|---|---|---|
| 1 | TEMP_004 증가 → defect 확률 증가 (단일, 선형) | MURA |
| 2 | PRESSURE_012 증가 → defect 확률 증가 | MURA |
| 3 | TEMP_004 × PRESSURE_012 interaction → 확률 급증 | MURA |
| 4 | SPEED_007 특정 범위 이탈 → 불량 증가 (비선형, U자형) | SCRATCH |
| 5 | TIME_003 + TEMP_004 조합 → 불량 증가 | PARTICLE |
| 6 | 특정 EQUIPMENT_ID baseline shift → 불량 증가 | OPEN_SHORT |
| 7 | 시간에 따른 공정 drift → 후기 불량 증가 | PARTICLE |
| 8 | DEFECT와 무관하지만 유의 변수와 강한 상관을 갖는 confounder 변수 → false positive 검증용 | — |

즉 "단일 변수 / 비선형 / interaction / 설비 차이 / 시간 변화 / noise / confounder"가
모두 포함된 dataset. QUALITY_SCORE는 DEFECT_FLAG에서 파생(→ feature 제외 대상).

CLI 실행 지원: `python -m src.data.generator` → data/raw/*.parquet 생성.

---

## 7. 분석 기능

### 7.1 기본 품질 현황 (Overview)
전체 LOT/Sample 수, 정상/불량 수, Defect Rate, Yield,
제품·공정·설비별 Defect Rate, 시간별 Defect Trend.

### 7.2 단일 변수 유의차 분석
정상군 vs 불량군, 각 numerical feature에 대해:
- 기술통계: Mean, Median, Std, IQR, Min, Max, Difference, Difference %
- 검정 자동 선택: 정규성/분산 검토 → **t-test(Welch) 또는 Mann-Whitney U**;
  다군 비교(설비 등)는 ANOVA / Kruskal-Wallis
- **Effect size 짝짓기**: t-test → Cohen's d, Mann-Whitney → Cliff's delta
- **Multiple testing correction**: Benjamini-Hochberg (FDR)
- Ranking: p-value 단독이 아니라 significance + effect size + sample size + robustness 조합
- 결과 컬럼: Feature / Normal Mean / Defect Mean / Difference / Difference % /
  P-value / Adjusted P-value / Effect Size / Significance / Rank

### 7.3 Correlation Analysis
- Pearson / Spearman / Mutual Information (DEFECT_FLAG, QUALITY_SCORE 대상)
- Linear / Monotonic / Non-linear 관계 구분, Heatmap + ranking
- 화면에 disclaimer 필수: *"본 분석은 통계적 연관성을 탐색하는 목적이며,
  인과관계를 직접 증명하지 않습니다."*

### 7.4 Distribution Analysis
정상 vs 불량 분포 비교: Box / Violin / Histogram / KDE / Scatter / ECDF.

---

## 8. Machine Learning

- 모델: Logistic Regression(baseline), Random Forest, XGBoost, LightGBM
- 지표: Accuracy, Precision, Recall, F1, ROC-AUC, **PR-AUC(imbalance 주지표)**, Confusion Matrix
- Imbalance: class_weight / scale_pos_weight
- **Split 규칙 (기본값)**: `GroupShuffleSplit(groups=LOT_ID)` — 같은 LOT이 train/test에
  걸치지 않도록. Time split 옵션도 지원.
- 학습된 모델/SHAP 값은 `models/`에 joblib 저장, Streamlit에서 `st.cache_resource`로 재사용
  (rerun마다 재학습 금지).

## 9. Feature Importance & SHAP

- RF impurity / XGB·LGBM gain / SHAP importance → **Model consensus ranking** 표 제공
- SHAP 필수 화면: Summary Plot, Bar Plot, Dependence Plot, (가능하면) Interaction Plot
- "값이 높아질수록 defect prediction에 미치는 영향" 방향성 표현

## 10. Multivariate / Interaction / Rule Discovery (핵심 기능)

- 계산량 고려: **상위 10~20개 feature에 대해서만** interaction 분석
- 방법: SHAP interaction, 2D Partial Dependence, Decision Tree rule extraction
- Rule 형식 예: `TEMP_004 > 85.2 AND PRESSURE_012 > 1.25 → Sample=420, Defect Rate=14.8%`
- Rule ranking: Support / Defect Rate / Lift / Risk Increase / (가능하면) 통계 유의성

## 11. Leakage 방지 규칙 (명문화)

ML feature에서 **반드시 제외**: 모든 식별자(LOT_ID 등), TIMESTAMP,
DEFECT_FLAG, DEFECT_TYPE, **YIELD, QUALITY_SCORE** (target 파생 컬럼).
Data Validation 단계에서 target과 상관 0.95+ 인 feature는 leakage 의심 warning.

## 12. Equipment / Process / Time Series 분석

- Equipment·Process·Product별 Defect Rate, 주요 feature 분포 비교, equipment offset 탐지
- "Equipment B의 defect rate가 높다면 → 어떤 feature가 달라졌는지" drill-down
- Time series: feature trend, defect trend, rolling mean, process drift
- 향후 확장 고려: "불량 발생 시점 이전 10/30/60/120분 데이터 분석" 가능한 구조

## 13. Defect Risk Prediction

학습된 모델로 현재 공정 조건 입력 → defect probability + Risk Level(LOW/MID/HIGH)
+ 주요 위험 요인 top-N (SHAP 기반).

## 14. AI Report (MVP = template 기반)

**MVP의 report는 LLM이 아니라 template 기반 자동 문장 생성**이다 — 분석 결과
수치를 한국어 문장 템플릿에 채워 Executive Summary 등을 생성한다. LLM 기반
자연어 분석은 향후 확장 항목(§20).

사용자가 Product / Process / Defect Type / 기간 선택 → HTML report 자동 생성:

1. Executive Summary / 2. Defect Overview / 3. Significant Variables /
4. Correlation / 5. ML Feature Importance / 6. SHAP / 7. Multivariate Interaction /
8. Rule Discovery / 9. Equipment Analysis / 10. Time Trend / 11. Key Findings /
12. Recommended Investigation Items / 13. Disclaimer

## 15. Streamlit Dashboard

Sidebar: Dataset / Product / Process / Defect Type / Date Range / Model Selection

Pages (제조 엔지니어의 실제 분석 순서 반영):
1. Overview(불량 현황) → 2. Statistical Analysis(무엇이 다른가) →
3. Correlation(무엇과 관련 있는가) → 4. ML Analysis(어떤 변수가 중요한가) →
5. SHAP/Explainability → 6. Multivariate(어떤 조합이 문제인가) →
7. Equipment → 8. Time Series → 9. AI Report(무엇을 우선 조사할 것인가)

"Run Analysis" 버튼 → 자동 pipeline (validation → 통계 → ML → SHAP →
interaction → rule → report) 진행상태 표시.

## 16. Data Validation

Missing rate / Constant / Duplicate / Data type / Cardinality / Outlier /
Class imbalance / Target leakage 자동 검사. 문제 feature는 제외 또는 warning.

## 17. 성능 전략 (feature 1,000+ 대응)

3단계 filtering:
1. **1차**: missing / constant / variance filtering
2. **2차**: statistical significance / MI 기반 상위 feature 선별
3. **3차**: Top 10~20 feature만 interaction / SHAP interaction

## 18. 분석 철학 (결과 문구 규칙)

이 시스템은 "자동으로 원인을 확정하는 AI"가 아니다.

- ❌ "TEMP_004가 불량의 원인이다."
- ✅ "TEMP_004는 정상군과 불량군 간 통계적으로 유의한 차이가 확인되었으며,
  ML 기반 분석에서도 높은 중요도를 나타냈습니다. 추가적인 공정 검증을 권고합니다."

구조: **Correlation → Evidence → Hypothesis → Engineer Validation**

## 19. 코드 품질 / 테스트

- Modular 함수, type hint, docstring, logging, exception handling, config 분리,
  seed 관리, Streamlit session state + cache
- pytest: generator / validation / 통계 검정 / effect size / correlation /
  model training / SHAP / rule extraction / report 생성 테스트

## 20. 향후 확장 (MVP 제외)

실제 MES 연결, SQL DB, REST API, Spark/Kafka/Data Lake, LLM 자연어 분석,
자동 RCA, Anomaly Detection, PdM, Bayesian Network, Causal Inference,
Digital Twin, Prescriptive Analytics.

## 21. 디렉토리 구조

```
Defect-Insight-AI/
├── app.py                      # Streamlit 진입점
├── views/                      # home.py + 01_overview.py ~ 10_deeplearning.py (st.navigation)
├── src/
│   ├── data/                   # generator.py, loader.py, validator.py
│   ├── statistics/             # significance.py, correlation.py, effect_size.py
│   ├── ml/                     # trainer.py, evaluation.py, feature_importance.py
│   ├── explainability/         # shap_analysis.py
│   ├── multivariate/           # interaction.py, rule_discovery.py
│   ├── report/                 # report_generator.py
│   └── utils/                  # config.py
├── data/raw/ · data/processed/
├── models/                     # 학습된 모델 joblib 저장
├── reports/                    # 생성된 HTML report
├── tests/
├── docs/                       # 이 명세 등
├── requirements.txt · README.md · .gitignore · .env.example · CLAUDE.md
```

## 22. 개발 Phase

1. 프로젝트 구조 설계 → 2. Synthetic Generator → 3. Validation/Preprocessing →
4. Statistical Analysis → 5. ML → 6. SHAP → 7. Multivariate/Interaction →
8. Streamlit Dashboard → 9. Automated Report → 10. Testing/README

각 Phase 완료 시 실제 실행하여 오류 확인·수정. 코드 작성만이 아니라
**실행 가능한 상태**(데이터 생성 → Streamlit 구동 → 테스트 통과)까지 완성한다.

## 23. 최종 수용 기준 (Definition of Done)

- `python -m src.data.generator` 실행 → parquet 데이터 생성 확인
- `streamlit run app.py` 정상 구동 (headless HTTP 응답 확인)
- Run Analysis 실행 시: 심어 놓은 mechanism(TEMP_004, PRESSURE_012, SPEED_007,
  interaction, Equipment shift)이 **상위 유의 변수/rule로 실제 검출**되는지 확인
- Confounder 변수(Case 8)가 어떻게 나타나는지 확인 및 disclaimer 동작
- pytest 전체 통과
- README.md: 아키텍처, 실행 방법, 분석 알고리즘 설명, MES 연결 확장 방법

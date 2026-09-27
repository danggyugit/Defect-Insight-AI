"""홈 — 프로젝트 쇼케이스: 문제 정의, 아키텍처, 핵심 성과, 시연 가이드."""
from __future__ import annotations

import streamlit as st

from src.utils.config import RAW_DATASET_PATH
from src.utils.st_helpers import DISCLAIMER, ensure_dataset_ready

ensure_dataset_ready()  # 클라우드 최초 부팅 시 데이터 자동 생성

st.title("🏭 Manufacturing Defect Insight AI")
st.markdown(
    "##### 수백 개의 공정 변수 속에서 불량 영향인자를 자동 탐색하는 분석 플랫폼 "
    "— 통계 × Machine Learning × Explainable AI × Deep Learning"
)

if not RAW_DATASET_PATH.exists():
    st.error("데이터셋 생성에 실패했습니다. 페이지를 새로고침해 주세요.")

# ---------------------------------------------------------------- 문제 정의
st.markdown("---")
left, right = st.columns([3, 2])
with left:
    st.subheader("🎯 해결하려는 문제")
    st.markdown(
        """
디스플레이·반도체·2차전지부터 식품·화학까지, **공정 변수와 합격/불합격 판정이 쌓이는
제조업이라면 공통으로 겪는 문제**입니다 — 불량이 발생하면 엔지니어는 MES·FDC·검사
데이터의 **수백~수천 개 변수** 중 원인 후보를 찾아야 합니다. 사람이 변수를 하나씩 보는 방식은
수일이 걸리고, 단순 상관분석은 **비선형 관계와 변수 조합 효과, 가짜 상관(confounder)** 을
놓치거나 오인합니다.

이 플랫폼은 그 과정을 자동화합니다:

> **"수백 개 변수 → 유의 변수 → 위험 조합 → 고위험 조건 규칙 → 보고서"**
> 를 버튼 하나로, 통계적으로 검증된 방법으로.
"""
    )
with right:
    st.subheader("📌 설계 원칙")
    st.markdown(
        """
- **원인을 단정하지 않는다** — 모든 결과는 '검증 권고 후보'
  (Correlation → Evidence → Hypothesis → **Engineer Validation**)
- **점수를 부풀리지 않는다** — LOT 단위 Group Split, 다중검정 보정,
  target 파생 변수 제외 (data leakage 3중 방어)
- **설명 없는 AI는 쓰지 않는다** — 모든 모델에 SHAP/saliency 근거 제시
- **데이터 형태에 맞는 도구** — 표에는 tree 모델, 파형에는 CNN
"""
    )

# ---------------------------------------------------------------- 아키텍처
st.markdown("---")
st.subheader("🏗️ 분석 파이프라인")
st.graphviz_chart(
    """
digraph {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor="#eff6ff",
          fontname="sans-serif", fontsize=11];
    edge [color="#94a3b8"];
    subgraph cluster_0 { label="데이터"; style=dashed; color="#cbd5e0";
        A [label="공정 데이터\\n30,000건 × 400변수"];
        A2 [label="FDC 센서 파형\\n3센서 × 128step"];
    }
    B [label="자동 검증\\nmissing·leakage·불균형"];
    C [label="통계 분석\\n유의차·effect size·BH보정"];
    D [label="상관·MI\\n비선형 관계 탐지"];
    E [label="ML 4종\\nLR·RF·XGB·LGBM"];
    F [label="SHAP\\n판단 근거 설명", fillcolor="#fef3c7"];
    G [label="조합·규칙 탐색\\ninteraction·rule", fillcolor="#fef3c7"];
    H [label="1D-CNN\\n파형 이상 + saliency", fillcolor="#fce7f3"];
    I [label="자동 리포트\\nHTML/PDF", fillcolor="#dcfce7"];
    A -> B -> C -> D -> E -> F -> G -> I;
    A2 -> H -> I;
}
"""
)

# ---------------------------------------------------------------- 핵심 성과
st.markdown("---")
st.subheader("🏆 검증된 성과 (본 데모 데이터 실측)")
st.caption(
    "데모 데이터에는 8종의 불량 메커니즘이 **정답으로 심어져** 있어, "
    "파이프라인이 이를 실제로 찾아내는지 검증할 수 있습니다."
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("심어진 영향인자 검출", "6 / 6", help="TEMP_004, PRESSURE_012, SPEED_007(U자형), TIME_003, 설비 shift, 조합효과 — 전부 상위 검출")
c2.metric("변수 조합 규칙 발견", "불량률 9.0×", help="TEMP_004>83 AND PRESSURE_012>1.27 → 불량률 51% (전체 평균 5.7%의 9배)")
c3.metric("파형 딥러닝 성능", "PR-AUC 0.71", delta="+0.16 vs tabular", help="요약값 최고 모델(LightGBM 0.55) 대비 — 파형 정보의 보완 효과 실증")
c4.metric("가짜 상관 식별", "2건 경고", help="불량과 무관하지만 유의 변수와 강하게 상관된 confounder 2건을 화면 경고로 안내")

st.markdown(
    """
| 검증 항목 | 결과 |
|---|---|
| 평균 비교로 안 잡히는 **U자형 변수** (SPEED_007) | 유의차 361위 → **MI 분석·ML에서 4~8위로 검출** — 다각도 분석의 가치 실증 |
| **변수 조합 효과** (온도×압력) | Interaction scan 1위 + 규칙 자동 추출 ("A>83 AND B>1.27 → 불량 51%") |
| **설비 이상** (EQ_E) | 불량률 8.6% vs 평균 5.3% 검출 + 원인 변수 drill-down |
| **파형 이상** (spike·진동·drift) | 1D-CNN이 검출하고 saliency로 근거 구간 표시 |
| 신뢰성 장치 | LOT GroupSplit(누출 방지) · BH 다중검정 보정 · PR-AUC 주지표 · pytest 55종 |
"""
)

# ---------------------------------------------------------------- 시연 가이드
st.markdown("---")
st.subheader("🗺️ 5분 시연 경로 (처음 보시는 분께)")
st.markdown(
    """
왼쪽 메뉴가 곧 **제조 엔지니어의 실제 분석 순서**입니다. 빠르게 보시려면:

1. **📊 불량 현황** — 설비 EQ_E의 불량률이 유독 높은 것을 확인 *(30초)*
2. **📈 유의차** — 400개 변수 중 TEMP_004·PRESSURE_012가 상위 검출, 가짜 용의자 경고 확인 *(1분)*
3. **🔗 상관·MI** — 'Pearson vs MI' 산점도 좌상단에서 숨은 비선형 변수(SPEED_007) 발견 *(1분)*
4. **🧩 조합 위험** — 2D 히트맵 우상단 구석의 불량률 51% 확인, 자동 추출된 규칙 읽기 *(1분)*
5. **📄 리포트** — [Run Analysis] 클릭 → 전 과정 자동 실행 → 보고서 생성 *(1.5분)*

각 페이지 상단의 **"📖 이 분석, 쉽게 이해하기"** 를 펼치면 비전문가용 설명이 있습니다.
"""
)

# ---------------------------------------------------------------- 기술 스택
with st.expander("🛠️ 기술 스택 & 프로젝트 구성"):
    st.markdown(
        """
| 영역 | 사용 기술 |
|---|---|
| 통계 | scipy, statsmodels — Welch t / Mann-Whitney 자동 선택, Cohen's d / Cliff's delta, Benjamini-Hochberg |
| ML | scikit-learn, XGBoost, LightGBM — GroupShuffleSplit, class imbalance 대응 |
| XAI | SHAP (TreeExplainer, interaction), gradient×input saliency |
| 딥러닝 | PyTorch 1D-CNN (Apple Silicon MPS 가속) |
| 데이터 | pandas, pyarrow(parquet), 합성 데이터 생성기 (8종 불량 메커니즘 주입) |
| 앱/시각화 | Streamlit, Plotly, 자동 HTML 리포트 (Jinja2) |
| 품질 | pytest 55 tests (데이터 재현성·leakage 방지·페이지 렌더링 포함) |

**확장 설계**: 실 MES 연결은 loader 교체만으로 가능 · feature 1,000+ 대응 3단계 필터링 ·
Anomaly Detection / LLM 리포트 확장 구조 준비
"""
    )

with st.expander("📚 용어 미니 사전 (비전문가용)"):
    st.markdown(
        """
| 용어 | 쉬운 설명 |
|---|---|
| **p-value** | "이 차이가 우연일 확률". 낮을수록 진짜 차이 (관례적 기준 0.05) |
| **Effect Size** | 차이의 '실질적 크기'. 데이터가 많으면 사소한 차이도 p-value는 통과하므로 크기를 따로 봄 |
| **다중검정 보정** | 400번 시험하면 우연히 통과하는 게 나옴 → 그만큼 기준을 엄격하게 보정 (BH) |
| **Mutual Information** | 모양 불문 "이 변수를 알면 예측에 도움 되는가" — U자형 관계도 잡음 |
| **PR-AUC** | 불량이 희소할 때의 공정한 성적표 — "전부 정상" 꼼수에 속지 않음 |
| **GroupSplit** | 같은 LOT(쌍둥이 데이터)이 학습/시험에 갈라지는 '컨닝'을 막는 시험 출제 방식 |
| **SHAP** | AI 판단을 변수별 기여도로 분해 — "온도 때문에 +40%p, 압력 때문에 +25%p" |
| **Confounder** | 진범과 붙어 다녀서 범인처럼 보이는 무고한 변수 (가짜 상관) |
| **Lift** | "이 조건에서 불량률이 평균의 몇 배인가" |
| **Saliency** | 딥러닝 버전 SHAP — 파형의 어느 구간을 보고 판정했는지 표시 |
"""
    )

st.info(DISCLAIMER)

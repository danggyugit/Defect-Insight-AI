"""Page 3: Correlation Analysis — 무엇과 관련이 있는가?"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.statistics.correlation import correlation_matrix, run_correlation_analysis
from src.utils.config import RANDOM_SEED, TARGET_COLUMN
from src.utils.st_helpers import (
    GROUP_COLOR_MAP,
    cached_load_dataset,
    cached_validate,
    guard_empty,
    insight_box,
    page_header,
    render_sidebar_filters,
    show_disclaimer,
)

page_header(
    3, "🔗 무엇과 관련 있나 — 상관·MI",
    "선형 상관으로는 안 보이는 숨은(비선형) 연관 변수는 무엇인가?",
    "correlation",
)

df_all = cached_load_dataset()
df = render_sidebar_filters(df_all)
if guard_empty(df):
    st.stop()

validation = cached_validate(df_all)


@st.cache_data(show_spinner="상관/MI 분석 중...")
def cached_correlation(df_in: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    return run_correlation_analysis(df_in, list(features))


result = cached_correlation(df, tuple(validation.valid_features))

# ----------------------------------------------------- 핵심 발견 자동 요약
_type_counts = result["relationship_type"].value_counts()
_nonlinear = (
    result[result["relationship_type"] == "non-linear"]
    .nlargest(3, "mutual_info")
)
_insights = [
    "관계 유형 분포: " + " · ".join(
        f"{t} **{c}개**" for t, c in _type_counts.items()
    ),
]
if len(_nonlinear):
    _insights.append(
        "MI 상위인데 |Pearson|<0.1인 **숨은 비선형 후보**: "
        + ", ".join(
            f"**{r.feature}** (MI {r.mutual_info:.3f}, Pearson {r.pearson:+.2f})"
            for r in _nonlinear.itertuples()
        )
        + " — **평균 비교(유의차 페이지)로는 놓쳤을 변수**입니다. "
        "SHAP dependence plot에서 U자형 여부를 확인하세요"
    )
else:
    _insights.append(
        "현재 필터 조건에서는 비선형(non-linear)으로 분류된 변수가 없습니다 — "
        "상위 변수 대부분이 선형/단조 관계입니다"
    )
insight_box(_insights)

# --------------------------------------------------------- ranking table
st.subheader("Target(DEFECT_FLAG) 연관성 Ranking")
st.caption(
    "Pearson=선형, Spearman=단조, Mutual Information=비선형 포함 일반 연관성. "
    "**선형 상관이 0이어도 MI가 높으면 비선형 관계**(예: 범위 이탈형)일 수 있습니다."
)
top_n = st.slider("표시 개수", 10, 100, 30, step=10)
st.dataframe(
    result.head(top_n).style.format(
        {"pearson": "{:+.3f}", "spearman": "{:+.3f}", "mutual_info": "{:.4f}",
         "rank_score": "{:.3f}"},
        subset=[c for c in ("pearson", "spearman", "mutual_info", "rank_score")
                if c in result.columns],
    ),
    use_container_width=True, height=420,
)

# ------------------------------------------- relationship type 분포/산점도
left, right = st.columns(2)
with left:
    st.subheader("관계 유형 분포")
    type_counts = result["relationship_type"].value_counts().reset_index()
    fig = px.bar(type_counts, x="relationship_type", y="count", height=320)
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Pearson vs MI (비선형 후보 탐지)")
    fig = px.scatter(
        result, x=result["pearson"].abs(), y="mutual_info",
        color="relationship_type", hover_name="feature", height=320,
        labels={"x": "|Pearson|", "mutual_info": "Mutual Information"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "**좌상단 = 숨은 비선형 용의자** — |Pearson|은 낮은데 MI가 높은 변수로, "
        "선형 상관분석만 돌렸으면 영원히 놓쳤을 후보입니다. 이 산점도는 이걸 찾으려고 그립니다."
    )

# -------------------------------------------------------------- heatmap
st.markdown("---")
st.subheader("상위 Feature 간 Correlation Heatmap")
n_heat = st.slider("Heatmap feature 수", 5, 30, 15)
method = st.radio("방법", ["pearson", "spearman"], horizontal=True)
top_feats = result.head(n_heat)["feature"].tolist()
corr_mat = correlation_matrix(df, top_feats, method=method)
fig = px.imshow(
    corr_mat, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    height=550, aspect="auto",
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "💡 서로 강하게 상관된 feature 쌍(예: 검사값 ↔ 공정 온도)은 한쪽이 confounder일 수 있습니다. "
    "둘 다 유의하게 나오면 공정 지식으로 어느 쪽이 실제 조작 가능한 변수인지 판단하세요."
)

# ------------------------------------------------------- 산점도 drill-down
st.markdown("---")
st.subheader("Feature vs Defect 산점도")
feature = st.selectbox("Feature", result["feature"].head(50).tolist())
sample = df.sample(min(5000, len(df)), random_state=RANDOM_SEED)
fig = px.strip(
    sample, x=TARGET_COLUMN, y=feature, color=TARGET_COLUMN, height=380,
    color_discrete_map=GROUP_COLOR_MAP,
    labels={TARGET_COLUMN: "DEFECT_FLAG (0=정상, 1=불량)"},
)
st.plotly_chart(fig, use_container_width=True)

show_disclaimer()

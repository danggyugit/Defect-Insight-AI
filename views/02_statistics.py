"""Page 2: Statistical Analysis — 단일 변수 유의차 분석 + 분포 비교."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde

from src.statistics.significance import run_significance_analysis
from src.utils.config import RANDOM_SEED, SIGNIFICANCE_ALPHA, TARGET_COLUMN
from src.utils.st_helpers import (
    COLOR_DEFECT,
    COLOR_NORMAL,
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
    2, "📈 무엇이 다른가 — 유의차 분석",
    "400개 공정 변수 중 정상군과 불량군에서 통계적으로 다른 변수는 무엇인가?",
    "statistics",
)

df_all = cached_load_dataset()
df = render_sidebar_filters(df_all)
if guard_empty(df):
    st.stop()

validation = cached_validate(df_all)


@st.cache_data(show_spinner="유의차 분석 중... (400 features)")
def cached_significance(df_in: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    return run_significance_analysis(df_in, list(features))


result = cached_significance(df, tuple(validation.valid_features))

n_sig = int(result["significant"].sum())
c1, c2, c3 = st.columns(3)
c1.metric("분석 Feature 수", f"{len(result):,}")
c2.metric(f"유의 Feature (FDR {SIGNIFICANCE_ALPHA})", f"{n_sig:,}")
c3.metric("검정 방법", "Welch t / Mann-Whitney U 자동 선택")

_top3 = result.head(3)
_large = result[(result["significant"]) & (result["effect_label"] == "large")]
insight_box(
    [
        f"{len(result)}개 변수 중 **{n_sig}개**가 통계적으로 유의 (BH 보정 후), "
        f"그중 효과 크기 'large'는 **{len(_large)}개**",
        "최상위 후보: " + ", ".join(
            f"**{r.feature}** (불량군에서 {'높음 ↑' if r.difference > 0 else '낮음 ↓'}, "
            f"effect {r.effect_size:+.2f})"
            for r in _top3.itertuples()
        ),
        "⚠️ 상위에는 진짜 영향 변수와 강하게 상관된 **가짜 용의자(confounder)** 가 섞일 수 "
        "있습니다 — ML·SHAP 페이지와 교차 확인이 설계된 사용법입니다",
    ]
)

st.caption(
    "Ranking은 p-value 단독이 아니라 **BH 보정 유의성 × Effect Size** 조합 기준입니다. "
    "Effect size: t-test → Cohen's d, Mann-Whitney → Cliff's delta."
)

# ------------------------------------------------------------- 결과 테이블
st.subheader("유의차 Ranking")
top_n = st.slider("표시 개수", 10, 100, 30, step=10)
display_cols = [
    "rank", "feature", "normal_mean", "defect_mean", "difference",
    "difference_pct", "p_value", "adjusted_p_value", "effect_size",
    "effect_label", "test", "significant",
]
st.dataframe(
    result.head(top_n)[display_cols].style.format(
        {
            "normal_mean": "{:.3f}", "defect_mean": "{:.3f}",
            "difference": "{:+.3f}", "difference_pct": "{:+.1f}%",
            "p_value": "{:.2e}", "adjusted_p_value": "{:.2e}",
            "effect_size": "{:.3f}",
        }
    ),
    use_container_width=True, height=420,
)

# ------------------------------------------------------- Effect size 시각화
st.subheader("Effect Size 상위 Feature")
top_effect = result.head(20).iloc[::-1]
fig = px.bar(
    top_effect, x="effect_size", y="feature", orientation="h",
    color="significant", height=500,
    labels={"effect_size": "Effect Size (defect - normal 방향)"},
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------- 분포 비교
st.markdown("---")
st.subheader("분포 비교 (정상 vs 불량)")
feature = st.selectbox("Feature 선택", result["feature"].tolist())

normal_vals = df.loc[df[TARGET_COLUMN] == 0, feature].dropna()
defect_vals = df.loc[df[TARGET_COLUMN] == 1, feature].dropna()

row = result[result["feature"] == feature].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Normal Mean", f"{row['normal_mean']:.3f}")
c2.metric("Defect Mean", f"{row['defect_mean']:.3f}", delta=f"{row['difference_pct']:+.1f}%")
c3.metric("Adjusted p-value", f"{row['adjusted_p_value']:.2e}")
c4.metric("Effect Size", f"{row['effect_size']:.3f} ({row['effect_label']})")

plot_df = pd.DataFrame(
    {
        "value": pd.concat([normal_vals, defect_vals]),
        "group": ["Normal"] * len(normal_vals) + ["Defect"] * len(defect_vals),
    }
)

tab1, tab2, tab3, tab4 = st.tabs(["Histogram/KDE", "Box Plot", "Violin", "ECDF"])
with tab1:
    fig = go.Figure()
    grid = np.linspace(
        plot_df["value"].min(), plot_df["value"].max(), 200
    )
    for name, vals, color in (
        ("Normal", normal_vals, COLOR_NORMAL), ("Defect", defect_vals, COLOR_DEFECT)
    ):
        fig.add_histogram(
            x=vals, name=f"{name} (hist)", histnorm="probability density",
            opacity=0.35, marker_color=color, nbinsx=60,
        )
        if len(vals) > 5 and vals.std() > 0:
            kde = gaussian_kde(vals.sample(min(3000, len(vals)), random_state=RANDOM_SEED))
            fig.add_scatter(
                x=grid, y=kde(grid), name=f"{name} (KDE)", mode="lines",
                line={"color": color},
            )
    fig.update_layout(barmode="overlay", height=400)
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    fig = px.box(plot_df, x="group", y="value", color="group",
                 color_discrete_map=GROUP_COLOR_MAP, height=400)
    st.plotly_chart(fig, use_container_width=True)
with tab3:
    fig = px.violin(plot_df, x="group", y="value", color="group", box=True,
                    color_discrete_map=GROUP_COLOR_MAP, height=400)
    st.plotly_chart(fig, use_container_width=True)
with tab4:
    fig = px.ecdf(plot_df, x="value", color="group",
                  color_discrete_map=GROUP_COLOR_MAP, height=400)
    st.plotly_chart(fig, use_container_width=True)

show_disclaimer()
st.caption(
    "💡 참고: 유의차 상위에는 실제 영향 변수와 강하게 상관된 **confounder**(예: 검사값이 "
    "공정 온도를 반영하는 경우)가 함께 나타날 수 있습니다. ML/SHAP 분석과 교차 확인하세요."
)

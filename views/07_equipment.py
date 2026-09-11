"""Page 7: Equipment / Process / Product 분석 — 어느 설비가 다른가?"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.statistics.significance import run_significance_analysis
from src.utils.st_helpers import (
    COLOR_DEFECT,
    cached_load_dataset,
    cached_validate,
    guard_empty,
    insight_box,
    page_header,
    render_sidebar_filters,
    show_disclaimer,
)

page_header(
    7, "🏗️ 어느 설비가 다른가",
    "불량률이 높은 설비는 어디고, 그 설비의 어떤 조건이 다른가?",
    "equipment",
)

df_all = cached_load_dataset()
df = render_sidebar_filters(df_all)
if guard_empty(df):
    st.stop()

validation = cached_validate(df_all)
group_col = st.radio(
    "분석 단위", ["EQUIPMENT_ID", "PROCESS_ID", "PRODUCT_ID"], horizontal=True
)

# ------------------------------------------------ group별 defect rate
summary = (
    df.groupby(group_col)["DEFECT_FLAG"]
    .agg(defect_rate="mean", samples="count", defects="sum")
    .reset_index()
    .sort_values("defect_rate", ascending=False)
)
overall = df["DEFECT_FLAG"].mean()

fig = px.bar(
    summary, x=group_col, y="defect_rate", text_auto=".2%",
    hover_data=["samples", "defects"],
    title=f"{group_col}별 Defect Rate (점선=전체 평균 {overall:.2%})",
)
fig.update_traces(marker_color=COLOR_DEFECT, opacity=0.85)
fig.add_hline(y=overall, line_dash="dash", line_color="red")
fig.update_layout(yaxis_tickformat=".1%", height=380)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(
    summary.style.format({"defect_rate": "{:.2%}"}), use_container_width=True
)

# ------------------------------------------- 특정 그룹 vs 나머지 비교
st.markdown("---")
st.subheader(f"특정 {group_col} 은 무엇이 다른가? (drill-down)")
worst = summary.iloc[0][group_col]
target_group = st.selectbox(
    f"비교할 {group_col} (기본: defect rate 최고)",
    summary[group_col].tolist(),
    index=0,
)


@st.cache_data(show_spinner="그룹 간 feature 차이 분석 중...")
def group_feature_diff(
    df_in: pd.DataFrame, group_col_in: str, group_val: str, features: tuple[str, ...]
) -> pd.DataFrame:
    """선택 그룹 vs 나머지의 feature 평균 차이를 effect size로 정량화."""
    sub = df_in.copy()
    # '해당 그룹 여부'를 임시 target으로 두고 유의차 엔진 재사용
    sub["DEFECT_FLAG"] = (sub[group_col_in] == group_val).astype(int)
    return run_significance_analysis(sub, list(features))


diff = group_feature_diff(df, group_col, target_group, tuple(validation.valid_features))
diff = diff.rename(
    columns={"normal_mean": "others_mean", "defect_mean": f"{target_group}_mean"}
)

# 핵심 발견 자동 요약
_worst_row = summary.iloc[0]
_large_vars = diff[diff["effect_label"] == "large"]["feature"].head(3).tolist()
_insights = [
    f"불량률 최고 그룹: **{_worst_row[group_col]}** — **{_worst_row['defect_rate']:.2%}** "
    f"(전체 평균 {overall:.2%}의 **{_worst_row['defect_rate'] / overall:.1f}배**)"
    if overall > 0
    else f"불량률 최고 그룹: **{_worst_row[group_col]}** — {_worst_row['defect_rate']:.2%}",
]
if _large_vars:
    _insights.append(
        f"**{target_group}** 만 유독 값이 다른 변수(effect 'large'): "
        + ", ".join(f"**{v}**" for v in _large_vars)
        + " — 최우선 점검 후보"
    )
else:
    _insights.append(
        f"**{target_group}** 에서 effect 'large' 수준으로 다른 변수는 없습니다 — "
        "설비 조건 차이보다 다른 요인 가능성"
    )
insight_box(_insights)

st.caption(
    f"**{target_group}** 과 나머지 그룹 간 평균 차이가 큰 feature 순위입니다. "
    "설비 offset·calibration 차이 후보를 찾는 용도입니다. "
    "**읽는 법**: effect size **'large'** = 이 설비만 유독 값이 다른 변수 → 최우선 점검 후보."
)
st.dataframe(
    diff.head(15)[
        ["rank", "feature", "others_mean", f"{target_group}_mean",
         "difference_pct", "adjusted_p_value", "effect_size", "effect_label"]
    ].style.format(
        {
            "others_mean": "{:.3f}", f"{target_group}_mean": "{:.3f}",
            "difference_pct": "{:+.1f}%", "adjusted_p_value": "{:.2e}",
            "effect_size": "{:.3f}",
        }
    ),
    use_container_width=True, height=400,
)

# ---------------------------------------- 선택 feature 그룹별 분포
feature = st.selectbox("분포를 비교할 Feature", diff["feature"].head(30).tolist())
fig = px.box(df, x=group_col, y=feature, color=group_col, height=380)
st.plotly_chart(fig, use_container_width=True)

show_disclaimer()

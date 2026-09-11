"""Page 1: Overview — 기본 품질 현황 KPI."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.utils.st_helpers import (
    cached_load_dataset,
    guard_empty,
    insight_box,
    page_header,
    render_sidebar_filters,
)

page_header(
    1, "📊 불량 현황 한눈에",
    "불량이 어디서(설비·제품·공정), 언제(추이) 발생하고 있는가?",
    "overview",
)

df_all = cached_load_dataset()
df = render_sidebar_filters(df_all)
if len(df) == 0:
    st.warning("선택된 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# ---------------------------------------------------------------- KPI row
n_samples = len(df)
n_lots = df["LOT_ID"].nunique()
n_defect = int(df["DEFECT_FLAG"].sum())
defect_rate = df["DEFECT_FLAG"].mean()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("전체 LOT", f"{n_lots:,}")
c2.metric("전체 Sample", f"{n_samples:,}")
c3.metric("정상", f"{n_samples - n_defect:,}")
c4.metric("불량", f"{n_defect:,}")
c5.metric("Defect Rate", f"{defect_rate:.2%}")
c6.metric("Yield", f"{1 - defect_rate:.2%}")

# 핵심 발견 자동 요약
_eq = df.groupby("EQUIPMENT_ID")["DEFECT_FLAG"].mean()
_worst_eq, _worst_rate = _eq.idxmax(), _eq.max()
_monthly = df.set_index("TIMESTAMP")["DEFECT_FLAG"].resample("ME").mean().dropna()
_insights = [
    f"전체 불량률 **{defect_rate:.2%}** ({n_defect:,}건 / {n_samples:,}건)",
]
if _worst_rate > defect_rate * 1.2:
    _insights.append(
        f"설비 **{_worst_eq}** 의 불량률이 **{_worst_rate:.2%}** 로 전체 평균의 "
        f"{_worst_rate / defect_rate:.1f}배 — '어느 설비가 다른가' 페이지에서 원인 변수 확인 권장"
    )
if len(_monthly) >= 3 and _monthly.iloc[-1] > _monthly.iloc[0] * 1.3:
    _insights.append(
        f"불량률이 기간 초 {_monthly.iloc[0]:.1%} → 말 {_monthly.iloc[-1]:.1%}로 "
        "상승 추세 — 공정 drift 가능성, '시간 추이' 페이지 확인 권장"
    )
insight_box(_insights)

st.markdown("---")

# ------------------------------------------------- Defect rate by category
left, right = st.columns(2)

with left:
    st.subheader("제품 / 공정별 Defect Rate")
    by_product = df.groupby("PRODUCT_ID")["DEFECT_FLAG"].agg(["mean", "count"]).reset_index()
    fig = px.bar(
        by_product, x="PRODUCT_ID", y="mean", text_auto=".2%",
        labels={"mean": "Defect Rate"}, title="Product별",
    )
    fig.update_layout(yaxis_tickformat=".1%", height=300)
    st.plotly_chart(fig, use_container_width=True)

    by_process = df.groupby("PROCESS_ID")["DEFECT_FLAG"].agg(["mean", "count"]).reset_index()
    fig = px.bar(
        by_process, x="PROCESS_ID", y="mean", text_auto=".2%",
        labels={"mean": "Defect Rate"}, title="Process별",
    )
    fig.update_layout(yaxis_tickformat=".1%", height=300)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("설비별 Defect Rate")
    by_equip = df.groupby("EQUIPMENT_ID")["DEFECT_FLAG"].agg(["mean", "count"]).reset_index()
    overall = df["DEFECT_FLAG"].mean()
    fig = px.bar(
        by_equip, x="EQUIPMENT_ID", y="mean", text_auto=".2%",
        labels={"mean": "Defect Rate"}, title="Equipment별 (점선=전체 평균)",
    )
    fig.add_hline(y=overall, line_dash="dash", line_color="red")
    fig.update_layout(yaxis_tickformat=".1%", height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Defect Type 분포")
    type_counts = (
        df.loc[df["DEFECT_FLAG"] == 1, "DEFECT_TYPE"].value_counts().reset_index()
    )
    if len(type_counts):
        fig = px.pie(type_counts, names="DEFECT_TYPE", values="count", height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("불량 샘플이 없습니다.")

# ---------------------------------------------------------- Time trend
st.markdown("---")
st.subheader("시간별 Defect Trend")
freq = st.radio("집계 단위", ["일", "주"], horizontal=True, index=1)
rule = "D" if freq == "일" else "W"
trend = (
    df.set_index("TIMESTAMP")["DEFECT_FLAG"]
    .resample(rule)
    .agg(["mean", "count"])
    .rename(columns={"mean": "defect_rate", "count": "samples"})
    .reset_index()
)
fig = px.line(
    trend, x="TIMESTAMP", y="defect_rate", markers=True,
    labels={"defect_rate": "Defect Rate"},
)
fig.update_layout(yaxis_tickformat=".1%", height=350)
st.plotly_chart(fig, use_container_width=True)
st.caption("기간 후반부 상승 추세가 보이면 공정 drift 가능성을 Time Series 페이지에서 확인하세요.")

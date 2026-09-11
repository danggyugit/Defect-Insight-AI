"""Page 8: Time Series — 시간에 따라 어떻게 변하나? (drift 탐지)"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.st_helpers import (
    COLOR_DEFECT,
    COLOR_NORMAL,
    cached_load_dataset,
    cached_validate,
    guard_empty,
    insight_box,
    page_header,
    render_sidebar_filters,
    show_disclaimer,
)

page_header(
    8, "⏱️ 시간 추이 · Drift",
    "공정이 시간에 따라 서서히 틀어지고(drift) 있는가?",
    "timeseries",
)

df_all = cached_load_dataset()
df = render_sidebar_filters(df_all)
if guard_empty(df):
    st.stop()

validation = cached_validate(df_all)

# -------------------------------------------------- defect trend + rolling
st.subheader("Defect Trend")
rule = st.radio("집계 단위", ["D", "W"], format_func=lambda x: {"D": "일", "W": "주"}[x], horizontal=True)
trend = (
    df.set_index("TIMESTAMP")["DEFECT_FLAG"].resample(rule).mean().rename("defect_rate")
)
rolling = trend.rolling(4, min_periods=1).mean()

fig = go.Figure()
fig.add_scatter(x=trend.index, y=trend.values, mode="lines+markers", name="Defect Rate",
                line={"color": COLOR_DEFECT})
fig.add_scatter(x=rolling.index, y=rolling.values, mode="lines", name="Rolling Mean(4)",
                line={"dash": "dash", "color": COLOR_NORMAL})
fig.update_layout(yaxis_tickformat=".1%", height=350)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------- feature drift
st.markdown("---")
st.subheader("Feature Drift")
st.caption("기간을 전/후반으로 나눠 평균 변화가 큰 feature를 찾습니다 (drift 후보).")


@st.cache_data(show_spinner="drift 계산 중...")
def compute_drift(df_in: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    mid = df_in["TIMESTAMP"].quantile(0.5)
    early = df_in[df_in["TIMESTAMP"] < mid]
    late = df_in[df_in["TIMESTAMP"] >= mid]
    rows = []
    for f in features:
        e_mean, l_mean = early[f].mean(), late[f].mean()
        std = df_in[f].std()
        rows.append(
            {
                "feature": f,
                "early_mean": e_mean,
                "late_mean": l_mean,
                "drift_z": (l_mean - e_mean) / std if std > 0 else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    out["abs_drift"] = out["drift_z"].abs()
    return out.sort_values("abs_drift", ascending=False)


drift = compute_drift(df, tuple(validation.valid_features))

# 핵심 발견 자동 요약: drift 상위 변수 + 불량률 전/후반 변화
_mid_ts = df["TIMESTAMP"].quantile(0.5)
_early_rate = df.loc[df["TIMESTAMP"] < _mid_ts, "DEFECT_FLAG"].mean()
_late_rate = df.loc[df["TIMESTAMP"] >= _mid_ts, "DEFECT_FLAG"].mean()
_top_drift = drift.head(3)
_insights = [
    "이동량(|drift_z|) 상위 변수: " + ", ".join(
        f"**{r.feature}** ({r.drift_z:+.2f}σ)" for r in _top_drift.itertuples()
    ),
]
if _early_rate > 0 and _late_rate > _early_rate * 1.15:
    _insights.append(
        f"불량률이 전반 **{_early_rate:.2%}** → 후반 **{_late_rate:.2%}** 로 "
        f"**상승** ({_late_rate / _early_rate:.1f}배) — 위 drift 변수와의 동반 이동 여부를 "
        "아래 이중축 차트에서 확인 권장"
    )
elif _early_rate > 0 and _late_rate < _early_rate * 0.85:
    _insights.append(
        f"불량률이 전반 **{_early_rate:.2%}** → 후반 **{_late_rate:.2%}** 로 하락 — "
        "공정 개선 또는 필터 조건 영향 가능성"
    )
else:
    _insights.append(
        f"불량률은 전반 **{_early_rate:.2%}** → 후반 **{_late_rate:.2%}** 로 "
        "큰 변화 없이 **안정** — drift 변수가 있어도 불량률과 무관할 수 있음"
    )
insight_box(_insights)

st.caption("**읽는 법**: drift_z = 전/후반 평균 이동량(표준편차 단위). 절대값이 클수록 크게 움직인 변수.")
st.dataframe(
    drift.head(15)[["feature", "early_mean", "late_mean", "drift_z"]].style.format(
        {"early_mean": "{:.3f}", "late_mean": "{:.3f}", "drift_z": "{:+.3f}"}
    ),
    use_container_width=True,
)

# --------------------------------------------------- feature trend 시각화
feature = st.selectbox(
    "Trend를 볼 Feature", drift["feature"].head(50).tolist()
)
ft = (
    df.set_index("TIMESTAMP")
    .resample(rule)
    .agg(value=(feature, "mean"), defect_rate=("DEFECT_FLAG", "mean"))
    .reset_index()
)
fig = go.Figure()
fig.add_scatter(x=ft["TIMESTAMP"], y=ft["value"], name=feature, yaxis="y1",
                line={"color": COLOR_NORMAL})
fig.add_scatter(
    x=ft["TIMESTAMP"], y=ft["defect_rate"], name="Defect Rate", yaxis="y2",
    line={"color": COLOR_DEFECT, "dash": "dot"},
)
fig.update_layout(
    height=380,
    yaxis={"title": feature},
    yaxis2={"title": "Defect Rate", "overlaying": "y", "side": "right", "tickformat": ".1%"},
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "💡 향후 확장: 불량 발생 시점 이전 10/30/60/120분 window의 공정 데이터 분석은 "
    "TIMESTAMP 기반 구조가 이미 갖춰져 있어 window aggregation 추가로 구현 가능합니다."
)

show_disclaimer()

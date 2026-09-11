"""Page 5: SHAP / Explainability — 왜 그렇게 판단했는가?"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import shap
import streamlit as st

from src.explainability.shap_analysis import ShapResult, compute_shap, shap_importance
from src.ml.trainer import MODEL_BUNDLE_FILENAME, load_models
from src.utils.config import MODELS_DIR
from src.utils.st_helpers import (
    cached_load_dataset,
    guard_empty,
    insight_box,
    page_header,
    render_sidebar_filters,
    show_disclaimer,
)

page_header(
    5, "🔍 왜 그렇게 판단했나 — SHAP",
    "AI 판단의 근거는 무엇이며, 각 변수는 어느 방향으로 위험한가?",
    "shap",
)

df_all = cached_load_dataset()
df = render_sidebar_filters(df_all)
if guard_empty(df):
    st.stop()

result = load_models()
if result is None:
    st.warning("학습된 모델이 없습니다. **ML Analysis 페이지에서 먼저 학습하세요.**")
    st.stop()

st.caption(
    "SHAP 값 = 각 샘플의 예측(log-odds)에 대한 feature별 기여도. "
    "**+ 방향일수록 defect 예측 확률을 높이는 방향**입니다. "
    "성능을 위해 X_test에서 최대 2,000개 층화 샘플(불량은 모두 포함)만 계산합니다. "
    "분석 대상 모델은 **4. ML Analysis에서 마지막으로 학습된 조건** 기준입니다."
)

model_names = list(result.models)
default_idx = model_names.index("lightgbm") if "lightgbm" in model_names else 0
model_name = st.selectbox("모델 선택", model_names, index=default_idx)


@st.cache_resource(show_spinner="SHAP 계산 중... (최초 1회, 이후 캐시)")
def cached_shap(name: str, bundle_mtime: float | None) -> ShapResult:
    """모델별 SHAP 계산 캐시.

    bundle mtime을 캐시 키에 포함 — 04 페이지에서 재학습해 models/가 바뀌면
    stale SHAP을 반환하지 않도록 한다.
    """
    trained = load_models()
    if trained is None:
        raise RuntimeError("No trained models found")
    return compute_shap(trained, model_name=name)


_bundle = MODELS_DIR / MODEL_BUNDLE_FILENAME
shap_result = cached_shap(
    model_name, _bundle.stat().st_mtime if _bundle.exists() else None
)
importance = shap_importance(shap_result)
X_sample = shap_result.X_sample

# ----------------------------------------------------- 핵심 발견 자동 요약
def _direction_label(feat: str) -> str:
    """SHAP 값과 feature 값의 상관 부호로 위험 방향을 자동 판정한다."""
    sv = shap_result.shap_values[:, X_sample.columns.get_loc(feat)]
    vals = X_sample[feat].to_numpy(dtype=float)
    if np.std(vals) == 0 or np.std(sv) == 0:
        return "방향 판정 불가"
    c = float(np.corrcoef(vals, sv)[0, 1])
    if abs(c) < 0.2:
        return "비단조(U자형 등 의심) ⚠️"
    return "높을수록 위험 ↑" if c > 0 else "낮을수록 위험 ↓"


_top3_feats = importance.head(3)
insight_box(
    [
        f"**{model_name}** 모델의 SHAP 중요도 Top 3: " + ", ".join(
            f"**{feat}** (mean|SHAP| {val:.3f}, {_direction_label(feat)})"
            for feat, val in _top3_feats.items()
        ),
        "방향은 변수값–SHAP값 상관 부호로 자동 판정한 것입니다. "
        "'비단조 의심' 변수는 아래 dependence plot에서 U자형 여부를 꼭 확인하세요",
    ]
)

# ------------------------------------------------------- (1) SHAP bar plot
st.subheader("SHAP Feature Importance — mean(|SHAP|) 상위 20")
top20 = importance.head(20)
fig = px.bar(
    x=top20.values[::-1],
    y=top20.index[::-1],
    orientation="h",
    labels={"x": "mean(|SHAP value|)", "y": "feature"},
    height=550,
)
fig.update_traces(marker_color="#1f77b4")
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "💡 ML Analysis의 consensus ranking과 비교하세요. SHAP은 개별 예측 기여도의 "
    "평균이므로 gain/impurity importance보다 방향성 해석에 안전합니다."
)

# ------------------------------------------------ (2) SHAP summary beeswarm
st.markdown("---")
st.subheader("SHAP Summary Plot (beeswarm)")
st.caption(
    "각 점 = 샘플 1개. x축 = SHAP 값(+면 defect 방향), 색 = feature 값(빨강=높음, "
    "파랑=낮음). 예: 빨간 점이 오른쪽에 몰리면 '값이 높을수록 불량 위험 증가'."
)
plt.figure()
shap.summary_plot(
    shap_result.shap_values, X_sample, max_display=20, show=False,
)
st.pyplot(plt.gcf(), clear_figure=True)
plt.close("all")

# ---------------------------------------------------- (3) dependence plot
st.markdown("---")
st.subheader("Dependence Plot — feature 값 vs SHAP 값")

feature = st.selectbox("Feature 선택", importance.index.tolist(), index=0)
feat_idx = X_sample.columns.get_loc(feature)
shap_col = shap_result.shap_values[:, feat_idx]

# 색상: 선택 feature와 값의 상관이 가장 높은 다른 feature (interaction 후보)
corr = (
    X_sample.corrwith(X_sample[feature])
    .drop(labels=[feature])
    .abs()
    .sort_values(ascending=False)
)
color_feature = corr.index[0] if len(corr) else feature

fig = px.scatter(
    x=X_sample[feature],
    y=shap_col,
    color=X_sample[color_feature],
    color_continuous_scale="RdBu_r",
    labels={
        "x": f"{feature} 값",
        "y": f"SHAP value ({feature})",
        "color": color_feature,
    },
    opacity=0.6,
    height=500,
    title=f"{feature} — 색상: {color_feature} (|corr|={corr.iloc[0]:.2f})"
    if len(corr) else feature,
)
fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
st.plotly_chart(fig, use_container_width=True)

if feature.startswith("SPEED"):
    st.caption(
        "🎯 **데모 포인트**: SPEED_007 같은 **범위 이탈형(U자형)** 변수는 정상 범위의 "
        "양끝에서만 SHAP 값이 +로 튀어오릅니다. 평균 비교(통계 페이지)로는 잡히지 않는 "
        "비선형 패턴을 SHAP dependence plot이 드러내는 사례입니다."
    )

# ------------------------------------------------------- (4) 해석 안내
st.markdown("---")
st.subheader("해석 가이드")
st.markdown(
    """
- **SHAP 값이 + (0보다 큼)** → 해당 샘플에서 그 feature 값이 **defect 예측 확률을
  높이는 방향**으로 작용했다는 뜻입니다. −면 정상 방향입니다.
- **Bar plot**은 '평균적으로 얼마나 크게 작용했는가'(크기), **beeswarm**은
  '어느 방향으로, 값이 높을 때/낮을 때 어떻게'(방향)를 보여줍니다.
- **Dependence plot**에서 점 색이 뚜렷한 패턴(색 구분에 따라 SHAP 값이 갈라짐)을
  보이면 두 변수의 **interaction** 가능성이 있습니다 — Multivariate 페이지에서
  interaction 분석으로 확인하세요.
- SHAP은 모델의 판단 근거이지 물리적 인과가 아닙니다. 상위 feature는
  **공정 검증이 권고되는 후보**로 해석해야 합니다.
"""
)

show_disclaimer()

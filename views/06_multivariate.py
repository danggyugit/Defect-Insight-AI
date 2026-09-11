"""Page 6: Multivariate Analysis — 어떤 조합이 문제인가?"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ml.trainer import TrainResult, load_models, MODEL_BUNDLE_FILENAME
from src.multivariate.interaction import (
    interaction_scan,
    pairwise_defect_rate_grid,
    pairwise_sample_count_grid,
    partial_dependence_2d,
)
from src.multivariate.rule_discovery import discover_rules, rules_to_dataframe
from src.statistics.correlation import run_correlation_analysis
from src.statistics.significance import run_significance_analysis
from src.utils.config import MODELS_DIR, TOP_N_FEATURES_INTERACTION
from src.utils.st_helpers import (
    cached_load_dataset,
    cached_validate,
    guard_empty,
    insight_box,
    page_header,
    render_sidebar_filters,
    show_disclaimer,
)

page_header(
    6, "🧩 어떤 조합이 위험한가 — Interaction·Rule",
    "단일 변수로는 안 보이는 위험한 변수 조합·조건은 무엇인가?",
    "multivariate",
)

st.caption(
    "두 변수가 **동시에 특정 구간에 들어갈 때** defect rate가 급증하는 결합 패턴을 탐색합니다. "
    "성능 규칙에 따라 상위 feature만 대상으로 하되, **유의차(평균 차이) 상위와 "
    "MI 포함 correlation 상위를 병합**해 U자형 같은 비선형 변수가 누락되지 않게 합니다."
)

df_all = cached_load_dataset()
df = render_sidebar_filters(df_all)
if guard_empty(df):
    st.stop()

validation = cached_validate(df_all)


@st.cache_data(show_spinner="상위 feature 선정 중... (유의차 + MI)")
def cached_top_features(
    df_in: pd.DataFrame, features: tuple[str, ...], top_n: int
) -> list[str]:
    """interaction 분석 대상 feature 선정.

    평균 차이 기반 유의차 상위와 MI 포함 correlation rank_score 상위를 병합한다 —
    유의차 검정만 쓰면 U자형(비단조) 변수가 구조적으로 누락되기 때문.
    """
    sig = run_significance_analysis(df_in, list(features))
    corr = run_correlation_analysis(df_in, list(features))
    sig_top = sig.head(top_n)["feature"].tolist()
    corr_top = corr.head(top_n)["feature"].tolist()
    # 두 ranking을 interleave — 한쪽 기준에만 잡히는 변수(비선형 등)도 공평하게 포함
    merged: list[str] = []
    for a, b in zip(sig_top, corr_top):
        for f in (a, b):
            if f not in merged:
                merged.append(f)
    return merged[:top_n]


@st.cache_data(show_spinner="Interaction 전수 스캔 중...")
def cached_interaction_scan(
    df_in: pd.DataFrame, features: tuple[str, ...], top_k: int
) -> pd.DataFrame:
    return interaction_scan(df_in, list(features), top_k=top_k)


@st.cache_data(show_spinner="규칙 탐색 중... (Decision Tree)")
def cached_rules(
    df_in: pd.DataFrame, features: tuple[str, ...], max_depth: int
) -> pd.DataFrame:
    return rules_to_dataframe(
        discover_rules(df_in, list(features), max_depth=max_depth)
    )


top_features = cached_top_features(
    df, tuple(validation.valid_features), TOP_N_FEATURES_INTERACTION
)

# ------------------------------------------------------- 1. Interaction Scan
st.subheader("1️⃣ Interaction Scan — 결합 위험 쌍 탐색")
st.caption(
    f"선정된 상위 {len(top_features)}개 feature의 모든 쌍을 스캔합니다. "
    "**synergy** = 최고위험 cell의 실제 defect rate − 두 변수 각각의 marginal 기대치 조합 "
    "(additive 기대 대비 초과분). 클수록 두 변수의 **결합 효과**가 강한 후보입니다."
)

scan = cached_interaction_scan(df, tuple(top_features), top_k=15)
if scan.empty:
    st.warning("스캔 결과가 없습니다. 필터를 조정하거나 데이터 규모를 확인하세요.")
    st.stop()

# ----------------------------------------------------- 핵심 발견 자동 요약
_top_pair = scan.iloc[0]
_rules_preview = cached_rules(df, tuple(top_features), 3)
_insights = [
    f"결합 위험 1위 쌍: **{_top_pair['feature_a']} × {_top_pair['feature_b']}** — "
    f"최고위험 구간 불량률 **{_top_pair['max_cell_rate']:.1%}** "
    f"(전체 평균의 {_top_pair['max_cell_lift']:.2f}×), "
    f"synergy **{_top_pair['synergy']:+.1%}** (단독 효과의 합 대비 초과분)",
]
if len(_rules_preview):
    _r = _rules_preview.iloc[0]
    _insights.append(
        f"최상위 규칙: **{_r['rule']}** → 불량률 **{_r['defect_rate']:.1%}** "
        f"(lift **{_r['lift']:.2f}×**, 해당 {_r['samples']:,}건)"
    )
_insights.append(
    "각 변수 단독으로는 관리 범위 안이라 경보가 없는 **SPC 사각지대** 후보 — "
    "해당 조건에 걸린 LOT의 공정 이력 검증을 권고합니다"
)
insight_box(_insights)

st.dataframe(
    scan.style.format(
        {
            "max_cell_rate": "{:.1%}",
            "max_cell_lift": "{:.2f}×",
            "synergy": "{:+.1%}",
            "n_max_cell": "{:,}",
        }
    ),
    use_container_width=True,
    height=380,
)

# ------------------------------------------- 2. 2D Defect Rate Heatmap
st.markdown("---")
st.subheader("2️⃣ 2D Defect Rate Heatmap — 조합 구간별 불량률")

best_a = str(scan.iloc[0]["feature_a"])
best_b = str(scan.iloc[0]["feature_b"])
c1, c2, c3 = st.columns([2, 2, 1])
feature_a = c1.selectbox(
    "Feature A (행)", top_features, index=top_features.index(best_a)
)
feature_b = c2.selectbox(
    "Feature B (열)", top_features, index=top_features.index(best_b)
)
n_bins = c3.slider("Bin 수", 3, 8, 5)

if feature_a == feature_b:
    st.warning("서로 다른 두 feature를 선택하세요.")
else:
    grid = pairwise_defect_rate_grid(df, feature_a, feature_b, n_bins=n_bins)
    counts = pairwise_sample_count_grid(df, feature_a, feature_b, n_bins=n_bins)

    overall_rate = float(df["DEFECT_FLAG"].mean())
    fig = px.imshow(
        grid,
        text_auto=".1%",
        color_continuous_scale="Reds",
        aspect="auto",
        labels={"color": "Defect Rate"},
        title=f"{feature_a} × {feature_b} — cell별 defect rate (전체 평균 {overall_rate:.1%})",
    )
    fig.update_traces(
        customdata=counts.to_numpy(),
        hovertemplate=(
            f"{feature_a}: %{{y}}<br>{feature_b}: %{{x}}<br>"
            "Defect Rate: %{z:.2%}<br>샘플 수: %{customdata:,}<extra></extra>"
        ),
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "각 축은 quantile bin(구간 경계값 라벨)입니다. cell hover 시 샘플 수를 함께 표시합니다. "
        "**우상단 구석만 진하게 붉으면 전형적인 결합 위험** — 두 변수 각각은 괜찮은데 "
        "동시에 높은 구간에서만 불량률이 급증하는 패턴입니다."
    )

# ------------------------------------------- 3. 2D Partial Dependence
st.markdown("---")
st.subheader("3️⃣ 2D Partial Dependence — 모델이 학습한 결합 효과")

@st.cache_resource(show_spinner="저장된 모델 로딩 중...")
def cached_load_models(bundle_mtime: float | None) -> TrainResult | None:
    """models/의 TrainResult 캐시 로딩 (파일 mtime 변경 시 재로딩)."""
    return load_models()


_bundle = MODELS_DIR / MODEL_BUNDLE_FILENAME
train_result: TrainResult | None = cached_load_models(
    _bundle.stat().st_mtime if _bundle.exists() else None
)
if train_result is None:
    st.info(
        "저장된 모델이 없습니다. **4. ML Analysis** 페이지에서 모델을 먼저 학습하세요. "
        "학습된 모델은 models/에 저장되어 이 페이지에서 재사용됩니다."
    )
elif feature_a == feature_b:
    st.info("위에서 서로 다른 두 feature를 선택하면 partial dependence를 계산합니다.")
elif feature_a not in train_result.feature_names or feature_b not in train_result.feature_names:
    st.warning(
        "선택한 feature가 학습된 모델의 feature 집합에 없습니다. "
        "4. ML Analysis 페이지에서 재학습 후 다시 시도하세요."
    )
else:
    model_name = st.selectbox("모델 선택", list(train_result.models))

    @st.cache_data(show_spinner="Partial dependence 계산 중... (1000 샘플)")
    def cached_pd_2d(
        name: str, fa: str, fb: str, bundle_mtime: float | None
    ) -> pd.DataFrame:
        """bundle mtime을 캐시 키에 포함 — 재학습 시 stale 결과 방지."""
        return partial_dependence_2d(train_result, name, fa, fb)

    pd_grid = cached_pd_2d(
        model_name, feature_a, feature_b,
        _bundle.stat().st_mtime if _bundle.exists() else None,
    )
    fig = px.imshow(
        pd_grid,
        text_auto=".1%",
        color_continuous_scale="Reds",
        aspect="auto",
        labels={"color": "Predicted P(defect)"},
        title=f"{model_name}: {feature_a} × {feature_b} partial dependence",
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "**모델이 학습한 결합 효과**입니다 — 두 변수 값을 grid로 움직이며 나머지 변수를 "
        "고정했을 때의 평균 예측 불량 확률(X_train 1000개 샘플 기준). "
        "위의 실측 heatmap과 패턴이 일치하면 결합 효과의 신뢰도가 높아집니다."
    )

# ------------------------------------------------- 4. Rule Discovery
st.markdown("---")
st.subheader("4️⃣ Rule Discovery — 불량 조건 규칙 추출")
st.caption(
    "Decision Tree 기반으로 'TEMP_004 > 85.2 AND PRESSURE_012 > 1.25' 형태의 "
    "고위험 조건 규칙을 추출합니다. p-value는 Fisher exact (규칙 내 vs 규칙 외)."
)

max_depth = st.slider("규칙 조건 최대 개수 (tree max_depth)", 2, 4, 3)
rules_df = cached_rules(df, tuple(top_features), max_depth)

if rules_df.empty:
    st.warning("조건을 만족하는 규칙이 없습니다 (lift ≥ 1.5, support ≥ 100).")
else:
    st.dataframe(
        rules_df.style.format(
            {
                "defect_rate": "{:.1%}",
                "lift": "{:.2f}×",
                "risk_increase": "{:+.1%}",
                "p_value": "{:.2e}",
                "samples": "{:,}",
                "defects": "{:,}",
            }
        ),
        use_container_width=True,
        height=380,
    )
    st.caption(
        "💡 규칙은 '해당 조건에서 통계적으로 유의한 불량률 상승 확인 → 공정 검증 권고' "
        "후보이며, 원인 단정이 아닙니다. SHAP 페이지의 개별 변수 기여도와 교차 확인하세요."
    )

show_disclaimer()

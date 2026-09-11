"""Page 4: ML Analysis — 어떤 변수가 중요한가?"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ml.evaluation import confusion_matrices, evaluate_models, roc_pr_curves
from src.ml.feature_importance import compute_importances, consensus_ranking
from src.ml.trainer import save_models, train_models
from src.utils.st_helpers import (
    cached_load_dataset,
    guard_empty,
    insight_box,
    page_header,
    render_sidebar_filters,
    show_disclaimer,
)

page_header(
    4, "🤖 어떤 변수가 중요한가 — ML 예측 모델",
    "400개 변수를 동시에 학습한 AI는 어떤 변수를 중요하게 보는가?",
    "ml",
)

df_all = cached_load_dataset()
df = render_sidebar_filters(df_all)
if guard_empty(df):
    st.stop()

st.caption(
    "Split: **GroupShuffleSplit(LOT_ID)** — 같은 LOT이 train/test에 걸치지 않도록 "
    "leakage를 방지합니다. Imbalance 대응: class_weight / scale_pos_weight. "
    "주지표: **PR-AUC**."
)

split = st.radio("Split 방법", ["group", "time"], horizontal=True,
                 format_func=lambda x: {"group": "Group (LOT_ID)", "time": "Time (앞 75%)"}[x])


@st.cache_resource(show_spinner="모델 학습 중... (LR/RF/XGB/LGBM)")
def cached_train(filter_key: tuple, split_method: str):
    """필터링된 데이터로 4개 모델 학습.

    캐시 키는 sidebar 필터 선택값 자체 — 행수 같은 파생값을 쓰면 서로 다른
    필터 조합이 우연히 같은 키가 되어 stale 모델을 돌려줄 수 있다.
    """
    result = train_models(df, split=split_method)
    save_models(result)
    return result


_dates = st.session_state.get("filter_dates")
filter_key = (
    tuple(sorted(st.session_state.get("filter_products") or [])),
    tuple(sorted(st.session_state.get("filter_processes") or [])),
    tuple(sorted(st.session_state.get("filter_defect_types") or [])),
    tuple(str(d) for d in _dates) if isinstance(_dates, tuple) else str(_dates),
)
if st.button("🚀 모델 학습 / 재학습", type="primary") or "ml_trained" in st.session_state:
    st.session_state["ml_trained"] = True
    result = cached_train(filter_key, split)

    # ------------------------------------------------------ 성능 지표
    st.subheader("모델 성능")
    metrics = evaluate_models(result)
    st.dataframe(metrics.round(4), use_container_width=True)
    st.caption(
        "지표 읽는 법 — **Precision**=불량이라 한 것 중 진짜 비율 · "
        "**Recall**=실제 불량 중 잡아낸 비율 · "
        "**PR-AUC**=불량 희소 상황의 공정한 종합 점수(주지표)"
    )

    best_model = metrics["pr_auc"].idxmax()

    importances = compute_importances(result)
    consensus = consensus_ranking(importances)
    _top3 = consensus.head(3)["feature"].tolist()
    insight_box(
        [
            f"PR-AUC 최고 모델: **{best_model}** "
            f"(**{metrics.loc[best_model, 'pr_auc']:.4f}**) — "
            f"Recall {metrics.loc[best_model, 'recall']:.2%}, "
            f"Precision {metrics.loc[best_model, 'precision']:.2%}",
            "4개 모델 합의(consensus) 중요도 Top 3: "
            + ", ".join(f"**{f}**" for f in _top3),
            "이 Top 3가 통계 유의차 페이지 상위와 겹치면 신뢰도가 높고, "
            "여기서만 상위인 변수는 **비선형 신호**일 수 있습니다 — "
            "SHAP 페이지에서 판단 근거와 방향을 확인하세요",
        ]
    )

    # ------------------------------------------------------ ROC/PR curve
    curves = roc_pr_curves(result)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for name, c in curves.items():
            fig.add_scatter(x=c["fpr"], y=c["tpr"], name=name, mode="lines")
        fig.add_scatter(x=[0, 1], y=[0, 1], line={"dash": "dash", "color": "gray"},
                        showlegend=False)
        fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        for name, c in curves.items():
            fig.add_scatter(x=c["recall"], y=c["precision"], name=name, mode="lines")
        fig.update_layout(title="PR Curve", xaxis_title="Recall", yaxis_title="Precision",
                          height=380)
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------- confusion matrix
    with st.expander("Confusion Matrix"):
        cms = confusion_matrices(result)
        cols = st.columns(len(cms))
        for col, (name, cm) in zip(cols, cms.items()):
            with col:
                fig = px.imshow(
                    cm, text_auto=True, title=name,
                    x=["예측 정상", "예측 불량"], y=["실제 정상", "실제 불량"],
                    color_continuous_scale="Blues",
                )
                fig.update_layout(height=300, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------ feature importance
    st.markdown("---")
    st.subheader("Feature Importance — Model Consensus Ranking")
    st.dataframe(consensus.head(20).round(4), use_container_width=True, height=420)

    top10 = consensus.head(10)["feature"].tolist()
    melted = importances[importances["feature"].isin(top10)].melt(
        id_vars="feature", var_name="model", value_name="importance"
    )
    fig = px.bar(
        melted, x="importance", y="feature", color="model", barmode="group",
        orientation="h", height=500, category_orders={"feature": top10[::-1]},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "💡 통계 유의차 페이지 상위와 비교하세요. 통계에서 안 보이던 변수가 여기서 "
        "상위에 있다면 **비선형 관계**(범위 이탈형 등)일 수 있습니다 — SHAP 페이지에서 확인."
    )
else:
    st.info("위 버튼을 눌러 모델을 학습하세요. 학습된 모델은 SHAP/Multivariate 페이지에서 재사용됩니다.")

show_disclaimer()

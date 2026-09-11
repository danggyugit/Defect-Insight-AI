"""Page 10: Deep Learning — FDC 센서 trace 파형 분석 (1D-CNN)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data.trace_generator import (
    SENSOR_NAMES,
    TRACE_PATH,
    load_traces,
)
from src.dl.trace_model import (
    TraceTrainResult,
    compute_saliency,
    save_trace_model,
    train_trace_model,
)
from src.ml.evaluation import evaluate_models
from src.ml.trainer import load_models
from src.utils.config import RANDOM_SEED
from src.utils.st_helpers import (
    COLOR_DEFECT,
    COLOR_NORMAL,
    cached_load_dataset,
    insight_box,
    page_header,
    show_disclaimer,
)

page_header(
    10, "🧠 파형 이상 탐지 — 딥러닝(1D-CNN)",
    "요약 숫자에서 사라진 파형의 이상을 딥러닝이 잡아낼 수 있는가?",
    "deeplearning",
)

st.markdown(
    """
이 페이지는 **FDC 센서 raw trace**(샘플당 3개 센서 × 128 timestep 파형)를 다룹니다 —
spike, 진동(oscillation), drift, level shift 같은 **파형 형태 이상**은 요약 통계로는
잡기 어렵고, **1D-CNN**이 파형에서 직접 학습합니다.
(2~6페이지의 요약값 기반 분석에는 tree 모델이 최적 — 데이터 형태에 맞는 도구 선택)
"""
)

df = cached_load_dataset()

if not TRACE_PATH.exists():
    st.error(
        "FDC trace 데이터가 없습니다. 터미널에서 먼저 실행하세요:\n\n"
        "```\npython -m src.data.trace_generator\n```"
    )
    st.stop()


@st.cache_data(show_spinner="Trace 데이터 로딩 중... (40MB)")
def cached_traces() -> tuple[np.ndarray, np.ndarray]:
    return load_traces()


traces, panel_ids = cached_traces()
# PANEL_ID → df row 정렬 (trace 순서 기준)
order = df.set_index("PANEL_ID").loc[panel_ids].reset_index()

# =====================================================================
# 1. 파형 예시 — 정상 vs defect type별
# =====================================================================
st.subheader("1️⃣ 파형 예시 — 정상 vs 불량 유형별")
st.caption(
    "각 불량 유형에 심어진 파형 이상: MURA=완만한 drift, PARTICLE=transient spike, "
    "SCRATCH=고주파 진동, OPEN_SHORT=계단형 level shift. "
    "평균값은 거의 같아서 **표(요약 통계) 분석으로는 구분 불가 — 모양이 다를 뿐**인 신호입니다. "
    "(파랑=정상 파형, 빨강=불량 유형 파형)"
)

rng = np.random.default_rng(RANDOM_SEED)
sensor_idx = st.selectbox(
    "센서 선택", range(len(SENSOR_NAMES)), format_func=lambda i: SENSOR_NAMES[i]
)

type_options = ["NONE(정상)", "MURA", "PARTICLE", "SCRATCH", "OPEN_SHORT"]
fig = make_subplots(rows=1, cols=len(type_options), subplot_titles=type_options,
                    shared_yaxes=True)
for col, tname in enumerate(type_options, start=1):
    key = "NONE" if tname.startswith("NONE") else tname
    line_color = COLOR_NORMAL if key == "NONE" else COLOR_DEFECT
    pool = order.index[
        (order["DEFECT_TYPE"] == key)
        & (order["DEFECT_FLAG"] == (0 if key == "NONE" else 1))
    ].to_numpy()
    for i in rng.choice(pool, size=min(3, len(pool)), replace=False):
        fig.add_scatter(
            y=traces[i, sensor_idx], mode="lines", opacity=0.7,
            line={"width": 1.2, "color": line_color}, showlegend=False, row=1, col=col,
        )
fig.update_layout(height=300, margin={"t": 40, "b": 20})
st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# 2. 1D-CNN 학습
# =====================================================================
st.markdown("---")
st.subheader("2️⃣ 1D-CNN 학습 및 성능")
st.caption(
    "Split: tabular ML과 동일하게 **GroupShuffleSplit(LOT_ID)** — LOT leakage 방지. "
    "Imbalance: BCEWithLogitsLoss(pos_weight). 정규화 파라미터는 train set에서만 계산."
)

epochs = st.slider("Epochs", 4, 20, 8)


@st.cache_resource(show_spinner="1D-CNN 학습 중... (CPU/MPS, 약 1~2분)")
def cached_train_cnn(n_epochs: int, trace_mtime: float) -> TraceTrainResult:
    result = train_trace_model(df, traces, panel_ids, epochs=n_epochs)
    save_trace_model(result)
    return result


if st.button("🚀 CNN 학습 / 재학습", type="primary") or "dl_trained" in st.session_state:
    st.session_state["dl_trained"] = True
    result = cached_train_cnn(epochs, TRACE_PATH.stat().st_mtime)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{result.metrics['roc_auc']:.4f}")
    c2.metric("PR-AUC (주지표)", f"{result.metrics['pr_auc']:.4f}")
    c3.metric("Precision", f"{result.metrics['precision']:.4f}")
    c4.metric("Recall", f"{result.metrics['recall']:.4f}")
    c5.metric("F1", f"{result.metrics['f1']:.4f}")

    # tabular 모델 비교 (insight + 우측 차트에서 공용)
    tabular = load_models()
    tab_metrics = (
        evaluate_models(tabular)[["roc_auc", "pr_auc"]] if tabular is not None else None
    )

    # 핵심 발견 자동 요약
    _cnn_pr = result.metrics["pr_auc"]
    if tab_metrics is not None and len(tab_metrics):
        _best_tab = tab_metrics["pr_auc"].idxmax()
        _best_pr = tab_metrics["pr_auc"].max()
        _ratio_txt = f" ({_cnn_pr / _best_pr:.1f}배)" if _best_pr > 0 else ""
        insight_box(
            [
                f"파형을 본 CNN의 PR-AUC **{_cnn_pr:.3f}** vs 요약값만 본 최고 tabular 모델 "
                f"**{_best_tab}** {_best_pr:.3f}{_ratio_txt}",
                "차이는 모델 우열이 아니라 **입력 정보의 차이** — 요약 숫자에서 사라진 "
                "파형 정보(spike·진동·drift·level shift)를 CNN이 보완적으로 잡아낸다는 증거",
            ]
        )
    else:
        insight_box(
            [
                f"CNN PR-AUC **{_cnn_pr:.3f}** — 요약 통계로는 보이지 않는 파형 이상을 "
                "직접 학습한 결과",
                "tabular 모델과 비교하려면 4페이지(ML)에서 모델을 먼저 학습하세요",
            ]
        )

    left, right = st.columns(2)
    with left:
        fig = px.line(
            y=result.history, markers=True, height=280,
            labels={"index": "Epoch", "y": "Train Loss"}, title="학습 곡선",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        # tabular 모델과 비교
        if tab_metrics is not None:
            comp = pd.concat(
                [
                    tab_metrics,
                    pd.DataFrame(
                        {"roc_auc": [result.metrics["roc_auc"]],
                         "pr_auc": [result.metrics["pr_auc"]]},
                        index=["trace_cnn (DL)"],
                    ),
                ]
            ).reset_index(names="model")
            fig = px.bar(
                comp.melt(id_vars="model", var_name="metric"), x="model", y="value",
                color="metric", barmode="group", height=280,
                title="Tabular ML vs Trace CNN",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "⚠️ 공정한 비교가 아닙니다 — tabular 모델은 요약값만, CNN은 파형을 봅니다. "
                "**서로 다른 정보를 보완적으로 쓴다**는 것이 핵심입니다."
            )
        else:
            st.info("tabular 모델 비교를 보려면 4페이지에서 모델을 먼저 학습하세요.")

    # =================================================================
    # 3. Saliency — 판단 근거 시간 구간
    # =================================================================
    st.markdown("---")
    st.subheader("3️⃣ Saliency — 파형의 어느 구간이 판단 근거인가?")
    st.caption(
        "gradient × input 기반 saliency입니다. SHAP이 'feature별 기여도'를 보여주듯, "
        "saliency는 '**파형의 어느 시간 구간·어느 센서**가 불량 판정에 기여했는지'를 보여줍니다."
    )

    # test set에서 확률 높은 불량 샘플 상위
    hit_order = np.argsort(-result.test_probs)
    hits = [
        int(result.test_indices[i]) for i in hit_order
        if result.test_labels[i] == 1
    ][:20]
    if not hits:
        st.warning("test set에 불량 샘플이 없습니다.")
    else:
        labels = {
            i: (f"{order.loc[i, 'PANEL_ID']} — {order.loc[i, 'DEFECT_TYPE']} "
                f"(P={result.test_probs[np.where(result.test_indices == i)[0][0]]:.2f})")
            for i in hits
        }
        chosen = st.selectbox(
            "불량 샘플 선택 (예측 확률 높은 순)", hits, format_func=lambda i: labels[i]
        )
        saliency = compute_saliency(result, traces, np.array([chosen]))[0]

        fig = make_subplots(
            rows=len(SENSOR_NAMES), cols=1, shared_xaxes=True,
            subplot_titles=[
                f"{name} (saliency 합 {saliency[k].sum():.1f})"
                for k, name in enumerate(SENSOR_NAMES)
            ],
        )
        for k in range(len(SENSOR_NAMES)):
            fig.add_scatter(
                y=traces[chosen, k], mode="lines", name=SENSOR_NAMES[k],
                line={"color": COLOR_NORMAL}, showlegend=False, row=k + 1, col=1,
            )
            # saliency를 배경 강조로 표시
            sal = saliency[k]
            if sal.max() > 0:
                sal_norm = sal / sal.max()
                fig.add_bar(
                    y=sal_norm * traces[chosen, k].max(), opacity=0.25,
                    marker_color=COLOR_DEFECT, showlegend=False, row=k + 1, col=1,
                )
        fig.update_layout(height=550, bargap=0, title="파란선=파형, 붉은 배경=saliency(판단 근거 구간)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "💡 PARTICLE 샘플은 spike 위치에, SCRATCH는 진동 구간 전체에, "
            "OPEN_SHORT는 level shift 지점에 saliency가 몰리면 모델이 "
            "심어진 파형 이상을 실제로 근거로 사용하고 있다는 뜻입니다."
        )
else:
    st.info("버튼을 눌러 CNN을 학습하세요. 학습된 모델은 models/trace_cnn.pt에 저장됩니다.")

show_disclaimer()

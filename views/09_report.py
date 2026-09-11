"""Page 9: AI Report — 자동 분석 파이프라인 실행 + HTML report 생성."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.validator import validate_dataset
from src.explainability.shap_analysis import compute_shap, shap_importance
from src.ml.evaluation import evaluate_models
from src.ml.feature_importance import compute_importances, consensus_ranking
from src.ml.trainer import train_models
from src.multivariate.interaction import interaction_scan
from src.multivariate.rule_discovery import discover_rules, rules_to_dataframe
from src.report.report_generator import ReportInput, generate_report
from src.statistics.correlation import run_correlation_analysis
from src.statistics.significance import run_significance_analysis
from src.utils.config import TOP_N_FEATURES_INTERACTION
from src.utils.st_helpers import (
    cached_load_dataset,
    guard_empty,
    insight_box,
    page_header,
    render_sidebar_filters,
)

page_header(
    9, "📄 자동 분석 리포트",
    "전체 분석을 누구나 읽을 수 있는 보고서로 어떻게 전달하는가?",
    "report",
)

df_all = cached_load_dataset()
df = render_sidebar_filters(df_all)
if guard_empty(df):
    st.stop()

st.markdown(
    "사이드바에서 Product / Process / Defect Type / 기간을 선택한 뒤 버튼을 누르세요.\n\n"
    "리포트는 template 기반 자동 문장 생성이며(LLM 아님), HTML로 저장되어 "
    "브라우저 인쇄로 PDF 변환이 가능합니다."
)

# 필터 조건 설명 문자열
_p = st.session_state.get("filter_products") or ["전체 제품"]
_pr = st.session_state.get("filter_processes") or ["전체 공정"]
_dt = st.session_state.get("filter_defect_types") or ["전체 불량유형"]
_dates = st.session_state.get("filter_dates")
_period = (
    f"{_dates[0]} ~ {_dates[1]}"
    if isinstance(_dates, tuple) and len(_dates) == 2
    else f"{df['TIMESTAMP'].min().date()} ~ {df['TIMESTAMP'].max().date()}"
)
filter_desc = (
    f"{', '.join(_p)} / {', '.join(_pr)} / {', '.join(_dt)} / "
    f"{_period} / {len(df):,} samples"
)
st.caption(f"현재 분석 조건: **{filter_desc}**")

_filter_key = (tuple(sorted(_p)), tuple(sorted(_pr)), tuple(sorted(_dt)), _period)


@st.cache_resource(show_spinner=False)
def cached_report_train(filter_key: tuple, features: tuple[str, ...]):
    """report 파이프라인용 모델 학습 캐시 (같은 필터로 재클릭 시 재학습 방지)."""
    return train_models(df, features=list(features))

st.caption(
    "이 버튼 하나로 **검증 → 통계 → 상관 → ML → SHAP → 조합 → 규칙 → 리포트**가 "
    "자동 실행됩니다 (약 1~2분)."
)
if st.button("🚀 Run Analysis", type="primary"):
    progress = st.progress(0, text="1/9 Data Validation...")
    validation = validate_dataset(df)
    features = validation.valid_features
    if validation.warnings:
        for w in validation.warnings[:5]:
            st.warning(w)

    progress.progress(10, text="2/9 Statistical Significance (BH 보정)...")
    sig = run_significance_analysis(df, features)

    progress.progress(22, text="3/9 Correlation / Mutual Information...")
    corr = run_correlation_analysis(df, features)

    progress.progress(35, text="4/9 ML Training (LR/RF/XGB/LGBM, GroupSplit)...")
    train_result = cached_report_train(_filter_key, tuple(features))
    metrics = evaluate_models(train_result)

    progress.progress(55, text="5/9 SHAP 분석...")
    shap_result = compute_shap(train_result)
    shap_imp = shap_importance(shap_result)

    progress.progress(65, text="6/9 Feature Importance Consensus (+SHAP)...")
    consensus = consensus_ranking(
        compute_importances(train_result), shap_importance=shap_imp
    )
    top_feats = consensus.head(TOP_N_FEATURES_INTERACTION)["feature"].tolist()

    progress.progress(75, text="7/9 Interaction Scan...")
    scan = interaction_scan(df, top_feats)
    interaction_pairs = [
        (str(r["feature_a"]), str(r["feature_b"]), float(r["synergy"]))
        for _, r in scan.head(5).iterrows()
    ]

    progress.progress(85, text="8/9 Rule Discovery...")
    rules = rules_to_dataframe(discover_rules(df, top_feats))

    progress.progress(93, text="9/9 Time Trend / Report 생성...")
    equipment_summary = (
        df.groupby("EQUIPMENT_ID")["DEFECT_FLAG"]
        .agg(defect_rate="mean", samples="count")
        .reset_index()
        .sort_values("defect_rate", ascending=False)
    )
    # drift: 기간 전/후반 평균 차이 |z| > 0.5 인 변수
    _mid = df["TIMESTAMP"].quantile(0.5)
    _early, _late = df[df["TIMESTAMP"] < _mid], df[df["TIMESTAMP"] >= _mid]
    drift_features = [
        f for f in top_feats
        if df[f].std() > 0
        and abs(_late[f].mean() - _early[f].mean()) / df[f].std() > 0.5
    ]
    defect_trend = (
        df.set_index("TIMESTAMP")["DEFECT_FLAG"].resample("W").mean()
        .rename("defect_rate").reset_index()
        .rename(columns={"TIMESTAMP": "period"})
    )
    defect_trend["defect_rate"] = defect_trend["defect_rate"].round(4)

    report_input = ReportInput(
        filter_desc=filter_desc,
        n_samples=len(df),
        n_lots=df["LOT_ID"].nunique(),
        n_defects=int(df["DEFECT_FLAG"].sum()),
        defect_rate=float(df["DEFECT_FLAG"].mean()),
        significance=sig,
        correlation=corr.head(20),
        consensus=consensus,
        ml_metrics=metrics,
        rules=rules,
        equipment_summary=equipment_summary,
        shap_top=shap_imp.head(10).index.tolist(),
        interaction_pairs=interaction_pairs,
        drift_features=drift_features,
        defect_trend=defect_trend,
    )
    path = generate_report(report_input)
    progress.progress(100, text="완료!")

    st.success(f"리포트 생성 완료: `{path}`")

    # 핵심 발견 자동 요약
    _n_sig = int(sig["significant"].sum()) if "significant" in sig.columns else 0
    _insights = [
        f"리포트 생성 완료 — 분석 조건: {filter_desc}",
        f"통계적으로 유의한 변수(BH 보정): **{_n_sig}개** / {len(sig)}개",
    ]
    if len(rules):
        _r = rules.iloc[0]
        _insights.append(
            f"최고 위험 규칙: **{_r['rule']}** → 불량률 {_r['defect_rate']:.1%} "
            f"(lift {_r['lift']:.1f}배)"
        )
    else:
        _insights.append("고위험 규칙은 발견되지 않았습니다")
    insight_box(_insights)

    # 요약 미리보기
    st.markdown("---")
    st.subheader("핵심 결과 요약")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Top Significant Features**")
        st.write(sig.head(5)[["rank", "feature", "effect_size"]].round(3))
    with c2:
        st.markdown("**ML Consensus Top 5**")
        st.write(consensus.head(5)[["feature", "consensus_rank"]])
    with c3:
        st.markdown("**High Risk Conditions**")
        if len(rules):
            st.write(rules.head(3)[["rule", "defect_rate", "lift"]].round(3))
        else:
            st.caption("발견된 고위험 규칙 없음")

    with open(path, encoding="utf-8") as f:
        html = f.read()
    st.download_button("📥 HTML Report 다운로드", html, file_name=path.name,
                       mime="text/html")
    with st.expander("리포트 미리보기", expanded=True):
        st.components.v1.html(html, height=800, scrolling=True)

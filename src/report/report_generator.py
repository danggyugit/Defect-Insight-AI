"""AI Report Generator — template 기반 HTML report 자동 생성.

LLM이 아니라 분석 결과 수치를 한국어 문장 템플릿에 채우는 방식이다
(docs/PROJECT_PROMPT.md §14). 브라우저 인쇄로 PDF 저장 가능.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Template

from src.utils.config import REPORTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class ReportInput:
    """report 생성에 필요한 분석 결과 모음."""

    # 필터 조건
    filter_desc: str                        # 예: "PROD_A / 전체 공정 / MURA / 2026-01~08"
    # overview
    n_samples: int
    n_lots: int
    n_defects: int
    defect_rate: float
    # 분석 결과 테이블
    significance: pd.DataFrame              # run_significance_analysis 출력
    correlation: pd.DataFrame               # run_correlation_analysis 출력
    consensus: pd.DataFrame                 # consensus_ranking 출력
    ml_metrics: pd.DataFrame                # evaluate_models 출력 (index=model)
    rules: pd.DataFrame                     # rules_to_dataframe 출력
    equipment_summary: pd.DataFrame         # EQUIPMENT_ID, defect_rate, samples
    # 선택 항목
    shap_top: list[str] = field(default_factory=list)
    interaction_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    drift_features: list[str] = field(default_factory=list)
    defect_trend: pd.DataFrame | None = None  # 컬럼: period, defect_rate (주 단위 등)


def _executive_summary(inp: ReportInput) -> str:
    """분석 결과 수치를 조합해 Executive Summary 문장을 생성한다."""
    sig_top = inp.significance.head(3)["feature"].tolist()
    cons_top = inp.consensus.head(3)["feature"].tolist() if len(inp.consensus) else []
    parts = [
        f"총 {inp.n_samples:,}개 sample({inp.n_lots:,}개 LOT)을 분석한 결과 "
        f"defect rate는 {inp.defect_rate:.1%}로 나타났습니다."
    ]
    if sig_top:
        parts.append(
            f"통계적 유의차 분석 결과 {', '.join(sig_top)} 등이 정상군과 불량군 간 "
            "유의한 차이를 보였습니다."
        )
    if cons_top:
        overlap = [f for f in cons_top if f in sig_top]
        if overlap:
            parts.append(
                f"ML 기반 분석에서도 {', '.join(cons_top)}이(가) 높은 중요도를 보여 "
                f"통계 분석과 {'일치' if len(overlap) >= 2 else '부분적으로 일치'}하는 결과를 확인했습니다."
            )
        else:
            parts.append(
                f"ML 기반 분석에서는 {', '.join(cons_top)}이(가) 높은 중요도를 보였습니다. "
                "통계 분석 결과와 차이가 있어 비선형 관계 가능성을 시사합니다."
            )
    if len(inp.rules):
        top_rule = inp.rules.iloc[0]
        parts.append(
            f"조건 조합 탐색에서는 [{top_rule['rule']}] 조건에서 defect rate가 "
            f"{top_rule['defect_rate']:.1%}(전체 대비 {top_rule['lift']:.1f}배)로 "
            "증가하는 패턴이 확인되었습니다."
        )
    if len(inp.equipment_summary):
        eq = inp.equipment_summary.sort_values("defect_rate", ascending=False).iloc[0]
        if eq["defect_rate"] > inp.defect_rate * 1.3:
            parts.append(
                f"설비별로는 {eq['EQUIPMENT_ID']}의 defect rate({eq['defect_rate']:.1%})가 "
                "전체 평균 대비 높아 해당 설비의 공정조건 및 calibration 상태에 대한 "
                "우선 검증을 권고합니다."
            )
    parts.append(
        "본 분석은 통계적 연관성 기반의 탐색 결과이며 인과관계를 직접 증명하는 것은 아닙니다."
    )
    return " ".join(parts)


def _recommendations(inp: ReportInput) -> list[str]:
    """우선 조사 권고 항목 생성."""
    recs: list[str] = []
    for _, row in inp.significance.head(3).iterrows():
        direction = "높게" if row["difference"] > 0 else "낮게"
        recs.append(
            f"{row['feature']}: 불량군에서 평균이 {direction} 관측됨 "
            f"(정상 {row['normal_mean']:.2f} → 불량 {row['defect_mean']:.2f}, "
            f"effect size {row['effect_size']:.2f}). 해당 공정 구간의 관리 상태 점검 권고."
        )
    if len(inp.rules):
        top_rule = inp.rules.iloc[0]
        recs.append(
            f"고위험 조건 [{top_rule['rule']}] 해당 LOT의 공정이력 우선 조사 권고 "
            f"(해당 조건 defect rate {top_rule['defect_rate']:.1%})."
        )
    if len(inp.equipment_summary):
        eq = inp.equipment_summary.sort_values("defect_rate", ascending=False).iloc[0]
        if eq["defect_rate"] > inp.defect_rate * 1.3:
            recs.append(f"{eq['EQUIPMENT_ID']} 설비 점검 및 동일 조건 재현성 확인 권고.")
    if inp.drift_features:
        recs.append(
            f"시간 drift가 관측된 변수({', '.join(inp.drift_features[:3])})의 "
            "관리 한계 재설정 검토 권고."
        )
    return recs


_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Defect Insight Report</title>
<style>
  body { font-family: -apple-system, 'Malgun Gothic', sans-serif; max-width: 960px;
         margin: 2rem auto; padding: 0 1rem; color: #1a202c; line-height: 1.6; }
  h1 { border-bottom: 3px solid #2b6cb0; padding-bottom: .5rem; }
  h2 { color: #2b6cb0; margin-top: 2rem; border-bottom: 1px solid #e2e8f0; }
  table { border-collapse: collapse; width: 100%; font-size: .85rem; margin: .5rem 0; }
  th, td { border: 1px solid #cbd5e0; padding: 4px 8px; text-align: right; }
  th { background: #ebf8ff; } td:first-child, th:first-child { text-align: left; }
  .summary { background: #f7fafc; border-left: 4px solid #2b6cb0; padding: 1rem; }
  .disclaimer { background: #fffaf0; border-left: 4px solid #dd6b20; padding: 1rem;
                font-size: .9rem; }
  .kpi { display: inline-block; margin-right: 2rem; }
  .kpi b { font-size: 1.4rem; }
  @media print { body { margin: 0; } }
</style>
</head>
<body>
<h1>🏭 Manufacturing Defect Insight Report</h1>
<p>생성일시: {{ generated_at }} · 분석 조건: {{ filter_desc }}</p>

<h2>1. Executive Summary</h2>
<div class="summary">{{ executive_summary }}</div>

<h2>2. Defect Overview</h2>
<div>
  <span class="kpi">Sample <b>{{ "{:,}".format(n_samples) }}</b></span>
  <span class="kpi">LOT <b>{{ "{:,}".format(n_lots) }}</b></span>
  <span class="kpi">불량 <b>{{ "{:,}".format(n_defects) }}</b></span>
  <span class="kpi">Defect Rate <b>{{ "%.2f%%"|format(defect_rate * 100) }}</b></span>
</div>

<h2>3. Significant Variables (통계적 유의차 상위)</h2>
{{ significance_table }}

<h2>4. Correlation / Mutual Information 상위</h2>
{{ correlation_table }}

<h2>5. ML Model Performance</h2>
{{ ml_metrics_table }}

<h2>6. ML Feature Importance (Model Consensus)</h2>
{{ consensus_table }}

{% if shap_top %}
<h2>7. SHAP 상위 영향 변수</h2>
<p>{{ shap_top | join(", ") }}</p>
{% endif %}

{% if interaction_pairs %}
<h2>8. Multivariate Interaction</h2>
<ul>
{% for a, b, s in interaction_pairs %}
  <li>{{ a }} × {{ b }} (interaction strength {{ "%.3f"|format(s) }})</li>
{% endfor %}
</ul>
{% endif %}

<h2>9. Rule Discovery (고위험 조건)</h2>
{{ rules_table }}

<h2>10. Equipment Analysis</h2>
{{ equipment_table }}

<h2>11. Time Trend</h2>
{% if drift_features %}
<p>기간 전/후반 비교에서 drift가 관측된 변수: {{ drift_features | join(", ") }}</p>
{% else %}
<p>뚜렷한 feature drift가 관측되지 않았습니다.</p>
{% endif %}
{{ defect_trend_table }}

<h2>12. Recommended Investigation Items</h2>
<ol>
{% for rec in recommendations %}
  <li>{{ rec }}</li>
{% endfor %}
</ol>

<h2>13. Disclaimer</h2>
<div class="disclaimer">
본 리포트는 통계적 연관성을 탐색한 결과이며 인과관계를 직접 증명하지 않습니다.
식별된 변수와 조건은 "추가 공정 검증이 권고되는 후보"입니다.
분석 흐름: Correlation → Evidence → Hypothesis → <b>Engineer Validation</b>.
유의차 상위 변수에는 실제 영향 변수와 상관된 confounder가 포함될 수 있으므로
ML/SHAP 결과와 교차 확인이 필요합니다.
</div>
</body>
</html>"""
)


def generate_report(inp: ReportInput, output_path: Path | None = None) -> Path:
    """HTML report를 생성하고 저장 경로를 반환한다."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"defect_insight_report_{stamp}.html"

    def fmt_table(df: pd.DataFrame, n: int = 15) -> str:
        if df is None or len(df) == 0:
            return "<p>해당 없음</p>"
        return df.head(n).to_html(index=False, float_format=lambda x: f"{x:.4g}")

    html = _TEMPLATE.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        filter_desc=inp.filter_desc,
        executive_summary=_executive_summary(inp),
        n_samples=inp.n_samples,
        n_lots=inp.n_lots,
        n_defects=inp.n_defects,
        defect_rate=inp.defect_rate,
        significance_table=fmt_table(
            inp.significance[
                [c for c in ("rank", "feature", "normal_mean", "defect_mean",
                             "difference_pct", "adjusted_p_value", "effect_size",
                             "effect_label") if c in inp.significance.columns]
            ]
        ),
        correlation_table=fmt_table(inp.correlation),
        ml_metrics_table=fmt_table(inp.ml_metrics.reset_index()),
        consensus_table=fmt_table(inp.consensus, n=10),
        shap_top=inp.shap_top,
        interaction_pairs=inp.interaction_pairs,
        rules_table=fmt_table(inp.rules, n=10),
        equipment_table=fmt_table(inp.equipment_summary),
        drift_features=inp.drift_features,
        defect_trend_table=(
            fmt_table(inp.defect_trend, n=40)
            if inp.defect_trend is not None else ""
        ),
        recommendations=_recommendations(inp),
    )
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report saved: %s", output_path)
    return output_path

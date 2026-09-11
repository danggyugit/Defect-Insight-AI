"""단일 변수 유의차 분석 (정상군 vs 불량군).

- 검정 자동 선택: 정규성(n>5000이면 정규성 검정 생략하고 분포 왜도 기준) →
  Welch t-test 또는 Mann-Whitney U
- Effect size 짝: t-test → Cohen's d, MWU → Cliff's delta
- Benjamini-Hochberg (FDR) 다중검정 보정
- Ranking: p-value 단독이 아니라 significance + effect size 조합
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.utils.config import SIGNIFICANCE_ALPHA, TARGET_COLUMN
from src.statistics.effect_size import cliffs_delta, cohens_d, effect_size_label

logger = logging.getLogger(__name__)

_SKEW_THRESHOLD = 2.0  # |skewness| 초과 시 비모수 검정 선택


def _select_test(normal: np.ndarray, defect: np.ndarray) -> str:
    """데이터 특성 기반 검정 방법 자동 선택.

    대표본에서 Shapiro 검정은 무의미하게 예민하므로, 왜도(skewness)를
    기준으로 심하게 비대칭이면 Mann-Whitney U, 아니면 Welch t-test.
    """
    skew = max(abs(stats.skew(normal)), abs(stats.skew(defect)))
    return "mannwhitney" if skew > _SKEW_THRESHOLD else "ttest"


def analyze_feature(
    normal: np.ndarray, defect: np.ndarray
) -> dict[str, float | str]:
    """단일 feature의 정상 vs 불량 유의차 분석."""
    normal = normal[~np.isnan(normal)]
    defect = defect[~np.isnan(defect)]
    if len(normal) < 5 or len(defect) < 5:
        return {"p_value": np.nan, "effect_size": np.nan, "test": "insufficient"}

    test = _select_test(normal, defect)
    if test == "ttest":
        _, p_value = stats.ttest_ind(defect, normal, equal_var=False)
        effect = cohens_d(defect, normal)
        es_method = "cohens_d"
    else:
        _, p_value = stats.mannwhitneyu(defect, normal, alternative="two-sided")
        effect = cliffs_delta(defect, normal)
        es_method = "cliffs_delta"

    normal_mean, defect_mean = float(normal.mean()), float(defect.mean())
    diff = defect_mean - normal_mean
    return {
        "normal_mean": normal_mean,
        "defect_mean": defect_mean,
        "normal_median": float(np.median(normal)),
        "defect_median": float(np.median(defect)),
        "normal_std": float(normal.std(ddof=1)),
        "defect_std": float(defect.std(ddof=1)),
        "difference": diff,
        "difference_pct": (diff / abs(normal_mean) * 100) if normal_mean != 0 else np.nan,
        "p_value": float(p_value),
        "effect_size": effect,
        "effect_size_method": es_method,
        "test": test,
    }


def run_significance_analysis(
    df: pd.DataFrame,
    features: list[str],
    alpha: float = SIGNIFICANCE_ALPHA,
) -> pd.DataFrame:
    """모든 feature에 대해 유의차 분석을 수행하고 ranking 테이블을 반환한다.

    Returns:
        컬럼: feature, normal_mean, defect_mean, difference, difference_pct,
        p_value, adjusted_p_value, effect_size, effect_size_method,
        effect_label, significant, rank_score, rank
    """
    mask = df[TARGET_COLUMN] == 1
    rows = []
    for feat in features:
        values = df[feat].to_numpy(dtype=float)
        res = analyze_feature(values[~mask.to_numpy()], values[mask.to_numpy()])
        res["feature"] = feat
        rows.append(res)

    result = pd.DataFrame(rows).set_index("feature")
    result = result[result["p_value"].notna()].copy()

    # Benjamini-Hochberg FDR 보정
    _, adj_p, _, _ = multipletests(result["p_value"], alpha=alpha, method="fdr_bh")
    result["adjusted_p_value"] = adj_p
    result["significant"] = result["adjusted_p_value"] < alpha
    result["effect_label"] = [
        effect_size_label(e, m)
        for e, m in zip(result["effect_size"], result["effect_size_method"])
    ]

    # Ranking: 유의성 통과 여부 × effect size 크기 조합
    # p-value 단독 ranking 금지 — -log10(adj_p)는 saturation시키고 effect 중심으로
    log_p = -np.log10(result["adjusted_p_value"].clip(lower=1e-300))
    result["rank_score"] = (
        result["significant"].astype(float)
        * result["effect_size"].abs()
        * (1.0 + np.minimum(log_p, 50.0) / 50.0)
    )
    result = result.sort_values("rank_score", ascending=False)
    result["rank"] = range(1, len(result) + 1)

    logger.info(
        "Significance analysis: %d features, %d significant (FDR %.2f)",
        len(result), int(result["significant"].sum()), alpha,
    )
    return result.reset_index()


def group_comparison_test(
    df: pd.DataFrame, feature: str, group_col: str
) -> dict[str, float | str]:
    """다군 비교 (설비/공정/제품별): ANOVA 또는 Kruskal-Wallis 자동 선택."""
    groups = [
        g[feature].dropna().to_numpy()
        for _, g in df.groupby(group_col, observed=True)
        if len(g) >= 5
    ]
    if len(groups) < 2:
        return {"p_value": np.nan, "test": "insufficient"}
    max_skew = max(abs(stats.skew(g)) for g in groups)
    if max_skew > _SKEW_THRESHOLD:
        _, p = stats.kruskal(*groups)
        return {"p_value": float(p), "test": "kruskal"}
    _, p = stats.f_oneway(*groups)
    return {"p_value": float(p), "test": "anova"}

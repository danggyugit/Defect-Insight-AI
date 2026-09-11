"""Feature ↔ target Correlation / Mutual Information 분석.

- Pearson(선형) / Spearman(단조) / Mutual Information(비선형 포함) 3종 병행
- 관계 유형 분류: linear / monotonic / non-linear / weak/none
  (U자형처럼 상관계수는 0에 가깝지만 MI가 높은 변수를 non-linear로 검출)
- feature 간 correlation matrix (heatmap용)

주의: 본 분석은 통계적 연관성 탐색 목적이며, 인과관계를 직접 증명하지 않는다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.utils.config import RANDOM_SEED, TARGET_COLUMN

logger = logging.getLogger(__name__)

# 관계 유형 분류 기준
_CORR_THRESHOLD = 0.1      # |corr| 미만이면 해당 유형의 상관 없음으로 간주
_MONOTONIC_GAP = 0.05      # |spearman| - |pearson| 초과 시 비선형 단조 관계
_MI_STRONG_NORM = 0.25     # 정규화 MI가 이 값 이상이면 "MI 기준 상위"로 간주
_MI_TOP_QUANTILE = 0.95    # 또는 MI가 전체 feature 중 상위 5% 이내면 상위로 간주
# 절대 기준(_MI_STRONG_NORM)만 쓰면 최상위 feature의 MI가 매우 클 때
# U자형처럼 MI 절대값이 작은 비선형 변수를 놓치므로 percentile 기준을 병행한다.

_MATRIX_METHODS = ("pearson", "spearman")


def _classify_relationship(
    pearson: float, spearman: float, mi_norm: float, mi_is_top: bool
) -> str:
    """단일 feature의 target 관계 유형을 분류한다.

    Args:
        pearson: Pearson 상관계수.
        spearman: Spearman 상관계수.
        mi_norm: 전체 feature 중 최대 MI로 정규화한 mutual information (0~1).
        mi_is_top: MI가 전체 feature 중 상위 percentile에 속하는지 여부.

    Returns:
        "linear" | "monotonic" | "non-linear" | "weak/none".
        - linear: 선형 상관이 뚜렷하고 Pearson≈Spearman
        - monotonic: 단조이지만 비선형 (Spearman이 Pearson보다 뚜렷이 큼)
        - non-linear: 상관계수로는 안 잡히지만 MI가 상위 (예: U자형)
        - weak/none: 위 어디에도 해당 없음
    """
    abs_p, abs_s = abs(pearson), abs(spearman)
    if np.isnan(pearson) or np.isnan(spearman):
        return "weak/none"
    if abs_p >= _CORR_THRESHOLD and abs_s - abs_p <= _MONOTONIC_GAP:
        return "linear"
    if abs_s >= _CORR_THRESHOLD and abs_s - abs_p > _MONOTONIC_GAP:
        return "monotonic"
    if (mi_norm >= _MI_STRONG_NORM or mi_is_top) and max(abs_p, abs_s) < _CORR_THRESHOLD:
        return "non-linear"
    return "weak/none"


def _compute_mutual_info(
    X: pd.DataFrame, y: pd.Series
) -> np.ndarray:
    """전체 feature의 mutual information을 한 번의 호출로 계산한다.

    per-feature 루프 금지 — mutual_info_classif는 (n_samples, n_features)
    행렬을 통째로 받아야 400 features × 30k rows에서도 수 초 내에 끝난다.
    NaN은 MI 계산용으로만 컬럼 median으로 대체한다 (원본 불변).
    """
    X_filled = X.fillna(X.median(numeric_only=True))
    return mutual_info_classif(
        X_filled.to_numpy(dtype=float),
        y.to_numpy(dtype=int),
        random_state=RANDOM_SEED,
    )


def run_correlation_analysis(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """각 feature와 target의 Pearson/Spearman/Mutual Information 분석.

    상관계수는 선형/단조 관계만 검출하므로 MI를 병행하여 U자형 등
    비선형 관계도 함께 탐지한다. 결과는 rank_score 내림차순 정렬.

    Args:
        df: target 컬럼을 포함한 데이터셋.
        features: 분석 대상 numeric feature 목록.
        target_col: 이진 target 컬럼명 (기본 DEFECT_FLAG).

    Returns:
        컬럼: feature, pearson, spearman, mutual_info,
        relationship_type, rank_score, rank.
        rank_score = 정규화 MI와 max(|pearson|, |spearman|)의 평균.
    """
    if not features:
        raise ValueError("features is empty")

    X = df[features]
    y = df[target_col]

    # Pearson / Spearman — corrwith로 전 feature 벡터화 계산
    pearson = X.corrwith(y)
    spearman = X.rank().corrwith(y.rank(), method="pearson")

    logger.info(
        "Computing mutual information for %d features × %d rows (single call)",
        len(features), len(df),
    )
    mi = _compute_mutual_info(X, y)
    mi_max = float(mi.max())
    mi_norm = mi / mi_max if mi_max > 0 else np.zeros_like(mi)
    # 상위 percentile 기준 (MI=0인 noise feature는 상위로 취급하지 않음)
    mi_top_cut = float(np.quantile(mi, _MI_TOP_QUANTILE))
    mi_is_top = (mi >= mi_top_cut) & (mi > 0)

    result = pd.DataFrame(
        {
            "feature": features,
            "pearson": pearson.to_numpy(),
            "spearman": spearman.to_numpy(),
            "mutual_info": mi,
        }
    )
    result["relationship_type"] = [
        _classify_relationship(p, s, m, t)
        for p, s, m, t in zip(
            result["pearson"], result["spearman"], mi_norm, mi_is_top
        )
    ]

    # Ranking: 선형 상관 강도와 MI(비선형 포함)를 절반씩 반영 —
    # 상관계수 단독 ranking은 U자형 변수를 놓치므로 금지.
    max_abs_corr = (
        result[["pearson", "spearman"]].abs().max(axis=1).fillna(0.0)
    )
    result["rank_score"] = 0.5 * mi_norm + 0.5 * max_abs_corr.to_numpy()
    result = result.sort_values("rank_score", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    type_counts = result["relationship_type"].value_counts().to_dict()
    logger.info("Correlation analysis done: %d features, types=%s",
                len(result), type_counts)
    return result


def correlation_matrix(
    df: pd.DataFrame, features: list[str], method: str = "pearson"
) -> pd.DataFrame:
    """feature 간 correlation matrix (heatmap용).

    feature 수가 많은 데이터셋을 대비해 전달받은 features만 계산한다 —
    호출 측에서 상위 feature로 좁혀서 넘길 것 (성능 규칙 7).

    Args:
        df: 데이터셋.
        features: matrix에 포함할 feature 목록.
        method: "pearson" 또는 "spearman".

    Returns:
        (len(features) × len(features)) correlation matrix DataFrame.
    """
    if method not in _MATRIX_METHODS:
        raise ValueError(
            f"method must be one of {_MATRIX_METHODS}, got {method!r}"
        )
    logger.info(
        "Computing %s correlation matrix for %d features", method, len(features)
    )
    return df[features].corr(method=method)

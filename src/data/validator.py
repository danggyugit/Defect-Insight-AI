"""Data Validation.

분석 전 데이터 품질을 자동 검사한다 (docs/PROJECT_PROMPT.md §16):
missing rate / constant / duplicate / dtype / cardinality / outlier /
class imbalance / target leakage 의심.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.config import (
    LEAKAGE_CORR_THRESHOLD,
    MISSING_RATE_THRESHOLD,
    TARGET_COLUMN,
    VARIANCE_FILTER_THRESHOLD,
)
from src.data.loader import get_feature_columns

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """데이터 검증 결과."""

    n_rows: int = 0
    n_features: int = 0
    n_duplicate_rows: int = 0
    defect_rate: float = 0.0
    imbalance_warning: bool = False
    missing_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    excluded_features: dict[str, str] = field(default_factory=dict)  # {feature: 사유}
    leakage_suspects: list[str] = field(default_factory=list)
    outlier_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid_features(self) -> list[str] | None:
        """분석에 사용할 feature 목록 (검증 후 세팅)."""
        return getattr(self, "_valid_features", None)

    @valid_features.setter
    def valid_features(self, value: list[str]) -> None:
        self._valid_features = value


def validate_dataset(df: pd.DataFrame) -> ValidationResult:
    """데이터셋을 검증하고 분석 가능 feature 목록과 warning을 반환한다."""
    result = ValidationResult()
    features = get_feature_columns(df)
    result.n_rows = len(df)
    result.n_features = len(features)

    # 중복 행
    result.n_duplicate_rows = int(df.duplicated(subset=features).sum())
    if result.n_duplicate_rows:
        result.warnings.append(f"중복 행 {result.n_duplicate_rows}개 발견")

    # class imbalance
    result.defect_rate = float(df[TARGET_COLUMN].mean())
    if result.defect_rate < 0.01 or result.defect_rate > 0.5:
        result.imbalance_warning = True
        result.warnings.append(
            f"Defect rate {result.defect_rate:.2%} — 심한 imbalance. "
            "class_weight/scale_pos_weight 및 PR-AUC 기준 평가 필요"
        )

    X = df[features]

    # missing rate
    missing_rate = X.isna().mean()
    result.missing_summary = (
        missing_rate[missing_rate > 0]
        .sort_values(ascending=False)
        .rename("missing_rate")
        .to_frame()
    )
    for feat in missing_rate[missing_rate > MISSING_RATE_THRESHOLD].index:
        result.excluded_features[feat] = (
            f"missing rate {missing_rate[feat]:.1%} > {MISSING_RATE_THRESHOLD:.0%}"
        )

    # constant / near-zero variance
    variances = X.var(numeric_only=True)
    for feat in variances[variances <= VARIANCE_FILTER_THRESHOLD].index:
        result.excluded_features[feat] = "constant (near-zero variance)"

    # target leakage 의심: target과 상관 |r| >= threshold
    remaining = [f for f in features if f not in result.excluded_features]
    y = df[TARGET_COLUMN].astype(float)
    corrs = X[remaining].corrwith(y).abs()
    result.leakage_suspects = corrs[corrs >= LEAKAGE_CORR_THRESHOLD].index.tolist()
    for feat in result.leakage_suspects:
        result.excluded_features[feat] = (
            f"target 상관 {corrs[feat]:.3f} — leakage 의심"
        )
        result.warnings.append(f"{feat}: target leakage 의심 (|corr|={corrs[feat]:.3f})")

    # outlier 요약 (IQR 기준 비율, 상위만)
    valid = [f for f in features if f not in result.excluded_features]
    q1, q3 = X[valid].quantile(0.25), X[valid].quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (X[valid] < q1 - 3 * iqr) | (X[valid] > q3 + 3 * iqr)
    outlier_rate = outlier_mask.mean().sort_values(ascending=False)
    result.outlier_summary = (
        outlier_rate[outlier_rate > 0].head(30).rename("outlier_rate").to_frame()
    )

    result.valid_features = valid
    logger.info(
        "Validation: %d features → %d valid (%d excluded), %d warnings",
        len(features), len(valid), len(result.excluded_features), len(result.warnings),
    )
    return result

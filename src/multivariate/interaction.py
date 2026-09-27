"""Multivariate Interaction — 두 변수 조합의 결합 불량 위험 분석.

- pairwise_defect_rate_grid: 2-feature quantile binning → cell별 defect rate (heatmap용)
- interaction_scan: 상위 feature 쌍 전수 스캔 — joint 고위험 cell의 lift와
  additive 기대 대비 초과분(synergy) 계산
- partial_dependence_2d: 학습 모델의 2-way partial dependence (모델이 학습한 결합 효과)

성능 규칙(CLAUDE.md #7): interaction은 사전 filtering된 상위 10~20 feature만 대상.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

from src.ml.trainer import TrainResult
from src.utils.config import RANDOM_SEED, TARGET_COLUMN

logger = logging.getLogger(__name__)

# interaction_scan에서 max cell 선정 시 최소 샘플 수 (극소 cell 과대평가 방지)
_MIN_CELL_SAMPLES = 30
# partial dependence 계산 시 X_train 샘플 상한
_PD_SAMPLE_SIZE = 1000


def _quantile_codes(
    s: pd.Series, n_bins: int
) -> tuple[np.ndarray, list[str]]:
    """Series를 quantile bin code와 경계값 라벨로 변환한다.

    Returns:
        (codes, labels): codes는 int array (결측/binning 불가 = -1),
        labels는 각 bin의 "[left, right]" 형식 경계값 문자열.
    """
    binned = pd.qcut(s, q=n_bins, duplicates="drop")
    codes = binned.cat.codes.to_numpy()  # NaN → -1
    labels = [
        f"[{iv.left:.3g}, {iv.right:.3g}]" for iv in binned.cat.categories
    ]
    return codes, labels


def _cell_stats(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    y: np.ndarray,
    n_a: int,
    n_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(bin_a, bin_b) cell별 (샘플 수, 불량 수) 행렬을 계산한다."""
    valid = (codes_a >= 0) & (codes_b >= 0) & ~np.isnan(y)
    flat = codes_a[valid] * n_b + codes_b[valid]
    counts = np.bincount(flat, minlength=n_a * n_b).reshape(n_a, n_b)
    defects = np.bincount(
        flat, weights=y[valid], minlength=n_a * n_b
    ).reshape(n_a, n_b)
    return counts, defects


def pairwise_defect_rate_grid(
    df: pd.DataFrame, feature_a: str, feature_b: str, n_bins: int = 5
) -> pd.DataFrame:
    """두 feature를 quantile bin으로 나눠 각 cell의 defect rate를 계산한다.

    Args:
        df: 데이터 (TARGET_COLUMN 포함).
        feature_a: 행(index)에 배치할 feature.
        feature_b: 열(columns)에 배치할 feature.
        n_bins: quantile bin 수.

    Returns:
        pivot DataFrame — index=feature_a bin 라벨, columns=feature_b bin 라벨,
        값=defect rate (샘플 없는 cell은 NaN). 라벨은 구간 경계값. heatmap용.
    """
    codes_a, labels_a = _quantile_codes(df[feature_a], n_bins)
    codes_b, labels_b = _quantile_codes(df[feature_b], n_bins)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    counts, defects = _cell_stats(
        codes_a, codes_b, y, len(labels_a), len(labels_b)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(counts > 0, defects / np.maximum(counts, 1), np.nan)

    grid = pd.DataFrame(rates, index=labels_a, columns=labels_b)
    grid.index.name = feature_a
    grid.columns.name = feature_b
    return grid


def pairwise_sample_count_grid(
    df: pd.DataFrame, feature_a: str, feature_b: str, n_bins: int = 5
) -> pd.DataFrame:
    """pairwise_defect_rate_grid와 동일한 binning의 cell별 샘플 수 grid.

    heatmap hover/주석에 각 cell의 support를 함께 표시하기 위한 보조 함수.
    """
    codes_a, labels_a = _quantile_codes(df[feature_a], n_bins)
    codes_b, labels_b = _quantile_codes(df[feature_b], n_bins)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    counts, _ = _cell_stats(codes_a, codes_b, y, len(labels_a), len(labels_b))
    grid = pd.DataFrame(counts, index=labels_a, columns=labels_b)
    grid.index.name = feature_a
    grid.columns.name = feature_b
    return grid


def interaction_scan(
    df: pd.DataFrame,
    features: list[str],
    top_k: int = 10,
    n_bins: int = 3,
) -> pd.DataFrame:
    """상위 feature 쌍을 전수 스캔하여 interaction(synergy) 후보를 찾는다.

    len(features) <= 20 가정 (C(20,2)=190쌍). 각 쌍에 대해 quantile binning 후
    - max_cell_rate: 최고위험 cell(샘플 30개 이상)의 실제 defect rate
    - max_cell_lift: max_cell_rate / 전체 defect rate
    - synergy: max_cell_rate - additive 기대치
      (기대치 = marginal_rate_a + marginal_rate_b - overall_rate,
      두 변수 효과가 단순 합산이라면 나올 값 — 초과분이 클수록 결합 효과)

    Args:
        df: 데이터 (TARGET_COLUMN 포함).
        features: 스캔 대상 feature (사전 filtering된 상위 10~20개 권장).
        top_k: 반환할 상위 쌍 수.
        n_bins: quantile bin 수 (전수 스캔이므로 3 권장).

    Returns:
        DataFrame[feature_a, feature_b, max_cell_rate, max_cell_lift,
        synergy, n_max_cell] — synergy 내림차순.
    """
    if len(features) > 20:
        logger.warning(
            "interaction_scan: %d features (>20) — 성능 규칙상 상위 20개만 사용",
            len(features),
        )
        features = features[:20]

    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    overall_rate = float(np.nanmean(y))
    if overall_rate == 0:
        return pd.DataFrame(
            columns=["feature_a", "feature_b", "max_cell_rate",
                     "max_cell_lift", "synergy", "n_max_cell"]
        )

    # feature별 bin code / marginal defect rate를 1회만 계산
    codes: dict[str, np.ndarray] = {}
    n_bins_of: dict[str, int] = {}
    marginal: dict[str, np.ndarray] = {}
    for feat in features:
        c, labels = _quantile_codes(df[feat], n_bins)
        codes[feat] = c
        n_bins_of[feat] = len(labels)
        valid = (c >= 0) & ~np.isnan(y)
        cnt = np.bincount(c[valid], minlength=len(labels))
        dfc = np.bincount(c[valid], weights=y[valid], minlength=len(labels))
        marginal[feat] = np.where(cnt > 0, dfc / np.maximum(cnt, 1), overall_rate)

    rows: list[dict[str, object]] = []
    for feat_a, feat_b in itertools.combinations(features, 2):
        n_a, n_b = n_bins_of[feat_a], n_bins_of[feat_b]
        counts, defects = _cell_stats(codes[feat_a], codes[feat_b], y, n_a, n_b)
        with np.errstate(invalid="ignore", divide="ignore"):
            rates = np.where(counts > 0, defects / np.maximum(counts, 1), np.nan)

        # additive 기대치: marginal_a[i] + marginal_b[j] - overall
        expected = (
            marginal[feat_a][:, None] + marginal[feat_b][None, :] - overall_rate
        )
        eligible = counts >= _MIN_CELL_SAMPLES
        if not eligible.any():
            continue
        masked = np.where(eligible, rates, -np.inf)
        i, j = np.unravel_index(int(np.argmax(masked)), masked.shape)
        max_rate = float(rates[i, j])
        rows.append(
            {
                "feature_a": feat_a,
                "feature_b": feat_b,
                "max_cell_rate": max_rate,
                "max_cell_lift": max_rate / overall_rate,
                "synergy": max_rate - float(np.clip(expected[i, j], 0.0, 1.0)),
                "n_max_cell": int(counts[i, j]),
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values("synergy", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        "interaction_scan: %d pairs scanned (features=%d, n_bins=%d)",
        len(result), len(features), n_bins,
    )
    return result.head(top_k)


def partial_dependence_2d(
    result: TrainResult,
    model_name: str,
    feature_a: str,
    feature_b: str,
    grid_size: int = 12,
) -> pd.DataFrame:
    """학습 모델의 2-way partial dependence를 계산한다.

    sklearn.inspection.partial_dependence 사용. 계산량 제한을 위해
    X_train에서 최대 1000개 샘플(RANDOM_SEED)로 계산한다.

    Args:
        result: trainer.TrainResult (models + X_train 포함).
        model_name: result.models의 키 (예: "lightgbm").
        feature_a: 행(index)에 배치할 feature.
        feature_b: 열(columns)에 배치할 feature.
        grid_size: 각 축 grid 해상도.

    Returns:
        pivot DataFrame — index=feature_a grid 값, columns=feature_b grid 값,
        값=predicted defect probability.
    """
    if model_name not in result.models:
        raise KeyError(
            f"Unknown model {model_name!r} (available: {list(result.models)})"
        )
    for feat in (feature_a, feature_b):
        if feat not in result.feature_names:
            raise KeyError(f"Feature {feat!r} not in trained feature set")

    model = result.models[model_name]
    # float64 캐스팅: pandas 3는 float32 컬럼에 float64 grid 값 대입을 거부
    # (LossySetitemError) — sklearn partial_dependence 내부 대입 호환용.
    X = result.X_train.astype("float64")
    if len(X) > _PD_SAMPLE_SIZE:
        X = X.sample(_PD_SAMPLE_SIZE, random_state=RANDOM_SEED)

    logger.info(
        "partial_dependence_2d: model=%s pair=(%s, %s) grid=%d n=%d",
        model_name, feature_a, feature_b, grid_size, len(X),
    )
    cols = list(X.columns)
    pd_result = partial_dependence(
        model,
        X,
        features=[(cols.index(feature_a), cols.index(feature_b))],
        grid_resolution=grid_size,
        kind="average",
    )
    values = pd_result["average"][0]  # shape: (len(grid_a), len(grid_b))
    grid_a, grid_b = pd_result["grid_values"]

    out = pd.DataFrame(
        values,
        index=[f"{v:.3g}" for v in grid_a],
        columns=[f"{v:.3g}" for v in grid_b],
    )
    out.index.name = feature_a
    out.columns.name = feature_b
    return out

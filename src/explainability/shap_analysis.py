"""SHAP 기반 explainability 분석.

- tree 모델(random_forest/xgboost/lightgbm)은 TreeExplainer,
  logistic(Pipeline: StandardScaler+LogisticRegression)은 LinearExplainer.
- 성능 규칙(CLAUDE.md #7) 준수: SHAP 값은 X_test에서 max_samples 층화 샘플만,
  interaction은 상위 feature 부분집합 + 경량 모델로만 계산한다.
- shap 라이브러리 버전에 따라 shap_values 반환형이 list / 2D·3D ndarray /
  Explanation으로 달라지므로 방어적으로 정규화한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

from src.ml.trainer import TrainResult
from src.utils.config import RANDOM_SEED

logger = logging.getLogger(__name__)

TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}


@dataclass
class ShapResult:
    """SHAP 계산 결과 묶음."""

    shap_values: np.ndarray        # (n_samples_subset, n_features) — positive class 기준
    X_sample: pd.DataFrame         # SHAP 계산에 사용한 샘플 (원 스케일)
    base_value: float
    model_name: str


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------
def _stratified_sample(
    X: pd.DataFrame, y: pd.Series, max_samples: int
) -> pd.DataFrame:
    """X에서 max_samples 층화 샘플링 — 불량(y=1) 샘플은 가능한 모두 포함.

    불량 샘플 수가 max_samples를 넘으면 불량도 샘플링한다.
    RANDOM_SEED로 재현성을 확보한다.
    """
    if len(X) <= max_samples:
        return X
    pos_idx = X.index[y.loc[X.index] == 1]
    neg_idx = X.index[y.loc[X.index] == 0]
    rng = np.random.default_rng(RANDOM_SEED)
    if len(pos_idx) >= max_samples:
        keep_pos = pd.Index(
            rng.choice(pos_idx.to_numpy(), size=max_samples, replace=False)
        )
        keep_neg = pd.Index([])
    else:
        keep_pos = pos_idx
        n_neg = max_samples - len(pos_idx)
        keep_neg = pd.Index(
            rng.choice(neg_idx.to_numpy(), size=n_neg, replace=False)
        )
    sampled = X.loc[keep_pos.append(keep_neg)].sort_index()
    logger.info(
        "Stratified SHAP sample: %d rows (pos=%d, neg=%d) from %d",
        len(sampled), len(keep_pos), len(keep_neg), len(X),
    )
    return sampled


def _to_2d_shap(raw: object, n_features: int) -> np.ndarray:
    """shap_values 반환값을 (n, n_features) 2D array로 정규화.

    버전에 따라 list[array](클래스별), 3D array(n, f, 2), Explanation 등으로
    반환되므로 positive class(index 1) 값을 방어적으로 추출한다.
    """
    if isinstance(raw, shap.Explanation):
        raw = raw.values
    if isinstance(raw, list):
        raw = raw[1] if len(raw) == 2 else raw[-1]
    arr = np.asarray(raw)
    if arr.ndim == 3:
        if arr.shape[-1] == 2 and arr.shape[1] == n_features:
            arr = arr[:, :, 1]          # (n, f, 2)
        elif arr.shape[0] == 2 and arr.shape[-1] == n_features:
            arr = arr[1]                # (2, n, f)
    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")
    return arr


def _to_scalar_base(raw: object) -> float:
    """expected_value 반환값(스칼라/배열/list)에서 positive class 기준 스칼라 추출."""
    if isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.asarray(raw).ravel()
        return float(arr[1]) if arr.size == 2 else float(arr[-1])
    return float(raw)


def _to_3d_interaction(raw: object, k: int) -> np.ndarray:
    """shap_interaction_values 반환값을 (n, k, k) 3D array로 정규화."""
    if isinstance(raw, list):
        raw = raw[1] if len(raw) == 2 else raw[-1]
    arr = np.asarray(raw)
    if arr.ndim == 4:
        if arr.shape[-1] == 2:
            arr = arr[..., 1]           # (n, k, k, 2)
        elif arr.shape[0] == 2:
            arr = arr[1]                # (2, n, k, k)
    if arr.ndim != 3 or arr.shape[1:] != (k, k):
        raise ValueError(f"Unexpected interaction shape: {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def compute_shap(
    result: TrainResult,
    model_name: str = "lightgbm",
    max_samples: int = 2000,
) -> ShapResult:
    """학습된 모델의 SHAP 값을 계산한다.

    tree 모델은 TreeExplainer, logistic은 LinearExplainer(스케일된 입력)를
    사용한다. X_test에서 max_samples 층화 샘플링(불량은 가능한 모두 포함)하며
    RANDOM_SEED로 재현성을 확보한다.

    Args:
        result: trainer.train_models / load_models의 TrainResult.
        model_name: result.models 내 모델명 (기본 lightgbm).
        max_samples: SHAP 계산 최대 샘플 수 (성능 제한).

    Returns:
        ShapResult (shap_values는 positive class=불량 기준).
    """
    if model_name not in result.models:
        raise KeyError(
            f"Model {model_name!r} not in TrainResult (has {list(result.models)})"
        )
    model = result.models[model_name]
    X_sample = _stratified_sample(result.X_test, result.y_test, max_samples)
    n_features = X_sample.shape[1]

    if model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_sample)
    elif model_name == "logistic":
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]
        X_scaled = pd.DataFrame(
            scaler.transform(X_sample),
            columns=X_sample.columns,
            index=X_sample.index,
        )
        explainer = shap.LinearExplainer(clf, X_scaled)
        raw = explainer.shap_values(X_scaled)
    else:
        raise ValueError(f"Unsupported model for SHAP: {model_name!r}")

    shap_values = _to_2d_shap(raw, n_features)
    base_value = _to_scalar_base(explainer.expected_value)
    logger.info(
        "Computed SHAP for %s: values=%s base=%.4f",
        model_name, shap_values.shape, base_value,
    )
    return ShapResult(
        shap_values=shap_values,
        X_sample=X_sample,
        base_value=base_value,
        model_name=model_name,
    )


def shap_importance(shap_result: ShapResult) -> pd.Series:
    """mean(|SHAP|) 기반 feature importance.

    Returns:
        index=feature명, 값=mean|SHAP|, 내림차순 정렬된 Series.
    """
    imp = pd.Series(
        np.abs(shap_result.shap_values).mean(axis=0),
        index=shap_result.X_sample.columns,
        name="mean_abs_shap",
    ).sort_values(ascending=False)
    logger.info("SHAP importance top3: %s", ", ".join(imp.index[:3]))
    return imp


def shap_interaction_values(
    result: TrainResult,
    top_features: list[str],
    max_samples: int = 500,
) -> tuple[np.ndarray, pd.DataFrame]:
    """상위 feature 부분집합에 대한 SHAP interaction 값 계산.

    전체 feature(400개)로는 (n, 400, 400) 계산이 불가능하므로, top_features만
    사용해 경량 LightGBM을 재학습한 뒤 TreeExplainer로 interaction을 구한다
    (성능 규칙 — max_samples 제한 필수).

    Args:
        result: TrainResult (X_train/y_train으로 경량 모델 학습).
        top_features: interaction 분석 대상 feature 목록 (10~20개 권장).
        max_samples: interaction 계산 최대 샘플 수.

    Returns:
        (interaction array (n, k, k), 계산에 사용한 X_sample[top_features])
    """
    missing = [f for f in top_features if f not in result.X_train.columns]
    if missing:
        raise KeyError(f"Features not in training data: {missing}")

    n_pos = int(result.y_train.sum())
    n_neg = int(len(result.y_train) - n_pos)
    lite = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        scale_pos_weight=n_neg / max(n_pos, 1),
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    lite.fit(result.X_train[top_features], result.y_train)

    X_sample = _stratified_sample(
        result.X_test[top_features], result.y_test, max_samples
    )
    explainer = shap.TreeExplainer(lite)
    raw = explainer.shap_interaction_values(X_sample)
    interaction = _to_3d_interaction(raw, len(top_features))
    logger.info(
        "Computed SHAP interactions: %s over %d features",
        interaction.shape, len(top_features),
    )
    return interaction, X_sample


def top_interaction_pairs(
    interaction: np.ndarray,
    feature_names: list[str],
    top_k: int = 10,
) -> pd.DataFrame:
    """mean(|interaction|) 기준 상위 feature 쌍 (대각=main effect 제외).

    Args:
        interaction: shap_interaction_values의 (n, k, k) array.
        feature_names: k개 feature 이름 (interaction 축 순서와 동일).
        top_k: 반환할 쌍 수.

    Returns:
        컬럼 feature_a, feature_b, strength — strength 내림차순.
    """
    mean_abs = np.abs(interaction).mean(axis=0)  # (k, k), 대칭
    rows = []
    k = len(feature_names)
    for i in range(k):
        for j in range(i + 1, k):
            rows.append(
                {
                    "feature_a": feature_names[i],
                    "feature_b": feature_names[j],
                    # off-diagonal 두 칸에 반씩 분배되므로 합산
                    "strength": float(mean_abs[i, j] + mean_abs[j, i]),
                }
            )
    pairs = (
        pd.DataFrame(rows)
        .sort_values("strength", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    if len(pairs):
        logger.info(
            "Top interaction pair: %s × %s (%.5f)",
            pairs.loc[0, "feature_a"], pairs.loc[0, "feature_b"],
            pairs.loc[0, "strength"],
        )
    return pairs

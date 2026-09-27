"""ML 모델 학습 모듈.

- 모델: Logistic Regression(baseline) / Random Forest / XGBoost / LightGBM
- Split: 기본 GroupShuffleSplit(groups=LOT_ID) — 같은 LOT이 train/test에
  걸치지 않도록. "time" 옵션은 TIMESTAMP 기준 앞 75% / 뒤 25%.
- Imbalance: class_weight="balanced" / scale_pos_weight 적용.
- 학습된 모델은 models/ 에 joblib으로 저장 후 재사용.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.loader import get_feature_columns
from src.utils.config import (
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
    TIMESTAMP_COLUMN,
)

logger = logging.getLogger(__name__)

MODEL_NAMES = ["logistic", "random_forest", "xgboost", "lightgbm"]
MODEL_BUNDLE_FILENAME = "train_result.joblib"


@dataclass
class TrainResult:
    """학습 결과 묶음 (모델 + split된 데이터 + 메타)."""

    models: dict[str, object] = field(default_factory=dict)
    X_train: pd.DataFrame = field(default_factory=pd.DataFrame)
    X_test: pd.DataFrame = field(default_factory=pd.DataFrame)
    y_train: pd.Series = field(default_factory=pd.Series)
    y_test: pd.Series = field(default_factory=pd.Series)
    feature_names: list[str] = field(default_factory=list)
    split_method: str = "group"


def _split_indices(
    df: pd.DataFrame, split: str
) -> tuple[pd.Index, pd.Index]:
    """train/test 행 인덱스를 반환한다.

    Args:
        df: 전체 데이터셋 (LOT_ID, TIMESTAMP 포함).
        split: "group"(GroupShuffleSplit, LOT_ID 기준) 또는
            "time"(TIMESTAMP 정렬 후 앞 75% train / 뒤 25% test).
    """
    if split == "group":
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        train_pos, test_pos = next(
            splitter.split(df, groups=df["LOT_ID"])
        )
        return df.index[train_pos], df.index[test_pos]
    if split == "time":
        order = df[TIMESTAMP_COLUMN].sort_values(kind="mergesort").index
        n_train = int(round(len(order) * (1.0 - TEST_SIZE)))
        return order[:n_train], order[n_train:]
    raise ValueError(f"Unknown split method: {split!r} (use 'group' or 'time')")


def _build_models(
    n_pos: int, n_neg: int, models_to_train: list[str]
) -> dict[str, object]:
    """학습할 모델 인스턴스를 생성한다 (imbalance 보정 포함)."""
    scale_pos_weight = n_neg / max(n_pos, 1)
    registry: dict[str, object] = {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbose=-1,
        ),
    }
    unknown = set(models_to_train) - set(registry)
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    return {name: registry[name] for name in models_to_train}


def train_models(
    df: pd.DataFrame,
    features: list[str] | None = None,
    split: str = "group",
    models_to_train: list[str] | None = None,
) -> TrainResult:
    """불량 예측 모델들을 학습한다.

    Args:
        df: 전체 데이터셋 (loader.load_dataset 결과).
        features: 사용할 feature 목록. None이면
            loader.get_feature_columns(df) (leakage 컬럼 자동 제외).
        split: "group"(기본, GroupShuffleSplit/LOT_ID) 또는 "time".
        models_to_train: 학습할 모델명 부분집합. None이면 전체 4개.

    Returns:
        TrainResult: 학습된 모델 dict + train/test 데이터 + 메타 정보.
    """
    if features is None:
        features = get_feature_columns(df)
    models_to_train = models_to_train or list(MODEL_NAMES)

    train_idx, test_idx = _split_indices(df, split)
    X_train = df.loc[train_idx, features]
    X_test = df.loc[test_idx, features]
    y_train = df.loc[train_idx, TARGET_COLUMN]
    y_test = df.loc[test_idx, TARGET_COLUMN]

    if split == "group":
        overlap = set(df.loc[train_idx, "LOT_ID"]) & set(df.loc[test_idx, "LOT_ID"])
        if overlap:  # GroupShuffleSplit이 보장하지만 방어적으로 확인
            raise RuntimeError(f"LOT_ID leakage across split: {sorted(overlap)[:5]}")

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    logger.info(
        "Training %s | split=%s | train=%d (pos=%d) test=%d | features=%d",
        models_to_train, split, len(y_train), n_pos, len(y_test), len(features),
    )

    models = _build_models(n_pos, n_neg, models_to_train)
    for name, model in models.items():
        model.fit(X_train, y_train)
        logger.info("Trained model: %s", name)

    return TrainResult(
        models=models,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(features),
        split_method=split,
    )


def save_models(result: TrainResult, path: Path = MODELS_DIR) -> None:
    """TrainResult 전체를 joblib으로 저장한다."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    target = path / MODEL_BUNDLE_FILENAME
    joblib.dump(result, target, compress=3)
    logger.info("Saved TrainResult to %s", target)


def load_models(path: Path = MODELS_DIR) -> TrainResult | None:
    """저장된 TrainResult를 로드한다. 파일이 없으면 None."""
    target = Path(path) / MODEL_BUNDLE_FILENAME
    if not target.exists():
        logger.info("No saved models at %s", target)
        return None
    result: TrainResult = joblib.load(target)
    logger.info(
        "Loaded TrainResult from %s (models=%s)", target, list(result.models)
    )
    return result

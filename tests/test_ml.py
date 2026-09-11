"""ML 엔진 테스트: trainer / evaluation / feature importance."""
from __future__ import annotations

import pandas as pd
import pytest

from src.ml.evaluation import confusion_matrices, evaluate_models, roc_pr_curves
from src.ml.feature_importance import compute_importances, consensus_ranking
from src.ml.trainer import TrainResult, train_models
from src.utils.config import NON_FEATURE_COLUMNS


@pytest.fixture(scope="module")
def train_result(small_df: pd.DataFrame) -> TrainResult:
    # 테스트 속도를 위해 2개 모델만
    return train_models(small_df, models_to_train=["logistic", "lightgbm"])


class TestTrainer:
    def test_no_lot_overlap(self, train_result: TrainResult, small_df: pd.DataFrame) -> None:
        """GroupSplit: train/test에 같은 LOT이 없어야 한다 (leakage 방지)."""
        train_lots = set(small_df.loc[train_result.X_train.index, "LOT_ID"])
        test_lots = set(small_df.loc[train_result.X_test.index, "LOT_ID"])
        assert not train_lots & test_lots

    def test_no_leakage_columns(self, train_result: TrainResult) -> None:
        assert not set(train_result.feature_names) & set(NON_FEATURE_COLUMNS)

    def test_models_trained(self, train_result: TrainResult) -> None:
        assert set(train_result.models) == {"logistic", "lightgbm"}
        for model in train_result.models.values():
            assert hasattr(model, "predict_proba")

    def test_time_split(self, small_df: pd.DataFrame) -> None:
        result = train_models(small_df, split="time", models_to_train=["logistic"])
        train_max = small_df.loc[result.X_train.index, "TIMESTAMP"].max()
        test_min = small_df.loc[result.X_test.index, "TIMESTAMP"].min()
        assert train_max <= test_min


class TestEvaluation:
    def test_metrics_reasonable(self, train_result: TrainResult) -> None:
        metrics = evaluate_models(train_result)
        assert {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"} <= set(
            metrics.columns
        )
        # 심어진 signal이 있으므로 random(0.5)보다 명확히 좋아야 함
        assert (metrics["roc_auc"] > 0.6).all()

    def test_confusion_and_curves(self, train_result: TrainResult) -> None:
        cms = confusion_matrices(train_result)
        assert all(cm.shape == (2, 2) for cm in cms.values())
        curves = roc_pr_curves(train_result)
        for c in curves.values():
            assert {"fpr", "tpr", "precision", "recall"} <= set(c)


class TestImportance:
    def test_consensus_contains_planted(self, train_result: TrainResult) -> None:
        importances = compute_importances(train_result)
        consensus = consensus_ranking(importances)
        top15 = set(consensus.head(15)["feature"])
        assert "TEMP_004" in top15
        assert "PRESSURE_012" in top15

    def test_consensus_with_shap_rank(self, train_result: TrainResult) -> None:
        importances = compute_importances(train_result)
        shap_imp = pd.Series(
            range(len(importances), 0, -1), index=importances["feature"]
        ).astype(float)
        consensus = consensus_ranking(importances, shap_importance=shap_imp)
        assert "rank_shap" in consensus.columns

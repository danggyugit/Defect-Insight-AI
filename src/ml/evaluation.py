"""모델 평가 모듈.

class imbalance 환경이므로 PR-AUC(average precision)를 주지표로 사용한다.
Plotly 시각화를 위한 ROC/PR curve raw 데이터도 제공한다.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.ml.trainer import TrainResult

logger = logging.getLogger(__name__)


def _predict_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    """positive class(defect=1) 확률을 반환한다."""
    return model.predict_proba(X)[:, 1]


def evaluate_models(result: TrainResult) -> pd.DataFrame:
    """test set에 대한 모델별 성능 지표 테이블.

    Returns:
        index=모델명, 컬럼: accuracy, precision, recall, f1,
        roc_auc, pr_auc (imbalance 주지표).
    """
    y_test = result.y_test
    rows: dict[str, dict[str, float]] = {}
    for name, model in result.models.items():
        proba = _predict_proba(model, result.X_test)
        pred = model.predict(result.X_test)
        rows[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
        }
    metrics = pd.DataFrame.from_dict(rows, orient="index")
    metrics.index.name = "model"
    logger.info(
        "Evaluated %d models (split=%s): best PR-AUC=%s (%.4f)",
        len(metrics), result.split_method,
        metrics["pr_auc"].idxmax(), metrics["pr_auc"].max(),
    )
    return metrics


def confusion_matrices(result: TrainResult) -> dict[str, np.ndarray]:
    """모델별 confusion matrix (rows=actual, cols=predicted, [[TN,FP],[FN,TP]])."""
    return {
        name: confusion_matrix(result.y_test, model.predict(result.X_test))
        for name, model in result.models.items()
    }


def roc_pr_curves(result: TrainResult) -> dict[str, dict[str, np.ndarray]]:
    """Plotly 그리기용 ROC / PR curve raw 데이터.

    Returns:
        {모델명: {"fpr", "tpr", "precision", "recall"}} — fpr/tpr는 ROC curve,
        precision/recall은 PR curve 좌표.
    """
    curves: dict[str, dict[str, np.ndarray]] = {}
    for name, model in result.models.items():
        proba = _predict_proba(model, result.X_test)
        fpr, tpr, _ = roc_curve(result.y_test, proba)
        precision, recall, _ = precision_recall_curve(result.y_test, proba)
        curves[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
        }
    return curves

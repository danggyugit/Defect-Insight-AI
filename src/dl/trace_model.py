"""딥러닝 분석 — FDC trace 기반 1D-CNN 불량 분류.

요약 통계로는 잡기 어려운 파형 형태 이상(spike/oscillation/drift/level shift)을
raw trace에서 직접 학습한다. saliency(gradient × input)로 "파형의 어느 시간
구간이 판단 근거였는지"를 설명한다.

핵심 규칙 준수:
- Split: LOT_ID 기준 GroupShuffleSplit (tabular ML과 동일)
- Imbalance: BCEWithLogitsLoss(pos_weight)
- Seed: RANDOM_SEED로 torch/numpy 시드 고정
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

from src.utils.config import MODELS_DIR, RANDOM_SEED, TARGET_COLUMN, TEST_SIZE

logger = logging.getLogger(__name__)

TRACE_MODEL_PATH = MODELS_DIR / "trace_cnn.pt"


def _device() -> torch.device:
    """가용 장치 선택 (Apple Silicon MPS > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TraceCNN(nn.Module):
    """FDC trace 불량 분류용 1D-CNN."""

    def __init__(self, n_sensors: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_sensors, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).squeeze(-1)).squeeze(-1)


@dataclass
class TraceTrainResult:
    """trace CNN 학습 결과."""

    model: TraceCNN
    metrics: dict[str, float]
    test_probs: np.ndarray          # test set 예측 확률
    test_labels: np.ndarray
    test_indices: np.ndarray        # df 기준 row index (saliency 시각화용)
    trace_mean: np.ndarray          # 정규화 파라미터 (sensor별)
    trace_std: np.ndarray
    history: list[float] = field(default_factory=list)  # epoch별 train loss


def _normalize(
    traces: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((traces - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def train_trace_model(
    df: pd.DataFrame,
    traces: np.ndarray,
    panel_ids: np.ndarray,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> TraceTrainResult:
    """1D-CNN을 학습하고 test set 성능을 반환한다.

    Args:
        df: main dataset (PANEL_ID, LOT_ID, DEFECT_FLAG 필요).
        traces: (n, sensors, T) trace 배열 — panel_ids 순서와 일치.
        panel_ids: trace 순서에 대응하는 PANEL_ID.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # PANEL_ID 기준으로 df 순서와 trace 순서 정렬
    order = df.set_index("PANEL_ID").loc[panel_ids]
    y = order[TARGET_COLUMN].to_numpy().astype(np.float32)
    groups = order["LOT_ID"].to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    train_idx, test_idx = next(splitter.split(traces, y, groups=groups))
    assert not set(groups[train_idx]) & set(groups[test_idx]), "LOT overlap!"

    # train 기준 정규화 (test 정보 누출 방지)
    mean = traces[train_idx].mean(axis=(0, 2))
    std = traces[train_idx].std(axis=(0, 2)) + 1e-8
    X_train = _normalize(traces[train_idx], mean, std)
    X_test = _normalize(traces[test_idx], mean, std)
    y_train, y_test = y[train_idx], y[test_idx]

    device = _device()
    model = TraceCNN(n_sensors=traces.shape[1]).to(device)
    pos_weight = torch.tensor(
        [(len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)], device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    logger.info("Training TraceCNN on %s: %d train / %d test, %d epochs",
                device, len(y_train), len(y_test), epochs)
    history: list[float] = []
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(yb)
        epoch_loss = total / len(y_train)
        history.append(epoch_loss)
        logger.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, epoch_loss)

    # 평가
    model.eval()
    probs = predict_proba(model, X_test, device=device)
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }
    logger.info("TraceCNN test metrics: %s", metrics)

    return TraceTrainResult(
        model=model.cpu(),
        metrics=metrics,
        test_probs=probs,
        test_labels=y_test,
        test_indices=test_idx,
        trace_mean=mean,
        trace_std=std,
        history=history,
    )


def predict_proba(
    model: TraceCNN, X: np.ndarray, device: torch.device | None = None,
    batch_size: int = 1024,
) -> np.ndarray:
    """정규화된 trace 배열의 불량 확률 예측."""
    device = device or _device()
    model = model.to(device)
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    model.cpu()
    return np.concatenate(out)


def compute_saliency(
    result: TraceTrainResult, traces: np.ndarray, sample_indices: np.ndarray
) -> np.ndarray:
    """gradient × input saliency — 파형의 어느 시간 구간이 판단에 기여했는지.

    Returns:
        (len(sample_indices), sensors, T) 절대값 saliency.
    """
    model = result.model.cpu()
    model.eval()
    X = _normalize(traces[sample_indices], result.trace_mean, result.trace_std)
    x = torch.from_numpy(X).requires_grad_(True)
    logits = model(x)
    logits.sum().backward()
    saliency = (x.grad * x).abs().detach().numpy()
    return saliency


def save_trace_model(result: TraceTrainResult, path: Path = TRACE_MODEL_PATH) -> None:
    """모델 state와 정규화 파라미터/지표 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "trace_mean": result.trace_mean,
            "trace_std": result.trace_std,
            "metrics": result.metrics,
        },
        path,
    )
    logger.info("Saved trace model to %s", path)


def load_trace_model(
    path: Path = TRACE_MODEL_PATH, n_sensors: int = 3
) -> tuple[TraceCNN, np.ndarray, np.ndarray, dict[str, float]] | None:
    """저장된 모델 로드. 없으면 None. 반환: (model, mean, std, metrics)."""
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = TraceCNN(n_sensors=n_sensors)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["trace_mean"], payload["trace_std"], payload["metrics"]

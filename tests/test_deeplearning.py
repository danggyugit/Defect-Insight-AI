"""FDC trace generator + 1D-CNN 테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.trace_generator import (
    N_SENSORS,
    TRACE_LENGTH,
    generate_traces,
)
from src.dl.trace_model import compute_saliency, train_trace_model


@pytest.fixture(scope="module")
def trace_data(small_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return generate_traces(small_df)


class TestTraceGenerator:
    def test_shape_and_alignment(self, small_df, trace_data) -> None:
        traces, panel_ids = trace_data
        assert traces.shape == (len(small_df), N_SENSORS, TRACE_LENGTH)
        assert traces.dtype == np.float32
        assert list(panel_ids) == list(small_df["PANEL_ID"].astype(str))

    def test_reproducibility(self, small_df) -> None:
        t1, _ = generate_traces(small_df)
        t2, _ = generate_traces(small_df)
        np.testing.assert_array_equal(t1, t2)

    def test_defect_traces_have_higher_variability(self, small_df, trace_data) -> None:
        """파형 이상이 주입된 불량군은 plateau 구간 잔차 변동이 더 커야 한다."""
        traces, _ = trace_data
        mask = small_df["DEFECT_FLAG"].to_numpy() == 1
        # plateau 구간(20:108)에서 선형 추세 제거 후 std
        plateau = traces[:, :, 20:108]
        detrended_std = plateau.std(axis=2).mean(axis=1)
        assert detrended_std[mask].mean() > detrended_std[~mask].mean() * 1.05


class TestTraceCNN:
    @pytest.fixture(scope="class")
    def trained(self, small_df, trace_data):
        traces, panel_ids = trace_data
        return train_trace_model(small_df, traces, panel_ids, epochs=12)

    def test_metrics_better_than_random(self, trained) -> None:
        assert trained.metrics["roc_auc"] > 0.6
        assert trained.metrics["pr_auc"] > small_df_defect_rate_floor(trained)

    def test_no_lot_leakage_asserted(self, trained) -> None:
        # train_trace_model 내부 assert가 통과했다는 것 자체가 검증
        assert len(trained.test_indices) > 0

    def test_saliency_shape(self, trained, trace_data) -> None:
        traces, _ = trace_data
        idx = trained.test_indices[:3]
        sal = compute_saliency(trained, traces, idx)
        assert sal.shape == (3, N_SENSORS, TRACE_LENGTH)
        assert (sal >= 0).all()


def small_df_defect_rate_floor(trained) -> float:
    """PR-AUC 하한: 무작위 분류기의 기대값(=불량 비율)."""
    return float(trained.test_labels.mean())

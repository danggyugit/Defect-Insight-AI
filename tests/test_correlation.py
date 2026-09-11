"""Correlation/MI 분석 테스트."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.validator import validate_dataset
from src.statistics.correlation import correlation_matrix, run_correlation_analysis


@pytest.fixture(scope="module")
def result(small_df: pd.DataFrame) -> pd.DataFrame:
    validation = validate_dataset(small_df)
    return run_correlation_analysis(small_df, validation.valid_features)


class TestCorrelation:
    def test_schema(self, result: pd.DataFrame) -> None:
        for col in ("feature", "pearson", "spearman", "mutual_info",
                    "relationship_type", "rank_score", "rank"):
            assert col in result.columns

    def test_planted_signals_top(self, result: pd.DataFrame) -> None:
        top10 = set(result.head(10)["feature"])
        assert "TEMP_004" in top10
        assert "PRESSURE_012" in top10

    def test_correlation_bounds(self, result: pd.DataFrame) -> None:
        assert result["pearson"].abs().max() <= 1.0
        assert result["spearman"].abs().max() <= 1.0
        assert (result["mutual_info"] >= 0).all()

    def test_matrix_shape_and_symmetry(self, small_df: pd.DataFrame) -> None:
        feats = ["TEMP_004", "PRESSURE_012", "SPEED_007"]
        m = correlation_matrix(small_df, feats)
        assert m.shape == (3, 3)
        assert (m.values.round(8) == m.values.T.round(8)).all()

    def test_invalid_method_raises(self, small_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            correlation_matrix(small_df, ["TEMP_004"], method="kendall")

"""통계 엔진 테스트: effect size, 유의차 분석."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validator import validate_dataset
from src.statistics.effect_size import cliffs_delta, cohens_d, effect_size_label
from src.statistics.significance import group_comparison_test, run_significance_analysis


class TestEffectSize:
    def test_cohens_d_known_value(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(1.0, 1.0, 5000)
        b = rng.normal(0.0, 1.0, 5000)
        assert cohens_d(a, b) == pytest.approx(1.0, abs=0.1)

    def test_cohens_d_zero_for_identical(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert cohens_d(x, x) == pytest.approx(0.0)

    def test_cliffs_delta_bounds(self) -> None:
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cliffs_delta(a, b) == pytest.approx(1.0)
        assert cliffs_delta(b, a) == pytest.approx(-1.0)

    def test_labels(self) -> None:
        assert effect_size_label(0.05, "cohens_d") == "negligible"
        assert effect_size_label(0.9, "cohens_d") == "large"
        assert effect_size_label(0.5, "cliffs_delta") == "large"


class TestSignificance:
    def test_planted_signals_ranked_top(self, small_df: pd.DataFrame) -> None:
        """심어진 TEMP_004 / PRESSURE_012가 상위 10위 안에 들어야 한다."""
        validation = validate_dataset(small_df)
        result = run_significance_analysis(small_df, validation.valid_features)
        top10 = set(result.head(10)["feature"])
        assert "TEMP_004" in top10
        assert "PRESSURE_012" in top10

    def test_bh_correction_monotone(self, small_df: pd.DataFrame) -> None:
        validation = validate_dataset(small_df)
        result = run_significance_analysis(small_df, validation.valid_features)
        assert (result["adjusted_p_value"] >= result["p_value"] - 1e-12).all()

    def test_result_schema(self, small_df: pd.DataFrame) -> None:
        validation = validate_dataset(small_df)
        result = run_significance_analysis(small_df, validation.valid_features)
        for col in (
            "feature", "normal_mean", "defect_mean", "difference",
            "p_value", "adjusted_p_value", "effect_size", "significant", "rank",
        ):
            assert col in result.columns

    def test_group_comparison(self, small_df: pd.DataFrame) -> None:
        res = group_comparison_test(small_df, "EQUIP_PARAM_003", "EQUIPMENT_ID")
        assert res["test"] in ("anova", "kruskal")
        assert 0 <= res["p_value"] <= 1

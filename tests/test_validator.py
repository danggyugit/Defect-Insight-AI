"""Data validation 테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validator import validate_dataset


class TestValidator:
    def test_basic_validation(self, small_df: pd.DataFrame) -> None:
        result = validate_dataset(small_df)
        assert result.n_rows == len(small_df)
        assert result.valid_features
        assert 0 < result.defect_rate < 0.2

    def test_constant_feature_excluded(self, small_df: pd.DataFrame) -> None:
        df = small_df.copy()
        df["TEMP_001"] = 1.0
        result = validate_dataset(df)
        assert "TEMP_001" in result.excluded_features
        assert "TEMP_001" not in result.valid_features

    def test_high_missing_feature_excluded(self, small_df: pd.DataFrame) -> None:
        df = small_df.copy()
        df.loc[df.sample(frac=0.8, random_state=0).index, "TEMP_002"] = np.nan
        result = validate_dataset(df)
        assert "TEMP_002" in result.excluded_features

    def test_leakage_suspect_detected(self, small_df: pd.DataFrame) -> None:
        df = small_df.copy()
        # target을 거의 그대로 복사한 leakage feature 주입
        df["TEMP_003"] = df["DEFECT_FLAG"] + np.random.default_rng(0).normal(0, 0.01, len(df))
        result = validate_dataset(df)
        assert "TEMP_003" in result.leakage_suspects
        assert "TEMP_003" not in result.valid_features

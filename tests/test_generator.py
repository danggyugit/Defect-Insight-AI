"""Synthetic data generator 테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.generator import generate_dataset
from src.data.loader import get_feature_columns
from src.utils.config import NON_FEATURE_COLUMNS


class TestGenerator:
    def test_shape_and_columns(self, small_df: pd.DataFrame, small_config) -> None:
        assert len(small_df) == small_config.n_samples
        for col in NON_FEATURE_COLUMNS:
            assert col in small_df.columns

    def test_defect_rate_near_target(self, small_df: pd.DataFrame, small_config) -> None:
        rate = small_df["DEFECT_FLAG"].mean()
        assert abs(rate - small_config.target_defect_rate) < 0.02

    def test_reproducibility(self, small_config) -> None:
        df1 = generate_dataset(small_config)
        df2 = generate_dataset(small_config)
        pd.testing.assert_frame_equal(df1, df2)

    def test_defect_type_consistency(self, small_df: pd.DataFrame) -> None:
        # 정상 샘플은 NONE, 불량 샘플은 NONE이 아니어야 함
        assert (small_df.loc[small_df["DEFECT_FLAG"] == 0, "DEFECT_TYPE"] == "NONE").all()
        assert (small_df.loc[small_df["DEFECT_FLAG"] == 1, "DEFECT_TYPE"] != "NONE").all()

    def test_planted_signal_temp004(self, small_df: pd.DataFrame) -> None:
        """심어진 mechanism: TEMP_004는 불량군에서 평균이 높아야 함."""
        normal = small_df.loc[small_df["DEFECT_FLAG"] == 0, "TEMP_004"]
        defect = small_df.loc[small_df["DEFECT_FLAG"] == 1, "TEMP_004"]
        assert defect.mean() > normal.mean() + 0.5

    def test_shifted_equipment_has_higher_defect_rate(
        self, small_df: pd.DataFrame, small_config
    ) -> None:
        rates = small_df.groupby("EQUIPMENT_ID")["DEFECT_FLAG"].mean()
        assert rates[small_config.shifted_equipment] == rates.max()

    def test_feature_columns_exclude_leakage(self, small_df: pd.DataFrame) -> None:
        features = get_feature_columns(small_df)
        assert not set(features) & set(NON_FEATURE_COLUMNS)
        assert all(np.issubdtype(small_df[f].dtype, np.number) for f in features)

    def test_timestamps_within_range(self, small_df: pd.DataFrame, small_config) -> None:
        assert small_df["TIMESTAMP"].min() >= pd.Timestamp(small_config.start_date) - pd.Timedelta(hours=3)
        assert small_df["TIMESTAMP"].max() <= pd.Timestamp(small_config.end_date) + pd.Timedelta(days=1)

"""Rule discovery 테스트."""
from __future__ import annotations

import pandas as pd

from src.multivariate.rule_discovery import discover_rules, rules_to_dataframe


class TestRuleDiscovery:
    def test_finds_planted_interaction(self, small_df: pd.DataFrame) -> None:
        """심어진 TEMP_004 × PRESSURE_012 interaction 규칙이 발견되어야 한다."""
        rules = discover_rules(
            small_df,
            ["TEMP_004", "PRESSURE_012", "SPEED_007", "TIME_003"],
            min_samples_leaf=50,
        )
        assert rules, "규칙이 하나도 발견되지 않음"
        top = rules[0]
        text = top.condition_text
        assert "TEMP_004" in text and "PRESSURE_012" in text
        assert top.lift > 2.0
        assert top.p_value < 0.01

    def test_rule_metrics_consistent(self, small_df: pd.DataFrame) -> None:
        rules = discover_rules(
            small_df, ["TEMP_004", "PRESSURE_012"], min_samples_leaf=50
        )
        overall = small_df["DEFECT_FLAG"].mean()
        for r in rules:
            assert r.defect_rate == r.n_defects / r.n_samples
            assert abs(r.lift - r.defect_rate / overall) < 1e-9
            assert 0 <= r.p_value <= 1

    def test_no_duplicate_feature_direction(self, small_df: pd.DataFrame) -> None:
        """같은 feature+방향 조건이 병합되어야 한다."""
        rules = discover_rules(
            small_df, ["TEMP_004", "PRESSURE_012"], max_depth=4, min_samples_leaf=30
        )
        for r in rules:
            keys = [tuple(c.split(" ")[:2]) for c in r.conditions]
            assert len(keys) == len(set(keys))

    def test_dataframe_conversion(self, small_df: pd.DataFrame) -> None:
        rules = discover_rules(small_df, ["TEMP_004", "PRESSURE_012"], min_samples_leaf=50)
        out = rules_to_dataframe(rules)
        assert list(out.columns) == [
            "rule", "samples", "defects", "defect_rate", "lift",
            "risk_increase", "p_value",
        ]
        assert len(out) == len(rules)

    def test_empty_when_no_defects(self, small_df: pd.DataFrame) -> None:
        df = small_df.copy()
        df["DEFECT_FLAG"] = 0
        assert discover_rules(df, ["TEMP_004"]) == []

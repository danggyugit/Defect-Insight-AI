"""Rule Discovery — Decision Tree 기반 불량 조건 규칙 추출.

예: "TEMP_004 > 85.2 AND PRESSURE_012 > 1.25 → Sample=420, Defect Rate=14.8%"

Ranking 기준: Support / Defect Rate / Lift / Risk Increase / 통계 유의성(Fisher).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.tree import DecisionTreeClassifier

from src.utils.config import RANDOM_SEED, TARGET_COLUMN

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """추출된 불량 조건 규칙."""

    conditions: list[str]           # ["TEMP_004 > 85.20", "PRESSURE_012 > 1.25"]
    n_samples: int
    n_defects: int
    defect_rate: float
    lift: float                     # defect_rate / 전체 defect_rate
    risk_increase: float            # defect_rate - 전체 defect_rate
    p_value: float                  # Fisher exact (규칙 내 vs 규칙 외)

    @property
    def condition_text(self) -> str:
        return " AND ".join(self.conditions)


def _extract_leaf_rules(
    tree: DecisionTreeClassifier, feature_names: list[str]
) -> list[tuple[list[str], int]]:
    """트리의 각 leaf까지의 경로 조건과 leaf node id를 추출한다."""
    t = tree.tree_
    rules: list[tuple[list[str], int]] = []

    def simplify(conds: list[tuple[str, str, float]]) -> list[str]:
        """같은 feature의 중복 조건 병합: '>'는 최대 임계값, '<='는 최소 임계값만 유지."""
        bounds: dict[tuple[str, str], float] = {}
        order: list[tuple[str, str]] = []
        for feat, op, thr in conds:
            key = (feat, op)
            if key not in bounds:
                order.append(key)
                bounds[key] = thr
            else:
                bounds[key] = max(bounds[key], thr) if op == ">" else min(bounds[key], thr)
        return [f"{feat} {op} {bounds[(feat, op)]:.3f}" for feat, op in order]

    def recurse(node: int, conditions: list[tuple[str, str, float]]) -> None:
        if t.children_left[node] == -1:  # leaf
            rules.append((simplify(conditions), node))
            return
        feat = feature_names[t.feature[node]]
        thr = float(t.threshold[node])
        recurse(t.children_left[node], conditions + [(feat, "<=", thr)])
        recurse(t.children_right[node], conditions + [(feat, ">", thr)])

    recurse(0, [])
    return rules


def discover_rules(
    df: pd.DataFrame,
    features: list[str],
    max_depth: int = 3,
    min_samples_leaf: int = 100,
    min_lift: float = 1.5,
    top_k: int = 10,
) -> list[Rule]:
    """Decision Tree로 불량률이 높은 조건 조합 규칙을 추출한다.

    Args:
        df: 데이터 (TARGET_COLUMN 포함).
        features: 규칙 탐색에 사용할 feature (사전 filtering된 상위 feature 권장).
        max_depth: 규칙 조건 최대 개수 (해석 가능성 위해 3 이하 권장).
        min_samples_leaf: 규칙의 최소 support.
        min_lift: 전체 defect rate 대비 최소 배수.
        top_k: 반환할 규칙 수.
    """
    X = df[features]
    y = df[TARGET_COLUMN]
    overall_rate = float(y.mean())
    if overall_rate == 0:
        return []

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    tree.fit(X, y)

    leaf_ids = tree.apply(X)
    rules: list[Rule] = []
    total_defects = int(y.sum())
    n_total = len(df)

    for conditions, node in _extract_leaf_rules(tree, features):
        mask = leaf_ids == node
        n = int(mask.sum())
        if n < min_samples_leaf or not conditions:
            continue
        n_def = int(y[mask].sum())
        rate = n_def / n
        lift = rate / overall_rate
        if lift < min_lift:
            continue
        # Fisher exact: [규칙 내 불량/정상] vs [규칙 외 불량/정상]
        table = [
            [n_def, n - n_def],
            [total_defects - n_def, (n_total - n) - (total_defects - n_def)],
        ]
        _, p = fisher_exact(table, alternative="greater")
        rules.append(
            Rule(
                conditions=conditions,
                n_samples=n,
                n_defects=n_def,
                defect_rate=rate,
                lift=lift,
                risk_increase=rate - overall_rate,
                p_value=float(p),
            )
        )

    # ranking: lift × log(support) 조합 (support 너무 작은 규칙 과대평가 방지)
    rules.sort(key=lambda r: r.lift * np.log10(max(r.n_samples, 10)), reverse=True)
    logger.info("Rule discovery: %d rules found (lift >= %.1f)", len(rules), min_lift)
    return rules[:top_k]


def rules_to_dataframe(rules: list[Rule]) -> pd.DataFrame:
    """규칙 목록을 표시용 DataFrame으로 변환한다."""
    if not rules:
        return pd.DataFrame(
            columns=["rule", "samples", "defects", "defect_rate", "lift",
                     "risk_increase", "p_value"]
        )
    return pd.DataFrame(
        [
            {
                "rule": r.condition_text,
                "samples": r.n_samples,
                "defects": r.n_defects,
                "defect_rate": r.defect_rate,
                "lift": r.lift,
                "risk_increase": r.risk_increase,
                "p_value": r.p_value,
            }
            for r in rules
        ]
    )

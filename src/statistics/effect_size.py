"""Effect size 계산.

검정 방법과 effect size 짝:
    - t-test (Welch)     → Cohen's d
    - Mann-Whitney U     → Cliff's delta
"""
from __future__ import annotations

import numpy as np


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d (pooled std 기준). group_a - group_b 방향."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a, var_b = group_a.var(ddof=1), group_b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((group_a.mean() - group_b.mean()) / pooled_std)


def cliffs_delta(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cliff's delta ∈ [-1, 1]. Mann-Whitney U 통계량 기반 O(n log n) 계산.

    delta = P(a > b) - P(a < b)
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a == 0 or n_b == 0:
        return float("nan")
    # U 통계량으로부터 계산 (동순위는 0.5 처리)
    from scipy.stats import mannwhitneyu

    u, _ = mannwhitneyu(group_a, group_b, alternative="two-sided")
    return float(2.0 * u / (n_a * n_b) - 1.0)


def effect_size_label(value: float, method: str) -> str:
    """effect size 크기를 관례적 기준으로 라벨링한다.

    Cohen's d: 0.2 small / 0.5 medium / 0.8 large
    Cliff's delta: 0.147 small / 0.33 medium / 0.474 large
    """
    v = abs(value)
    if np.isnan(v):
        return "unknown"
    if method == "cohens_d":
        thresholds = (0.2, 0.5, 0.8)
    else:  # cliffs_delta
        thresholds = (0.147, 0.33, 0.474)
    if v < thresholds[0]:
        return "negligible"
    if v < thresholds[1]:
        return "small"
    if v < thresholds[2]:
        return "medium"
    return "large"

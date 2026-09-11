"""모델별 feature importance 및 model consensus ranking.

- logistic: 표준화된 입력에 대한 |coefficient| (Pipeline 내 StandardScaler 덕분에
  스케일 영향 없이 비교 가능)
- random_forest: impurity 기반 feature_importances_
- xgboost / lightgbm: gain 기반 importance
- 각 모델 importance는 0~1 정규화(max=1) 후, 모델별 rank의 평균으로
  consensus ranking을 만든다. SHAP importance는 선택적으로 결합.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.ml.trainer import TrainResult

logger = logging.getLogger(__name__)

_RANK_COLUMN_MAP = {
    "logistic": "rank_logistic",
    "random_forest": "rank_rf",
    "xgboost": "rank_xgb",
    "lightgbm": "rank_lgbm",
}


def _logistic_importance(model: object, features: list[str]) -> pd.Series:
    """Pipeline(StandardScaler+LogisticRegression)에서 |coef| 추출."""
    coef = model.named_steps["clf"].coef_.ravel()
    return pd.Series(np.abs(coef), index=features)


def _xgboost_gain(model: object, features: list[str]) -> pd.Series:
    """XGBoost gain importance (split에 안 쓰인 feature는 0)."""
    score = model.get_booster().get_score(importance_type="gain")
    return pd.Series({f: score.get(f, 0.0) for f in features})


def _lightgbm_gain(model: object, features: list[str]) -> pd.Series:
    """LightGBM gain importance."""
    gain = model.booster_.feature_importance(importance_type="gain")
    return pd.Series(gain, index=features)


def _normalize(s: pd.Series) -> pd.Series:
    """0~1 정규화 (최댓값 기준). 전부 0이면 그대로 반환."""
    max_val = s.max()
    return s / max_val if max_val > 0 else s


def compute_importances(result: TrainResult) -> pd.DataFrame:
    """모델별 feature importance 테이블.

    Returns:
        컬럼: feature + 학습된 모델별 컬럼(logistic, random_forest,
        xgboost, lightgbm) — 각 컬럼 0~1 정규화.
    """
    features = result.feature_names
    extractors = {
        "logistic": _logistic_importance,
        "random_forest": lambda m, f: pd.Series(m.feature_importances_, index=f),
        "xgboost": _xgboost_gain,
        "lightgbm": _lightgbm_gain,
    }
    table = pd.DataFrame({"feature": features}).set_index("feature")
    for name, model in result.models.items():
        if name not in extractors:
            logger.warning("No importance extractor for model %s — skipped", name)
            continue
        table[name] = _normalize(extractors[name](model, features))
    logger.info(
        "Computed importances for %d features × %d models",
        len(table), table.shape[1],
    )
    return table.reset_index()


def _trimmed_mean_rank(ranks: pd.DataFrame) -> pd.Series:
    """행별 robust mean rank: rank source가 3개 이상이면 최악 rank 1개 제외.

    선형 baseline(logistic)은 비단조(U자형) 효과를 구조적으로 잡지 못해
    해당 feature에 극단적으로 나쁜 rank를 주는데, 단순 평균은 이 한 모델의
    'veto'로 tree 모델들이 일치 검출한 feature를 묻어버린다. 최악 rank
    1개를 제외한 trimmed mean으로 이를 방지한다.
    """
    if ranks.shape[1] < 3:
        return ranks.mean(axis=1, skipna=True)
    total = ranks.sum(axis=1, skipna=True)
    worst = ranks.max(axis=1, skipna=True)
    count = ranks.notna().sum(axis=1)
    return (total - worst) / (count - 1).clip(lower=1)


def consensus_ranking(
    importances: pd.DataFrame,
    shap_importance: pd.Series | None = None,
) -> pd.DataFrame:
    """모델별 rank(+선택적 SHAP rank)의 평균으로 consensus ranking 생성.

    mean_rank는 최악 rank 1개를 제외한 trimmed mean (rank source 3개 이상일
    때) — 비선형 효과에 눈먼 단일 모델이 consensus를 왜곡하지 않도록.

    Args:
        importances: compute_importances 결과 (feature + 모델별 컬럼).
        shap_importance: index=feature명인 SHAP importance. None이면 미포함.

    Returns:
        컬럼: feature, rank_logistic, rank_rf, rank_xgb, rank_lgbm,
        [rank_shap], mean_rank, consensus_rank — consensus_rank 오름차순 정렬.
    """
    table = importances.set_index("feature")
    ranks = pd.DataFrame(index=table.index)
    for model_col, rank_col in _RANK_COLUMN_MAP.items():
        if model_col in table.columns:
            ranks[rank_col] = table[model_col].rank(ascending=False, method="min")

    if shap_importance is not None:
        shap_aligned = shap_importance.reindex(table.index)
        ranks["rank_shap"] = shap_aligned.rank(ascending=False, method="min")

    ranks["mean_rank"] = _trimmed_mean_rank(ranks)
    ranks["consensus_rank"] = (
        ranks["mean_rank"].rank(ascending=True, method="min").astype(int)
    )
    ranks = ranks.sort_values("consensus_rank")
    logger.info(
        "Consensus ranking over %d features (%d rank sources); top: %s",
        len(ranks), ranks.shape[1] - 2, ", ".join(ranks.index[:3]),
    )
    return ranks.reset_index()

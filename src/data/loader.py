"""데이터 로딩 및 feature/target 분리 유틸리티."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.config import (
    NON_FEATURE_COLUMNS,
    RAW_DATASET_PATH,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


def load_dataset(path: Path | str = RAW_DATASET_PATH) -> pd.DataFrame:
    """parquet 데이터셋을 로드한다.

    Args:
        path: parquet 파일 경로.

    Raises:
        FileNotFoundError: 파일이 없을 때 (generator 먼저 실행 필요).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run `python -m src.data.generator` first."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded dataset: %d rows × %d cols from %s", *df.shape, path)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """ML/통계 분석 대상 numeric feature 컬럼 목록.

    leakage 방지: 식별자, TIMESTAMP, target, DEFECT_TYPE, YIELD,
    QUALITY_SCORE는 제외한다 (config.NON_FEATURE_COLUMNS).
    """
    excluded = set(NON_FEATURE_COLUMNS)
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X, y, groups=LOT_ID) 튜플을 반환한다."""
    features = get_feature_columns(df)
    return df[features], df[TARGET_COLUMN], df["LOT_ID"]


def filter_dataset(
    df: pd.DataFrame,
    products: list[str] | None = None,
    processes: list[str] | None = None,
    defect_types: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """대시보드 sidebar 조건으로 데이터를 필터링한다.

    defect_types 필터는 '해당 타입 불량 + 전체 정상'을 남긴다
    (불량 vs 정상 비교 구조 유지).
    """
    out = df
    if products:
        out = out[out["PRODUCT_ID"].isin(products)]
    if processes:
        out = out[out["PROCESS_ID"].isin(processes)]
    if defect_types:
        out = out[(out["DEFECT_FLAG"] == 0) | (out["DEFECT_TYPE"].isin(defect_types))]
    if date_range:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        # end 날짜 당일 포함
        out = out[(out["TIMESTAMP"] >= start) & (out["TIMESTAMP"] < end + pd.Timedelta(days=1))]
    return out

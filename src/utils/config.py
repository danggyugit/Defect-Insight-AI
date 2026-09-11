"""프로젝트 전역 설정.

모든 상수/경로/시드는 이 모듈에서 관리한다. 다른 모듈에서 하드코딩 금지.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATASET_PATH = DATA_RAW_DIR / "manufacturing_data.parquet"

# ---------------------------------------------------------------------------
# 재현성
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 컬럼 정의
# ---------------------------------------------------------------------------
ID_COLUMNS = ["LOT_ID", "PANEL_ID", "PRODUCT_ID", "PROCESS_ID", "EQUIPMENT_ID"]
TIMESTAMP_COLUMN = "TIMESTAMP"
TARGET_COLUMN = "DEFECT_FLAG"
DEFECT_TYPE_COLUMN = "DEFECT_TYPE"
# target에서 파생된 품질 컬럼 — ML feature에서 반드시 제외 (leakage)
QUALITY_DERIVED_COLUMNS = ["QUALITY_SCORE", "YIELD"]

# ML feature에서 제외할 전체 컬럼 목록
NON_FEATURE_COLUMNS = (
    ID_COLUMNS
    + [TIMESTAMP_COLUMN, TARGET_COLUMN, DEFECT_TYPE_COLUMN]
    + QUALITY_DERIVED_COLUMNS
)

DEFECT_TYPES = ["NONE", "MURA", "PARTICLE", "SCRATCH", "OPEN_SHORT"]

# ---------------------------------------------------------------------------
# 분석 파라미터
# ---------------------------------------------------------------------------
SIGNIFICANCE_ALPHA = 0.05          # BH 보정 후 유의 판정 기준
TOP_N_FEATURES_INTERACTION = 15    # interaction/SHAP 분석 대상 상위 feature 수
TOP_N_FEATURES_REPORT = 20         # report에 표시할 상위 feature 수
VARIANCE_FILTER_THRESHOLD = 1e-10  # 1차 filtering: 상수 feature 제거
MISSING_RATE_THRESHOLD = 0.5       # missing rate 초과 시 feature 제외
LEAKAGE_CORR_THRESHOLD = 0.95      # target 상관 초과 시 leakage 의심 warning

TEST_SIZE = 0.25                   # GroupShuffleSplit test 비율
RISK_LEVEL_THRESHOLDS = {"LOW": 0.3, "MID": 0.6}  # 초과 시 다음 등급, 0.6+ = HIGH


# ---------------------------------------------------------------------------
# Synthetic data generator 설정
# ---------------------------------------------------------------------------
@dataclass
class GeneratorConfig:
    """Synthetic manufacturing dataset 생성 파라미터."""

    n_samples: int = 30_000
    n_lots: int = 1_200
    target_defect_rate: float = 0.06
    start_date: str = "2026-01-01"
    end_date: str = "2026-08-31"
    seed: int = RANDOM_SEED

    # feature 수 (합계 = 400)
    n_temp: int = 60
    n_pressure: int = 60
    n_speed: int = 50
    n_current: int = 50
    n_voltage: int = 40
    n_time: int = 40
    n_flow: int = 40
    n_equip_param: int = 40
    n_inspection: int = 20

    products: list[str] = field(
        default_factory=lambda: ["PROD_A", "PROD_B", "PROD_C"]
    )
    processes: list[str] = field(
        default_factory=lambda: ["PROC_CVD", "PROC_PHOTO", "PROC_ETCH", "PROC_TEST"]
    )
    equipments: list[str] = field(
        default_factory=lambda: [f"EQ_{c}" for c in "ABCDEF"]
    )
    # baseline shift가 심어진 설비 (Case 6)
    shifted_equipment: str = "EQ_E"

    @property
    def total_features(self) -> int:
        return (
            self.n_temp + self.n_pressure + self.n_speed + self.n_current
            + self.n_voltage + self.n_time + self.n_flow
            + self.n_equip_param + self.n_inspection
        )


DEFAULT_GENERATOR_CONFIG = GeneratorConfig()


def ensure_directories() -> None:
    """데이터/모델/리포트 디렉토리 생성."""
    for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

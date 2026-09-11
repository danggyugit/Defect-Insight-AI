"""Synthetic Manufacturing Dataset Generator.

실제 제조공정을 모사한 합성 데이터를 생성한다.

심어진 불량 mechanism (docs/PROJECT_PROMPT.md §6):
    Case 1: TEMP_004 증가 → defect 확률 증가 (선형)              → MURA
    Case 2: PRESSURE_012 증가 → defect 확률 증가                 → MURA
    Case 3: TEMP_004 × PRESSURE_012 interaction → 확률 급증       → MURA
    Case 4: SPEED_007 정상 범위 이탈 (U자형, 비선형)              → SCRATCH
    Case 5: TIME_003 + TEMP_004 조합                              → PARTICLE
    Case 6: 특정 EQUIPMENT baseline shift                         → OPEN_SHORT
    Case 7: 시간에 따른 공정 drift → 후기 불량 증가               → PARTICLE
    Case 8: confounder — 불량과 무관하지만 유의 변수와 강한 상관
            (INSPECTION_005 ~ TEMP_004, EQUIP_PARAM_010 ~ PRESSURE_012)

실행: python -m src.data.generator
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.config import (
    DEFAULT_GENERATOR_CONFIG,
    RAW_DATASET_PATH,
    GeneratorConfig,
    ensure_directories,
)

logger = logging.getLogger(__name__)

# feature 유형별 (prefix, 개수 필드, 평균, 표준편차)
_FEATURE_SPECS = [
    ("TEMP", "n_temp", 82.0, 3.0),
    ("PRESSURE", "n_pressure", 1.15, 0.08),
    ("SPEED", "n_speed", 120.0, 8.0),
    ("CURRENT", "n_current", 4.5, 0.6),
    ("VOLTAGE", "n_voltage", 220.0, 6.0),
    ("TIME", "n_time", 45.0, 5.0),
    ("FLOW", "n_flow", 30.0, 4.0),
    ("EQUIP_PARAM", "n_equip_param", 50.0, 10.0),
    ("INSPECTION", "n_inspection", 0.0, 1.0),
]

# mechanism 기여도 → DEFECT_TYPE 매핑에 사용하는 키
_MECHANISM_TYPES = {
    "temp": "MURA",
    "pressure": "MURA",
    "interaction": "MURA",
    "speed": "SCRATCH",
    "time_temp": "PARTICLE",
    "drift": "PARTICLE",
    "equipment": "OPEN_SHORT",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(
    logits: np.ndarray, target_rate: float, lo: float = -15.0, hi: float = 5.0
) -> float:
    """평균 defect 확률이 target_rate가 되도록 intercept를 이분탐색으로 보정."""
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _sigmoid(logits + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def generate_dataset(config: GeneratorConfig | None = None) -> pd.DataFrame:
    """합성 제조 데이터셋을 생성한다.

    Args:
        config: 생성 파라미터. None이면 기본값 사용.

    Returns:
        식별자 + 품질 + 공정변수 컬럼을 가진 DataFrame.
    """
    cfg = config or DEFAULT_GENERATOR_CONFIG
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_samples

    logger.info(
        "Generating dataset: %d samples, %d lots, %d features, target defect rate %.1f%%",
        n, cfg.n_lots, cfg.total_features, cfg.target_defect_rate * 100,
    )

    # ------------------------------------------------------------------
    # 1) 식별자 / 시간축
    # ------------------------------------------------------------------
    lot_indices = rng.integers(0, cfg.n_lots, size=n)
    # LOT별 속성 (product/process/equipment는 LOT 단위로 고정)
    lot_products = rng.choice(cfg.products, size=cfg.n_lots)
    lot_processes = rng.choice(cfg.processes, size=cfg.n_lots)
    lot_equipments = rng.choice(cfg.equipments, size=cfg.n_lots)

    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    total_seconds = (end - start).total_seconds()
    # LOT 시각을 기간 내 균등 분포시키고 LOT 내 샘플은 ±2시간 산포
    lot_time_frac = np.sort(rng.uniform(0, 1, size=cfg.n_lots))
    sample_offset = rng.uniform(-7200, 7200, size=n)
    timestamps = start + pd.to_timedelta(
        lot_time_frac[lot_indices] * total_seconds + sample_offset, unit="s"
    )
    time_frac = np.clip(
        (timestamps - start).total_seconds() / total_seconds, 0, 1
    )  # 0(기간 초) ~ 1(기간 말) — drift 항에 사용

    # ------------------------------------------------------------------
    # 2) 공정 변수 생성
    # ------------------------------------------------------------------
    feature_names: list[str] = []
    columns: dict[str, np.ndarray] = {}

    equipment_codes = pd.Categorical(
        lot_equipments[lot_indices], categories=cfg.equipments
    ).codes

    for prefix, count_attr, mean, std in _FEATURE_SPECS:
        count = getattr(cfg, count_attr)
        base = rng.normal(mean, std, size=(n, count))

        # 일부 변수는 skewed distribution (각 유형의 20%)
        n_skewed = max(1, count // 5)
        skew_idx = rng.choice(count, size=n_skewed, replace=False)
        base[:, skew_idx] = mean + std * rng.gamma(2.0, 1.0, size=(n, n_skewed)) - 2 * std

        # 일부 변수 간 correlation (인접 변수와 0.6 수준)
        n_corr = max(1, count // 4)
        corr_idx = rng.choice(count - 1, size=n_corr, replace=False)
        for j in corr_idx:
            base[:, j + 1] = 0.6 * base[:, j] + 0.4 * base[:, j + 1] + 0.4 * std * rng.normal(size=n)

        # 일부 변수는 설비별 offset (각 유형의 25%)
        n_offset = max(1, count // 4)
        offset_idx = rng.choice(count, size=n_offset, replace=False)
        equip_offsets = rng.normal(0, 0.5 * std, size=(len(cfg.equipments), n_offset))
        base[:, offset_idx] += equip_offsets[equipment_codes]

        # 일부 변수는 시간 drift (각 유형의 15%)
        n_drift = max(1, count // 7)
        drift_idx = rng.choice(count, size=n_drift, replace=False)
        drift_slopes = rng.normal(0, 0.8 * std, size=n_drift)
        base[:, drift_idx] += np.outer(time_frac, drift_slopes)

        for j in range(count):
            name = f"{prefix}_{j + 1:03d}"
            feature_names.append(name)
            columns[name] = base[:, j]

    # ------------------------------------------------------------------
    # 3) 불량 mechanism 주입
    # ------------------------------------------------------------------
    def z(name: str) -> np.ndarray:
        v = columns[name]
        return (v - v.mean()) / v.std()

    # Case 6: shifted equipment는 일부 EQUIP_PARAM baseline이 밀려 있음
    is_shifted_eq = (
        lot_equipments[lot_indices] == cfg.shifted_equipment
    ).astype(float)
    for pname in ("EQUIP_PARAM_003", "EQUIP_PARAM_007"):
        columns[pname] = columns[pname] + is_shifted_eq * 8.0

    z_temp = z("TEMP_004")
    z_press = z("PRESSURE_012")
    z_speed = z("SPEED_007")
    z_time3 = z("TIME_003")

    contrib = {
        # Case 1, 2: 단일 변수 선형 효과
        "temp": 0.75 * z_temp,
        "pressure": 0.6 * z_press,
        # Case 3: 두 변수 모두 높을 때 급증하는 interaction
        "interaction": 1.2 * np.clip(z_temp, 0, None) * np.clip(z_press, 0, None),
        # Case 4: SPEED_007 범위 이탈 (U자형 비선형)
        "speed": 1.4 * np.clip(np.abs(z_speed) - 0.9, 0, None),
        # Case 5: TIME_003 + TEMP_004 조합
        "time_temp": 0.9 * np.clip(z_time3, 0, None) * np.clip(z_temp, 0, None),
        # Case 6: equipment baseline shift
        "equipment": 1.0 * is_shifted_eq,
        # Case 7: 시간 drift에 따른 불량 증가 (기간 후반부)
        "drift": 0.8 * np.clip(time_frac - 0.6, 0, None) / 0.4,
    }

    logits = sum(contrib.values())
    intercept = _calibrate_intercept(np.asarray(logits), cfg.target_defect_rate)
    defect_prob = _sigmoid(logits + intercept)
    defect_flag = (rng.uniform(size=n) < defect_prob).astype(int)

    # Case 8: confounder — 유의 변수와 강한 상관, 불량 확률에는 미기여
    columns["INSPECTION_005"] = 0.85 * z_temp + 0.5 * rng.normal(size=n)
    columns["EQUIP_PARAM_010"] = 50.0 + 8.5 * z_press + 5.0 * rng.normal(size=n)

    # ------------------------------------------------------------------
    # 4) DEFECT_TYPE — 불량 샘플별 지배 mechanism 기준으로 할당
    # ------------------------------------------------------------------
    mech_keys = list(_MECHANISM_TYPES.keys())
    contrib_matrix = np.stack([np.broadcast_to(contrib[k], (n,)) for k in mech_keys], axis=1)
    dominant = np.array(mech_keys)[contrib_matrix.argmax(axis=1)]
    max_contrib = contrib_matrix.max(axis=1)

    defect_type = np.full(n, "NONE", dtype=object)
    defect_mask = defect_flag == 1
    # 지배 mechanism이 뚜렷하면 해당 타입, 아니면 random background 불량
    typed = defect_mask & (max_contrib > 0.3)
    defect_type[typed] = [_MECHANISM_TYPES[k] for k in dominant[typed]]
    background = defect_mask & ~typed
    defect_type[background] = rng.choice(
        ["MURA", "PARTICLE", "SCRATCH", "OPEN_SHORT"], size=int(background.sum())
    )

    # ------------------------------------------------------------------
    # 5) 품질 파생 컬럼 (leakage 검증용 — ML feature에서 제외 대상)
    # ------------------------------------------------------------------
    quality_score = np.clip(
        100.0 - 55.0 * defect_prob - 15.0 * defect_flag + rng.normal(0, 3, size=n),
        0, 100,
    )
    lot_defect_rate = pd.Series(defect_flag).groupby(lot_indices).transform("mean")
    yield_col = (1.0 - lot_defect_rate) * 100.0

    # ------------------------------------------------------------------
    # 6) DataFrame 조립
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "LOT_ID": [f"LOT_{i:05d}" for i in lot_indices],
            "PANEL_ID": [f"PNL_{i:06d}" for i in range(n)],
            "PRODUCT_ID": lot_products[lot_indices],
            "PROCESS_ID": lot_processes[lot_indices],
            "EQUIPMENT_ID": lot_equipments[lot_indices],
            "TIMESTAMP": timestamps,
            "DEFECT_FLAG": defect_flag,
            "DEFECT_TYPE": defect_type,
            "QUALITY_SCORE": quality_score,
            "YIELD": yield_col.to_numpy(),
        }
    )
    feature_df = pd.DataFrame({k: columns[k] for k in feature_names})
    df = pd.concat([df, feature_df], axis=1).sort_values("TIMESTAMP").reset_index(drop=True)

    logger.info(
        "Generated: %d rows × %d cols, defect rate %.2f%%",
        len(df), df.shape[1], df["DEFECT_FLAG"].mean() * 100,
    )
    return df


def main() -> None:
    """데이터셋을 생성하여 parquet으로 저장한다."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ensure_directories()
    df = generate_dataset()
    df.to_parquet(RAW_DATASET_PATH, index=False)
    logger.info("Saved to %s", RAW_DATASET_PATH)

    # 요약 로깅
    logger.info("Shape: %s", df.shape)
    logger.info("Defect rate: %.4f", df["DEFECT_FLAG"].mean())
    logger.info("Period: %s ~ %s", df["TIMESTAMP"].min(), df["TIMESTAMP"].max())
    logger.info(
        "Defect type distribution:\n%s",
        df.loc[df["DEFECT_FLAG"] == 1, "DEFECT_TYPE"].value_counts().to_string(),
    )
    logger.info(
        "Defect rate by equipment:\n%s",
        df.groupby("EQUIPMENT_ID")["DEFECT_FLAG"].mean().round(4).to_string(),
    )


if __name__ == "__main__":
    main()

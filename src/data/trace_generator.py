"""FDC Sensor Trace Generator — 딥러닝 분석용 시계열 파형 데이터.

기존 tabular 데이터셋의 각 샘플에 대해 FDC 센서 raw trace(파형)를 생성한다.
요약 통계(평균 등)로는 잡기 어려운 **파형 형태 이상**을 defect type별로 주입하여,
1D-CNN 같은 딥러닝 모델이 의미를 갖는 데이터 구조를 만든다.

심어진 파형 이상 (DEFECT_TYPE과 연결):
    MURA       → plateau 구간 내 완만한 drift (기울기)
    PARTICLE   → 짧은 transient spike (랜덤 위치)
    SCRATCH    → 고주파 oscillation (ripple)
    OPEN_SHORT → 중간 지점 level shift (계단형 변화)
    정상       → 깨끗한 파형 (3% 확률로 경미한 랜덤 이상 — false positive 현실성)

실행: python -m src.data.trace_generator  (main dataset 생성 후 실행할 것)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.utils.config import (
    DATA_RAW_DIR,
    RANDOM_SEED,
    ensure_directories,
)

logger = logging.getLogger(__name__)

TRACE_PATH = DATA_RAW_DIR / "fdc_traces.npz"

# trace 구조
N_SENSORS = 3
SENSOR_NAMES = ["RF_POWER", "CHAMBER_TEMP", "GAS_FLOW"]
TRACE_LENGTH = 128
_RAMP_UP_END = 20      # 0~20: ramp-up
_RAMP_DOWN_START = 108  # 108~128: ramp-down
_SENSOR_LEVELS = np.array([100.0, 65.0, 40.0])  # plateau 기준 레벨
_NOISE_STD = 1.0

# 정상 샘플에도 경미한 이상이 섞일 확률 (현실성 / false positive 검증용)
_NORMAL_ANOMALY_RATE = 0.03
# 불량 샘플에 해당 type의 파형 이상이 주입될 확률
_DEFECT_ANOMALY_RATE = 0.9


def _base_traces(rng: np.random.Generator, n: int) -> np.ndarray:
    """정상 파형 생성: ramp-up → plateau → ramp-down + noise + 샘플별 레벨 산포."""
    t = np.arange(TRACE_LENGTH, dtype=np.float32)
    profile = np.ones(TRACE_LENGTH, dtype=np.float32)
    profile[:_RAMP_UP_END] = t[:_RAMP_UP_END] / _RAMP_UP_END
    profile[_RAMP_DOWN_START:] = 1.0 - (
        (t[_RAMP_DOWN_START:] - _RAMP_DOWN_START) / (TRACE_LENGTH - _RAMP_DOWN_START)
    )
    # (n, sensors, T)
    traces = profile[None, None, :] * _SENSOR_LEVELS[None, :, None]
    level_jitter = rng.normal(0, 1.0, size=(n, N_SENSORS, 1)).astype(np.float32)
    noise = rng.normal(0, _NOISE_STD, size=(n, N_SENSORS, TRACE_LENGTH)).astype(np.float32)
    return (traces + level_jitter + noise).astype(np.float32)


def _inject_drift(trace: np.ndarray, rng: np.random.Generator) -> None:
    """MURA: plateau 구간 내 완만한 drift."""
    sensor = rng.integers(0, N_SENSORS)
    slope = rng.uniform(3.0, 6.0) * rng.choice([-1, 1])
    span = np.linspace(0, 1, _RAMP_DOWN_START - _RAMP_UP_END, dtype=np.float32)
    trace[sensor, _RAMP_UP_END:_RAMP_DOWN_START] += slope * span


def _inject_spike(trace: np.ndarray, rng: np.random.Generator) -> None:
    """PARTICLE: 짧은 transient spike (1~2개, 폭 2~5 step)."""
    sensor = rng.integers(0, N_SENSORS)
    for _ in range(rng.integers(1, 3)):
        pos = rng.integers(_RAMP_UP_END + 5, _RAMP_DOWN_START - 10)
        width = rng.integers(2, 6)
        amp = rng.uniform(4.0, 8.0) * rng.choice([-1, 1])
        trace[sensor, pos : pos + width] += amp


def _inject_oscillation(trace: np.ndarray, rng: np.random.Generator) -> None:
    """SCRATCH: plateau 구간 고주파 ripple."""
    sensor = rng.integers(0, N_SENSORS)
    span = _RAMP_DOWN_START - _RAMP_UP_END
    cycles = rng.uniform(8, 15)
    amp = rng.uniform(1.5, 3.0)
    phase = rng.uniform(0, 2 * np.pi)
    ripple = amp * np.sin(np.linspace(0, cycles * 2 * np.pi, span) + phase)
    trace[sensor, _RAMP_UP_END:_RAMP_DOWN_START] += ripple.astype(np.float32)


def _inject_level_shift(trace: np.ndarray, rng: np.random.Generator) -> None:
    """OPEN_SHORT: 중간 지점 계단형 level shift."""
    sensor = rng.integers(0, N_SENSORS)
    pos = rng.integers(_RAMP_UP_END + 15, _RAMP_DOWN_START - 15)
    shift = rng.uniform(3.0, 6.0) * rng.choice([-1, 1])
    trace[sensor, pos:_RAMP_DOWN_START] += shift


_INJECTORS = {
    "MURA": _inject_drift,
    "PARTICLE": _inject_spike,
    "SCRATCH": _inject_oscillation,
    "OPEN_SHORT": _inject_level_shift,
}


def generate_traces(
    df: pd.DataFrame, seed: int = RANDOM_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """main dataset의 각 샘플에 대응하는 FDC trace를 생성한다.

    Args:
        df: main dataset (PANEL_ID, DEFECT_FLAG, DEFECT_TYPE 필요).
        seed: 랜덤 시드.

    Returns:
        (traces (n, N_SENSORS, TRACE_LENGTH) float32, panel_ids (n,) str)
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    logger.info("Generating FDC traces: %d samples × %d sensors × %d steps",
                n, N_SENSORS, TRACE_LENGTH)

    traces = _base_traces(rng, n)
    defect_types = df["DEFECT_TYPE"].to_numpy()
    defect_flags = df["DEFECT_FLAG"].to_numpy()
    anomaly_types = list(_INJECTORS)

    n_injected = 0
    for i in range(n):
        if defect_flags[i] == 1 and defect_types[i] in _INJECTORS:
            if rng.uniform() < _DEFECT_ANOMALY_RATE:
                _INJECTORS[defect_types[i]](traces[i], rng)
                n_injected += 1
        elif rng.uniform() < _NORMAL_ANOMALY_RATE:
            # 정상 샘플의 경미한 랜덤 이상 (진폭 절반 수준)
            snapshot = traces[i].copy()
            _INJECTORS[anomaly_types[rng.integers(0, len(anomaly_types))]](traces[i], rng)
            traces[i] = snapshot + 0.5 * (traces[i] - snapshot)

    logger.info("Injected waveform anomalies into %d/%d defective samples",
                n_injected, int(defect_flags.sum()))
    return traces, df["PANEL_ID"].to_numpy().astype(str)


def save_traces(traces: np.ndarray, panel_ids: np.ndarray) -> None:
    """trace를 npz로 저장한다."""
    ensure_directories()
    np.savez_compressed(TRACE_PATH, traces=traces, panel_ids=panel_ids)
    logger.info("Saved traces to %s (%.1f MB)", TRACE_PATH,
                TRACE_PATH.stat().st_size / 1e6)


def load_traces() -> tuple[np.ndarray, np.ndarray]:
    """저장된 trace 로드. 없으면 FileNotFoundError."""
    if not TRACE_PATH.exists():
        raise FileNotFoundError(
            f"Traces not found: {TRACE_PATH}. "
            "Run `python -m src.data.trace_generator` first."
        )
    data = np.load(TRACE_PATH, allow_pickle=False)
    return data["traces"], data["panel_ids"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = load_dataset()
    traces, panel_ids = generate_traces(df)
    save_traces(traces, panel_ids)
    logger.info("Trace shape: %s, dtype: %s", traces.shape, traces.dtype)


if __name__ == "__main__":
    main()

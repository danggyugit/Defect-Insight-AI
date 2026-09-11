"""공용 fixture: 작은 규모의 synthetic dataset (테스트 속도용)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.generator import generate_dataset
from src.utils.config import GeneratorConfig


@pytest.fixture(scope="session")
def small_config() -> GeneratorConfig:
    return GeneratorConfig(
        n_samples=4000,
        n_lots=160,
        n_temp=10, n_pressure=15, n_speed=10, n_current=8,
        n_voltage=8, n_time=8, n_flow=8, n_equip_param=12, n_inspection=8,
    )


@pytest.fixture(scope="session")
def small_df(small_config: GeneratorConfig) -> pd.DataFrame:
    return generate_dataset(small_config)

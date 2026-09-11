"""Streamlit 페이지 렌더링 테스트 (AppTest — 예외 없이 렌더되는지 확인).

ML 학습이 필요한 페이지(04~06, 09)는 버튼 클릭 전 초기 렌더만 검증한다.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.utils.config import PROJECT_ROOT, RAW_DATASET_PATH

PAGES_DIR = PROJECT_ROOT / "views"

needs_dataset = pytest.mark.skipif(
    not RAW_DATASET_PATH.exists(),
    reason="dataset 없음 — python -m src.data.generator 먼저 실행",
)


def _run_page(path: Path, timeout: int = 120) -> AppTest:
    at = AppTest.from_file(str(path), default_timeout=timeout)
    at.run()
    assert not at.exception, f"{path.name} 렌더 중 예외: {at.exception}"
    return at


def test_app_entry() -> None:
    _run_page(PROJECT_ROOT / "app.py", timeout=30)


@needs_dataset
@pytest.mark.parametrize(
    "page",
    sorted(p.name for p in PAGES_DIR.glob("*.py")),
)
def test_page_renders(page: str) -> None:
    _run_page(PAGES_DIR / page)

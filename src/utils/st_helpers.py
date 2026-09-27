"""Streamlit 공통 헬퍼: 데이터 캐시, sidebar 필터, 페이지 헤더/설명, 디자인 토큰."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.loader import filter_dataset, load_dataset
from src.data.validator import ValidationResult, validate_dataset
from src.utils.config import DEFECT_TYPES
from src.utils.explanations import get_explanation

DISCLAIMER = (
    "⚠️ 본 분석은 통계적 연관성을 탐색하는 목적이며, "
    "인과관계를 직접 증명하지 않습니다. "
    "식별된 변수는 '추가 공정 검증이 권고되는 후보'로 해석해야 합니다."
)

# ---------------------------------------------------------------------------
# 디자인 토큰 — 전 페이지 공통 색상 (정상=파랑, 불량=빨강 고정)
# ---------------------------------------------------------------------------
COLOR_NORMAL = "#3b82f6"
COLOR_DEFECT = "#ef4444"
COLOR_ACCENT = "#8b5cf6"
GROUP_COLOR_MAP = {"Normal": COLOR_NORMAL, "Defect": COLOR_DEFECT,
                   0: COLOR_NORMAL, 1: COLOR_DEFECT, "0": COLOR_NORMAL, "1": COLOR_DEFECT}

# 분석 여정 (breadcrumb 용) — (짧은 라벨, 핵심 질문)
ANALYSIS_JOURNEY = [
    ("현황", "어디서 불량이 나나"),
    ("유의차", "무엇이 다른가"),
    ("상관·MI", "무엇과 관련 있나"),
    ("ML", "어떤 변수가 중요한가"),
    ("SHAP", "왜 그렇게 판단했나"),
    ("조합·규칙", "어떤 조합이 위험한가"),
    ("설비", "어느 설비가 다른가"),
    ("추이", "시간에 따라 변하나"),
    ("리포트", "무엇을 보고하나"),
    ("딥러닝", "파형에 이상은 없나"),
]


def page_header(step: int, title: str, question: str, explain_key: str | None = None) -> None:
    """공통 페이지 헤더: 제목 + 분석 여정 breadcrumb + '쉽게 이해하기' expander.

    Args:
        step: 분석 여정에서의 순서 (1~10).
        title: 페이지 제목 (이모지 포함).
        question: 이 페이지가 답하는 질문.
        explain_key: explanations.py의 설명 키. None이면 expander 생략.
    """
    st.title(title)
    crumbs = []
    for i, (label, _) in enumerate(ANALYSIS_JOURNEY, start=1):
        crumbs.append(f"**:violet[{label}]**" if i == step else label)
    st.caption("분석 여정 : " + " › ".join(crumbs))
    st.markdown(f"#### ❓ {question}")
    if explain_key:
        with st.expander("📖 이 분석, 쉽게 이해하기 (처음이라면 펼쳐보세요)"):
            st.markdown(get_explanation(explain_key))
    st.markdown("---")


def insight_box(lines: list[str], title: str = "💡 이 화면의 핵심 발견") -> None:
    """분석 결과 자동 요약 박스 — '그래서 결론이 뭔데'에 답한다."""
    if not lines:
        return
    body = "\n".join(f"- {line}" for line in lines)
    st.success(f"**{title}**\n\n{body}")


@st.cache_resource(show_spinner=False)
def ensure_dataset_ready() -> bool:
    """데이터셋이 없으면 자동 생성 (Streamlit Cloud 최초 부팅 대응).

    합성 데이터는 seed 고정이라 언제 생성해도 동일 — git에 넣지 않고
    배포 환경에서 최초 1회 생성한다 (재부팅 시 재생성, 약 1분).
    """
    from src.utils.config import RAW_DATASET_PATH, ensure_directories

    if not RAW_DATASET_PATH.exists():
        with st.spinner("⏳ 최초 실행: 데모 데이터셋 생성 중... (30,000건 × 400변수, 약 1분 — 최초 1회만)"):
            from src.data.generator import generate_dataset

            ensure_directories()
            df = generate_dataset()
            df.to_parquet(RAW_DATASET_PATH, index=False)
    return True


@st.cache_data(show_spinner="데이터 로딩 중...")
def cached_load_dataset() -> pd.DataFrame:
    """parquet 데이터셋 캐시 로딩 (없으면 자동 생성)."""
    ensure_dataset_ready()
    return load_dataset()


@st.cache_data(show_spinner="데이터 검증 중...")
def cached_validate(df: pd.DataFrame) -> ValidationResult:
    """데이터 검증 결과 캐시."""
    return validate_dataset(df)


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """공통 sidebar 필터를 렌더링하고 필터링된 DataFrame을 반환한다.

    선택값은 session_state에 저장되어 페이지 간 유지된다.
    """
    st.sidebar.header("🔍 분석 조건")

    products = st.sidebar.multiselect(
        "Product", sorted(df["PRODUCT_ID"].unique()), key="filter_products"
    )
    processes = st.sidebar.multiselect(
        "Process", sorted(df["PROCESS_ID"].unique()), key="filter_processes"
    )
    defect_types = st.sidebar.multiselect(
        "Defect Type", [t for t in DEFECT_TYPES if t != "NONE"], key="filter_defect_types"
    )

    min_d, max_d = df["TIMESTAMP"].min().date(), df["TIMESTAMP"].max().date()
    date_range = st.sidebar.date_input(
        "기간", value=(min_d, max_d), min_value=min_d, max_value=max_d,
        key="filter_dates",
    )

    filtered = filter_dataset(
        df,
        products=products or None,
        processes=processes or None,
        defect_types=defect_types or None,
        date_range=(
            (str(date_range[0]), str(date_range[1]))
            if isinstance(date_range, tuple) and len(date_range) == 2
            else None
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.metric("선택된 Sample", f"{len(filtered):,}")
    st.sidebar.metric("Defect Rate", f"{filtered['DEFECT_FLAG'].mean():.2%}" if len(filtered) else "—")

    if len(filtered) < 500:
        st.sidebar.warning("샘플이 500개 미만입니다. 통계 검정력이 부족할 수 있습니다.")

    return filtered


def show_disclaimer() -> None:
    """상관≠인과 disclaimer 표시 (분석 결과 페이지 필수)."""
    st.info(DISCLAIMER)


def guard_empty(df: pd.DataFrame) -> bool:
    """필터 결과가 비었으면 안내 후 True 반환 (페이지에서 조기 return용)."""
    if len(df) == 0:
        st.warning("선택된 조건에 해당하는 데이터가 없습니다. 필터를 조정하세요.")
        return True
    if df["DEFECT_FLAG"].sum() == 0:
        st.warning("선택된 조건에 불량 샘플이 없습니다. 필터를 조정하세요.")
        return True
    return False

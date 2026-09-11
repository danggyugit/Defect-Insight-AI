"""Manufacturing Defect Insight AI — Streamlit 진입점.

st.navigation 기반 멀티페이지: 제조 엔지니어의 실제 분석 순서대로
메뉴를 단계별 그룹으로 구성한다.

실행: streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Manufacturing Defect Insight AI",
    page_icon="🏭",
    layout="wide",
)

pages = {
    "": [
        st.Page("views/home.py", title="프로젝트 소개", icon="🏭", default=True),
    ],
    "STEP 1 · 현황 파악": [
        st.Page("views/01_overview.py", title="불량 현황 한눈에", icon="📊"),
    ],
    "STEP 2 · 원인 탐색 (통계)": [
        st.Page("views/02_statistics.py", title="무엇이 다른가 — 유의차", icon="📈"),
        st.Page("views/03_correlation.py", title="무엇과 관련 있나 — 상관·MI", icon="🔗"),
    ],
    "STEP 3 · AI 모델링": [
        st.Page("views/04_ml_analysis.py", title="어떤 변수가 중요한가 — ML", icon="🤖"),
        st.Page("views/05_shap.py", title="왜 그렇게 판단했나 — SHAP", icon="🔍"),
        st.Page("views/10_deeplearning.py", title="파형 이상 탐지 — 딥러닝", icon="🧠"),
    ],
    "STEP 4 · 심화 분석": [
        st.Page("views/06_multivariate.py", title="어떤 조합이 위험한가", icon="🧩"),
        st.Page("views/07_equipment.py", title="어느 설비가 다른가", icon="🏗️"),
        st.Page("views/08_timeseries.py", title="시간 추이 · Drift", icon="⏱️"),
    ],
    "STEP 5 · 결과 전달": [
        st.Page("views/09_report.py", title="자동 분석 리포트", icon="📄"),
    ],
}

st.navigation(pages).run()

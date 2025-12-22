import streamlit as st
from dotenv import load_dotenv

# [Refactored] 분리된 UI 모듈 가져오기
from src.ui import render_sidebar, render_chat_interface
from src.inference import RAGEngine

# 1. 설정 로드
load_dotenv()
st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")
st.markdown("<style>.stChatFloatingInputContainer {bottom: 20px;}</style>", unsafe_allow_html=True)

# 2. 전역 State 초기화
if 'target_company' not in st.session_state:
    st.session_state['target_company'] = "삼성전자"
if 'target_year' not in st.session_state:
    st.session_state['target_year'] = "2024"

# 3. RAG 엔진 로드 (캐싱)
@st.cache_resource
def load_engine():
    return RAGEngine()

# ==========================================
# [Main Application Flow]
# ==========================================
def main():
    # A. 사이드바 렌더링 (설정 및 데이터 처리)
    render_sidebar()

    # B. 엔진 준비
    try:
        rag_engine = load_engine()
    except Exception as e:
        st.error("RAG 엔진 로드 실패. 관리자에게 문의하세요.")
        st.stop()

    # C. 채팅 인터페이스 렌더링 (메인 화면)
    render_chat_interface(rag_engine)

if __name__ == "__main__":
    main()
import streamlit as st
import os
import time
from dotenv import load_dotenv

# [설정] 모듈 가져오기
from src.ingestion import DartCollector
from src.parsing import DartIntegratedParser
from src.embedding import VectorStoreBuilder
from src.inference import RAGEngine

# .env 로드
load_dotenv()

# ==========================================
# [UI 설정] 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (채팅창 스타일 개선)
st.markdown("""
<style>
    .stChatFloatingInputContainer {bottom: 20px;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# [사이드바] 데이터 파이프라인 제어 (Control Panel)
# ==========================================
with st.sidebar:
    st.title("🛠️ 데이터 관리자")
    
    # 1. API 설정
    with st.expander("🔐 API 설정", expanded=False):
        default_api_key = os.getenv("DART_API_KEY", "")
        api_key = st.text_input("DART API Key", value=default_api_key, type="password")

    # 2. 타겟 설정
    st.divider()
    st.subheader("🎯 분석 타겟 설정")
    target_company = st.text_input("회사명", value="삼성전자")
    target_year = st.text_input("연도", value="2024")

    # 3. 데이터 처리 파이프라인 (버튼)
    st.divider()
    st.subheader("⚙️ 지식 베이스 구축")

    # Step 1: 다운로드
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("1. DART 보고서 다운로드")
    with col2:
        if st.button("수집"):
            if not api_key:
                st.toast("❌ API Key가 필요합니다!", icon="🚫")
            else:
                with st.status("DART 서버 통신 중...", expanded=True) as status:
                    try:
                        collector = DartCollector(api_key=api_key)
                        path = collector.download_report(target_company, target_year)
                        if path:
                            st.session_state['xml_path'] = path
                            status.update(label="다운로드 완료!", state="complete", expanded=False)
                            st.toast(f"{target_company} 보고서 다운로드 성공!", icon="✅")
                        else:
                            status.update(label="보고서 없음", state="error")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Step 2: 파싱
    col3, col4 = st.columns([3, 1])
    with col3:
        st.caption("2. 텍스트 추출 및 가공")
    with col4:
        if st.button("가공"):
            xml_path = st.session_state.get('xml_path', "")
            if not xml_path or not os.path.exists(xml_path):
                st.toast("먼저 보고서를 수집해주세요!", icon="⚠️")
            else:
                with st.status("문서 분석 중...", expanded=True) as status:
                    parser = DartIntegratedParser()
                    parser.parse_file(xml_path)
                    status.update(label="파싱 완료!", state="complete", expanded=False)
                    st.toast("문서 가공 완료!", icon="✅")

    # Step 3: DB 구축
    col5, col6 = st.columns([3, 1])
    with col5:
        st.caption("3. 벡터 DB 저장 (AI 학습)")
    with col6:
        if st.button("학습"):
            with st.status("지식 베이스 구축 중 (GPU)...", expanded=True) as status:
                builder = VectorStoreBuilder()
                docs = builder.load_documents()
                if docs:
                    chunks = builder.split_documents(docs)
                    builder.build_database(chunks)
                    status.update(label="DB 구축 완료!", state="complete", expanded=False)
                    st.toast("AI 학습 완료! 이제 대화해보세요.", icon="🎉")
                else:
                    status.update(label="처리할 데이터 없음", state="error")
    
    st.divider()
    if st.button("🗑️ 대화 내용 초기화", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# [메인] 채팅 인터페이스 (ChatGPT Style)
# ==========================================

# 1. 헤더 (간단하게)
st.header(f"💬 DocuMind AI : {target_company} ({target_year})")
st.caption("기업 보고서 기반 RAG 챗봇")

# 2. RAG 엔진 로드 (캐싱)
@st.cache_resource
def load_engine():
    return RAGEngine()

try:
    rag_engine = load_engine()
except Exception as e:
    st.error("RAG 엔진을 로드할 수 없습니다. DB가 구축되었는지 확인하세요.")
    st.stop()

# 3. 대화 기록 초기화 및 표시
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"안녕하세요! **{target_company} {target_year}년** 보고서에 대해 무엇이든 물어보세요."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요... (예: 주요 사업 내용은?)"):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 답변 생성 및 표시
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 사이드바의 설정을 필터로 사용
        filters = {"company": target_company, "year": target_year}
        
        # 스트리밍 출력
        try:
            # rag_engine.chat은 generator이므로 for문으로 한 글자씩 받음
            for chunk in rag_engine.chat(prompt, filters=filters):
                full_response += chunk
                # 커서 효과(|) 추가
                message_placeholder.markdown(full_response + "▌")
            
            # 최종 출력 (커서 제거)
            message_placeholder.markdown(full_response)
            
            # 기록 저장
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
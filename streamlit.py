import streamlit as st
import os
import time
from dotenv import load_dotenv

# [설정] 모듈 가져오기 (새로 추가된 Ingestor/Parser 포함)
from src.ingestion import DartCollector, FileIngestor
from src.parsing import DartIntegratedParser, PDFParser
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

# 커스텀 CSS (채팅창 하단 고정 및 스타일 개선)
st.markdown("""
<style>
    .stChatFloatingInputContainer {bottom: 20px;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# [사이드바] 데이터 파이프라인 (Data Pipeline)
# ==========================================
with st.sidebar:
    st.title("🛠️ 데이터 관리자")
    
    # -------------------------------------------------------------------------
    # 1. 공통 설정 (Target Configuration)
    # -------------------------------------------------------------------------
    st.subheader("🎯 분석 타겟 설정")
    st.caption("업로드할 문서나 수집할 보고서의 기준 정보를 입력하세요.")
    
    # 이 정보는 메타데이터 태깅 및 채팅 필터링에 사용됩니다.
    target_company = st.text_input("회사명 (Company)", value="삼성전자")
    target_year = st.text_input("연도 (Year)", value="2024")
    
    st.divider()

    # -------------------------------------------------------------------------
    # 2. 데이터 소스 선택 (Hybrid Pipeline)
    # -------------------------------------------------------------------------
    st.subheader("📂 데이터 소스")
    data_source = st.radio(
        "데이터 확보 방식을 선택하세요:",
        ["DART API (자동)", "파일 직접 업로드"]
    )

    # -------------------------------------------
    # MODE A: DART API 사용
    # -------------------------------------------
    if data_source == "DART API (자동)":
        with st.expander("🔐 API 키 설정", expanded=True):
            default_api_key = os.getenv("DART_API_KEY", "")
            api_key = st.text_input("API Key", value=default_api_key, type="password")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.caption("1. 보고서 다운로드")
        with col2:
            if st.button("수집"):
                if not api_key:
                    st.toast("API Key가 필요합니다!", icon="🚫")
                else:
                    with st.status("DART 서버 통신 중...", expanded=True) as status:
                        try:
                            collector = DartCollector(api_key=api_key)
                            path = collector.download_report(target_company, target_year)
                            if path:
                                st.session_state['xml_path'] = path
                                status.update(label="다운로드 완료!", state="complete", expanded=False)
                                st.toast(f"{target_company} 보고서 저장 완료!", icon="✅")
                            else:
                                status.update(label="보고서 없음", state="error")
                        except Exception as e:
                            st.error(f"Error: {e}")

        # DART는 다운로드 후 파싱을 별도로 수행 (단계적 처리)
        col3, col4 = st.columns([2, 1])
        with col3:
            st.caption("2. XML 파싱 및 가공")
        with col4:
            if st.button("가공"):
                xml_path = st.session_state.get('xml_path', "")
                if not xml_path or not os.path.exists(xml_path):
                    st.toast("먼저 보고서를 수집해주세요!", icon="⚠️")
                else:
                    with st.status("문서 구조 분석 중...", expanded=True) as status:
                        parser = DartIntegratedParser()
                        parser.parse_file(xml_path)
                        status.update(label="파싱 완료!", state="complete", expanded=False)
                        st.toast("문서 가공 완료!", icon="✅")

    # -------------------------------------------
    # MODE B: 파일 직접 업로드
    # -------------------------------------------
    else:
        st.info("PDF, XML 파일을 직접 업로드하여 분석합니다.")
        uploaded_file = st.file_uploader("파일 선택", type=["pdf", "xml"])

        if uploaded_file is not None:
            if st.button("업로드 및 처리 시작", type="primary"):
                with st.status("데이터 처리 파이프라인 가동...", expanded=True) as status:
                    try:
                        # Step 1: 파일 저장
                        status.write("1. 서버에 파일 저장 중...")
                        ingestor = FileIngestor()
                        saved_path = ingestor.save_uploaded_file(uploaded_file)
                        
                        # Step 2: 확장자에 따른 자동 파싱
                        status.write("2. 문서 내용 추출 및 변환 중...")
                        file_ext = os.path.splitext(saved_path)[1].lower()
                        
                        if file_ext == ".xml":
                            parser = DartIntegratedParser()
                            parser.parse_file(saved_path)
                        elif file_ext == ".pdf":
                            parser = PDFParser()
                            parser.parse_file(saved_path)
                        else:
                            st.error("지원하지 않는 파일 형식입니다.")
                            st.stop()
                            
                        status.update(label="업로드 및 가공 완료!", state="complete", expanded=False)
                        st.toast("문서 처리 완료! DB 구축을 진행하세요.", icon="✅")
                        
                    except Exception as e:
                        st.error(f"처리 중 오류 발생: {e}")

    st.divider()

    # -------------------------------------------------------------------------
    # 3. DB 구축 (공통 단계)
    # -------------------------------------------------------------------------
    st.subheader("지식 베이스(DB) 업데이트")
    st.caption("가공된 데이터를 벡터 DB에 저장합니다. (필수)")
    
    if st.button("DB 학습 시작", use_container_width=True):
        with st.status("임베딩 및 벡터 저장 중 (GPU)...", expanded=True) as status:
            try:
                builder = VectorStoreBuilder()
                docs = builder.load_documents()
                if docs:
                    chunks = builder.split_documents(docs)
                    builder.build_database(chunks)
                    status.update(label="DB 구축 완료!", state="complete", expanded=False)
                    st.toast("AI 학습 완료! 이제 대화가 가능합니다.", icon="🎉")
                else:
                    status.update(label="처리할 데이터가 없습니다.", state="error")
                    st.warning("먼저 데이터를 수집/업로드하고 가공해주세요.")
            except Exception as e:
                st.error(f"DB 구축 실패: {e}")

    # 대화 초기화
    st.divider()
    if st.button("🗑️ 대화 내용 초기화"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# [메인] 채팅 인터페이스 (ChatGPT Style)
# ==========================================

# 1. 헤더 영역
st.header(f"DocuMind AI : {target_company} ({target_year})")
st.caption("기업 보고서 기반 RAG 지능형 질의응답 시스템")

# 2. RAG 엔진 로드 (캐싱으로 성능 최적화)
@st.cache_resource
def load_engine():
    return RAGEngine()

try:
    rag_engine = load_engine()
except Exception as e:
    st.error("RAG 엔진 로드 실패. `config.py` 설정이나 DB 상태를 확인하세요.")
    st.stop()

# 3. 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"안녕하세요! **{target_company} {target_year}년** 관련 문서에 대해 무엇이든 물어보세요."}
    ]

# 4. 이전 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. 사용자 입력 및 답변 생성
if prompt := st.chat_input("질문을 입력하세요... (예: 이 회사의 주요 리스크는?)"):
    # 사용자 질문 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 답변 표시 (Streaming)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 사이드바에서 설정한 회사/연도를 필터로 적용
        # (업로드 모드일 때도 사용자가 입력한 회사명/연도를 기준으로 검색)
        filters = {"company": target_company, "year": target_year}
        
        try:
            # rag_engine.chat은 generator -> 한 글자씩 받아옴
            for chunk in rag_engine.chat(prompt, filters=filters):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌") # 커서 효과
            
            # 최종 출력
            message_placeholder.markdown(full_response)
            
            # 대화 기록 저장
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
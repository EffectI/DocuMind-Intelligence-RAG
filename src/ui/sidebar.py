# src/ui/sidebar.py
import streamlit as st
import os
from src.utils import parse_filename_meta
from src.ingestion import DartCollector, FileIngestor
from src.parsing import DartIntegratedParser, PDFParser
from src.embedding import VectorStoreBuilder

def render_sidebar():
    """사이드바 UI 및 이벤트 처리 로직"""
    with st.sidebar:
        st.title("🛠️ 데이터 관리자")
        
        # 1. 소스 선택
        st.subheader("📂 데이터 소스")
        data_source = st.radio(
            "방식 선택", ["DART API (자동)", "파일 직접 업로드"],
            label_visibility="collapsed"
        )
        is_upload_mode = (data_source == "파일 직접 업로드")
        st.divider()

        # 2. 타겟 설정 (State 관리)
        st.subheader("🎯 분석 타겟 (Metadata)")
        if is_upload_mode:
            st.info("ℹ️ 파일 업로드 시 메타데이터가 자동 동기화됩니다.")

        target_company = st.text_input("회사명", key="target_company", disabled=is_upload_mode)
        target_year = st.text_input("연도", key="target_year", disabled=is_upload_mode)
        st.divider()

        # 3. 데이터 처리 로직 (DART vs Upload)
        if not is_upload_mode:
            _render_dart_mode()
        else:
            _render_upload_mode()

        st.divider()
        
        # 4. DB 구축 (공통)
        st.subheader("⚙️ 지식 베이스 업데이트")
        if st.button("🚀 DB 학습 시작", use_container_width=True):
            with st.status("Vector DB 업데이트 중...", expanded=True) as status:
                try:
                    builder = VectorStoreBuilder()
                    docs = builder.load_documents()
                    if docs:
                        chunks = builder.split_documents(docs)
                        builder.build_database(chunks)
                        status.update(label="학습 완료!", state="complete", expanded=False)
                        st.toast("학습 완료! 대화 가능.", icon="🎉")
                    else:
                        status.update(label="데이터 없음", state="error")
                except Exception as e:
                    st.error(f"실패: {e}")

        if st.button("🗑️ 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

def _render_dart_mode():
    """DART 모드 내부 로직"""
    with st.expander("🔐 API 키 설정", expanded=True):
        default_api_key = os.getenv("DART_API_KEY", "")
        api_key = st.text_input("API Key", value=default_api_key, type="password")

    if st.button("1. 수집 (Download)", use_container_width=True):
        if not api_key:
            st.toast("API Key 필요!", icon="🚫")
            return
        
        with st.status("DART 통신 중...", expanded=True) as status:
            try:
                collector = DartCollector(api_key=api_key)
                path = collector.download_report(st.session_state['target_company'], st.session_state['target_year'])
                if path:
                    st.session_state['xml_path'] = path
                    status.update(label="완료!", state="complete", expanded=False)
                    st.toast("저장 완료!", icon="✅")
                else:
                    status.update(label="보고서 없음", state="error")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("2. 가공 (Parsing)", use_container_width=True):
        xml_path = st.session_state.get('xml_path', "")
        if not xml_path:
            st.toast("먼저 수집해주세요!", icon="⚠️")
            return
            
        with st.status("분석 중...", expanded=True) as status:
            parser = DartIntegratedParser()
            parser.parse_file(xml_path)
            status.update(label="완료!", state="complete", expanded=False)
            st.toast("가공 완료!", icon="✅")

def _render_upload_mode():
    """업로드 모드 내부 로직"""
    uploaded_file = st.file_uploader("PDF/XML 업로드", type=["pdf", "xml"])
    
    if uploaded_file:
        # 메타데이터 자동 동기화 로직
        meta = parse_filename_meta(uploaded_file.name)
        if (meta['company'] != st.session_state['target_company']) or \
           (meta['year'] != st.session_state['target_year']):
            if meta['company']: st.session_state['target_company'] = meta['company']
            if meta['year']: st.session_state['target_year'] = meta['year']
            st.rerun()

        if st.button("업로드 및 처리 시작", type="primary", use_container_width=True):
            with st.status("처리 중...", expanded=True) as status:
                try:
                    ingestor = FileIngestor()
                    saved_path = ingestor.save_uploaded_file(uploaded_file)
                    
                    file_ext = os.path.splitext(saved_path)[1].lower()
                    if file_ext == ".xml":
                        DartIntegratedParser().parse_file(saved_path)
                    elif file_ext == ".pdf":
                        PDFParser().parse_file(saved_path)
                        
                    status.update(label="완료!", state="complete", expanded=False)
                    st.toast("처리 완료!", icon="✅")
                except Exception as e:
                    st.error(f"오류: {e}")
# src/ui/chat.py
import streamlit as st

def render_chat_interface(rag_engine):
    """채팅 인터페이스 렌더링"""
    curr_company = st.session_state.get('target_company', "Unknown")
    curr_year = st.session_state.get('target_year', "Unknown")

    st.header(f"💬 DocuMind : {curr_company} ({curr_year})")

    # 1. 초기 메시지
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"안녕하세요! **{curr_company} {curr_year}년** 문서를 분석할 준비가 되었습니다."}
        ]

    # 2. 히스토리 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. 입력 및 답변
    if prompt := st.chat_input("질문하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            msg_placeholder = st.empty()
            full_res = ""
            filters = {"company": curr_company, "year": curr_year}
            
            try:
                for chunk in rag_engine.chat(prompt, filters=filters):
                    full_res += chunk
                    msg_placeholder.markdown(full_res + "▌")
                msg_placeholder.markdown(full_res)
                st.session_state.messages.append({"role": "assistant", "content": full_res})
            except Exception as e:
                st.error(f"Error: {e}")
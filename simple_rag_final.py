import os
import torch
from threading import Thread

# 1. 랭체인 필수 요소
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# 2. 허깅페이스 모델 & 토크나이저 & 설정
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TextIteratorStreamer
)

# ==========================================
# [설정] 경로 및 모델
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")

LLM_MODEL_ID = "beomi/Llama-3-Open-Ko-8B"
EMBEDDING_MODEL = "BAAI/bge-m3"

def main():
    # 1. GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"========== [1] 시스템 점검 ({device}) ==========")
    
    # 2. 임베딩 모델 로드
    print("Loading Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device}
    )

    # 3. 벡터 DB 로드
    if not os.path.exists(DB_PATH):
        print("DB가 없습니다. simple_rag.py를 먼저 실행하세요.")
        return

    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="samsung_report_db"
    )
    # 검색기 생성 (유사도 높은 3개 문서 추출)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4. LLM 로드
    print(f"Loading LLM ({LLM_MODEL_ID}) with 4-bit Quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config, 
        device_map="auto",
    )

    # 5. 프롬프트 템플릿 설정
    template = """당신은 삼성전자 사업보고서를 분석하는 AI 비서입니다.
아래의 [참고 문서]를 바탕으로 질문에 대해 명확하고 간결하게 답변하세요.
만약 문서에 없는 내용이라면 "문서에 해당 내용이 없습니다"라고 말하세요.

[참고 문서]
{context}

질문: {question}
답변:"""
    prompt = PromptTemplate.from_template(template)

    # =========================================================
    # 스트리밍 함수 구현
    # =========================================================
    def stream_response(query):
        print(f"\n질문: {query}")
        print("AI 답변: ", end="", flush=True)

        # 1. 문서 검색 (Retrieval)
        docs = retriever.invoke(query)
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # 2. 프롬프트 완성
        final_prompt = prompt.format(context=context_text, question=query)
        
        # 3. 토큰화
        inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)

        # 4. 스트리머 준비
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # 5. 생성 시작
        generation_kwargs = dict(
            inputs, 
            streamer=streamer, 
            max_new_tokens=512, 
            temperature=0.1,
            repetition_penalty=1.1
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 6. 실시간 출력
        generated_text = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            generated_text += new_text
        print("\n")
        return generated_text

    # 7. 실행
    print("\n========== [2] RAG 파이프라인 가동 (Streaming) ==========")
    query = "삼성전자의 DX 부문 주요 제품은 무엇인가요?"
    stream_response(query)

if __name__ == "__main__":
    main()
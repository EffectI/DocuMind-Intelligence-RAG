import os
import torch
from threading import Thread
from collections import deque

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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
    print(f"[1] System Check: {device}")
    
    # 2. 임베딩 모델 로드
    print("Loading Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device}
    )

    # 3. 벡터 DB 로드
    if not os.path.exists(DB_PATH):
        print("Error: DB Not Found. Run simple_rag.py first.")
        return

    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="samsung_report_db"
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4. LLM 로드
    print(f"Loading LLM ({LLM_MODEL_ID})...")
    
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

    # =========================================================
    # 대화 기록 저장소
    # =========================================================
    chat_history = deque(maxlen=5) 

    # =========================================================
    # 스트리밍 함수
    # =========================================================
    def stream_response(query, history):
        print(f"\nQuestion: {query}")
        
        # 1. 문서 검색
        docs = retriever.invoke(query)

        # 검색된 문서 내용 로그 출력
        print("\n" + "="*60)
        print(f"🔍 [DEBUG] Retriever가 찾아온 문서 ({len(docs)}개)")
        print("="*60)
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown Source')
            page = doc.metadata.get('page', 'Unknown Page')
            content = doc.page_content.strip()
            
            print(f"[Chunk {i+1}] (Source: {source}, Page: {page})")
            print("-" * 30)
            print(content) # 문서 내용 전체 출력
            print("-" * 60)
        print("============================================================\n")

        context_text = "\n\n".join([d.page_content for d in docs])
        
        print("AI Answer: ", end="", flush=True)

        # [시스템 프롬프트 유지]
        system_prompt = (
            "당신은 삼성전자 사업보고서를 분석하는 AI 비서입니다. "
            "제공된 [참고 문서]를 바탕으로 질문에 대해 핵심만 간결하게 답변하세요. "
            "답변이 끝나면 불필요한 부연 설명이나 단위 변환 목록(예: 1GB=...)을 절대 작성하지 말고 즉시 대화를 종료하세요."
        )

        # 2. 메시지 구조 생성
        messages = [{"role": "system", "content": system_prompt}]

        for old_query, old_answer in history:
            messages.append({"role": "user", "content": old_query})
            messages.append({"role": "assistant", "content": old_answer})

        messages.append({
            "role": "user", 
            "content": f"[참고 문서]\n{context_text}\n\n질문: {query}"
        })
        
        # 3. 토큰화
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        # 4. 스트리머 준비
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # [종료 토큰 설정 유지]
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        ]

        # 5. 생성 시작
        generation_kwargs = dict(
            input_ids=input_ids, 
            streamer=streamer, 
            max_new_tokens=512, 
            temperature=0.09,        
            repetition_penalty=1.15, 
            do_sample=True,        
            eos_token_id=terminators 
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 6. 실시간 출력 및 텍스트 수집
        generated_text = ""
        
        stop_keywords = ["질문:", "User:", "Question:", "사용자:", "<|eot_id|>", "<|end_of_text|>"]

        for new_text in streamer:
            should_stop = False
            for keyword in stop_keywords:
                if keyword in new_text:
                    should_stop = True
                    break
            
            if should_stop:
                break

            print(new_text, end="", flush=True)
            generated_text += new_text
            
        print("\n")
        return generated_text

    # 7. 대화형 인터페이스 실행
    print("\n[2] RAG Chatbot Started (Type 'q' to exit)")
    
    while True:
        try:
            query = input("\nInput: ")
            if query.lower() in ['q', 'quit', 'exit']:
                break
            if not query.strip():
                continue
                
            response_text = stream_response(query, chat_history)
            chat_history.append((query, response_text))
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
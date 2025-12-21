import os
import torch
from threading import Thread
from collections import deque
from typing import Optional, Dict, List

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
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 상황에 맞춰 상위 경로로 조정 필요
DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")

LLM_MODEL_ID = "beomi/Llama-3-Open-Ko-8B"
EMBEDDING_MODEL = "BAAI/bge-m3"

class RAGEngine:
    def __init__(self):
        # 1. 하드웨어 설정 (AMD 이슈 대비 CPU/CUDA 자동)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Init] System Device: {self.device}")
        
        # 2. 자원 로드
        self._load_vector_db()
        self._load_llm()
        
        # 3. 대화 기록 (인스턴스 변수로 관리)
        self.chat_history = deque(maxlen=5)

    def _load_vector_db(self):
        print("[Init] Loading Vector DB...")
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"DB Not Found at {DB_PATH}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': self.device}, # 임베딩은 가벼워서 CPU도 OK
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=self.embeddings,
            collection_name="samsung_report_db"
        )

    def _load_llm(self):
        print(f"[Init] Loading LLM ({LLM_MODEL_ID})...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        ]

    def search(self, query: str, filters: Optional[Dict] = None, k: int = 3):
        """
        메타데이터 필터링을 적용하여 문서 검색
        filters 예시: {"company": "삼성전자", "year": "2024"}
        """
        search_kwargs = {"k": k}
        
        # ChromaDB 필터 문법 적용
        if filters:
            conditions = [{key: {"$eq": val}} for key, val in filters.items()]
            if len(conditions) > 1:
                search_kwargs["filter"] = {"$and": conditions}
            elif len(conditions) == 1:
                search_kwargs["filter"] = conditions[0]

        print(f"\n[Search] Query: '{query}' | Filter: {filters}")
        docs = self.vector_store.similarity_search(query, **search_kwargs)
        return docs

    def chat(self, query: str, filters: Optional[Dict] = None):
        """질문하고 답변을 스트리밍으로 출력"""
        
        # 1. 검색 (Retrieve)
        docs = self.search(query, filters)
        
        # 검색 결과 디버깅 및 문맥 생성
        context_parts = []
        sources = []
        
        print("="*40)
        for i, doc in enumerate(docs):
            meta = doc.metadata
            src = f"{meta.get('company', 'Unknown')} {meta.get('year', '')}"
            content = doc.page_content.strip()
            
            print(f"[Chunk {i+1}] ({src}) {content[:50]}...") # 로그
            
            context_parts.append(f"[{src}]\n{content}")
            sources.append(f"- {meta.get('source', 'Unknown')} ({src})")
        print("="*40)

        context_text = "\n\n".join(context_parts)

        # 2. 프롬프트 구성 (Augment)
        system_prompt = (
            "당신은 기업 보고서 분석 AI입니다. [참고 문서]를 기반으로 질문에 답변하세요. "
            "없는 내용을 지어내지 말고, 수치와 사실 위주로 설명하세요. "
            "답변 후 불필요한 반복이나 부연 설명을 하지 마십시오."
        )

        messages = [{"role": "system", "content": system_prompt}]
        
        # 과거 대화 포함
        for old_q, old_a in self.chat_history:
            messages.append({"role": "user", "content": old_q})
            messages.append({"role": "assistant", "content": old_a})

        messages.append({
            "role": "user", 
            "content": f"[참고 문서]\n{context_text}\n\n질문: {query}"
        })

        # 3. 생성 (Generate)
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.1,
            repetition_penalty=1.15,
            do_sample=True,
            eos_token_id=self.terminators
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # 4. 스트리밍 출력 및 후처리
        print(f"\nAI: ", end="", flush=True)
        full_response = ""
        stop_keywords = ["질문:", "User:", "Question:"]

        for new_text in streamer:
            # 조기 종료 체크
            if any(k in new_text for k in stop_keywords):
                break
            
            print(new_text, end="", flush=True)
            full_response += new_text

        # 출처 정보 덧붙이기
        source_footer = "\n\n[참고 자료]\n" + "\n".join(set(sources))
        print(source_footer)
        
        # 대화 기록 저장 (출처 제외)
        self.chat_history.append((query, full_response))
        
        return full_response

# ==========================================
# [실행부] 테스트용
# ==========================================
if __name__ == "__main__":
    # 1. 엔진 초기화 (여기서 시간 소요)
    rag = RAGEngine()
    
    print("\nRAG 엔진 준비 완료. (종료: /exit)")
    
    # 2. 예시: 필터 설정 (필요시 변경)
    # current_filter = {"company": "삼성전자", "year": "2024"} 
    current_filter = None 

    while True:
        try:
            q = input("\n질문: ")
            if q in ["/exit", "q"]: break
            
            # 엔진 호출
            rag.chat(q, filters=current_filter)
            
        except KeyboardInterrupt:
            break
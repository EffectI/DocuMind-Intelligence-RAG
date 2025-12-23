import os
import sys
import torch
from threading import Thread
from collections import deque
from typing import Optional, Dict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder 

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TextIteratorStreamer
)

from config import (
    DB_PATH, 
    LLM_MODEL_ID, 
    EMBEDDING_MODEL_ID, 
    RERANKER_MODEL_ID, # [New] config에서 가져오기
    DEVICE, 
    MAX_NEW_TOKENS, 
    TEMPERATURE
)

class RAGEngine:
    def __init__(self):
        # Config의 DEVICE 사용
        self.device = DEVICE
        print(f"[Init] Device: {self.device}")
        
        self._load_vector_db()
        self._load_reranker() 
        self._load_llm()
        self.chat_history = deque(maxlen=3)

    def _load_vector_db(self):
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"DB Not Found at {DB_PATH}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_ID,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=self.embeddings,
            collection_name="samsung_report_db"
        )

    def _load_reranker(self):
        """[New] Cross-Encoder Reranker 로드"""
        print(f"[Init] Reranker 로딩 ({RERANKER_MODEL_ID})...")
        # automodel_args를 사용하여 torch_dtype 설정
        self.reranker = CrossEncoder(
            RERANKER_MODEL_ID, 
            device=self.device,
            automodel_args={"torch_dtype": "auto"}
        )

    def _load_llm(self):
        print(f"[Init] LLM 로딩 ({LLM_MODEL_ID})...")
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
        [Upgrade] 2단계 검색 시스템
        1. Vector Search로 후보군(3배수) 추출
        2. Cross-Encoder로 정밀 채점(Reranking) 후 Top-k 반환
        """
        # 1. 초기 후보군 검색 (최종 k의 3배수 정도 가져옴)
        initial_k = k * 3 
        
        search_kwargs = {"k": initial_k}
        if filters:
            conditions = [{key: {"$eq": val}} for key, val in filters.items()]
            if len(conditions) > 1:
                search_kwargs["filter"] = {"$and": conditions}
            elif len(conditions) == 1:
                search_kwargs["filter"] = conditions[0]

        # ChromaDB에서 1차 검색
        docs = self.vector_store.similarity_search(query, **search_kwargs)
        
        if not docs:
            return []

        # 2. [Reranking] 정밀 채점
        # (질문, 문서내용) 쌍을 생성
        pairs = [[query, doc.page_content] for doc in docs]
        
        # CrossEncoder가 문맥 연관성 점수 계산 (높을수록 좋음)
        scores = self.reranker.predict(pairs)

        # 3. 점수와 문서 결합 및 정렬
        scored_docs = []
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = float(score) # 메타데이터에 점수 기록 (디버깅용)
            scored_docs.append(doc)

        # 점수 내림차순 정렬
        scored_docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

        # 상위 k개만 선택
        final_docs = scored_docs[:k]
        
        # (옵션) 로그 출력
        if final_docs:
            print(f"[Search] Top score: {final_docs[0].metadata['rerank_score']:.4f}")

        return final_docs

    def chat(self, query: str, filters: Optional[Dict] = None):
        """Generator 방식으로 UI에 스트리밍"""
        
        # 1. Retrieve (Reranker가 적용된 search 호출)
        docs = self.search(query, filters, k=3)
        
        context_parts = []
        sources = []
        for doc in docs:
            meta = doc.metadata
            src = f"{meta.get('company', 'Unknown')} {meta.get('year', '')}"
            page = meta.get('page', '?') # 페이지 정보가 있다면 표시
            
            # 컨텍스트 조립
            context_parts.append(f"[{src} p.{page}]\n{doc.page_content.strip()}")
            
            # 출처 목록 조립
            filename = os.path.basename(meta.get('source', 'Unknown'))
            sources.append(f"- 📄 **{filename}** (p.{page})")

        context_text = "\n\n".join(context_parts)

        # 2. Prompt Setup
        system_prompt = (
            "당신은 기업 보고서 분석 AI입니다. [참고 문서]를 기반으로 질문에 답변하세요. "
            "없는 내용을 지어내지 말고, 수치와 사실 위주로 설명하세요."
        )

        messages = [{"role": "system", "content": system_prompt}]
        for old_q, old_a in self.chat_history:
            messages.append({"role": "user", "content": old_q})
            messages.append({"role": "assistant", "content": old_a})
        messages.append({"role": "user", "content": f"[참고 문서]\n{context_text}\n\n질문: {query}"})

        # 3. Generation Setup
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            repetition_penalty=1.15,
            do_sample=True,
            eos_token_id=self.terminators
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # 4. Yield Stream
        full_response = ""
        for new_text in streamer:
            if any(k in new_text for k in ["질문:", "User:"]): break
            full_response += new_text
            yield new_text

        # 5. Sources
        if sources:
            source_footer = "\n\n**[참고 문서]**\n" + "\n".join(sorted(list(set(sources)))) # 중복 제거 및 정렬
            yield source_footer
            full_response += source_footer
        
        self.chat_history.append((query, full_response))
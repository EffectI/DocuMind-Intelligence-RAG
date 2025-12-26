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
    RERANKER_MODEL_ID, 
    DEVICE, 
    MAX_NEW_TOKENS, 
    TEMPERATURE
)

class RAGEngine:
    def __init__(self):
        self.device = DEVICE
        print(f"[Init] Device: {self.device}")
        
        # [안전 장치] 프로세스가 GPU 메모리의 90%까지만 쓰도록 제한 (OS 멈춤 방지)
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(0.9)
            except Exception as e:
                print(f"[Warning] Failed to set memory fraction: {e}")
        
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
        """Cross-Encoder Reranker 로드"""
        print(f"[Init] Reranker 로딩 ({RERANKER_MODEL_ID})...")
        # [수정] DeprecationWarning 해결: automodel_args -> model_kwargs, torch_dtype -> dtype
        self.reranker = CrossEncoder(
            RERANKER_MODEL_ID, 
            device=self.device,
            model_kwargs={"dtype": "auto"}
        )

    def _load_llm(self):
        print(f"[Init] LLM 로딩 ({LLM_MODEL_ID})...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        
        # [수정] Pad Token Warning 해결
        # Llama-3는 pad_token이 없으므로 eos_token으로 설정
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right" # 생성 시에는 right padding 권장

        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
        # 모델 설정에도 pad_token_id 반영
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        ]

    def search(self, query: str, filters: Optional[Dict] = None, k: int = 3):
        """
        2단계 검색 시스템
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
        
        # CrossEncoder가 문맥 연관성 점수 계산
        scores = self.reranker.predict(pairs)

        # 3. 점수와 문서 결합 및 정렬
        scored_docs = []
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = float(score)
            scored_docs.append(doc)

        # 점수 내림차순 정렬
        scored_docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

        # 상위 k개만 선택
        final_docs = scored_docs[:k]
        
        return final_docs

    def chat(self, query: str, filters: Optional[Dict] = None):
        """Generator 방식으로 UI에 스트리밍 (Thread 사용)"""
        
        # 1. Retrieve
        docs = self.search(query, filters, k=3)
        
        context_parts = []
        sources = []
        for doc in docs:
            meta = doc.metadata
            src = f"{meta.get('company', 'Unknown')} {meta.get('year', '')}"
            page = meta.get('page', '?')
            
            context_parts.append(f"[{src} p.{page}]\n{doc.page_content.strip()}")
            
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
        # [수정] attention_mask 생성 및 반환
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt",
            return_dict=True  # 딕셔너리 형태로 반환 (input_ids, attention_mask 포함)
        ).to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = dict(
            **inputs, # input_ids와 attention_mask가 같이 전달됨
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            repetition_penalty=1.15,
            do_sample=True,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id # 명시적 설정
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
            source_footer = "\n\n**[참고 문서]**\n" + "\n".join(sorted(list(set(sources))))
            yield source_footer
            full_response += source_footer
        
        self.chat_history.append((query, full_response))

    def generate_answer(self, query: str, filters: Optional[Dict] = None) -> str:
        """
        [Evaluation 전용] 스트리밍 없이 한 번에 답변 생성
        """
        # 1. Retrieve
        docs = self.search(query, filters, k=3)
        
        context_parts = []
        for doc in docs:
            meta = doc.metadata
            src = f"{meta.get('company', 'Unknown')} {meta.get('year', '')}"
            page = meta.get('page', '?')
            context_parts.append(f"[{src} p.{page}]\n{doc.page_content.strip()}")

        context_text = "\n\n".join(context_parts)

        # 2. Prompt Setup
        system_prompt = (
            "당신은 기업 보고서 분석 AI입니다. [참고 문서]를 기반으로 질문에 답변하세요. "
            "없는 내용을 지어내지 말고, 수치와 사실 위주로 설명하세요."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[참고 문서]\n{context_text}\n\n질문: {query}"}
        ]

        # [수정] attention_mask 자동 생성을 위해 return_dict=True 사용
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt",
            return_dict=True
        ).to(self.device)

        # 3. Generate (No Thread, No Streamer, No Grad)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, # input_ids, attention_mask 전달
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                repetition_penalty=1.15,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 4. Decode
        # 입력 길이만큼 자르고 생성된 부분만 디코딩
        generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response
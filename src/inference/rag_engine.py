import os
import sys
import torch
from threading import Thread
from collections import deque
from typing import Optional, Dict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
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
        self._load_llm()
        self.chat_history = deque(maxlen=5)

    def _load_vector_db(self):
        # Config의 DB_PATH 사용
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"DB Not Found at {DB_PATH}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_ID, # Config 사용
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=self.embeddings,
            collection_name="samsung_report_db"
        )

    def _load_llm(self):
        print(f"[Init] LLM 로딩 ({LLM_MODEL_ID})...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        # Config의 LLM_MODEL_ID 사용
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
        search_kwargs = {"k": k}
        if filters:
            conditions = [{key: {"$eq": val}} for key, val in filters.items()]
            if len(conditions) > 1:
                search_kwargs["filter"] = {"$and": conditions}
            elif len(conditions) == 1:
                search_kwargs["filter"] = conditions[0]

        return self.vector_store.similarity_search(query, **search_kwargs)

    def chat(self, query: str, filters: Optional[Dict] = None):
        """Generator 방식으로 UI에 스트리밍"""
        
        # 1. Retrieve
        docs = self.search(query, filters)
        
        context_parts = []
        sources = []
        for doc in docs:
            meta = doc.metadata
            src = f"{meta.get('company', 'Unknown')} {meta.get('year', '')}"
            context_parts.append(f"[{src}]\n{doc.page_content.strip()}")
            sources.append(f"- {meta.get('source', 'Unknown')} ({src})")

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
        
        # Config의 파라미터 사용
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
        source_footer = "\n\n[참고 자료]\n" + "\n".join(set(sources))
        yield source_footer
        full_response += source_footer
        
        self.chat_history.append((query, full_response))
import os
import sys
import torch
from threading import Thread
from collections import deque
from typing import Optional, Dict, List, Tuple

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
    TEMPERATURE,
    RAG_SYSTEM_PROMPT
)

class RAGEngine:
    def __init__(self):
        self.device = DEVICE
        print(f"[Init] Device: {self.device}")
        
        self._setup_memory_limit()
        self._load_vector_db()
        self._load_reranker() 
        self._load_llm()
        self.chat_history = deque(maxlen=3)

    def _setup_memory_limit(self):
        """[Optimization] Limit GPU memory usage to prevent system freeze"""
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(0.9)
            except Exception as e:
                print(f"[Warning] Failed to set memory fraction: {e}")

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
        print(f"[Init] Reranker Loading ({RERANKER_MODEL_ID})...")
        # [Fix] Resolved DeprecationWarning: automodel_args -> model_kwargs
        self.reranker = CrossEncoder(
            RERANKER_MODEL_ID, 
            device=self.device,
            model_kwargs={"dtype": "auto"}
        )

    def _load_llm(self):
        print(f"[Init] LLM Loading ({LLM_MODEL_ID})...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        
        # [Fix] Set pad_token_id to eos_token_id for Llama-3
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right" 

        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        ]

    def search(self, query: str, filters: Optional[Dict] = None, k: int = 3):
        """Two-Stage Retrieval: Vector Search -> Cross-Encoder Reranking"""
        initial_k = k * 3 
        
        search_kwargs = {"k": initial_k}
        if filters:
            conditions = [{key: {"$eq": val}} for key, val in filters.items()]
            if len(conditions) > 1:
                search_kwargs["filter"] = {"$and": conditions}
            elif len(conditions) == 1:
                search_kwargs["filter"] = conditions[0]

        docs = self.vector_store.similarity_search(query, **search_kwargs)
        
        if not docs:
            return []

        # Reranking
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)

        scored_docs = []
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = float(score)
            scored_docs.append(doc)

        scored_docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
        return scored_docs[:k]

    # [Refactor] Helper: Format context and extract sources
    def _format_context(self, docs) -> Tuple[str, List[str]]:
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
        return context_text, sources

    # [Refactor] Helper: Build chat messages using centralized config prompt
    def _build_messages(self, query: str, context_text: str, use_history: bool = False):
        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        
        if use_history:
            for old_q, old_a in self.chat_history:
                messages.append({"role": "user", "content": old_q})
                messages.append({"role": "assistant", "content": old_a})
                
        messages.append({"role": "user", "content": f"[참고 문서]\n{context_text}\n\n질문: {query}"})
        return messages

    def chat(self, query: str, filters: Optional[Dict] = None):
        """Generator method for UI streaming (Uses Thread)"""
        # 1. Retrieve
        docs = self.search(query, filters, k=3)
        
        # Use helpers
        context_text, sources = self._format_context(docs)
        messages = self._build_messages(query, context_text, use_history=True)

        # 2. Input Setup
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt",
            return_dict=True
        ).to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            repetition_penalty=1.15,
            do_sample=True,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # 3. Yield Stream
        full_response = ""
        for new_text in streamer:
            if any(k in new_text for k in ["질문:", "User:"]): break
            full_response += new_text
            yield new_text

        # 4. Sources
        if sources:
            source_footer = "\n\n**[참고 문서]**\n" + "\n".join(sorted(list(set(sources))))
            yield source_footer
            full_response += source_footer
        
        self.chat_history.append((query, full_response))

    def generate_answer(self, query: str, filters: Optional[Dict] = None) -> str:
        """[For Evaluation] Synchronous generation without streaming/threading"""
        # 1. Retrieve
        docs = self.search(query, filters, k=3)
        
        # Use helpers (No history for evaluation)
        context_text, _ = self._format_context(docs) 
        messages = self._build_messages(query, context_text, use_history=False)

        # 2. Input Setup
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt",
            return_dict=True
        ).to(self.device)

        # 3. Generate (No Thread, No Grad for optimization)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                repetition_penalty=1.15,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 4. Decode
        generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response
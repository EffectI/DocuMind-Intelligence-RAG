# DocuMind-Intelligence-RAG

**RAG System: 다중 문서 기반 지능형 보고서 질의응답 시스템**

본 프로젝트는 기업용 보고서(DART 사업보고서 등) 데이터셋에서 사용자의 질문에 관련된 정보를 정확하게 추출하고, 이를 바탕으로 답변을 생성하는 **On-Premise RAG(Retrieval-Augmented Generation)** 시스템입니다.

---

## Architecture & Features

### Retrieval & Ranking (고도화됨)
- **Dense Vector Search:** 의미 기반 검색(Semantic Search)을 위해 `BAAI/bge-m3` 임베딩 모델을 사용합니다. 단순 키워드 매칭을 넘어 문맥적 유사도가 높은 문서를 추출합니다.
- **Advanced Reranking:** 1차 검색된 문서들에 대해 **Cross-Encoder(`BAAI/bge-reranker-v2-m3`)**를 적용하여 정밀 재순위화(Re-ranking)를 수행, 검색 정확도를 대폭 향상시켰습니다.
- **ChromaDB Integration:** 로컬 벡터 저장소인 ChromaDB를 활용하여 빠르고 영구적인 데이터 인덱싱을 지원합니다.

### Inference Engine
- **Local LLM Pipeline:** HuggingFace `transformers` 파이프라인 기반의 `Llama-3-Open-Ko-8B` 모델을 사용합니다.
- **Efficient Quantization:** `BitsAndBytes`를 활용한 **4-bit 양자화(Quantization)**를 적용하여, 소비자용 GPU 환경에서도 고성능 모델 구동이 가능합니다.
- **Streaming Response:** `TextIteratorStreamer`를 적용하여 대기 시간 없이 실시간으로 답변이 생성되는 과정을 시각화합니다.

### Data Ingestion & Parsing
- **Custom PDF Parser:** `pdfplumber`를 기반으로 자체 개발한 파서를 적용하여, 단순 텍스트뿐만 아니라 **표(Table) 데이터를 Markdown 형식으로 변환**해 LLM의 해독력을 높였습니다.
- **Metadata-Aware Chunking:** 페이지 번호, 섹션 헤더 등 메타데이터를 보존하며 문서를 청킹(Chunking)하여, 답변 시 정확한 출처(Source)를 표기합니다.

---

## Technical Pipeline

**DocuMind AI**는 다음과 같은 기술적 흐름으로 동작합니다.

| Stage | Technology | Description |
| :--- | :--- | :--- |
| **1. Ingestion** | `pdfplumber`, Custom Logic | PDF/XML 문서의 텍스트 및 표 구조 추출, Markdown 변환 |
| **2. Embedding** | `BAAI/bge-m3` | 문서를 Dense Vector로 변환하여 의미론적 인덱싱 수행 |
| **3. Storage** | `ChromaDB` | 벡터 데이터 및 메타데이터(출처, 페이지) 영구 저장 |
| **4. Retrieval** | `Vector Similarity Search` | 사용자 질문과 가장 유사한 문맥을 1차 검색 (Top-k * 3) |
| **5. Reranking** | `Cross-Encoder` | 1차 검색된 문서와 질문의 연관성을 정밀 채점하여 최종 Top-k 선별 |
| **6. Inference** | `Llama-3`, `BnB` (4-bit) | 선별된 문맥을 바탕으로 Local LLM이 답변 생성 |
| **7. Evaluation** | `Ragas`, `Gemini Pro` | 생성된 답변의 사실성(Faithfulness) 및 정확도 정량 평가 |



## 🗺️ Roadmap & Status

### Phase 1: MVP (Completed) ✅
- [x] **Data Pipeline:** DART API 연동 및 PDF 파싱 구현
- [x] **Basic RAG:** ChromaDB + Dense Retrieval 구현
- [x] **UI:** Streamlit 기반 대화형 인터페이스 구축

### Phase 2: Accuracy & Retrieval (Current Focus) 🚧
- [x] **Reranking:** Cross-Encoder 기반 검색 결과 재순위화(Re-ranking) 적용 완료
- [ ] **Advanced Search:** Hybrid Search (BM25 + Vector) 도입 예정
- [x] **Golden Dataset:** 자동 생성 스크립트(`generate_dataset.py`) 구축 완료
- [x] **Evaluation Pipeline:** Ragas + Gemini 기반 정량 평가 시스템 구축 완료 (정확도 최적화 진행 중)

### Phase 3: Performance & Serving (Planned) 📅
- [ ] **Inference Engine:** vLLM 도입을 통한 추론 속도 가속
- [ ] **Deployment:** Docker Container 패키징

---

## 🛠️ Recent Fixes & Optimizations

### 1. Evaluation Stability & Memory Leak Fix (2025.12.25)
**문제 상황 (Issue):**
Ragas 평가 시 대량의 데이터(Batch)를 처리할 때, 기존 UI용 스트리밍 함수(`chat`)를 재사용함에 따라 **쓰레드(Thread) 적체** 및 **GPU VRAM 누수**가 발생하여 시스템이 멈추는 현상(Freeze) 확인.

**해결 방안 (Solution):**
- **Evaluation 전용 메서드 구현:** `RAGEngine` 내에 쓰레드와 스트리머를 사용하지 않는 `generate_answer()` 메서드를 추가하여 오버헤드 제거.
- **Memory Management:** 매 평가 턴마다 `gc.collect()` 및 `torch.cuda.empty_cache()`를 호출하여 VRAM 파편화 방지.
- **HuggingFace Warning Fix:** `pad_token_id` 설정 및 `attention_mask` 명시적 전달로 생성 품질 안정화.

**결과 (Result):**
- 시스템 멈춤 없이 전체 데이터셋에 대한 연속 평가 가능해짐.

### 2. Ragas Cost & Rate Limit Optimization (2025.12.26)
**문제 상황 (Issue):**
평가(Judge) 모델로 Gemini Pro 사용 시, 무료 티어의 **RPM(분당 요청 수) 제한**에 걸려 평가가 중단되거나 무한 대기하는 현상 발생.

**해결 방안 (Solution):**
- **Model Switching:** 평가 모델을 속도가 빠르고 제한이 넉넉한 **`gemini-1.5-flash`**로 변경.
- **Sequential Execution:** `RunConfig(max_workers=1)` 설정을 통해 병렬 처리를 비활성화하고 순차 처리로 변경하여 API 제한 회피.
- **Metric Initialization:** Ragas v0.2+ 호환성을 위해 메트릭 객체 초기화 방식 수정 (Deprecation Warning 해결).
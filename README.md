# DocuMind-Intelligence-RAG

**RAG System: 다중 문서 기반 지능형 보고서 질의응답 시스템**

본 프로젝트는 기업용 보고서(DART 사업보고서 등) 데이터셋에서 사용자의 질문에 관련된 정보를 정확하게 추출하고, 이를 바탕으로 답변을 생성하는 **On-Premise RAG(Retrieval-Augmented Generation)** 시스템입니다.

---

## ⚙️ Architecture & Features

### 🔍 Retrieval & Ranking (Advanced)
- **Dense Vector Search:** 의미 기반 검색(Semantic Search)을 위해 `BAAI/bge-m3` 임베딩 모델을 사용합니다. 단순 키워드 매칭을 넘어 문맥적 유사도가 높은 문서를 추출합니다.
- **Precision Reranking:** 1차 검색된 문서들에 대해 **Cross-Encoder(`BAAI/bge-reranker-v2-m3`)**를 적용하여 정밀 재순위화(Re-ranking)를 수행, 검색 정확도를 대폭 향상시켰습니다.
- **ChromaDB Integration:** 로컬 벡터 저장소인 ChromaDB를 활용하여 빠르고 영구적인 데이터 인덱싱을 지원합니다.

### 🧠 Inference Engine
- **Local LLM Pipeline:** HuggingFace `transformers` 파이프라인 기반의 `Llama-3-Open-Ko-8B` 모델을 사용합니다.
- **Efficient Quantization:** `BitsAndBytes`를 활용한 **4-bit 양자화(Quantization)**를 적용하여, 소비자용 GPU 환경에서도 고성능 모델 구동이 가능합니다.
- **Streaming Response:** `TextIteratorStreamer`를 적용하여 대기 시간 없이 실시간으로 답변이 생성되는 과정을 시각화합니다.

### 📄 Data Ingestion & Parsing
- **Custom PDF Parser:** `pdfplumber`를 기반으로 자체 개발한 파서를 적용하여, 단순 텍스트뿐만 아니라 **표(Table) 데이터를 Markdown 형식으로 변환**해 LLM의 해독력을 높였습니다.
- **Metadata-Aware Chunking:** 페이지 번호, 섹션 헤더 등 메타데이터를 보존하며 문서를 청킹(Chunking)하여, 답변 시 정확한 출처(Source)를 표기합니다.

---

## 🚀 Technical Pipeline

**DocuMind AI**는 다음과 같은 기술적 흐름으로 동작합니다.

| Stage | Technology | Description |
| :--- | :--- | :--- |
| **1. Ingestion** | `pdfplumber`, Custom Logic | PDF/XML 문서의 텍스트 및 표 구조 추출, Markdown 변환 |
| **2. Embedding** | `BAAI/bge-m3` | 문서를 Dense Vector로 변환하여 의미론적 인덱싱 수행 |
| **3. Storage** | `ChromaDB` | 벡터 데이터 및 메타데이터(출처, 페이지) 영구 저장 |
| **4. Retrieval** | Vector Similarity Search | 사용자 질문과 가장 유사한 문맥을 1차 검색 (Top-k * 3) |
| **5. Reranking** | `Cross-Encoder` | 1차 검색된 문서와 질문의 연관성을 정밀 채점하여 최종 Top-k 선별 |
| **6. Inference** | `Llama-3`, `BnB` (4-bit) | 선별된 문맥을 바탕으로 Local LLM이 답변 생성 |
| **7. Evaluation** | `Ragas`, `Gemini` | 생성된 답변의 사실성(Faithfulness) 및 정확도 정량 평가 |

---

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
평가 모델(Judge LLM)로 Gemini Pro/Flash 사용 시, 무료 티어의 **RPM(분당 요청 수) 제한** 이슈로 인해 평가가 중단됨.

**해결 방안 (Solution):**
- **Model Switching:** **`gemini-2.5-flash-lite`**로 변경
- **Batch Processing:** 데이터를 10개 단위로 나누어 처리하고, 배치 사이 휴식(Sleep) 시간을 두어 Rate Limit을 회피.
- **Sequential Execution:** `RunConfig(max_workers=1)` 설정을 통해 API 충돌 방지.
- **Metric Initialization:** Ragas v0.2+ 호환성을 위해 메트릭 객체 초기화 방식 수정 (Deprecation Warning 해결).

---

## ⚠️ Current Issues & Future Challenges

현재 시스템이 해결해야 할 주요 과제(Pain Points)와 향후 개선 방향입니다.

### 1. External API Dependency (Evaluation)
- **문제점:** 평가(Judge) 모델로 외부 API(Google Gemini Free Tier)를 사용함에 따라, **모델 수명 주기(Deprecation)** 및 **엄격한 속도 제한(Rate Limit)**에 종속적임. 평가 속도가 느리며(순차 처리), 서비스 정책 변경 시 코드를 수정해야 함.
- **해결 과제:** - 평가 전용 로컬 소형 LLM(Prometheus 등) 도입 검토.
    - 또는 안정적인 유료 API 티어 전환 고려.

### 2. Evaluation Speed
- **문제점:** API Rate Limit 회피를 위해 **단일 쓰레드(Sequential) + 대기 시간(Sleep)** 방식을 적용하여, 전체 데이터셋 평가에 소요되는 시간이 김.
- **해결 과제:** - 병렬 처리(Parallel Processing)가 가능한 환경 구축.
    - 평가 데이터셋을 핵심 케이스 위주로 경량화하거나 샘플링 전략 도입.

### 3. Inference Latency (Local LLM)
- **문제점:** 현재 `HuggingFace Pipeline`을 사용하고 있어, 긴 문맥 처리가 필요한 RAG 시스템 특성상 토큰 생성 속도(TPS)가 최적화되지 않음.
- **해결 과제:** - **Phase 3 목표:** **vLLM** 엔진 도입 및 PagedAttention 기술 적용을 통해 추론 속도 2배 이상 가속화 필요.
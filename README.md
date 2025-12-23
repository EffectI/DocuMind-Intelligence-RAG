# DocuMind-Intelligence-RAG

**RAG System: 다중 문서 기반 지능형 보고서 질의응답 시스템**


본 프로젝트는 기업용 보고서 데이터셋에서 사용자의 질문에 관련된 정보를 정확하게 추출하고, 
이를 바탕으로 답변을 생성하는 RAG(Retrieval-Augmented Generation) 시스템 구축을 목표로 함.


## Architecture & Features

본 프로젝트는 기업 보고서(PDF, XML) 분석에 최적화된 **On-Premise RAG 파이프라인**으로 구축되었습니다.

### Retrieval & Ranking
- **Dense Vector Search:** 의미 기반 검색(Semantic Search)을 위해 고차원 벡터 임베딩을 사용합니다. 단순 키워드 매칭을 넘어 문맥적 유사도가 높은 문서를 추출합니다.
- **Similarity-Based Ranking:** Cosine Similarity(코사인 유사도) 스코어를 기준으로 가장 연관성 높은 Top-k 문서를 선별하여 LLM에 전달합니다.
- **ChromaDB Integration:** 로컬 벡터 저장소인 ChromaDB를 활용하여 빠르고 영구적인 데이터 인덱싱을 지원합니다.

### Inference Engine
- **Local LLM Pipeline:** HuggingFace `transformers` 파이프라인을 기반으로 구축되었습니다.
- **Efficient Quantization:** `BitsAndBytes`를 활용한 4-bit 양자화(Quantization)를 적용하여, 제한된 GPU 메모리 환경에서도 Llama 3 등 고성능 모델 구동이 가능합니다.
- **Streaming Response:** `TextIteratorStreamer`를 적용하여 대기 시간 없이 실시간으로 답변이 생성되는 과정을 시각화합니다.

### Data Ingestion & Parsing
- **Custom PDF Parser:** `pdfplumber`를 기반으로 자체 개발한 파서가 적용되었습니다. 단순 텍스트뿐만 아니라 보고서 내 **표(Table)** 데이터를 Markdown 형식으로 변환하여 LLM의 해독력을 높였습니다.
- **Metadata-Aware Chunking:** 페이지 번호, 섹션 헤더 등 메타데이터를 보존하며 문서를 청킹(Chunking)하여, 답변 시 정확한 출처(Source)를 표기합니다.
- **Direct Query Processing:** 사용자의 자연어 질의(Raw Query)를 전처리 없이 즉각적으로 검색 엔진에 투입하여 Latency를 최소화합니다.


## Technical Pipeline

**DocuMind AI**는 다음과 같은 순서로 동작함.

| Stage | Technology | Description |
| :--- | :--- | :--- |
| **1. Ingestion** | `pdfplumber`, Custom Logic | PDF/XML 문서의 텍스트 및 표 구조 추출, Markdown 변환 |
| **2. Embedding** | HuggingFace Embeddings | 문서를 Dense Vector로 변환하여 의미론적 인덱싱 수행 |
| **3. Storage** | `ChromaDB` | 벡터 데이터 및 메타데이터(출처, 페이지) 영구 저장 |
| **4. Retrieval** | Vector Similarity Search | 사용자 질문과 가장 유사한 문맥을 Cosine Similarity로 검색 |
| **5. Inference** | `Transformers`, `BnB` (4-bit) | 검색된 문맥을 바탕으로 Local LLM(Llama 3)이 답변 생성 |
| **6. UI** | `Streamlit` | 대화형 인터페이스 및 실시간 스트리밍 답변 제공 |





## Roadmap

### Phase 1: MVP (Completed)
- [x] **Data Pipeline:** DART API 연동 및 PDF 파싱 구현
- [x] **Basic RAG:** ChromaDB + Dense Retrieval 구현
- [x] **UI:** Streamlit 기반 대화형 인터페이스 구축

### Phase 2: Accuracy & Retrieval (Current Focus)
- [ ] **Advanced Search:** Hybrid Search (BM25 + Vector) 도입
- [x] **Reranking:** Cross-Encoder 기반 검색 결과 재순위화(Re-ranking) 적용 
- [ ] **Evaluation:** Ragas 프레임워크를 활용한 검색 정확도 정량 평가

### Phase 3: Performance & Serving (Planned)
- [ ] **Inference Engine:** vLLM 도입을 통한 추론 속도 가속
- [ ] **Deployment:** Docker Container 패키징







## SWOT 분석
| 강점 (Strength) | 약점 (Weakness) |
| :--- | :--- |
| 특정 도메인 특화 프롬프트 및 템플릿 제공 가능 | 빅테크 대비 인프라 비용 및 데이터 확보의 한계 |
| **기회 (Opportunity)** | **위협 (Threat)** |
| 클라우드 보안 규제로 인한 독립적 RAG 수요 증가 | GPT-5 등 상위 모델의 성능 개선으로 인한 기술적 해자 축소 |




1. 임베딩 모델 (Embedding Model: 데이터 벡터화)
성능 최우선 (API형): OpenAI text-embedding-3-large
한국어 특화 (API형): Upstage solar-embedding-1-v2
오픈소스 (로컬 구축형): BAAI/bge-m3 또는 intfloat/multilingual-e5-large 


2. 리랭커 모델 (Reranker Model: 검색 결과 재정렬)
범용 리랭커: BAAI/bge-reranker-v2-m3
한국어 성능 강화: Upstage solar-reranker
성능 중심: Cohere Rerank v3


3. 거대 언어 모델 (LLM: 답변 생성)
종합 성능 1위: GPT-4o
복잡한 추론 및 긴 문맥: Claude 3.5 Sonnet
한국 특화 및 보안: Naver HyperCLOVA X
전문 도메인 특화: LG AI Research EXAONE 3.5
오픈소스: Llama 3.1/3.3 (Fine-tuned for Korean)
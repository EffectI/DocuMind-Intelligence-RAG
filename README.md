# DocuMind-Intelligence-RAG

**RAG System: 다중 문서 기반 지능형 보고서 질의응답 시스템**


본 프로젝트는 기업용 보고서 데이터셋에서 사용자의 질문에 관련된 정보를 정확하게 추출하고, 
이를 바탕으로 답변을 생성하는 RAG(Retrieval-Augmented Generation) 시스템 구축을 목표로 함.

## SWOT 분석
| 강점 (Strength) | 약점 (Weakness) |
| :--- | :--- |
| 특정 도메인 특화 프롬프트 및 템플릿 제공 가능 | 빅테크 대비 인프라 비용 및 데이터 확보의 한계 |
| **기회 (Opportunity)** | **위협 (Threat)** |
| 클라우드 보안 규제로 인한 독립적 RAG 수요 증가 | GPT-5 등 상위 모델의 성능 개선으로 인한 기술적 해자 축소 |


## 핵심 특징 (Features)
- **경량화**: 온프레미스(On-premise) 환경에서도 효율적으로 동작하는 SLM 활용
- **한국어 특화**: 국내 비즈니스 환경 및 전문 용어에 최적화된 자연어 처리
- **범용성**: 다양한 산업군(금융, 법률, 제조 등)의 보고서 양식 대응
- **보안성**: 내부 데이터 유출 방지를 위한 독립적 시스템 구성 가능


## 데이터 수집 전략
다음의 소스를 우선적으로 수집함.

1. 공공 및 정책 연구 보고서 (Public & Policy Reports)
국가정책연구포털 (NKIS): 국책연구단지 내 26개 경제·인문사회 연구기관의 연구 성과물을 통합 제공합니다. 가장 공신력 있는 정책 보고서 수집처임
알리오 (ALIO) 연구보고서 공시: 370여 개 공공기관에서 작성한 연구보고서 현황을 확인할 수 있음
NTIS (국가과학기술지식정보서비스): 국가 R&D 사업을 통해 산출된 연구보고서와 기술 동향 자료를 수집할 수 있음
공공데이터포털: 다양한 공공기관의 연구보고서 데이터를 API 형태로 제공

2. 기업 및 산업 분석 보고서 (Corporate & Industry Reports)
Open DART (전자공시시스템): 상장사들의 사업보고서, 분기보고서 원문을 XML/JSON 형태로 가져올 수 있는 API를 제공합니다. 재무 데이터와 텍스트 데이터가 결합된 형태.
한경컨센서스: 국내 모든 증권사 애널리스트들이 작성한 기업/산업/시장 분석 리포트를 한데 모아 제공, PDF 파일 수집에 용이함
한국IR협의회: 상장 기업들의 IR 자료와 기술 분석 보고서를 무료로 배포
네이버 금융 (리서치): 증권사 리포트를 카테고리별로 모아서 볼 수 있는 대중적인 사이트

3. 경제 및 전문 분야 보고서 (Economic & Specialized Reports)
주요 경제연구소: 삼성경제연구소(SERI), LG경영연구원, 현대경제연구원 등에서 발간하는 트렌드 리포트 존재
한국콘텐츠진흥원 (콘텐츠산업정보포털): 매년 발간되는 산업 백서와 트렌드 보고서의 품질이 높음
ScienceOn (KISTI): 국내외 과학기술 학술정보, 논문, 보고서를 통합 검색하고 수집할 수 있는 플랫폼


### 1순위 (FIRST PRIORITY)
- **한경컨센서스**: 국내 증권사 산업/기업 분석 리포트 (PDF 형태)
- **Open DART**: 상장사 사업보고서 및 재무 데이터 (XML/JSON/Text)









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
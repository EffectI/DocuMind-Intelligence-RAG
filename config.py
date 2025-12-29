import os
import torch
from dotenv import load_dotenv

# 1. 환경 변수 로드 (.env 파일에서 API KEY 등을 가져옴)
load_dotenv()

# ==========================================
# [경로 설정] Path Configuration
# ==========================================
# 프로젝트 루트: 이 파일(config.py)이 있는 위치를 기준으로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 데이터 관련 경로
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "dart")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed", "sections")
DB_PATH = os.path.join(DATA_DIR, "vector_db")

# [New] 평가 관련 경로 추가
EVAL_DIR = os.path.join(DATA_DIR, "evaluation")
EVAL_DATASET_PATH = os.path.join(EVAL_DIR, "eval_dataset.csv")
EVAL_RESULT_PATH = os.path.join(EVAL_DIR, "evaluation_result.csv")

# ==========================================
# [모델 설정] Model Configuration
# ==========================================
# LLM (생성 모델)
LLM_MODEL_ID = "beomi/Llama-3-Open-Ko-8B"

# reranker (재순위 모델)
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"

# Embedding (벡터 변환 모델) - DB 구축과 검색에 공통 사용
EMBEDDING_MODEL_ID = "BAAI/bge-m3"

# 디바이스 설정 (CUDA 우선, 없으면 CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# [파라미터 설정] Parameters
# ==========================================
# 1. 수집기 (Collector)
TARGET_REPORT_KIND = 'A'  # A: 사업보고서, F: 분기, S: 반기

# 2. 파서 (Parser) - 추출할 섹션 키워드
TARGET_SECTION_KEYWORDS = [
    "사업의 내용", 
    "재무에 관한 사항", 
    "이사의 경영진단"
]

# 3. RAG (Chunking)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# 4. 생성 (Generation)
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1

# 5. 평가 (Evaluation)
RAG_SYSTEM_PROMPT = (
    "당신은 기업 보고서 분석 AI입니다. [참고 문서]를 기반으로 질문에 답변하세요. "
    "없는 내용을 지어내지 말고, 수치와 사실 위주로 설명하세요."
)
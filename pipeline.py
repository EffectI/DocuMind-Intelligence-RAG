import os
import sys
from dotenv import load_dotenv
from src.ingestion import DartCollector
from src.parsing import DartIntegratedParser
from src.embedding import VectorStoreBuilder
from src.inference import RAGEngine

# ==========================================
# [설정] 파이프라인 타겟 설정
# ==========================================
TARGET_COMPANY = "삼성전자"
TARGET_YEAR = "2024"

def run_pipeline():
    print("="*60)
    print(f"[DocuMind] RAG 데이터 파이프라인 가동 시작")
    print(f"타겟: {TARGET_COMPANY} ({TARGET_YEAR})")
    print("="*60)

    # ---------------------------------------------------------
    # 1. 환경 설정 로드
    # ---------------------------------------------------------
    load_dotenv()
    api_key = os.getenv("DART_API_KEY")
    if not api_key:
        print("[Error] .env 파일에서 DART_API_KEY를 찾을 수 없습니다.")
        return

    # ---------------------------------------------------------
    # 2. Ingestion (데이터 수집)
    # ---------------------------------------------------------
    print("\n[Step 1] DART 보고서 다운로드 (Ingestion)...")
    collector = DartCollector(api_key=api_key)
    
    # 다운로드 실행 (이미 있으면 스킵하거나 덮어씀)
    xml_path = collector.download_report(TARGET_COMPANY, TARGET_YEAR)
    
    if not xml_path:
        print("[Stop] 파일 수집 실패로 파이프라인을 중단합니다.")
        return

    # ---------------------------------------------------------
    # 3. Parsing (데이터 가공)
    # ---------------------------------------------------------
    print("\n[Step 2] XML 파싱 및 섹션 추출 (Parsing)...")
    parser = DartIntegratedParser()
    
    # 파싱 수행 -> processed 폴더에 .md 파일들 생성됨
    parser.parse_file(xml_path)

    # ---------------------------------------------------------
    # 4. Embedding (벡터 DB 구축)
    # ---------------------------------------------------------
    print("\n[Step 3] 임베딩 및 벡터 DB 저장 (Embedding)...")
    builder = VectorStoreBuilder()
    
    # 가공된 문서 로드
    docs = builder.load_documents()
    
    if not docs:
        print("[Warn] 임베딩할 문서가 없습니다. 파싱 단계를 확인하세요.")
    else:
        # 청킹(Chunking) 및 태깅
        chunks = builder.split_documents(docs)
        
        # ChromaDB 저장 (증분 업데이트 방식)
        builder.build_database(chunks)

    # ---------------------------------------------------------
    # 5. Verification (검증 테스트)
    # ---------------------------------------------------------
    print("\n[Step 4] 파이프라인 검증 테스트 (Inference Check)...")
    print(" -> RAG 엔진을 로드하여 데이터가 잘 검색되는지 확인합니다.")
    
    try:
        rag = RAGEngine()
        test_query = f"{TARGET_COMPANY}의 주요 사업 내용 요약해줘"
        
        print(f"\n질문: {test_query}")
        print("-" * 30)
        
        # 스트리밍 방식이라 한 글자씩 출력
        print("답변: ", end="")
        for chunk in rag.chat(test_query, filters={"company": TARGET_COMPANY, "year": TARGET_YEAR}):
            print(chunk, end="", flush=True)
            
        print("\n" + "-" * 30)
        print("검증 완료.")

    except Exception as e:
        print(f"\n[Error] 엔진 로드 중 오류 발생: {e}")

    print("\n" + "="*60)
    print("전체 파이프라인 실행 완료!")
    print("이제 'streamlit run app.py'를 실행하여 웹 서비스를 시작하세요.")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
import os
import torch
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================================
# [설정] 경로 및 모델
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(BASE_DIR, "data", "processed", "sections")
DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")
EMBEDDING_MODEL = "BAAI/bge-m3"

print(f"데이터 경로 확인: {SOURCE_DIR}")
print(f"DB 저장 경로: {DB_PATH}")
print(f"사용 모델: {EMBEDDING_MODEL}")

def main():
    print(f"========== [1] GPU 환경 점검 ==========")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device}")
    if device == "cuda":
        print(f"그래픽카드: {torch.cuda.get_device_name(0)}")

    print(f"\n========== [2] 데이터 로드 ==========")
    if not os.path.exists(SOURCE_DIR):
        print(f"오류: '{SOURCE_DIR}' 폴더가 없습니다. 파싱 코드를 먼저 실행했나요?")
        return

    # 마크다운 파일(.md)만 골라서 로드
    loader = DirectoryLoader(SOURCE_DIR, glob="**/*.md", loader_cls=TextLoader, show_progress=True)
    docs = loader.load()
    
    if not docs:
        print("로드된 문서가 없습니다. 파싱 결과물을 확인하세요.")
        return
        
    print(f"총 {len(docs)}개의 문서 파일을 읽었습니다.")
    # 예시로 첫 번째 문서의 출처 출력
    print(f"   -> 첫 번째 문서: {docs[0].metadata['source']}")

    print(f"\n========== [3] 텍스트 분할 (Chunking) ==========")
    # 문맥 유지를 위해 1000자 단위로 자르고 200자 겹치게 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"전체 문서를 {len(chunks)}개의 청크(Chunk)로 분할했습니다.")

    print(f"\n========== [4] 벡터 DB 구축 (Indexing) ==========")
    print(f"임베딩 모델({EMBEDDING_MODEL}) 로드 중... (GPU 사용)")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device}, # 여기서 7800XT 가속 활용
        encode_kwargs={'normalize_embeddings': True}
    )

    print("벡터 DB 생성 및 저장 중... (시간이 조금 걸릴 수 있습니다)")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_name="samsung_report_db"
    )
    print(f"DB 저장 완료: {DB_PATH}")

    print(f"\n========== [5] 검색 성능 테스트 (Retrieval) ==========")
    # 실제로 잘 찾아지는지 테스트
    query = "반도체 부문의 주요 제품은 무엇인가요?"
    print(f"질문: {query}")
    
    # 유사도 기반 검색 (상위 3개)
    results = vector_store.similarity_search(query, k=3)
    
    print("\n[검색 결과 확인]")
    for i, res in enumerate(results):
        print(f"--- [결과 {i+1}] (출처: {os.path.basename(res.metadata['source'])}) ---")
        print(res.page_content[:200] + "...") # 내용 앞부분만 출력
        print("----------------------------------------------------")

    print("\nRAG 파이프라인의 '검색(Retrieval)' 단계 검증 완료")

if __name__ == "__main__":
    main()
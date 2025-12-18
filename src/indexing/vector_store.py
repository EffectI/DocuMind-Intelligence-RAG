import os
import traceback
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# [설정] 인덱싱 설정
# ==========================================
INPUT_DIR = "data/processed/sections"      # 파싱된 마크다운 파일 경로
DB_DIR = "data/vector_db"                  # 벡터 DB 저장 경로
MODEL_NAME = "jhgan/ko-sbert-nli"          # 한국어 특화 경량 임베딩 모델
CHUNK_SIZE = 500                           # 자르는 크기 (토큰/글자 수)
CHUNK_OVERLAP = 50                         # 겹치는 구간 (문맥 유실 방지)

class VectorStoreBuilder:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={'device': 'cpu'}, # GPU가 있다면 'cuda'로 변경
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None

    def load_documents(self):
        """저장된 마크다운 파일들을 로드합니다."""
        if not os.path.exists(INPUT_DIR):
            print(f"오류: 데이터 폴더가 없습니다 ({INPUT_DIR})")
            return []
            
        print(f"문서 로딩 중... ({INPUT_DIR})")
        # glob="*.md"로 마크다운 파일만 로드
        loader = DirectoryLoader(INPUT_DIR, glob="*.md", loader_cls=TextLoader)
        documents = loader.load()
        print(f" -> 총 {len(documents)}개의 문서 파일 로드됨")
        return documents

    def split_documents(self, documents):
        """문서를 작은 청크(Chunk) 단위로 쪼갭니다."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""] # 문단 > 줄바꿈 > 단어 순으로 자름
        )
        chunks = text_splitter.split_documents(documents)
        print(f" -> 총 {len(chunks)}개의 청크로 분할됨")
        return chunks

    def build_database(self, chunks):
        """청크를 임베딩하여 ChromaDB에 저장합니다."""
        if not chunks:
            print("저장할 청크가 없습니다.")
            return

        print("벡터 DB 생성 및 저장 중... (시간이 조금 걸릴 수 있습니다)")
        
        # ChromaDB 생성 및 저장
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=DB_DIR,
            collection_name="dart_reports"
        )
        print(f" -> DB 저장 완료: {DB_DIR}")

    def test_search(self, query):
        """간단한 검색 테스트를 수행합니다."""
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=DB_DIR, 
                embedding_function=self.embedding_model,
                collection_name="dart_reports"
            )
            
        print(f"\n[검색 테스트] 질문: '{query}'")
        results = self.vector_store.similarity_search(query, k=3)
        
        for i, doc in enumerate(results):
            print(f"\n--- 결과 {i+1} (출처: {os.path.basename(doc.metadata['source'])}) ---")
            print(doc.page_content[:200] + "...") # 앞부분만 출력

if __name__ == "__main__":
    builder = VectorStoreBuilder()
    
    # 1. 문서 로드
    docs = builder.load_documents()
    
    if docs:
        # 2. 분할 (Chunking)
        chunks = builder.split_documents(docs)
        
        # 3. DB 구축 (Vectorization)
        # 이미 DB가 있다면 이 단계는 건너뛰거나, 기존 DB를 지우고 새로 만들 수 있습니다.
        builder.build_database(chunks)
        
        # 4. 검색 테스트
        builder.test_search("반도체 시장 전망은?")
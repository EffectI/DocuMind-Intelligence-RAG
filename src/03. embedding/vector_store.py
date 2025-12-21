import os
import shutil
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================================
# [설정] 경로
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "sections")
DB_DIR = os.path.join(BASE_DIR, "data", "vector_db")

MODEL_NAME = "BAAI/bge-m3" 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

class VectorStoreBuilder:
    def __init__(self):
        print(f"[Init] 모델 로딩 중: {MODEL_NAME}")
        # GPU 사용 가능 여부 확인
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={'device': device}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None

    def load_documents(self):
        if not os.path.exists(INPUT_DIR):
            print(f"오류: 데이터 폴더가 없습니다 ({INPUT_DIR})")
            return []
            
        print(f"[1] 문서 로딩... ({INPUT_DIR})")
        loader = DirectoryLoader(INPUT_DIR, glob="*.md", loader_cls=TextLoader)
        documents = loader.load()
        print(f" -> {len(documents)}개 파일 로드 완료")
        return documents

    # [추가됨] 파일명에서 태그(회사, 연도)를 추출하는 핵심 함수
    def _extract_metadata_from_filename(self, filepath):
        filename = os.path.basename(filepath)
        meta = {}

        # 파일명 예시: Samsung_2024_사업의내용.md
        # 1. 연도 추출 (4자리 숫자)
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            meta['year'] = year_match.group(0)
        else:
            meta['year'] = "unknown"

        # 2. 회사명 추출 (첫 번째 언더바 앞부분)
        if "_" in filename:
            meta['company'] = filename.split("_")[0]
        else:
            meta['company'] = "unknown"
            
        return meta

    def split_documents(self, documents):
        print(f"[2] 문서 분할 및 자동 태깅(Tagging)...")

        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        all_splits = []
        for doc in documents:
            # [핵심] 여기서 파일명을 분석해 태그를 생성합니다.
            auto_meta = self._extract_metadata_from_filename(doc.metadata['source'])
            
            # 문서 객체에 태그를 심습니다.
            doc.metadata.update(auto_meta)

            # 마크다운 헤더로 1차 분할
            md_splits = markdown_splitter.split_text(doc.page_content)
            
            # 분할된 조각들에도 태그를 전파합니다.
            for split in md_splits:
                split.metadata.update(doc.metadata)
            
            # 텍스트 길이로 2차 분할
            final_splits = text_splitter.split_documents(md_splits)
            all_splits.extend(final_splits)

        print(f" -> 총 {len(all_splits)}개 청크 생성 (태그 적용 완료)")
        
        # [확인] 첫 번째 청크의 태그가 잘 들어갔는지 출력
        if all_splits:
            print(f"    [Sample Meta] {all_splits[0].metadata}")
            
        return all_splits

    def build_database(self, chunks):
        # 1. DB 로드 또는 생성
        if not os.path.exists(DB_DIR):
            print(f"[Info] 신규 DB를 생성합니다.")
            self.vector_store = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=DB_DIR,
                collection_name="samsung_report_db"
            )
        else:
            print(f"[Info] 기존 DB를 불러옵니다.")
            self.vector_store = Chroma(
                persist_directory=DB_DIR,
                embedding_function=self.embedding_model,
                collection_name="samsung_report_db"
            )

        # 2. 중복 확인 (이미 있는 파일은 건너뛰기)
        existing_data = self.vector_store.get()
        existing_sources = set()
        
        if existing_data['metadatas']:
            for meta in existing_data['metadatas']:
                if meta and 'source' in meta:
                    existing_sources.add(meta['source'])
        
        print(f"[Check] 현재 DB 문서 수: {len(existing_sources)}개")

        # 3. 새로운 것만 필터링
        new_chunks = []
        skipped_count = 0
        
        for chunk in chunks:
            source = chunk.metadata.get('source')
            if source not in existing_sources:
                new_chunks.append(chunk)
            else:
                skipped_count += 1

        # 4. 저장
        if new_chunks:
            print(f"[Update] 새로운 청크 {len(new_chunks)}개를 DB에 추가합니다... (중복 {skipped_count}개 제외)")
            self.vector_store.add_documents(new_chunks)
            print(f" -> [완료] DB 업데이트 성공.")
        else:
            print(f"[Info] 추가할 새로운 내용이 없습니다.")

    def test_search(self, query):
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=DB_DIR, 
                embedding_function=self.embedding_model,
                collection_name="samsung_report_db"
            )
            
        print(f"\n[검색 테스트] Q: '{query}'")
        # 메타데이터도 같이 출력해서 태그가 잘 달렸는지 확인
        results = self.vector_store.similarity_search(query, k=3)
        
        for i, doc in enumerate(results):
            meta = doc.metadata
            print(f"\n[결과 {i+1}] {meta.get('company')} / {meta.get('year')}")
            print(f"{doc.page_content[:100]}...")

if __name__ == "__main__":
    builder = VectorStoreBuilder()
    
    docs = builder.load_documents()
    if docs:
        chunks = builder.split_documents(docs)
        builder.build_database(chunks)
        builder.test_search("반도체 시장 전망")
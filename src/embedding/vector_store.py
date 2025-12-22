import os
import sys
import shutil
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import (
    PROCESSED_DATA_DIR, 
    DB_PATH, 
    EMBEDDING_MODEL_ID, 
    DEVICE,
    CHUNK_SIZE, 
    CHUNK_OVERLAP
)

class VectorStoreBuilder:
    def __init__(self):
        print(f"[Init] 임베딩 모델 로딩: {EMBEDDING_MODEL_ID}")
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_ID, # Config 사용
            model_kwargs={'device': DEVICE}, # Config 사용
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None

    def load_documents(self):
        # Config 사용 (PROCESSED_DATA_DIR)
        if not os.path.exists(PROCESSED_DATA_DIR):
            print(f"오류: 데이터 폴더 없음 ({PROCESSED_DATA_DIR})")
            return []
            
        print(f"[Loader] 문서 로딩 중... ({PROCESSED_DATA_DIR})")
        loader = DirectoryLoader(PROCESSED_DATA_DIR, glob="*.md", loader_cls=TextLoader)
        documents = loader.load()
        print(f" -> {len(documents)}개 파일 로드됨")
        return documents

    def _extract_metadata_from_filename(self, filepath):
        filename = os.path.basename(filepath)
        meta = {"company": "unknown", "year": "unknown"}
        
        year_match = re.search(r'20\d{2}', filename)
        if year_match: meta['year'] = year_match.group(0)
        
        if "_" in filename: meta['company'] = filename.split("_")[0]
        return meta

    def split_documents(self, documents):
        print(f"[Splitter] 청킹 작업 시작 (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")

        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        # Config 변수 사용
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        all_splits = []
        for doc in documents:
            auto_meta = self._extract_metadata_from_filename(doc.metadata['source'])
            doc.metadata.update(auto_meta)

            md_splits = md_splitter.split_text(doc.page_content)
            for split in md_splits:
                split.metadata.update(doc.metadata)
            
            final_splits = text_splitter.split_documents(md_splits)
            all_splits.extend(final_splits)

        print(f" -> 총 {len(all_splits)}개 청크 생성됨")
        return all_splits

    def build_database(self, chunks):
        # Config 사용 (DB_PATH)
        if not os.path.exists(DB_PATH):
            print(f"[Info] 신규 DB 생성 ({DB_PATH})")
            self.vector_store = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=DB_PATH,
                collection_name="samsung_report_db"
            )
        else:
            print(f"[Info] 기존 DB 로드 ({DB_PATH})")
            self.vector_store = Chroma(
                persist_directory=DB_PATH,
                embedding_function=self.embedding_model,
                collection_name="samsung_report_db"
            )

        # 증분 업데이트 (중복 방지)
        existing_data = self.vector_store.get()
        existing_sources = set()
        if existing_data['metadatas']:
            for meta in existing_data['metadatas']:
                if meta and 'source' in meta: existing_sources.add(meta['source'])
        
        new_chunks = []
        for chunk in chunks:
            if chunk.metadata.get('source') not in existing_sources:
                new_chunks.append(chunk)

        if new_chunks:
            print(f"[Update] 신규 청크 {len(new_chunks)}개 저장 중...")
            self.vector_store.add_documents(new_chunks)
            print(" -> 저장 완료")
        else:
            print(" -> 추가할 데이터가 없습니다.")
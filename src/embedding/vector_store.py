import os
import re
import hashlib
import pickle
from typing import List, Dict, Any

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    PROCESSED_DATA_DIR, 
    DB_PATH, 
    EMBEDDING_MODEL_ID, 
    DEVICE,
    CHUNK_SIZE, 
    CHUNK_OVERLAP,
    BM25_INDEX_PATH
)

class VectorStoreBuilder:
    def __init__(self):
        print(f"[Init] 임베딩 모델 로딩: {EMBEDDING_MODEL_ID} ({DEVICE})")
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_ID,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None

    def load_documents(self) -> List[Document]:
        """지정된 디렉토리에서 Markdown 파일 로드"""
        if not os.path.exists(PROCESSED_DATA_DIR):
            print(f"[Error] 데이터 폴더를 찾을 수 없습니다: {PROCESSED_DATA_DIR}")
            return []
            
        print(f"[Loader] 문서 로딩 시작: {PROCESSED_DATA_DIR}")
        
        # [Optimization] glob 패턴을 명확히 하고, 에러 발생 시 처리
        loader = DirectoryLoader(PROCESSED_DATA_DIR, glob="**/*.md", loader_cls=TextLoader)
        try:
            documents = loader.load()
            print(f" -> {len(documents)}개 파일 로드 성공")
            return documents
        except Exception as e:
            print(f"[Error] 문서 로딩 중 오류 발생: {e}")
            return []

    def _extract_metadata_from_filename(self, filepath: str) -> Dict[str, str]:
        """파일명에서 회사명과 연도 추출 (정규식 강화)"""
        filename = os.path.basename(filepath)
        meta = {"company": "unknown", "year": "unknown", "source_file": filename}
        
        # [Refactor] 연도 추출 정규식 강화 (1990~2029 범위로 한정)
        year_match = re.search(r'(19|20)\d{2}', filename)
        if year_match: 
            meta['year'] = year_match.group(0)
        
        # [Refactor] 회사명 추출 (첫 번째 언더바 앞부분을 회사명으로 간주)
        if "_" in filename: 
            meta['company'] = filename.split("_")[0]
            
        return meta

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서를 의미 단위(Header)와 문자 길이 기준으로 청킹"""
        if not documents:
            return []
            
        print(f"[Splitter] 청킹 시작 (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")

        # 1. Markdown Header Split (논리적 분할)
        headers_to_split_on = [
            ("#", "Header 1"), 
            ("##", "Header 2"), 
            ("###", "Header 3")
        ]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        # 2. Character Split (물리적 분할 - 토큰 제한 고려)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""] # 분할 우선순위 지정
        )

        all_splits = []
        
        for doc in documents:
            # 메타데이터 보강
            file_meta = self._extract_metadata_from_filename(doc.metadata.get('source', ''))
            doc.metadata.update(file_meta)

            # 1차 분할 (Markdown 구조 기반)
            md_splits = md_splitter.split_text(doc.page_content)
            
            # 2차 분할 (길이 기반) & 메타데이터 전파
            for split in md_splits:
                # 원본 문서의 메타데이터를 상속받되, Header 정보는 유지됨
                combined_meta = doc.metadata.copy()
                combined_meta.update(split.metadata)
                split.metadata = combined_meta
                
            final_splits = text_splitter.split_documents(md_splits)
            all_splits.extend(final_splits)

        print(f" -> 총 {len(all_splits)}개 청크 생성 완료")
        return all_splits

    def _generate_chunk_id(self, chunk: Document) -> str:
        """[Optimization] 청크 내용 기반 해시 ID 생성 (중복 저장 방지용)"""
        # 내용(page_content) + 소스파일(source)을 합쳐서 해시 생성
        # 내용이 조금이라도 바뀌면 ID가 바뀌어 새로 저장됨
        content_hash = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()
        source_hash = hashlib.md5(chunk.metadata.get('source', '').encode('utf-8')).hexdigest()
        return f"{source_hash}_{content_hash}"

    def build_database(self, chunks: List[Document]):
        """ChromaDB 저장(증분) 및 BM25 인덱스 생성(전체)"""
        if not chunks:
            print("[Info] 저장할 청크가 없습니다.")
            return

        # ---------------------------------------------------------
        # 1. ChromaDB 저장 (증분 업데이트)
        # ---------------------------------------------------------
        print(f"[DB] 데이터베이스 연결: {DB_PATH}")
        
        # [Refactor] Chroma 초기화 로직 단순화
        self.vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=self.embedding_model,
            collection_name="samsung_report_db"
        )

        # [Optimization] 증분 업데이트 (Incremental Update)
        # 기존 DB에 있는 ID들을 가져옴
        existing_ids = set(self.vector_store.get()['ids'])
        
        new_chunks = []
        new_ids = []
        
        for chunk in chunks:
            chunk_id = self._generate_chunk_id(chunk)
            
            # ID가 존재하지 않을 때만 추가 목록에 포함
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(chunk_id)
                existing_ids.add(chunk_id)

        if new_chunks:
            print(f"[Update] 신규 청크 {len(new_chunks)}개 저장 중...")
            self.vector_store.add_documents(documents=new_chunks, ids=new_ids)
            print(" -> ChromaDB 저장 완료")
        else:
            print(" -> ChromaDB는 이미 최신 상태입니다. (업데이트 없음)")

        # ---------------------------------------------------------
        # 2. BM25 인덱스 생성 및 저장 (Hybrid Search용)
        # ---------------------------------------------------------
        # BM25는 전체 문서 통계(IDF)가 중요하므로, 로드된 전체 chunks로 새로 갱신합니다.
        print("[BM25] 키워드 검색 인덱스 생성 중...")
        try:
            # BM25Retriever 생성
            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = 3  # 기본 검색 개수 설정
            
            # 저장 경로 디렉토리 확보
            bm25_dir = os.path.dirname(BM25_INDEX_PATH)
            if not os.path.exists(bm25_dir):
                os.makedirs(bm25_dir, exist_ok=True)

            # pickle로 저장
            with open(BM25_INDEX_PATH, "wb") as f:
                pickle.dump(bm25_retriever, f)
            print(f" -> BM25 인덱스 저장 완료 ({BM25_INDEX_PATH})")
            
        except Exception as e:
            print(f"[Error] BM25 인덱스 생성 실패: {e}")
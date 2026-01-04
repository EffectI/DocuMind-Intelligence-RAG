# # unset HSA_OVERRIDE_GFX_VERSION
# import torch
# import sys
# import os

# print("========== 환경 진단 시작 ==========")
# print(f"Python Version: {sys.version.split()[0]}")
# print(f"PyTorch Version: {torch.__version__}")

# # 핵심: 우리가 설정한 환경 변수가 잘 들어갔는지 확인
# print(f"\n[환경 변수 확인]")
# print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', '설정안됨(Not Set)')}")
# print(f"ROCM_PATH: {os.environ.get('ROCM_PATH', '설정안됨(Not Set)')}")

# print(f"\n[GPU 연결 테스트]")
# try:
#     # GPU 사용 가능 여부 확인
#     is_available = torch.cuda.is_available()
#     print(f"torch.cuda.is_available(): {is_available}")

#     if is_available:
#         print(f"Make/Model: {torch.cuda.get_device_name(0)}")
#         print(f"Device Count: {torch.cuda.device_count()}")
        
#         # 실제 텐서 연산 테스트 (메모리에 올리기)
#         x = torch.tensor([1.0, 2.0, 3.0]).cuda()
#         print(f"Tensor Test: 성공! (값: {x})")
#     else:
#         print("❌ 실패: GPU를 인식하지 못했습니다.")
        
# except Exception as e:
#     print(f"❌ 에러 발생: {e}")

# print("====================================")


# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI

# # 1. 환경 변수 로드
# load_dotenv()

# def test_gemini_connection():
#     print("Checking Google API connection...")
    
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         print("❌ Error: GOOGLE_API_KEY not found in .env file")
#         return

#     try:
#         # 2. 모델 초기화 (가벼운 Flash 모델 사용)
#         llm = ChatGoogleGenerativeAI(
#             model="gemini-2.5-flash",
#             temperature=0
#         )
        
#         # 3. 간단한 질문 전송
#         print("Sending request to Gemini...")
#         response = llm.invoke("Hello! Are you working?")
        
#         # 4. 결과 출력
#         print("\n✅ Success! Gemini Response:")
#         print(f"Content: {response.content}")
        
#     except Exception as e:
#         print(f"\n❌ Connection Failed: {e}")

# if __name__ == "__main__":
#     test_gemini_connection()

import os
import sys
import time
import pandas as pd
from tabulate import tabulate # pip install tabulate 필요 (없으면 print로 대체 가능)

# 프로젝트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import RAGEngine
from config import BM25_INDEX_PATH

def print_separator(title):
    print(f"\n{'='*20} [ {title} ] {'='*20}")

def debug_retrieval_system():
    print_separator("1. 시스템 초기화 상태 점검")
    
    # 1. BM25 인덱스 파일 확인
    if os.path.exists(BM25_INDEX_PATH):
        print(f"✅ BM25 인덱스 발견: {BM25_INDEX_PATH}")
        size_mb = os.path.getsize(BM25_INDEX_PATH) / (1024 * 1024)
        print(f"   - 크기: {size_mb:.2f} MB")
    else:
        print(f"❌ BM25 인덱스 없음: {BM25_INDEX_PATH}")
        print("   -> Hybrid Search가 작동하지 않고 Vector Search만 수행됩니다.")
        print("   -> 해결: 'python pipeline.py' 또는 UI에서 DB 학습을 다시 실행하세요.")

    # 2. 엔진 로드
    print("\n[Engine] RAGEngine 로딩 중...")
    start_time = time.time()
    try:
        rag = RAGEngine()
        print(f"✅ 엔진 로드 완료 ({time.time() - start_time:.2f}초)")
    except Exception as e:
        print(f"❌ 엔진 로드 실패: {e}")
        return

    # 3. 검색 품질 테스트
    print_separator("2. Hybrid Search & Reranking 상세 디버깅")
    
    test_query = "삼성전자의 2024년 주요 경영 전략은 무엇인가?" # 실제 데이터에 맞는 질문으로 변경 가능
    print(f"🔍 테스트 질문: \"{test_query}\"")

    try:
        # RAGEngine.search() 호출 (k=5로 넉넉하게 확인)
        retrieved_docs = rag.search(test_query, k=5)
        
        if not retrieved_docs:
            print("❌ 검색된 문서가 없습니다. (DB가 비어있거나 필터링 문제)")
            return

        # 결과 테이블 생성
        debug_data = []
        for rank, doc in enumerate(retrieved_docs, 1):
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            score = doc.metadata.get('rerank_score', 0.0)
            content_preview = doc.page_content[:50].replace('\n', ' ') + "..."
            
            debug_data.append([
                rank, 
                f"{score:.4f}", 
                source, 
                content_preview
            ])

        # 결과 출력
        headers = ["Rank", "Rerank Score", "Source File", "Content Preview"]
        try:
            print(tabulate(debug_data, headers=headers, tablefmt="grid"))
        except ImportError:
            # tabulate가 없는 경우 기본 출력
            print(f"{'Rank':<5} {'Score':<10} {'Source':<30} {'Content'}")
            for row in debug_data:
                print(f"{row[0]:<5} {row[1]:<10} {row[2]:<30} {row[3]}")

        # 점수 분석
        top_score = retrieved_docs[0].metadata.get('rerank_score', 0)
        if top_score < 0.0: # CrossEncoder 점수는 보통 Logit 값이므로 음수일 수 있음 (Sigmoid 전)
             # 모델에 따라 다르지만, 보통 0보다 크거나 Sigmoid 적용 시 0.5 이상이어야 관련성 있음
             print("\n⚠️ [주의] 상위 문서의 점수가 낮습니다. 질문과 관련 없는 문서일 수 있습니다.")
        else:
             print(f"\n✅ 상위 문서 신뢰도 양호 (Top Score: {top_score:.4f})")

    except Exception as e:
        print(f"❌ 검색 도중 에러 발생: {e}")
        import traceback
        traceback.print_exc()

    # 4. 생성 테스트 (옵션)
    print_separator("3. 답변 생성 테스트 (Generation)")
    try:
        print("🤖 답변 생성 중...", end=" ")
        # 평가용 동기 메서드 사용
        answer = rag.generate_answer(test_query)
        print("완료!\n")
        print(f"[AI 답변]\n{answer}")
    except Exception as e:
        print(f"\n❌ 답변 생성 실패: {e}")

if __name__ == "__main__":
    debug_retrieval_system()
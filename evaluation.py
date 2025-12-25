import os
import time
import gc
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from dotenv import load_dotenv

# Ragas & LangChain (Gemini)
from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerCorrectness, ContextRecall
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 사용자 정의 모듈
from src.inference import RAGEngine
from config import EVAL_DATASET_PATH, EVAL_RESULT_PATH

# 환경 변수 로드
load_dotenv()

def load_ground_truth(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # engine='python'과 on_bad_lines='skip'으로 로딩 안정성 확보
    return pd.read_csv(file_path, engine="python", on_bad_lines='skip')

def generate_rag_responses(df, rag_engine):
    """
    RAG 엔진을 사용하여 답변을 생성합니다.
    GPU 메모리 누수 방지를 위해 GC 및 캐시 정리를 수행합니다.
    """
    answers = []
    contexts = []
    
    print(f"Generating responses for {len(df)} questions...")

    # tqdm으로 진행률 표시
    for i, q in enumerate(tqdm(df['question'])):
        try:
            # -------------------------------------------------
            # [안전 장치 1] GPU 메모리 정리 (매 턴마다 수행)
            # -------------------------------------------------
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 1. 문서 검색
            retrieved_docs = rag_engine.search(q, k=3)
            doc_contents = [doc.page_content for doc in retrieved_docs]
            
            # 2. 답변 생성
            full_response = ""
            # chat 함수는 내부적으로 쓰레드를 사용하므로 제너레이터를 끝까지 소비해야 함
            for chunk in rag_engine.chat(q):
                full_response += chunk
            
            # 소스 표기 제거 (평가 정확도 향상용)
            if "[참고 문서]" in full_response:
                clean_answer = full_response.split("[참고 문서]")[0].strip()
            elif "**[참고 문서]**" in full_response:
                clean_answer = full_response.split("**[참고 문서]**")[0].strip()
            else:
                clean_answer = full_response.strip()

            answers.append(clean_answer)
            contexts.append(doc_contents)
            
            # -------------------------------------------------
            # [안전 장치 2] 과열 방지 휴식 (0.5초)
            # -------------------------------------------------
            time.sleep(0.5)

        except Exception as e:
            print(f"\n[Error at index {i}] {e}")
            answers.append("Error occurred")
            contexts.append([])
            
            # 에러 발생 시에도 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return answers, contexts

def run_evaluation():
    # config에 정의된 경로 사용
    csv_path = EVAL_DATASET_PATH
    output_path = EVAL_RESULT_PATH

    print(f"Loading dataset from: {csv_path}")

    # 1. 데이터 로드
    try:
        df = load_ground_truth(csv_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. RAG 엔진 초기화 (피평가자: 학생)
    try:
        rag_engine = RAGEngine()
    except Exception as e:
        print(f"Error initializing RAGEngine: {e}")
        return

    # 3. 답변 생성
    answers, contexts = generate_rag_responses(df, rag_engine)

    # 4. 데이터셋 생성
    data_dict = {
        "question": df['question'].tolist(),
        "answer": answers,
        "contexts": contexts,
        "ground_truth": df['ground_truth'].tolist()
    }
    dataset = Dataset.from_dict(data_dict)

    # =========================================================
    # 5. 평가 설정 (심판: Gemini)
    # =========================================================
    print("Initializing Gemini for evaluation...")
    
    try:
        # 채점관 LLM: Gemini 1.5 Pro
        judge_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0
        )

        # 임베딩 모델: Gemini Embedding
        judge_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )

        print("Starting Ragas evaluation with Gemini 1.5 Pro...")
        
        results = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(),
                AnswerCorrectness(),
                ContextRecall()
            ],
            llm=judge_llm,
            embeddings=judge_embeddings
        )
        
        print("\n=== Evaluation Results ===")
        print(results)
        
        # 결과 저장 폴더가 없으면 생성 (안전장치)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results.to_pandas().to_csv(output_path, index=False)
        print(f"Saved evaluation results to: {output_path}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Please check your GOOGLE_API_KEY in .env file.")

if __name__ == "__main__":
    run_evaluation()
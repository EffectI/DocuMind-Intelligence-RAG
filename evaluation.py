import os
import time
import gc
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerCorrectness, ContextRecall
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# [사용자 정의 모듈]
from src.inference import RAGEngine
from config import EVAL_DATASET_PATH, EVAL_RESULT_PATH

# 환경 변수 로드
load_dotenv()

def load_ground_truth(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, engine="python", on_bad_lines='skip')

def generate_rag_responses(df, rag_engine):
    """
    RAG 엔진을 사용하여 답변을 생성합니다.
    rag_engine.chat() 대신 generate_answer()를 사용하여 
    쓰레드 생성 오버헤드와 메모리 누수를 방지합니다.
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
            # 검색 결과도 저장 (Context Recall 평가용)
            retrieved_docs = rag_engine.search(q, k=3)
            doc_contents = [doc.page_content for doc in retrieved_docs]
            
            # 2. 답변 생성 (안정적인 generate_answer 사용)
            if hasattr(rag_engine, 'generate_answer'):
                full_response = rag_engine.generate_answer(q)
            else:
                full_response = ""
                for chunk in rag_engine.chat(q):
                    full_response += chunk

            # 소스 표기 제거 (평가 시에는 순수 답변만 필요하므로)
            if "[참고 문서]" in full_response:
                clean_answer = full_response.split("[참고 문서]")[0].strip()
            elif "**[참고 문서]**" in full_response:
                clean_answer = full_response.split("**[참고 문서]**")[0].strip()
            else:
                clean_answer = full_response.strip()

            answers.append(clean_answer)
            contexts.append(doc_contents)
            
            # -------------------------------------------------
            # [안전 장치 2] 과열 방지 휴식
            # -------------------------------------------------
            time.sleep(0.2)

        except Exception as e:
            print(f"\n[Error at index {i}] {e}")
            answers.append("Error occurred")
            contexts.append([])
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return answers, contexts

def run_evaluation():
    csv_path = EVAL_DATASET_PATH
    output_path = EVAL_RESULT_PATH

    print(f"Loading dataset from: {csv_path}")

    # 1. 데이터 로드
    try:
        df = load_ground_truth(csv_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. RAG 엔진 초기화
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
    # 5. 평가 설정 (Gemini)
    # =========================================================
    print("Initializing Gemini for evaluation...")
    
    try:
        judge_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            temperature=0,
            max_retries=10
        )

        judge_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )

        # 실행 설정 (Rate Limit 방지)
        run_config = RunConfig(
            max_workers=1, # 1개씩 순차 처리 (멈춤 방지)
            timeout=120
        )

        print("Starting Ragas evaluation (Sequential Mode with Gemini 2.5 Flash)...")
        
        results = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(),      
                AnswerCorrectness(), 
                ContextRecall()    
            ],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config
        )
        
        print("\n=== Evaluation Results ===")
        print(results)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results.to_pandas().to_csv(output_path, index=False)
        print(f"Saved evaluation results to: {output_path}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Please check your GOOGLE_API_KEY in .env file or check Rate Limits.")

if __name__ == "__main__":
    run_evaluation()
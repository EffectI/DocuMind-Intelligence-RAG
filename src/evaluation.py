import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_correctness
)
from src.inference import RAGEngine

# 1. 정답지 로드 (csv)
# 컬럼: ['question', 'ground_truth']
df = pd.read_csv("DocuMind-Intelligence-RAG/data/evaluation/eval_dataset.csv")


rag_engine = RAGEngine()

answers = []
contexts = []

print("평가 데이터 생성 중...")
for q in df['question']:
    # 검색된 문서 (Context) 가져오기
    # search 함수를 활용해 문서 내용만 리스트로 추출
    retrieved_docs = rag_engine.search(q, k=3)
    doc_contents = [doc.page_content for doc in retrieved_docs]
    

    full_response = ""
    for chunk in rag_engine.chat(q):
        full_response += chunk
    
    answers.append(full_response)
    contexts.append(doc_contents)

# 3. Ragas 데이터셋 포맷으로 변환
data_dict = {
    "question": df['question'].tolist(),
    "answer": answers, # 내 AI 답변
    "contexts": contexts, # 검색된 문서들
    "ground_truth": df['ground_truth'].tolist() # 정답지
}
dataset = Dataset.from_dict(data_dict)

# 4. 채점 시작 (OpenAI API Key 필요)
print("Ragas 평가 시작 (GPT-4)...")
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_correctness,
        context_recall
    ]
)

# 5. 결과 저장
print(results)
results.to_pandas().to_csv("evaluation_result.csv", index=False)
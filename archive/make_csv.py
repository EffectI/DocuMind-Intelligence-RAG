import pandas as pd
import os
from config import EVAL_DATASET_PATH, EVAL_DIR

# 1. 저장할 폴더가 없으면 생성
os.makedirs(EVAL_DIR, exist_ok=True)

# 2. 데이터 정의 (오류 없는 깨끗한 데이터)
data = [
    {
        "question": "2024년 3분기 삼성전자의 연결 기준 총 매출액은 얼마인가?",
        "ground_truth": "2024년 3분기 매출은 225조 826억 원입니다.",
        "source": "1. 사업의 개요, 4. 매출 및 수주상황"
    },
    {
        "question": "DX 부문의 2024년 3분기 매출 비중은 몇 퍼센트인가?",
        "ground_truth": "DX 부문의 매출 비중은 59.7%입니다.",
        "source": "2. 주요 제품 및 서비스 - 가. 주요 제품 매출"
    },
    {
        "question": "DS 부문의 주요 원재료 중 'Wafer'의 주요 매입처는 어디인가?",
        "ground_truth": "Wafer의 주요 매입처는 SK실트론(주), SILTRONIC 등입니다.",
        "source": "3. 원재료 및 생산설비 - 가. 주요 원재료 현황"
    },
    {
        "question": "2024년 3분기 말 기준, 삼성전자가 전 세계적으로 보유한 총 특허 건수는?",
        "ground_truth": "2024년 3분기 말 현재 세계적으로 총 260,602건의 특허를 보유하고 있습니다.",
        "source": "7. 기타 참고사항 - 가. 지적재산권 관련"
    },
    {
        "question": "삼성전자의 5대 주요 매출처를 나열하시오.",
        "ground_truth": "주요 매출처는 Apple, Deutsche Telekom, Hong Kong Techtronics, Supreme Electronics, Verizon입니다.",
        "source": "4. 매출 및 수주상황 - 마. 주요 매출처"
    },
    {
        "question": "2024년 3분기 연구개발비용 총계는 얼마인가?",
        "ground_truth": "2024년 3분기 연구개발비용 총계는 24조 7,465억 원입니다.",
        "source": "6. 주요계약 및 연구개발활동 - 나. 연구개발활동의 개요 및 연구개발비용"
    },
    {
        "question": "2024년 3분기 TV 및 모니터의 가동률은 몇 퍼센트인가?",
        "ground_truth": "TV, 모니터 등의 가동률은 80.6%입니다.",
        "source": "3. 원재료 및 생산설비 - 다. 생산능력, 생산실적, 가동률"
    },
    {
        "question": "2023년 삼성전자의 온실가스 배출량은 얼마인가?",
        "ground_truth": "2023년 온실가스 배출량은 17,337,196 tCO2-eq입니다.",
        "source": "7. 기타 참고사항 - 나. 환경 관련 규제사항"
    },
    {
        "question": "DS 부문의 2024년 3분기 시설투자 규모는 얼마인가?",
        "ground_truth": "DS 부문의 시설투자는 30조 3,111억 원입니다.",
        "source": "3. 원재료 및 생산설비 - 라. 생산설비 및 투자 현황 등"
    },
    {
        "question": "Google과 체결한 주요 계약 중 2024년 12월 31일까지 연장된 계약의 내용은 무엇인가?",
        "ground_truth": "유럽 32개국(EEA) 대상으로 Play Store, YouTube 등 구글 앱 사용에 대한 라이선스 계약(EMADA)입니다.",
        "source": "6. 주요계약 및 연구개발활동 - 가. 경영상의 주요 계약 등"
    }
]

# 3. 데이터프레임 생성 및 저장
df = pd.DataFrame(data)

# utf-8-sig: 엑셀에서도 한글 안 깨지게 저장
# index=False: 불필요한 인덱스 번호 제외
# quoting=1: 모든 텍스트에 쌍따옴표(")를 붙여서 쉼표 충돌 방지 (가장 안전)
df.to_csv(EVAL_DATASET_PATH, index=False, encoding="utf-8-sig", quoting=1)

print(f"CSV 파일이 정상적으로 재생성되었습니다:\n{EVAL_DATASET_PATH}")
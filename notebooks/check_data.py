import os
from bs4 import BeautifulSoup

# 수집 데이터 경로
DATA_DIR = "data/raw/dart"

def inspect_data():
    # 1. 디렉토리 및 파일 존재 여부 확인
    if not os.path.exists(DATA_DIR):
        print(f"[경고] 데이터 디렉토리를 찾을 수 없습니다: {DATA_DIR}")
        return

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.xml') or f.endswith('.txt')])
    
    if not files:
        print("[경고] 수집된 파일이 없습니다.")
        return

    print(f"=== 수집 현황 확인 (총 {len(files)}개 파일) ===")
    for f in files:
        file_path = os.path.join(DATA_DIR, f)
        size_kb = os.path.getsize(file_path) / 1024
        print(f"- {f} : {size_kb:.2f} KB")

    # 2. 가장 최신 파일 하나를 골라 상세 내용 검사
    target_file = files[-1]
    target_path = os.path.join(DATA_DIR, target_file)
    print(f"\n=== 상세 검사: {target_file} ===")

    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check 1: Raw 데이터 앞부분 (XML 헤더 등 확인)
        print("\n[1] Raw Data Preview (Top 200 chars):")
        print(content[:200])
        print("..." if len(content) > 200 else "")

        # Check 2: 텍스트 추출 시뮬레이션 (BeautifulSoup)
        print("\n[2] Parsed Text Preview (본문 추출 테스트):")
        soup = BeautifulSoup(content, 'html.parser')
        
        # 불필요한 태그 제거
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
            
        text = soup.get_text(separator='\n')
        
        # 공백 정리 후 의미 있는 라인만 추출
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 0]
        
        # 앞쪽 10줄과 뒤쪽 10줄 출력
        print("--- 시작 부분 ---")
        for line in lines[:10]:
            print(line)
        print("\n--- (중략) ---\n")
        print("--- 끝 부분 ---")
        for line in lines[-10:]:
            print(line)

        # 간단한 품질 평가
        if len(lines) < 10:
            print("\n[주의] 추출된 텍스트 라인이 너무 적습니다. 데이터가 비어있거나 파싱이 필요합니다.")
        else:
            print(f"\n[성공] 총 {len(lines)} 라인의 텍스트가 추출되었습니다. 데이터가 정상적으로 보입니다.")

    except Exception as e:
        print(f"[오류] 파일을 읽거나 파싱하는 중 문제가 발생했습니다: {e}")

if __name__ == "__main__":
    inspect_data()
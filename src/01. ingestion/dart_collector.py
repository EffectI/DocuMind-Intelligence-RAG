import os
import traceback
import OpenDartReader
from dotenv import load_dotenv

# ==========================================
# [설정] 변동 가능한 주요 변수
# ==========================================
DEFAULT_RAW_DATA_PATH = "data/raw/dart"

# 보고서 종류 (A: 사업보고서, F: 분기보고서, S: 반기보고서)
TARGET_REPORT_KIND = 'A' 

# ==========================================
# [클래스] DART 데이터 수집기
# ==========================================
class DartCollector:
    def __init__(self, api_key, save_path=DEFAULT_RAW_DATA_PATH):
        """
        :param api_key: DART API Key
        :param save_path: 파일 저장 경로 (기본값: DEFAULT_RAW_DATA_PATH)
        """
        self.dart = OpenDartReader(api_key)
        self.save_path = save_path
        self._ensure_directory_exists()

    def _ensure_directory_exists(self):
        """저장 경로가 존재하지 않으면 생성합니다."""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f"저장 디렉토리 생성됨: {self.save_path}")

    def download_report(self, corp_name, year):
        """
        특정 기업의 연간 사업보고서를 다운로드합니다.
        
        :param corp_name: 기업명 (예: 삼성전자)
        :param year: 대상 연도 (문자열, 예: '2022')
        :return: 저장된 파일 경로 (실패 시 None)
        """
        try:
            # 1. 보고서 목록 조회
            reports = self.dart.list(
                corp_name, 
                start=f"{year}-01-01", 
                end=f"{year}-12-31", 
                kind=TARGET_REPORT_KIND
            )
            
            if reports is None or reports.empty:
                print(f"[{corp_name}] {year}년 보고서를 찾을 수 없습니다.")
                return None

            # 2. 접수번호(rcept_no) 컬럼명 확인 및 추출
            # API 버전에 따라 컬럼명이 다를 수 있어 호환성 처리
            col_name = 'rcept_no' if 'rcept_no' in reports.columns else 'rcp_no'
            
            if col_name not in reports.columns:
                print(f"오류: DataFrame에 접수번호 컬럼이 없습니다. 현재 컬럼: {reports.columns}")
                return None

            # 가장 최근 보고서 선택
            target_report = reports.iloc[0]
            rcp_no = target_report[col_name]
            report_nm = target_report['report_nm']
            
            print(f"[{corp_name}] {report_nm} (접수번호: {rcp_no}) 다운로드 시작...")

            # 3. 본문 XML/텍스트 수집
            xml_text = self.dart.document(rcp_no)
            
            if not xml_text:
                print(f"[{corp_name}] 본문 내용을 가져오는 데 실패했습니다.")
                return None

            # 4. 파일 저장
            file_name = f"{corp_name}_{year}_business_report.xml"
            full_path = os.path.join(self.save_path, file_name)
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(xml_text)
                
            print(f"[{corp_name}] 수집 및 저장 완료: {full_path}")
            return full_path

        except Exception as e:
            print(f"[{corp_name}] 데이터 수집 중 치명적 오류 발생: {e}")
            traceback.print_exc()
            return None

# ==========================================
# [실행] 메인 로직
# ==========================================
if __name__ == "__main__":
    # 1. 환경 변수 로드
    load_dotenv()
    api_key = os.getenv("DART_API_KEY")
    
    # 2. 수집 대상 설정
    TARGET_CORP = "삼성전자"
    TARGET_YEAR = "2022"

    if not api_key:
        print("오류: .env 파일에서 DART_API_KEY를 찾을 수 없습니다.")
    else:
        # 수집기 초기화 (필요시 경로 변경 가능)
        # 예: collector = DartCollector(api_key, save_path="data/custom_path")
        collector = DartCollector(api_key)
        
        # 수집 실행
        collector.download_report(TARGET_CORP, TARGET_YEAR)
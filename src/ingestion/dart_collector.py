import os
import sys
import traceback
import OpenDartReader
from config import RAW_DATA_DIR, TARGET_REPORT_KIND

class DartCollector:
    def __init__(self, api_key, save_path=RAW_DATA_DIR):
        """
        save_path 기본값을 config의 RAW_DATA_DIR로 설정
        """
        self.dart = OpenDartReader(api_key)
        self.save_path = save_path
        self._ensure_directory_exists()

    def _ensure_directory_exists(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f"[Info] 저장 디렉토리 확인: {self.save_path}")

    def download_report(self, corp_name, year):
        try:
            reports = self.dart.list(
                corp_name, 
                start=f"{year}-01-01", 
                end=f"{year}-12-31", 
                kind=TARGET_REPORT_KIND # Config 변수 사용
            )
            
            if reports is None or reports.empty:
                print(f"[{corp_name}] {year}년 보고서를 찾을 수 없습니다.")
                return None

            col_name = 'rcept_no' if 'rcept_no' in reports.columns else 'rcp_no'
            target_report = reports.iloc[0]
            rcp_no = target_report[col_name]
            report_nm = target_report['report_nm']
            
            print(f"[{corp_name}] {report_nm} (접수번호: {rcp_no}) 다운로드 중...")

            xml_text = self.dart.document(rcp_no)
            
            if not xml_text:
                return None

            file_name = f"{corp_name}_{year}_business_report.xml"
            full_path = os.path.join(self.save_path, file_name)
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(xml_text)
                
            print(f"[{corp_name}] 저장 완료: {full_path}")
            return full_path

        except Exception as e:
            print(f"[{corp_name}] 수집 중 오류: {e}")
            traceback.print_exc()
            return None
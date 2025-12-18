import OpenDartReader

class DartCollector:
    def __init__(self, api_key):
        self.dart = OpenDartReader(api_key)

    def download_report(self, corp_name, year):
        # 특정 기업의 사업보고서 텍스트 혹은 재무지표 수집 로직
        pass
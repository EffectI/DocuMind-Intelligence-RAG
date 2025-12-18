import os
import re
import traceback
from bs4 import BeautifulSoup

# ==========================================
# [설정] 파싱 설정 및 상수
# ==========================================
# 저장될 기본 디렉토리
DEFAULT_OUTPUT_DIR = "data/processed/sections"

# 추출하고자 하는 핵심 섹션 키워드 리스트
TARGET_SECTION_KEYWORDS = [
    "사업의 내용", 
    "재무에 관한 사항", 
    "이사의 경영진단"
]

# 섹션 제목 패턴 (예: "II. 사업의 내용", "IV. 이사의 경영진단")
# 로마자(I, II, III...) + 점(.) + 공백 + 제목
SECTION_HEADER_PATTERN = re.compile(r'^[IVX]+\.\s*(.+)$')

# ==========================================
# [클래스] DART 섹션 파서
# ==========================================
class DartSectionParser:
    def __init__(self, output_dir=DEFAULT_OUTPUT_DIR):
        """
        :param output_dir: 파싱된 파일을 저장할 경로
        """
        self.output_dir = output_dir
        self.target_keywords = TARGET_SECTION_KEYWORDS
        self._ensure_directory_exists()

    def _ensure_directory_exists(self):
        """저장 경로 생성"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def parse_file(self, file_path):
        """
        DART XML 파일을 읽어 타겟 섹션별로 내용을 추출합니다.
        :param file_path: XML 파일 경로
        :return: {섹션명: 본문내용} 딕셔너리 (실패 시 None)
        """
        if not os.path.exists(file_path):
            print(f"오류: 파일을 찾을 수 없습니다 ({file_path})")
            return None

        try:
            print(f"파싱 시작: {os.path.basename(file_path)}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 1. BeautifulSoup을 이용한 텍스트 추출 (lxml-xml 파서 권장)
            soup = BeautifulSoup(content, 'lxml-xml')
            
            # 2. 텍스트 라인 단위로 분리
            full_text = soup.get_text(separator='\n')
            lines = [line.strip() for line in full_text.splitlines() if line.strip()]

            # 3. 섹션 분리 로직 실행
            return self._extract_sections_from_lines(lines)

        except Exception as e:
            print(f"파싱 중 오류 발생: {e}")
            traceback.print_exc()
            return None

    def _extract_sections_from_lines(self, lines):
        """텍스트 라인 리스트에서 정규식을 이용해 섹션을 분리합니다."""
        extracted_data = {}
        current_section_title = None
        buffer = []

        for line in lines:
            # 정규식으로 섹션 헤더인지 확인
            match = SECTION_HEADER_PATTERN.match(line)
            
            if match:
                # [이전 섹션 저장] 버퍼에 내용이 있고, 타겟 섹션이었던 경우 저장
                if current_section_title and buffer:
                    extracted_data[current_section_title] = "\n".join(buffer)

                # [새 섹션 시작]
                raw_title = match.group(1).strip()  # 예: "사업의 내용"
                
                # 타겟 키워드가 포함된 섹션인지 확인
                is_target = any(keyword in raw_title for keyword in self.target_keywords)
                
                if is_target:
                    print(f"   -> 타겟 섹션 발견: {line}")
                    current_section_title = raw_title
                    buffer = []  # 버퍼 초기화
                else:
                    current_section_title = None # 타겟이 아니면 무시

            elif current_section_title:
                # 현재 타겟 섹션 내부를 읽는 중이라면 버퍼에 추가
                buffer.append(line)

        # [마지막 섹션 저장] 루프 종료 후 남은 버퍼 처리
        if current_section_title and buffer:
            extracted_data[current_section_title] = "\n".join(buffer)

        return extracted_data

    def save_sections(self, extracted_data, original_filename):
        """
        추출된 데이터를 개별 Markdown 파일로 저장합니다.
        :return: 저장된 파일 경로 리스트
        """
        if not extracted_data:
            print("저장할 데이터가 없습니다.")
            return []

        base_name = os.path.splitext(original_filename)[0]
        saved_files = []

        for section_title, content in extracted_data.items():
            # 파일명에 쓸 수 없는 특수문자 제거
            safe_title = re.sub(r'[\\/*?:"<>|]', "", section_title)
            file_name = f"{base_name}_{safe_title}.md"
            save_path = os.path.join(self.output_dir, file_name)
            
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    # Markdown 헤더 추가
                    f.write(f"# {section_title}\n\n")
                    f.write(content)
                
                saved_files.append(save_path)
                print(f"저장 완료: {save_path}")
            
            except IOError as e:
                print(f"파일 저장 실패 ({save_path}): {e}")

        return saved_files

# ==========================================
# [실행] 메인 로직
# ==========================================
if __name__ == "__main__":
    # 1. 파서 초기화
    parser = DartSectionParser()
    
    # 2. 테스트 대상 파일 설정 (앞서 수집한 파일 경로로 수정하세요)
    # 예: data/raw/dart/삼성전자_2024_business_report.xml
    TARGET_FILE_PATH = "data/raw/dart/삼성전자_2024_business_report.xml"
    
    # 3. 실행
    if os.path.exists(TARGET_FILE_PATH):
        results = parser.parse_file(TARGET_FILE_PATH)
        
        if results:
            parser.save_sections(results, os.path.basename(TARGET_FILE_PATH))
            print("\n모든 작업이 완료되었습니다.")
        else:
            print("\n추출된 섹션이 없습니다. 정규식 패턴이나 파일 내용을 확인하세요.")
    else:
        print(f"테스트 파일을 찾을 수 없습니다: {TARGET_FILE_PATH}")
        print("먼저 dart_collector.py를 실행하여 데이터를 수집해주세요.")
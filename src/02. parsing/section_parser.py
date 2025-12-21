import os
import re
import traceback
from bs4 import BeautifulSoup

# ==============================================================================
# [1] 경로 설정 (Project Root 기준)
# ==============================================================================
# 1. 현재 파일(dart_parser.py)의 위치
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 프로젝트 루트 (Project Root): .../DocuMind-Intelligence-RAG
# 부모(src) -> 부모의 부모(Project Root)로 이동
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE_DIR))

# [확인용]
print(f"[Info] Project Root: {BASE_DIR}")

# ==============================================================================
# [2] 사용자 설정 (Configuration)
# ==============================================================================
PATH_DATA_ROOT = os.path.join(BASE_DIR, "data")
PATH_RAW_DART_DIR = os.path.join(PATH_DATA_ROOT, "raw", "dart")        
PATH_OUTPUT_DIR = os.path.join(PATH_DATA_ROOT, "processed", "sections") 

# 테스트 대상 파일명 (테스트 시 변경 가능)
TARGET_FILENAME = "삼성전자_2024_business_report.xml"

# 추출할 섹션(챕터) 키워드 리스트
TARGET_SECTION_KEYWORDS = [
    "사업의 내용", 
    "재무에 관한 사항", 
    "이사의 경영진단"
]

# 정규식 패턴 (Section Header): 로마자+점+공백 (예: "II. 사업의 내용")
REGEX_SECTION_HEADER = re.compile(r'^[IVX]+\.\s*(.+)$')

# ==============================================================================
# [3] 핵심 로직 클래스
# ==============================================================================
class DartIntegratedParser:
    def __init__(self, output_dir=PATH_OUTPUT_DIR):
        """
        :param output_dir: 처리된 마크다운 파일이 저장될 경로
        """
        self.output_dir = output_dir
        self._ensure_directory_exists()

    def _ensure_directory_exists(self):
        """저장 경로 생성"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[Info] 출력 디렉토리 확인: {self.output_dir}")

    def parse_file(self, xml_path):
        """
        XML 파일을 읽어 메타데이터 추출 -> 표 변환 -> 섹션 분리 -> 표준화된 이름으로 저장
        """
        if not os.path.exists(xml_path):
            print(f"[Error] 파일을 찾을 수 없습니다: {xml_path}")
            return

        print(f"\n[Process Start] 파싱 대상: {os.path.basename(xml_path)}")
        
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                # lxml-xml 파서 사용 (속도 및 정확도 우수)
                soup = BeautifulSoup(f, 'lxml-xml') 

            # ---------------------------------------------------------
            # 1. [메타데이터 추출] 파일명이 아닌 '내용'에서 회사/연도 추출
            # ---------------------------------------------------------
            real_meta = self._extract_real_metadata(soup, xml_path)
            print(f" -> [메타데이터 확보] 회사: {real_meta['company']} / 연도: {real_meta['year']}")

            # ---------------------------------------------------------
            # 2. [전처리] 표(Table)를 Markdown 텍스트로 변환
            # ---------------------------------------------------------
            self._convert_tables_to_markdown(soup)

            # ---------------------------------------------------------
            # 3. [본문 추출] 텍스트 추출 및 라인 분리
            # ---------------------------------------------------------
            body_text = soup.get_text(separator='\n')
            lines = [line.strip() for line in body_text.splitlines() if line.strip()]

            # ---------------------------------------------------------
            # 4. [섹션 분할] 정규식 기반 챕터 나누기
            # ---------------------------------------------------------
            sections = self._split_sections(lines)
            
            # ---------------------------------------------------------
            # 5. [저장] 추출된 메타데이터를 이용해 표준화된 이름으로 저장
            # ---------------------------------------------------------
            self._save_sections(sections, real_meta)

        except Exception as e:
            print(f"[Error] 파싱 중 예외 발생: {e}")
            traceback.print_exc()

    def _extract_real_metadata(self, soup, original_filepath):
        """
        XML 내부 태그를 분석하여 정확한 회사명과 연도를 추출합니다.
        실패 시 파일명에서 정보를 가져옵니다(Fallback).
        """
        meta = {"company": "UnknownCorp", "year": "UnknownYear"}
        
        # [1] 회사명 추출 (<COMPANY-NAME> 태그)
        company_tag = soup.find("COMPANY-NAME")
        if company_tag:
            meta["company"] = company_tag.get_text(strip=True)
        else:
            # 태그 없으면 파일명에서 추출 시도
            filename = os.path.basename(original_filepath)
            if "_" in filename:
                meta["company"] = filename.split("_")[0]

        # [2] 연도 추출 (<FORMULA-VERSION ADATE="..."> 태그)
        # ADATE="20240312" 형식에서 앞 4자리 추출
        formula_tag = soup.find("FORMULA-VERSION")
        if formula_tag and formula_tag.has_attr("ADATE"):
            adate = formula_tag["ADATE"]
            if len(adate) >= 4:
                meta["year"] = adate[:4]
        else:
            # 태그 없으면 파일명에서 4자리 숫자 추출 시도
            filename = os.path.basename(original_filepath)
            year_match = re.search(r'20\d{2}', filename)
            if year_match:
                meta["year"] = year_match.group(0)

        return meta

    def _convert_tables_to_markdown(self, soup):
        """
        <TABLE> 태그를 찾아서 Markdown 표 포맷으로 텍스트 치환
        """
        # 대소문자 모두 대응
        tables = soup.find_all(["table", "TABLE"])

        for table in tables:
            rows = table.find_all(["tr", "TR"])
            markdown_lines = ["\n"] 

            for i, row in enumerate(rows):
                cols = row.find_all(["td", "TD", "th", "TH"])
                
                # 셀 내용 추출 및 파이프(|) 충돌 방지
                col_texts = [ele.get_text(strip=True).replace("|", "&#124;") for ele in cols]
                
                if not any(col_texts) and not col_texts: 
                    continue

                # Row 생성: | 값 | 값 |
                row_str = "| " + " | ".join(col_texts) + " |"
                markdown_lines.append(row_str)

                # Header 구분선 (첫 행 아래)
                if i == 0:
                    sep_str = "| " + " | ".join(["---"] * len(col_texts)) + " |"
                    markdown_lines.append(sep_str)
            
            markdown_lines.append("\n")
            
            # HTML Table을 Markdown Text로 교체
            table.replace_with("\n".join(markdown_lines))

    def _split_sections(self, lines):
        """
        라인들을 순회하며 정규식(로마자 헤더)에 따라 섹션을 분리
        """
        sections = {}
        current_title = "Intro"
        buffer = []

        for line in lines:
            match = REGEX_SECTION_HEADER.match(line)
            
            if match:
                # 이전 섹션 저장
                if buffer:
                    sections[current_title] = "\n".join(buffer)
                
                # 새 섹션 시작
                current_title = match.group(0).strip()
                buffer = [f"# {current_title}"] # Markdown 제목 추가
                print(f"  -> 섹션 감지: {current_title}")
            else:
                buffer.append(line)
        
        # 마지막 섹션 처리
        if buffer:
            sections[current_title] = "\n".join(buffer)
            
        return sections

    def _save_sections(self, sections, meta):
        """
        [핵심] 추출된 메타데이터를 사용하여 '표준화된 파일명'으로 저장
        파일명 형식: {회사명}_{연도}_{섹션명}.md
        """
        # 파일명에 쓸 수 없는 특수문자 제거
        safe_company = re.sub(r'[\\/*?:"<>|]', "", meta['company'])
        safe_year = meta['year']
        
        count = 0

        for title, content in sections.items():
            # 타겟 키워드가 제목에 포함되어 있는지 확인
            is_target = any(k in title for k in TARGET_SECTION_KEYWORDS)
            
            if is_target:
                safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
                
                # 표준화된 파일명 생성
                file_name = f"{safe_company}_{safe_year}_{safe_title}.md"
                save_path = os.path.join(self.output_dir, file_name)
                
                try:
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"  [Saved] {file_name} ({len(content)} chars)")
                    count += 1
                except IOError as e:
                    print(f"  [Save Failed] {save_path}: {e}")

        if count == 0:
            print("\n[Warn] 저장된 섹션이 없습니다. TARGET_SECTION_KEYWORDS를 확인하세요.")
        else:
            print(f"\n[Complete] 총 {count}개의 표준화된 섹션 파일이 생성되었습니다.")

# ==============================================================================
# [4] 실행부
# ==============================================================================
if __name__ == "__main__":
    # 경로 조립
    target_file_path = os.path.join(PATH_RAW_DART_DIR, TARGET_FILENAME)
    
    # 실행
    if os.path.exists(target_file_path):
        parser = DartIntegratedParser(output_dir=PATH_OUTPUT_DIR)
        parser.parse_file(target_file_path)
    else:
        print("=" * 60)
        print(f"[Error] 입력 파일을 찾을 수 없습니다.")
        print(f"경로: {target_file_path}")
        print("=" * 60)
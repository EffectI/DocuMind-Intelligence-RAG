import os
import sys
import re
import traceback
from bs4 import BeautifulSoup
from config import PROCESSED_DATA_DIR, TARGET_SECTION_KEYWORDS


REGEX_SECTION_HEADER = re.compile(r'^[IVX]+\.\s*(.+)$')


class DartIntegratedParser:
    def __init__(self, output_dir=PROCESSED_DATA_DIR):
        self.output_dir = output_dir
        self._ensure_directory_exists()

    def _ensure_directory_exists(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def parse_file(self, xml_path):
        if not os.path.exists(xml_path):
            print(f"[Error] 파일 없음: {xml_path}")
            return

        print(f"\n[Parsing] 대상: {os.path.basename(xml_path)}")
        
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'lxml-xml') 

            # 1. 메타데이터 추출
            real_meta = self._extract_real_metadata(soup, xml_path)
            print(f" -> [Meta] {real_meta['company']} {real_meta['year']}")

            # 2. 표 변환
            self._convert_tables_to_markdown(soup)

            # 3. 본문 추출 및 섹션 분리
            body_text = soup.get_text(separator='\n')
            lines = [line.strip() for line in body_text.splitlines() if line.strip()]
            sections = self._split_sections(lines)
            
            # 4. 저장
            self._save_sections(sections, real_meta)

        except Exception as e:
            print(f"[Error] 파싱 실패: {e}")
            traceback.print_exc()

    def _extract_real_metadata(self, soup, original_filepath):
        meta = {"company": "Unknown", "year": "Unknown"}
        
        company_tag = soup.find("COMPANY-NAME")
        if company_tag:
            meta["company"] = company_tag.get_text(strip=True)
        else:
            filename = os.path.basename(original_filepath)
            if "_" in filename: meta["company"] = filename.split("_")[0]

        formula_tag = soup.find("FORMULA-VERSION")
        if formula_tag and formula_tag.has_attr("ADATE"):
            meta["year"] = formula_tag["ADATE"][:4]
        else:
            filename = os.path.basename(original_filepath)
            year_match = re.search(r'20\d{2}', filename)
            if year_match: meta["year"] = year_match.group(0)

        return meta

    def _convert_tables_to_markdown(self, soup):
        tables = soup.find_all(["table", "TABLE"])
        for table in tables:
            rows = table.find_all(["tr", "TR"])
            md_lines = ["\n"] 
            for i, row in enumerate(rows):
                cols = row.find_all(["td", "TD", "th", "TH"])
                col_texts = [ele.get_text(strip=True).replace("|", "&#124;") for ele in cols]
                if not any(col_texts) and not col_texts: continue
                
                md_lines.append("| " + " | ".join(col_texts) + " |")
                if i == 0:
                    md_lines.append("| " + " | ".join(["---"] * len(col_texts)) + " |")
            
            md_lines.append("\n")
            table.replace_with("\n".join(md_lines))

    def _split_sections(self, lines):
        sections = {}
        curr_title = "Intro"
        buffer = []
        for line in lines:
            match = REGEX_SECTION_HEADER.match(line)
            if match:
                if buffer: sections[curr_title] = "\n".join(buffer)
                curr_title = match.group(0).strip()
                buffer = [f"# {curr_title}"]
            else:
                buffer.append(line)
        if buffer: sections[curr_title] = "\n".join(buffer)
        return sections

    def _save_sections(self, sections, meta):
        safe_company = re.sub(r'[\\/*?:"<>|]', "", meta['company'])
        safe_year = meta['year']
        count = 0

        for title, content in sections.items():
            # Config에서 가져온 키워드(TARGET_SECTION_KEYWORDS) 사용
            if any(k in title for k in TARGET_SECTION_KEYWORDS):
                safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
                fname = f"{safe_company}_{safe_year}_{safe_title}.md"
                path = os.path.join(self.output_dir, fname)
                
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                count += 1
        
        if count > 0:
            print(f" -> {count}개 섹션 저장 완료")
        else:
            print(" -> 저장된 섹션 없음 (키워드 불일치)")
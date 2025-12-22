import os
import pdfplumber
from config import PROCESSED_DATA_DIR

class PDFParser:
    def __init__(self, output_dir=PROCESSED_DATA_DIR):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def parse_file(self, pdf_path):
        """PDF 텍스트 추출 -> 마크다운 저장"""
        if not os.path.exists(pdf_path):
            return

        print(f"[Parsing] PDF 분석 시작: {os.path.basename(pdf_path)}")
        
        # 1. 텍스트 추출
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
        
        # 2. 메타데이터 (파일명에서 유추)
        filename = os.path.basename(pdf_path)
        # 예: 삼성전자_2024_IR자료.pdf -> company:삼성전자, year:2024
        # (파일명 규칙을 강제하거나, 추후 AI로 추출하는 로직 고도화 필요)
        
        # 3. 저장 (.md 파일로)
        # 파일 확장자만 .pdf -> .md로 변경
        save_name = filename.replace(".pdf", ".md").replace(".PDF", ".md")
        save_path = os.path.join(self.output_dir, save_name)
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        print(f"[Parsing] 변환 완료: {save_path}")
        return save_path
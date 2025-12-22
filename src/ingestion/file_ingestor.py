import os
import shutil
from config import RAW_DATA_DIR

class FileIngestor:
    def __init__(self, save_dir=None):
        # 기본 저장 경로: data/raw/uploads (없으면 생성)
        self.save_dir = save_dir if save_dir else os.path.join(RAW_DATA_DIR, "uploads")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_uploaded_file(self, uploaded_file):
        """
        Streamlit의 UploadedFile 객체를 받아서 디스크에 저장
        """
        try:
            # 파일명 안전하게 처리 (띄어쓰기 -> 언더바)
            file_name = uploaded_file.name.replace(" ", "_")
            file_path = os.path.join(self.save_dir, file_name)
            
            # 바이트 쓰기
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            print(f"[Ingestor] 사용자 파일 저장 완료: {file_path}")
            return file_path
        except Exception as e:
            print(f"[Error] 파일 저장 실패: {e}")
            return None
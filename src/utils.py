# src/utils.py
import re
import os

def parse_filename_meta(filename):
    """
    파일명에서 회사명과 연도를 추출합니다.
    예: '삼성전자_2024_사업보고서.pdf' -> {'company': '삼성전자', 'year': '2024'}
    """
    meta = {'company': '', 'year': ''}
    
    # 1. 연도 추출
    year_match = re.search(r'20\d{2}', filename)
    if year_match:
        meta['year'] = year_match.group(0)
    
    # 2. 회사명 추출
    name_body = os.path.splitext(filename)[0]
    if "_" in name_body:
        meta['company'] = name_body.split("_")[0]
    elif " " in name_body:
        meta['company'] = name_body.split(" ")[0]
    else:
        if meta['year']:
            meta['company'] = name_body.replace(meta['year'], "").strip()
        else:
            meta['company'] = name_body 

    return meta
import os
import json
import camelot
import pdfplumber
import pandas as pd
from pathlib import Path
import fitz
import numpy as np
# from pykospacing import Spacing
import re
from tqdm import tqdm

def extract_text_with_pdfplumber(pdf_path, pages_to_try):
    """
    pdfplumber 를 통해 ISMS pdf 를 읽는 함수
    pdf 종류, page, text, word 개수, line 개수, 이미지 개수 등을 추출
    pdf_path : pdf 가 있는 path
    pages_to_try : 이미지가 있을 때 이미지를 불러올 page 를 저장하는 list

    output : pages_info (무슨 pdf, page, 내용, word 개수, line 개수, 이미지 개수)
    """
    pages_info = []
    with pdfplumber.open(pdf_path, repair=True) as pdf:
        for pageno, page in tqdm(enumerate(pdf.pages, start=1), total=len(pdf.pages)):
            # 텍스트 추출(레이아웃 보존력 높은 방식)
            text_simple = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            # 단어/라인 단위 정보(필요 시 사용)
            words = page.extract_words()
            lines = page.lines  # 벡터 선 정보(테이블 힌트)
            images = page.images  # 이미지 메타데이터
            pages_info.append({
                "pdf": str(pdf_path).split("/")[-1][:-4],
                "page": pageno,
                "text": text_simple,
                "words_count": len(words),
                "lines_count": len(lines),
                "images_count": len(images)
            })

            if len(images) > 0:
                pages_to_try.append(pageno)
    return pages_info

def strip_special(text: str, keep_punct: str = "().,:%-/_[]") -> str:
    """
    한글, 영어, 숫자, 일부 예외 특수 기호 제외 모두 제거하는 전처리 함수
    text : 전처리할 text
    keep_punct : 지우고 싶지 않은 특수 기호들
    output : 한글, 영어, 숫자, 일부 예외 특수 기호 제외 모두 제거된 text
    """
    HANGUL = r"\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3"
    # 다양한 대시류를 하이픈으로 통일(선택)
    text = re.sub(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]", "-", str(text))
    pattern = fr"[^{HANGUL}A-Za-z0-9\s{re.escape(keep_punct)}]"
    cleaned = re.sub(pattern, " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def save_txt(JSON_PATH, txt_out_dir):
    """
    JSON 데이터를 읽고 .txt 파일로 저장하는 함수
    JSON_PATH : json 파일 path
    txt_out_dir : txt 파일을 저장할 폴더 path

    txt 로 저장하는 데이터는 전처리한 text 
    """
    with JSON_PATH.open("r", encoding="utf-8") as f:
        pages = json.load(f)

    saved = []
    for item in pages:
        pdf_name = item.get("pdf", "") or ""
        page_no = int(item.get("page", 0))  # 1-based 가정
        base_text = item.get("text", "") or ""
        base_text = strip_special(base_text)

        out_path = txt_out_dir / f"{pdf_name}_p{page_no:03d}.txt"
        out_path.write_text(base_text, encoding="utf-8")
        saved.append(str(out_path))

    print("saved files:", len(saved))

if __name__ == "__main__":
    OUT_DIR = Path("../data/ISMS")
    path_list = [OUT_DIR / i for i in os.listdir(OUT_DIR) if i.endswith(".pdf")]
    pages_to_try = []
    pages = []

    for i in path_list: 
        page = extract_text_with_pdfplumber(i, pages_to_try)
        pages.extend(page)
        
    with open(OUT_DIR / "text_pages.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    txt_out_dir = Path("../data/ISMS/merged_text")
    txt_out_dir.mkdir(parents=True, exist_ok=True)

    save_txt(OUT_DIR / "text_pages.json", txt_out_dir)
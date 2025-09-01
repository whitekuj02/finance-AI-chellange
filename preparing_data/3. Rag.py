import re
import os
import itertools
import numpy as np
import pandas as pd
from typing import Any, Dict
import random
import yaml
import argparse
import glob
from tqdm import tqdm
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.utils import DistanceStrategy
import faiss
import pdfplumber
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import warnings

def split_by_article(text):
    """
    제N조의 N항을 기준으로 법을 나누는 함수
    """
    # '제 N조' 앞에서 split
    pattern = r'(제\s*\d+조(?:의\d+)?\s*\([^)]+\)[\s\S]*?)(?=제\s*\d+조(?:의\d+)?\s*\([^)]+\)|\Z)'
    return re.findall(pattern, text)

def load_law(law_path):
    """
    법 pdf 를 pdfplumber 로 읽는 함수
    """
    pdf_path = [law_path + f"/{i}" for i in os.listdir(law_path)]

    pdf_text = {
        i : "" for i in os.listdir(law_path)
    }

    for i, key in zip(pdf_path, pdf_text.keys()):
        text = ""
        with pdfplumber.open(i) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n" 
        
        pdf_text[key] = text

    pdf_split_data = []

    for article, text in pdf_text.items():
        split_text = split_by_article(text)
        for i in split_text:
            pdf_split_data.append(f"{article[:-4]} {i}")

    return pdf_split_data

def load_tech(tech_path):
    """
    보안 객관 주관 데이터를 읽는 함수 (미리 만들어 둔 CSV)
    """
    tech_data = pd.read_csv(tech_path)
    tech_array = list(tech_data['data'])

    return tech_array

def load_isms(isms_path):
    """
    ISMS 데이터를 읽는 함수 (미리 만들어 둔 CSV)
    """
    ISMS_path = Path(isms_path)
    ISMS_list = [ISMS_path / Path(f"{i}") for i in os.listdir(ISMS_path)]

    ISMS_text = []
    for i in ISMS_list:
        text = i.read_text(encoding="utf-8")
        ISMS_text.append(text)

    return ISMS_text

def recursive_split(pdf_all):
    """
    RecursiveCharacterTextSplitter 를 통해 text 데이터를 Chunk로 나누는 함수
    일부 후처리도 포함 : <> 나 [] 로 덮여있는 데이터 삭제
    """
    pdf_split_data_recursive = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    for i in pdf_all:
        clean_text = re.sub(r'\s+', ' ', i).strip()
        clean_text = re.sub(r"[<\[].*?[>\]]", "", clean_text).strip()
        chunks = splitter.split_text(clean_text)
        pdf_split_data_recursive.extend(chunks)

    pdf_split_data_recursive = [re.sub(r"[<\[].*?[>\]]", "", text).strip() for text in pdf_split_data_recursive]
    return pdf_split_data_recursive

if __name__ == "__main__":
    pdf_law = load_law("../data/law")
    pdf_tech = load_tech("../data/tech/tech_data.csv")
    pdf_isms = load_isms("../data/ISMS/merged_text")

    pdf_all = []
    pdf_all.extend(pdf_law)
    pdf_all.extend(pdf_tech)
    pdf_all.extend(pdf_isms)

    pdf_split_data = [i for i in pdf_all if len(i) > 100]
    pdf_split_data = recursive_split(pdf_split_data)

    embd_path = "./model/gte-multilingual-base"
    embedding = HuggingFaceEmbeddings(
        model_name=embd_path,
        model_kwargs={"device": "cuda", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},  # cosine에 필수 권장
    )


    # 문서 객체 + 메타데이터(doc_id)
    docs = [Document(page_content=t, metadata={"doc_id": i}) for i, t in enumerate(pdf_split_data)]

    # Dense(FAISS)
    vector_store = FAISS.from_documents(docs, embedding, distance_strategy=DistanceStrategy.COSINE)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    save_path = "../model/faiss_index"
    vector_store.save_local(save_path)
# models_runtime.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import threading
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
    AutoModelForSequenceClassification,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import numpy as np
import re
from collections import Counter
import math

# --- 전역 캐시 & 락 ---
_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()

@dataclass
class ModelBundle:
    retriever: FAISS
    embedding: HuggingFaceEmbeddings
    reranker_tokenizer: AutoTokenizer
    reranker_model: AutoModelForSequenceClassification
    llm_generate_config: GenerationConfig
    llm_model: AutoModelForCausalLM
    llm_tokenizer: AutoTokenizer

def all_model_load(base_path: str | Path) -> None:
    """
    같은 base_path로는 딱 한 번만 로드되어 전역 캐시에 저장됩니다.
    이후 infer()에서 재사용합니다.
    """
    key = str(Path(base_path).resolve())
    if key in _MODEL_CACHE:  # 이미 로드됨
        return

    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return

        base = Path(base_path)
        embd_id = base / "gte-multilingual-base"
        reranker_id = base / "bge-reranker-v2-m3"
        model_id = base / "midm2_bf16"

        # 1) Embedding (CPU)
        embedding = HuggingFaceEmbeddings(
            model_name=str(embd_id),
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 2) Reranker
        reranker_tokenizer = AutoTokenizer.from_pretrained(str(reranker_id))
        reranker_model = AutoModelForSequenceClassification.from_pretrained(str(reranker_id))

        # 3) LLM (BF16 / device_map="auto")
        llm_model = AutoModelForCausalLM.from_pretrained(
            str(model_id),
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        llm_tokenizer = AutoTokenizer.from_pretrained(str(model_id))
        # (선택) generation_config를 폴더에 저장해두었다면 이렇게 불러와도 됩니다:
        generation_config = GenerationConfig.from_pretrained(str(model_id))

        faiss_path = base / "faiss_index"
        loaded_vector_store = FAISS.load_local(
            faiss_path,
            embedding,
            allow_dangerous_deserialization=True  # 필수 (pickle 사용)
        )

        retriever = loaded_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})


        _MODEL_CACHE[key] = ModelBundle(
            retriever=retriever,
            embedding=embedding,
            reranker_tokenizer=reranker_tokenizer,
            reranker_model=reranker_model,
            llm_generate_config=generation_config,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
        )

def _get_models(base_path: str | Path) -> ModelBundle:
    key = str(Path(base_path).resolve())
    if key not in _MODEL_CACHE:
        # 필요 시 지연 로드(또는 에러로 바꿔도 됨)
        all_model_load(key)
    return _MODEL_CACHE[key]

# (선택) 메모리 해제용 유틸
def unload_models(base_path: str | Path | None = None) -> None:
    """
    특정 base_path만 제거하거나, None이면 전부 제거.
    GPU 메모리 해제까지 수행.
    """
    import gc
    if base_path is None:
        _MODEL_CACHE.clear()
    else:
        key = str(Path(base_path).resolve())
        _MODEL_CACHE.pop(key, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def exp_normalize(x):
    return 1 / (1 + np.exp(-x))

# 객관식 여부 판단 함수
def is_multiple_choice(question_text):
    """
    객관식 여부를 판단: 2개 이상의 숫자 선택지가 줄 단위로 존재할 경우 객관식으로 간주
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2


# 질문과 선택지 분리 함수
def extract_question_and_choices(full_text):
    """
    전체 질문 문자열에서 질문 본문과 선택지 리스트를 분리
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []

    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    
    question = " ".join(q_lines)
    return question, options

# 후처리 함수
def extract_answer_only(text: str, original_question: str) -> str:
    """
    LLM 정답 아웃풋 후처리 함수
    객관식 : 점수만 추출
    주관식 : 그대로 통과
    """
    # 공백 또는 빈 문자열일 경우 기본값 지정
    if not text:
        return "미응답"

    # 객관식 여부 판단
    is_mc = is_multiple_choice(original_question)

    if is_mc:
        # 숫자만 추출
        match = re.match(r"\D*([1-9][0-9]?)", text)
        if match:
            return match.group(1)
        else:
            # 숫자 추출 실패 시 "0" 반환
            return "0"
    else:
        return text
    
def make_prompt_auto(text, rag):
    """
    프롬포트 함수
    """

    # RAG 문서 안내
    if rag != "":
        rag_prompt = (
            "다음은 질문과 관련된 문서의 일부입니다.\n"
            f"[문서 정보]\n{rag}\n\n"
        )
    else:
        rag_prompt = ""
        
    # 객관식
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        numbered_options = [f"{i+1}. {opt}" for i, opt in enumerate(options)]

        prompt = (
            "다음 객관식 질문에 대해 **가장 적절한 선택지의 번호**를 정답으로 작성하세요. (1,2,3,4 or 5)\n\n"
            "출력 형식:\n정답: 번호(번호만)\n\n"
            f"{rag_prompt}"
            f"[질문]\n{question.strip()}\n\n"
            "[선택지]\n" + "\n".join(numbered_options) + "\n\n"
            "정답:"
        )

    # 주관식
    else:
        prompt = (
            "다음 주관식 질문에 대해 **정확한 정답**을 주요 키워드를 활용하여 작성하시오.\n"
            "정답은 정답에 사용할 키워드를 뽑아본 후 이를 활용하여 3문장 내로 서술하시오.\n"
            "출력 형식:\n키워드: 키워드\n정답: 내용\n\n"
            f"{rag_prompt}"
            f"[질문]\n{text.strip()}\n\n"
        )

    # 시스템 프롬프트
    system_prompt = (
        "당신은 한국의 금융 보안 및 법률에 정통한 전문가입니다.\n"
        "출력 형식을 반드시 지키세요.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return messages

def inference(question, doc, score, tokenizer, model, generation_config):
    """
    모델 inference 함수
    """

    rag = doc if score > 0.75 else ""

    input_message = make_prompt_auto(question, rag)
    # print(input_message)

    text_prompt = tokenizer.apply_chat_template(
        input_message,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True
    )

    model_inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)
    model_inputs.pop("token_type_ids", None)

    generated_ids = model.generate(
        **model_inputs,
        generation_config=generation_config,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    text = content.split("정답:")[-1].strip()
    reason = content.split("정답:")[0].strip()

    return reason, text


def infer(question: str, base_path: str | Path = "./model") -> str:
    """
    다른 파일에서 바로 infer(question)만 호출해도 되도록,
    내부에서 전역 캐시된 모델을 꺼내 씁니다.
    """
    bundle = _get_models(base_path)

    retriever=bundle.retriever
    embedding=bundle.embedding
    reranker_tokenizer=bundle.reranker_tokenizer
    reranker_model=bundle.reranker_model
    llm_generate_config=bundle.llm_generate_config
    llm_model=bundle.llm_model
    llm_tokenizer=bundle.llm_tokenizer

    # 1. retriver
    docs_obj = retriever.get_relevant_documents(question)
    docs = [d.page_content for d in docs_obj]

    # 2. reranker
    inputs = reranker_tokenizer(docs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
    scores = exp_normalize(scores.detach().numpy())

    # 3. 최고 스코어 문서 선택
    best_idx = int(np.argmax(scores))
    best_doc = docs[best_idx]
    best_score = float(scores[best_idx])

    # 4. SCD inference
    answer_candidates = []
    reasoning_candidates = []

    for _ in range(5):  # 반복 횟수 조절 가능
        thinking_content, content = inference(question, best_doc, best_score, llm_tokenizer, llm_model, llm_generate_config)
        pred_answer = extract_answer_only(content, original_question=question)

        answer_candidates.append(pred_answer)
        reasoning_candidates.append(thinking_content)

    is_mc = is_multiple_choice(question)


    if is_mc:
        # ✅ 객관식 → majority voting
        final_answer = Counter(answer_candidates).most_common(1)[0][0]
        matched_reasoning = next(
            (r for a, r in zip(answer_candidates, reasoning_candidates) if a == final_answer),
            reasoning_candidates[0]
        )

    else:
        def extract_keywords_from_text(t: str) -> list[str]:
            m = re.search(r"키워드\s*:\s*(.+)", t)
            if not m:
                return []
            return [k.strip().lower() for k in m.group(1).split(",") if k.strip()]

        # 1) 키워드 추출
        keywords_list = [extract_keywords_from_text(i) for i in reasoning_candidates]
        keywords_list = [[k for k in kws if k] for kws in keywords_list]

        # 2) df/idf 계산 (코퍼스=후보들)
        N = len(keywords_list)
        all_unique = sorted({k for kws in keywords_list for k in set(kws)})

        # df: 몇 개의 후보가 이 키워드를 갖고 있는가?
        df = {k: sum(1 for kws in keywords_list if k in set(kws)) for k in all_unique}

        # idf: log 스무딩
        idf = {k: math.log((N + 1) / (df[k] + 1)) + 1.0 for k in all_unique}
        max_idf = max(idf.values()) if idf else 1.0

        def keyword_weight(k: str, alpha: float = 0.7) -> float:
            """합의(consensus) + (너무 흔한 키워드 패널티) 를 섞은 하이브리드 가중치"""
            if k not in df:
                return 0.0
            # 합의도 (본인 제외 0~1)
            cf = (df[k] - 1) / max(1, N - 1)
            # IDF 정규화 (희소할수록 1)
            idf_norm = (idf[k] - 1.0) / max(1e-8, (max_idf - 1.0))
            # 흔함 점수 = 1 - idf_norm (너무 흔하면 1에 가깝지 않음)
            commonness = 1.0 - idf_norm
            return alpha * cf + (1 - alpha) * commonness

        # 3) 후보 점수 계산
        def score_candidate(kws: list[str], alpha: float = 0.7) -> float:
            # 중복 제거 후 각 키워드 가중치 합
            uniq = set(kws)
            return sum(keyword_weight(k, alpha=alpha) for k in uniq)

        scores = [score_candidate(kws, alpha=0.7) for kws in keywords_list]

        # 4) tie-breaker: (a) 기존 교집합 수, (b) 키워드 개수 많은 쪽
        def overlap_score(i: int) -> int:
            others = set().union(*[set(x) for j, x in enumerate(keywords_list) if j != i])
            return len(set(keywords_list[i]) & others)

        best_idx = max(
            range(len(scores)),
            key=lambda i: (scores[i], overlap_score(i), len(set(keywords_list[i])))
        )

        print("best_idx:", best_idx)
        print("chosen keywords:", keywords_list[best_idx])
        print("hybrid score:", scores[best_idx])

        final_answer = answer_candidates[best_idx]
        matched_reasoning = reasoning_candidates[best_idx]

    return final_answer
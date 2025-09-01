from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

embd_id = "Alibaba-NLP/gte-multilingual-base"
reranker_id = "BAAI/bge-reranker-v2-m3"
model_id = "K-intelligence/Midm-2.0-Base-Instruct"

embedding = HuggingFaceEmbeddings(
    model_name=embd_id,
    model_kwargs={"device": "cpu", "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},  # cosine에 필수 권장
)

reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_id)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_id)

llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

llm_tokenizer = AutoTokenizer.from_pretrained(model_id)

# ==== 저장 폴더 설정 ====
root = Path("./")
EMB_DIR = root / "gte-multilingual-base"
RER_DIR = root / "bge-reranker-v2-m3"
LLM_DIR = root / "midm2_bf16"

for d in [EMB_DIR, RER_DIR, LLM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==== 1) Embedding (LangChain 래퍼 → 내부 ST 모델 저장) ====
# LangChain의 HuggingFaceEmbeddings 내부는 SentenceTransformer 객체를 씁니다.
# 보유 중인 `embedding` 객체에서 직접 저장 시도, 실패 시 fallback로 ST 재로드 후 저장.
try:
    if hasattr(embedding, "client") and hasattr(embedding.client, "save"):
        embedding.client.save(str(EMB_DIR))  # SentenceTransformer.save()
    elif hasattr(embedding, "model") and hasattr(embedding.model, "save_pretrained"):
        embedding.model.save_pretrained(str(EMB_DIR))
    else:
        # 확실히 저장되도록 fallback
        SentenceTransformer(embd_id).save(str(EMB_DIR))
except Exception as e:
    print(f"[WARN] LangChain 임베딩 저장 실패, fallback 사용: {e}")
    SentenceTransformer(embd_id).save(str(EMB_DIR))
print(f"✅ Embedding saved to: {EMB_DIR}")

# ==== 2) Reranker ====
reranker_model.save_pretrained(RER_DIR, safe_serialization=True)
reranker_tokenizer.save_pretrained(RER_DIR)
print(f"✅ Reranker saved to: {RER_DIR}")

# ==== 3) LLM (BF16) ====
llm_model.save_pretrained(LLM_DIR, safe_serialization=True)
llm_tokenizer.save_pretrained(LLM_DIR)
GenerationConfig.from_pretrained(model_id).save_pretrained(LLM_DIR)
print(f"✅ LLM saved to: {LLM_DIR}")


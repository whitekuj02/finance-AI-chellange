import json
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

def process_trendyol(trendyol_name: str):
    ds = load_dataset(trendyol_name, split="train")
    for rec in tqdm(ds, desc="Trendyol→JSONL"):
        q = (rec.get("user") or "").strip()
        a = (rec.get("assistant") or "").strip()
        if q and a:
            yield {"question": q, "answer": a}

def process_cybermetric(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 최상위가 {"questions": [...]} 형식
    items = data.get("questions", [])
    for it in tqdm(items, desc="CyberMetric→JSONL"):
        q = (it.get("question") or "").strip()

        # answers가 dict로 { "A": "text", ... }
        answers = it.get("answers") or {}
        # key 대소문자 혼용 방지
        answers_norm = {str(k).upper(): v for k, v in answers.items()}

        sol = it.get("solution")
        # 정답 키(문자) → 텍스트
        ans_text = ""
        if isinstance(sol, str):
            sol_u = sol.strip().upper()
            ans_text = (answers_norm.get(sol_u) or sol.strip())
        elif isinstance(sol, int):
            # 숫자면 A=0, B=1…로 가정하여 인덱싱
            letters = sorted(answers_norm.keys())
            if 0 <= sol < len(letters):
                ans_text = answers_norm[letters[sol]]
        elif isinstance(sol, dict):
            # 혹시 {"label": "D"} 같은 형식
            for k in ("label", "answer", "value", "text"):
                if k in sol:
                    sol_u = str(sol[k]).strip().upper()
                    ans_text = answers_norm.get(sol_u, str(sol[k]).strip())
                    break

        # 백업: 정답 키가 없고 텍스트로 들어온 경우
        if not ans_text and isinstance(answers, dict) and len(answers) == 1:
            ans_text = next(iter(answers.values()))

        if q and ans_text:
            yield {"question": q, "answer": str(ans_text).strip()}

def save_jsonl(records, out_path: str):
    n = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"Saved {n} lines → {out_path}")

def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data

def save_csv(data, save_file_name):
    rag_data = []

    for d in tqdm(data, total=len(data), desc="csv processing"):
        question = d['question']
        answer = d['answer']

        full_text = question + " " + answer

        rag_data.append(full_text)

    result = pd.DataFrame({
        "data" : rag_data
    })

    result.to_csv(save_file_name, index=False)

if __name__ == "__main__":
    base_path = "../data/tech"
    trendyol_name = "../data/tech/Trendyol-Cybersecurity"
    cybermetric_path = f"{base_path}/CyberMetric/CyberMetric-10000-v1.json"
    out_path = f"{base_path}/combined.jsonl"
    csv_out_path = f"{base_path}/tech_data.csv"

    # 순서: Trendyol → CyberMetric
    all_records = []
    all_records.extend(process_trendyol(trendyol_name))
    all_records.extend(process_cybermetric(cybermetric_path))
    save_jsonl(all_records, out_path)

    data = read_jsonl(out_path)
    save_csv(data, csv_out_path)

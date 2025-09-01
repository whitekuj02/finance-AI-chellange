from huggingface_hub import snapshot_download

# 원하는 폴더 지정
local_dir = "./Trendyol-Cybersecurity"

snapshot_download(
    repo_id="Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
    repo_type="dataset",
    local_dir=local_dir
)

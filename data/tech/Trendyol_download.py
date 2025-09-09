from huggingface_hub import snapshot_download


def main():
    """
    주관식 Trendyol-Cybersecurity 데이터를 huggingface 에서 다운 받아 ./Trendyol-Cybersecurity 에 저장 하는 함수
    객관식인 CyberMetric 또한 huggingface 에서 다운로드 가능하지만 github 에서 clone 하여 사용하였음.
    """
    # 원하는 폴더 지정
    local_dir = "./Trendyol-Cybersecurity"

    snapshot_download(
        repo_id="Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
        repo_type="dataset",
        local_dir=local_dir
    )

if __name__ == "__main__":
    main()
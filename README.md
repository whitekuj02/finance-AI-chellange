# Finance-AI-chellange
![panel](./asset/panel.png)

# 2025 금융 AI Challenge : 금융 AI 모델 경쟁
- 주제 : 금융보안 실무에 적합한 개발 방법론 및 AI 모델을 발굴하기 위해 FSKU 평가지표를 기반으로 AI 모델의 성능을 경쟁
- 기간 : 2025.08.01 ~ 2025.08.29
- **Private Score : 0.67721 (1st)**

<br />

## TEAM 뛰어 🧑‍💻👩‍💻

| 이상혁 | 김의진 | 장희진 | 정승민 |
| :---: | :---: | :---: | :---: |
| <img src="https://avatars.githubusercontent.com/u/110239629?v=4" width=300> | <img src="https://avatars.githubusercontent.com/u/94896197?v=4" width=300> | <img src="https://avatars.githubusercontent.com/u/105128163?v=4" width=300> | <img src="https://avatars.githubusercontent.com/u/105360496?v=4" width=300> |

<br />

## Library

- 자세한 라이브러리는 ./environment 참고

<br />

## Pipeline
- **RAG system**
  
![RAG_system](./asset/RAG_system.png)

- **LLM system**
  
![LLM_system](./asset/LLM_system.png)

<br />

## Code

- Model download

```bash
python ./model/model_download.py
```

- Preprocessing

```bash
python ./preparing_data/1.\ tech.py
python ./preparing_data/2.\ ISMS.py
python ./preparing_data/3.\ Rag.py
```

- Streamlit

```bash
streamlit run streamlit_app.py
```

- original --> test.csv inference
```bash
original_code/1.\ RAG.ipynb
original_code/2.\ Inference.ipynb
```
<br />

## Model
- RAG 임베딩 모델 : "Alibaba-NLP/gte-multilingual-base" (https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
- Reranker 모델 : "BAAI/bge-reranker-v2-m3" (https://huggingface.co/BAAI/bge-reranker-v2-m3)
- LLM 모델 : "K-intelligence/Midm-2.0-Base-Instruct" (https://huggingface.co/K-intelligence/Midm-2.0-Base-Instruct)

<br />

## Dataset
- ISMS PDF : https://isms.kisa.or.kr/main/ispims/notice/ (KISA 한국인터넷진흥원)
- Law PDF : https://www.law.go.kr/ (국가법령정보센터)
- Tech dataset (subjective) : Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset (https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset)
- Tech dataset (multiple choice) : https://github.com/cybermetric/CyberMetric (Github) / https://huggingface.co/datasets/tihanyin/CyberMetric (Huggingface)


## 요기 있습니다.
- 외부 문서 및 금융보안 지식베이스 구축 방법 및 구조 상세 설명
- 임베딩 모델 및 벡터 DB 구성 방식, 문서 검색 및 지식 추출 과정 기술

1. RAG 데이터 셋 선정 과정
    - 금융보안 질문의 유형은 크게 2가지로 나눌 수 있었음
    - 1) 금융 관련 법, 2) 정보 보안 기술
    - 다음과 같은 2개의 domain 에서 RAG document를 search
    - 1) 금융 관련 법
        - 법의 참고 문서는 ISMS-P 인증기준 안내서(2023.11.23.).pdf 의 46p 에서 찾을 수 있었음 + 또 다른 정보보안 관련 법들을 찾아서 법 PDF 모음집을 구축
        - test 데이터와 비슷한 domain 으로 예측되어 KISA의 안내서 3개를 RAG 데이터로 사용 ( ISMS-P 인증기준 안내서, ISMS-P 인증제도 안내서, KISA_클라우드서비스_보안인증제도_안내서 )
        - ** ISMS-P 인증기준 안내서는 공공누리집 4유형으로 인용과 chunking 은 가능한 라이센스, 법 PDF 는 국가법령정보센터에서 다운로드 **
        - 법 목록 : 가상자산 이용자 보호 등에 관한 법률, 개인정보 보호법, 금융 기관 검사 및 제재에 관한 규정, 금융위원회의 설치 등에 관한 법률, 금융회사의 정보처리 업무 위탁에 관한 규정, 부정 경쟁 방지 및 영업 비밀 보호에 관한 법률, 사이버 안보 업무 규정, 산업기술의 유출방지 및 보호에 관한 법률, 산업표준화법, 소득세법, 소프트웨어 진흥법, 소프트웨어 진흥법 시행령, 신용정보의 이용 및 보호에 관한 법률, 여신전문금융업법, 위치정보의 보호 및 이용 등에 관한 법률, 은행법, 저작권법, 전기통신사업법, 전자금융감독규정, 전자금융거래법, 전자금융거래법 시행령, 전자문서 및 전자거래 기본법, 전자상거래 등에서의 소비자보호에 관한 법률, 전자서명법, 전자정부법, 정보보호 관리 등급 부여에 관한 고시, 정보보호산업의 진흥에 관한 법률, 정보보호산업의 진흥에 관한 법률 시행규칙, 정보통신기반 보호법, 정보통신망 이용촉진 및 정보보호 등에 관한 법률, 클라우드컴퓨팅 발전 및 이용자 보호에 관한 법률, 통신비밀보호법, 특정 금융거래정보의 보고 및 이용 등에 관한 법률 
    - 2) 정보 보안 기술
        - 한글로 된 정보 보안 기술 데이터 셋은 찾을 수 없었음
        - 영어로 확장하여 search 해본 결과 다음과 같은 2개의 데이터 셋을 찾을 수 있었음
        - 1. Trendyol-Cybersecurity-Instruction-Tuning-Dataset, 2. cybermetric
        - 1번은 주관식 QA 허깅페이스 데이터셋, 2번은 객관식 QA 허깅페이스 데이터셋임

2. 데이터 셋 전처리 과정
    - 전처리 과정은 3가지의 경우로 설명
    - 1) KISA PDF
    - 특징 : 안내서로 표, 그림, 아이콘 등 다양한 요소들이 있는 PDF, text에 noise 가 있을 수 있음
    - 방법 : 먼저 PDFplumber 를 통해 text 를 모두 읽고 각 page 마다 대시류 통일 후에 한글, 영어, 숫자, 일부 예외 특수 기호를 제외한 모든 문자를 제거 --> 정보 손실이 있지만 noise 를 줄여 의미있는 정보들을 살리는 방향
    - 2) 법 PDF
    - 특징 : 제N조의 N항()으로 text 의 형태가 정형화 되어있음
    - 방법 : 제N조의 N항()을 기준으로 모든 항 분리 후 앞에 법의 이름을 붙힘 ex) 전자금융거래법 제1조(목적) 이 법은 전자금융거래의 ...
    - 3) 보안 허깅페이스 데이터 셋
    - 특징 : 따로 노이즈는 없는 깔끔한 text 데이터 셋
    - 방법 : 주관식, 객관식 모두 질문 + 정답의 형태로 질문과 정답을 하나의 text 로 이어서 제작. 객관식의 경우 오답은 사용하지 않음.

3. Chunk 구성 과정
    - 1. 짧은 text 들은 사용하지 않음 : 리트리버의 similarity 계산에서 방해를 줌, 100 자 이상으로 필터링
    - 2. RecursiveCharacterTextSplitter로 Chunk split : size 2000, overlap 500 기준으로 \n\n, \n, “ ”, “” 을 separators 로 구성
    - 3. [], <> 등의 필요 없는 설명 부분 ( 주로 법 ) 부분 삭제 후처리

4. RAG
    - 1. Embedding 모델은 한, 영 모두를 고려 할 수 있는 model 선정 : 
        Alibaba-NLP/gte-multilingual-base, normalize_embeddings 사용
    - 2. FAISS vectorDB 사용 : similarity, k는 10

5. Reranking
    - 1. Raranking model 또한 한, 영 모두를 고려 할 수 있는 model 선정 :
        BAAI/bge-reranker-v2-m3
    - 2. query 에 대해 retriver 로 나온 10 개의 chunk 에 대해 모두 Reranking, logit 처리는 sigmoid --> 0~1 사이의 score 가 나오고 이 때 가장 높은 score 를 달성한 하나의 chunk만 사용
    - 3. score threshold 0.75 를 넘지 못한다면 RAG Chunk를 사용하지 않음
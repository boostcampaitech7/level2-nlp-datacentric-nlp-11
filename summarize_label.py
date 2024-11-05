import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from tqdm import tqdm
import re
from langchain_ollama import OllamaLLM

# CSV 파일 경로
input_csv_path = (
    "/data/ephemeral/home/level2-nlp-datacentric-nlp-11/label_category_counts.csv"
)

# CSV 파일 읽기
df = pd.read_csv(input_csv_path)

# 라벨과 정규화된 카테고리별로 출현 횟수 계산

# LangChain 프롬프트 템플릿 설정
prompt_label_sum = ChatPromptTemplate.from_template(
    """
### Prompt:

아래에 label과 category, counts 정보를 보고 해당 label이 어떤 category를 의미하는지를 하나의 category로 정의해주세요.

1. 하나의 label엔 하나의 category만 존재해야합니다.
2. 최종 출력의 형식을 엄격히 지켜서 출력해주세요.

### 입력 데이터(label, category, counts)

{input}

### 최종 출력 형식:

label, category
"""
)

# LangChain LLM 설정 (Ollama)
llm_label_sum = OllamaLLM(
    model="gemma2:27b", temperature=0.1
)  # 실제 모델 이름으로 변경하세요
chain_label_sum = prompt_label_sum | llm_label_sum

# 결과를 저장할 리스트 초기화
label_summary_list = []

# 라벨 목록 추출 (0부터 6까지)
labels = range(0, 7)

# 프로그레스 바 설정
pbar = tqdm(total=len(labels), desc="Summarizing Labels")

for label in labels:
    # 해당 라벨의 카테고리와 counts 추출
    label_data = df[df["label"] == label][["category", "counts"]].copy()
    if label_data.empty:
        # 해당 라벨에 데이터가 없을 경우 기본값 설정
        label_summary_list.append({"label": label, "category": "Uncategorized"})
        pbar.update(1)
        continue

    # 입력 데이터 포맷팅
    input_text = ""
    for _, row in label_data.iterrows():
        input_text += f"label: {label}, {row['category']}, {row['counts']}\n"
    # breakpoint()
    # 프롬프트에 데이터 삽입
    prompt_input = {"input": input_text}

    try:
        # LLM에 프롬프트 전달 및 응답 받기
        response = chain_label_sum.invoke(prompt_input)
        print(f"Label {label} summary response:\n{response}\n")

        # 응답에서 label 번호와 category 추출
        match = re.search(r"(\d+),\s*([\w/가-힣]+)", response)
        if match:
            summarized_category = match[2]
            print(label)
            print(summarized_category)
            label_summary_list.append({"label": label, "category": summarized_category})
        else:
            print(f"Warning: 라벨 {label}의 응답 형식이 올바르지 않습니다.")
            label_summary_list.append({"label": label, "category": "Uncategorized"})

    except Exception as e:
        print(f"Exception Occurred while processing label {label}: {e}")
        label_summary_list.append({"label": label, "category": "Uncategorized"})

    pbar.update(1)

# 프로그레스 바 닫기
pbar.close()

# 결과를 데이터프레임으로 변환
df_label_summary = pd.DataFrame(label_summary_list)

# 결과 출력
print("\n### 최종 라벨별 대표 카테고리 ###")
print(df_label_summary)

# CSV 파일로 저장
df_label_summary.to_csv("label_summary_categories.csv", index=False)

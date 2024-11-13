import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
import re
from langchain_ollama import OllamaLLM
from math import ceil

# CSV 파일 경로
input_csv_path = (
    "/data/ephemeral/home/level2-nlp-datacentric-nlp-11/df_labels_with_reasoning.csv"
)

# CSV 파일 읽기
df = pd.read_csv(input_csv_path)
df = df[
    (df["category"] != "Uncategorized") & (df["reasoning"] != "No reasoning provided")
].copy()

# LangChain 프롬프트 템플릿 설정
prompt_label_sum = ChatPromptTemplate.from_template(
    """
### Prompt:

아래에 label과 category, reasoning 정보를 보고 해당 label이 어떤 category를 의미하는지를 하위 tag로 정의해주세요.


1. label, category, reasoning 정보에 근거해서 하위 카테고리 #태그1 #태그2...를 붙여야합니다.
2. category는 뉴스 기사의 카테고리를 의미합니다.
3. 태그는 category와 관련있어야 하고, 구체적인 내용보다 추상적인 카테고리에 대한 정보를 담고 있어야 합니다.
4. 최종 출력의 형식을 엄격히 지켜서 출력해주세요.

### 입력 데이터(label, category, reasoning)
{input}

### 최종 출력 형식:
label, category, #tag1 #tag2...
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
batch_size = 10  # 필요에 따라 조정하세요
# 프로그레스 바 설정
pbar = tqdm(total=len(labels), desc="Summarizing Labels")

for label in labels:
    # 해당 라벨의 카테고리와 counts 추출
    label_data = df[df["label"] == label][["category", "reasoning"]].copy()
    if label_data.empty:
        # 해당 라벨에 데이터가 없을 경우 기본값 설정
        label_summary_list.append({"label": label, "category": "Uncategorized"})
        pbar.update(1)
        continue

        # 배치 수 계산
    num_batches = ceil(len(label_data) / batch_size)
    batch_summaries = []

    for batch_num in range(num_batches):
        # 배치 데이터 추출
        batch_data = label_data.iloc[
            batch_num * batch_size : (batch_num + 1) * batch_size
        ]

        # 입력 데이터 포맷팅
        input_text = ""
        for _, row in batch_data.iterrows():
            input_text += f"label: {label}, {row['category']}, {row['reasoning']}\n"

        # 프롬프트에 데이터 삽입
        prompt_input = {"input": input_text}
        try:
            # LLM에 프롬프트 전달 및 응답 받기
            response = chain_label_sum.invoke(prompt_input)
            # 응답에서 label 번호와 category, 태그 추출
            matches = re.findall(
                r"(\d+),\s*([\w/가-힣]+),\s*((?:#[\w]+\s*)+)", response
            )
            if matches:
                for match in matches:
                    label_num, summarized_category, tags = match
                    # 카테고리 정규화 (띄어쓰기로 분할)
                    category_parts = summarized_category.split()
                    batch_summaries.append(
                        {"category": category_parts, "tags": tags.strip()}
                    )
            else:
                print(
                    f"Warning: 레이블 {label}의 배치 {batch_num+1} 응답 형식이 올바르지 않습니다."
                )
                batch_summaries.append({"category": "Uncategorized", "tags": "NoTag"})

        except Exception as e:
            print(
                f"Exception Occurred while processing label {label}, batch {batch_num+1}: {e}"
            )
            batch_summaries.append({"category": "Uncategorized", "tags": "NoTag"})

    # 배치 요약을 기반으로 최종 레이블 요약 생성
    # 여기서는 단순히 모든 배치의 태그를 결합하는 방식으로 처리합니다.
    # 필요에 따라 더 정교한 집계 방식을 적용할 수 있습니다.
    try:
        final_categories = [
            batch["category"]
            for batch in batch_summaries
            if batch["category"] != "Uncategorized"
        ]
        final_tags = [
            batch["tags"] for batch in batch_summaries if batch["tags"] != "NoTag"
        ]
    except:
        pass

    if final_categories:
        # 각 요소가 리스트일 경우 튜플로 변환하여 set 사용 가능하게 처리
        hashed_categories = [tuple(category) for category in final_categories]
        # 가장 빈번하게 등장하는 카테고리를 선택
        summarized_category = max(set(hashed_categories), key=hashed_categories.count)
        # 튜플을 리스트로 변환하여 출력 (필요 시)
        summarized_category = list(summarized_category)

    else:
        summarized_category = "Uncategorized"

    if final_tags:
        # 모든 태그를 중복 없이 결합
        tags_set = set()
        for tag_str in final_tags:
            tags_set.update(tag_str.split())
        summarized_tags = " ".join(sorted(tags_set))
    else:
        summarized_tags = "NoTag"

    # 최종 요약을 label_summary_list에 추가
    label_summary_list.append(
        {"label": label, "category": summarized_category, "tag": summarized_tags}
    )

    pbar.update(1)

# 프로그레스 바 닫기
pbar.close()

for summary in label_summary_list:
    print(summary)
# 결과를 데이터프레임으로 변환
df_label_summary = pd.DataFrame(label_summary_list)

# 결과 출력
print("\n### 최종 라벨별 대표 카테고리 ###")
print(df_label_summary)

# CSV 파일로 저장
df_label_summary.to_csv("label_summary_categories_with_tags.csv", index=False)

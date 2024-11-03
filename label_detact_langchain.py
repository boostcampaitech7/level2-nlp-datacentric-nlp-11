import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from tqdm import tqdm
import re
from langchain_ollama import OllamaLLM

# CSV 파일 경로
input_csv_path = "/data/ephemeral/home/level2-nlp-datacentric-nlp-11/data/processed_is_noise_train_2step_ollama.csv"

# CSV 파일 읽기
df = pd.read_csv(input_csv_path)
df = df[(df["is_noise"] == 1.0) & (df["modified_text_or_0"] != "0")].copy()
# df = df[:1000]

# 텍스트 컬럼 확인
if "text" not in df.columns:
    raise ValueError("CSV 파일에 'text' 컬럼이 존재하지 않습니다.")

# 프롬프트 템플릿 설정
prompt_label_info = ChatPromptTemplate.from_template(
    """
### Prompt:

아래에 번호와 줄바꿈으로 나누어진 `{batch_size}`개의 텍스트를 보고 아래의 규칙에 따라 작업을 수행해주세요.

1. 각 텍스트를 보고 target의 숫자가 어떤 뉴스의 카테고리를 의미하는지 유추해야합니다.
2. target에 있는 번호는 카테고리 정보를 추가해야합니다.
3. target에 없는 번호는 정보 없음으로 출력합니다.
4. 최종 출력 형식의 0~6은 target값을 의미합니다.
5. 최종 출력의 형식을 엄격하게 지켜서 작성해 주세요.

### 입력 데이터(ID, target, text)

{input}

### 최종 출력 형식(target 번호: 카테고리 정보, 근거):

0: 카테고리 정보, 근거
1: 카테고리 정보, 근거
2: 카테고리 정보, 근거
3: 카테고리 정보, 근거
4: 카테고리 정보, 근거
5: 카테고리 정보, 근거
6: 카테고리 정보, 근거


    """
)

# LangChain LLM 설정 (Ollama)
llm_label_info = OllamaLLM(model="gemma2:27b")
chain_label_info = prompt_label_info | llm_label_info

# 배치 크기 설정
batch_size = 10

# 결과를 저장할 리스트 초기화
label_category_list = []

# 총 배치 수 계산
total_batches = (len(df) + batch_size - 1) // batch_size

# 프로그레스 바 설정
pbar = tqdm(total=total_batches, desc="Processing Batches")

# 데이터프레임을 배치 단위로 처리
for i in range(0, len(df), batch_size):
    # 현재 배치의 텍스트 추출
    batch_texts = df["modified_text_or_0"].iloc[i : i + batch_size].tolist()
    batch_IDs = df["ID"].iloc[i : i + batch_size].tolist()
    batch_target = df["target"].iloc[i : i + batch_size].tolist()

    numbered_texts = [
        f"ID: {ID}, target: {target}, text: {text}"
        for ID, target, text in zip(batch_IDs, batch_target, batch_texts)
    ]
    input_text = "\n".join(numbered_texts)
    print("###########Input_text##############")
    print(input_text)
    try:
        response_label_info = chain_label_info.invoke(
            {"batch_size": len(batch_texts), "input": input_text}
        )
        print("###########Response_reasoning##############")
        print(response_label_info)
        # 응답을 라인별로 분리하여 저장
        responses = [
            line.strip()
            for line in response_label_info.strip().split("\n")
            if line.strip()
        ]
        parsed_output = []
        for resp in responses:
            # 예시 응답 형식: "0: 경제 또는 0, 근거 또는 0"
            match = re.match(r"(\d+):\s*(.*?),\s*(.*)", resp)
            if match:
                idx = match.group(1)
                category = match.group(2).strip()
                reasoning = match.group(3).strip()
                parsed_output.append(f"{idx}: {category}, {reasoning}")
                label_category_list.append((idx, category))
            else:
                # 형식이 맞지 않으면 기본값 설정
                idx = resp.split(":")[0] if ":" in resp else "0"
                category = "Uncategorized"
                parsed_output.append(f"{idx}: {category}, No reasoning provided")
                # 라벨과 기본 카테고리를 리스트에 저장
                label_category_list.append((idx, category))
    except Exception as e:
        print("Exception Occurred:", e)
    finally:
        pbar.update(1)

# 프로그레스 바 닫기
pbar.close()

# 라벨과 카테고리 정보를 데이터프레임으로 변환
df_labels = pd.DataFrame(label_category_list, columns=["label", "category"])

df_labels["category"] = (
    df_labels["category"].str.replace(r"\*\*", "", regex=True).str.strip()
)

# 라벨과 카테고리별로 출현 횟수 계산
counts = df_labels.groupby(["label", "category"]).size().reset_index(name="counts")

# 'Uncategorized'와 '0'을 제외
counts_filtered = counts[~counts["category"].isin(["Uncategorized", "0", "정보 없음"])]

# 각 라벨별로 가장 많이 나온 카테고리 추출
most_common = counts_filtered.loc[
    counts_filtered.groupby("label")["counts"].idxmax()
].reset_index(drop=True)
# 결과 출력
print("### 라벨별 카테고리 출현 횟수 ###")
print(counts_filtered)

print("\n### 라벨별 가장 많이 나온 카테고리 ###")
print(most_common)

# CSV 파일로 저장 (필요에 따라)
counts_filtered.to_csv("label_category_counts.csv", index=False)
most_common.to_csv("most_common_category_per_label.csv", index=False)

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from tqdm import tqdm
import re
from langchain_ollama import OllamaLLM

# CSV 파일 경로
input_csv_path = "/data/ephemeral/home/level2-nlp-datacentric-nlp-11/data/2step_predicted_targets_assignment.csv"

# CSV 파일 읽기
df = pd.read_csv(input_csv_path)
# breakpoint()
df = df[(df["is_noise"] != 1.0) & (df["modified_text_or_0"] == "0")].copy()
# 라벨과 정규화된 카테고리별로 출현 횟수 계산
# df = df[:100]
# LangChain 프롬프트 템플릿 설정
prompt_label = ChatPromptTemplate.from_template(
    """
## Prompt:
### label,category
0,문화예술
1,스포츠
2,정치
3,사회
4,IT/과학기술
5,경제
6,국제

주어진 입력 데이터에 대해 어떤 category에 속하는지 판단해서 label을 매겨주세요.
1. label은 숫자를 의미하고 category는 label에 대한 정보를 의미합니다.
2. 하나의 입력엔 하나의 label만 존재해야합니다.
3. 7개의 label 중 하나가 무조건 선택되어야 합니다.
2. 최종 출력의 형식을 엄격히 지켜서 출력해주세요.

### 입력 데이터(ID: ID, text: text)
{input}

### 최종 출력 형식:
ID, label
ID, label
...
"""
)

# LangChain LLM 설정 (Ollama)
llm_label = OllamaLLM(
    model="gemma2:27b", temperature=0.1
)  # 실제 모델 이름으로 변경하세요
chain_label = prompt_label | llm_label

batch_size = 5

# 결과를 저장할 리스트 초기화
label_list = []


total_batches = (len(df) + batch_size - 1) // batch_size


pbar = tqdm(total=total_batches, desc="Processing Batches")

# breakpoint()
for i in range(0, len(df), batch_size):
    # 현재 배치의 텍스트 추출
    batch_texts = df["text"].iloc[i : i + batch_size].tolist()
    batch_IDs = df["ID"].iloc[i : i + batch_size].tolist()

    numbered_texts = [
        f"ID: {ID}, text: {text}" for ID, text in zip(batch_IDs, batch_texts)
    ]
    input_text = "\n".join(numbered_texts)
    print("###########Input_text##############")
    print(input_text)
    try:
        response_label_info = chain_label.invoke(
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
        for resp in responses:
            print("#########resp###########")
            print(resp)
            # 예시 응답 형식: "0: 경제 또는 0, 근거 또는 0"
            pattern = r"^(?:```)?\s*(?:ID:\s*)?(ynat-v1_train_\d+),\s*(?:label:\s*)?(\d+)\s*(?:```)?$"
            match = re.match(pattern, resp)
            print("#########match###########")
            print(match)
            # breakpoint()
            if match:
                ID = match.group(1)
                label = match.group(2)
                label_list.append((ID, label))
            else:
                # 형식이 맞지 않으면 기본값 설정
                # ID 추출 시도
                id_match = re.search(pattern, resp)
                print(id_match)
                if id_match:
                    ID = id_match.group(1)
                else:
                    # ID를 응답에서 추출하지 못한 경우, 배치의 ID 중 하나를 할당하거나 로깅
                    # 여기서는 해당 응답을 로깅하고 스킵
                    print(f"매치되지 않은 응답 (ID 추출 실패): {resp}")
                    continue  # 다음 응답으로 이동
                # 기본 라벨을 -1로 설정
                label = "-1"
                label_list.append((ID, label))

        # 배치 처리 완료 후 진행 상황 업데이트
        pbar.update(1)
    except Exception as e:
        print(f"Error processing batch {i//batch_size + 1}: {e}")
        # 에러가 발생한 배치의 모든 ID에 대해 기본 라벨 할당
        for ID in batch_IDs:
            label_list.append((ID, "-1"))
        # 진행 상황 업데이트
        pbar.update(1)

# 프로그레스 바 닫기
pbar.close()

# 라벨 리스트를 DataFrame으로 변환
labels_df = pd.DataFrame(label_list, columns=["ID", "label"])

# 원본 DataFrame과 라벨 DataFrame 병합
df = df.merge(labels_df, on="ID", how="left")

# 병합 후 누락된 라벨을 -1로 채움
df["label"] = df["label"].fillna(-1).astype(int)

# 결과를 새로운 CSV 파일로 저장
output_csv_path = "/data/ephemeral/home/level2-nlp-datacentric-nlp-11/data/2step_predicted_targets_assignment_with_labels.csv"
df.to_csv(output_csv_path, index=False)

print("라벨링이 완료되었습니다. 결과를 저장했습니다.")

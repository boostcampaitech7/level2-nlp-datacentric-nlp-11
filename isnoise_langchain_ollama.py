import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from tqdm import tqdm
import re
from langchain_ollama import OllamaLLM

# CSV 파일 경로
input_csv_path = (
    "/data/ephemeral/home/level2-nlp-datacentric-nlp-11/data/is_noise_train_ollama.csv"
)
output_csv_path = "/data/ephemeral/home/level2-nlp-datacentric-nlp-11/data/processed_is_noise_train_2step_llm.csv"

# CSV 파일 읽기
df = pd.read_csv(input_csv_path)
df_noise = df[df["is_noise"] == 1.0].copy()
# df_noise = df_noise[:10]
# df = df[:40]
# 텍스트 컬럼 확인
if "text" not in df.columns:
    raise ValueError("CSV 파일에 'text' 컬럼이 존재하지 않습니다.")

# 2. LangChain 프롬프트 템플릿 설정
# 첫 번째 프롬프트 템플릿 설정 (근거 생성)
prompt_noise_reasoning = ChatPromptTemplate.from_template(
    """
### Prompt:

아래에 번호와 줄바꿈으로 나누어진 `{batch_size}`개의 텍스트에는 한글을 제외한 다른 ASCII 코드로 임의의 character가 20%에서 80% 대체된 경우가 포함되어 있습니다. 아래의 규칙에 따라 각 텍스트를 뉴스 기사 제목으로 변환해 주세요:

1. 각 텍스트에서 한글을 유지하고, 전체 의미가 자연스러우며 명확한 뉴스 기사 제목으로 수정해 주세요.
2. 문장 의미를 파악할 수 없을 만큼 많은 문자가 ASCII 코드로 대체된 경우, 해당 데이터는 무시하고 `0`을 출력해 주세요.
3. 뉴스 기사의 제목은 총 3가지의 옵션으로 출력해야 합니다.
4. 최종 출력의 형식을 엄격하게 지켜서 작성해 주세요.

### 최종 출력 형식:

1. 텍스트 ID값, 수정된 텍스트1 or 0, 수정된 텍스트2 or 0, 수정된 텍스트3 or 0
2. 텍스트 ID값, 수정된 텍스트1 or 0, 수정된 텍스트2 or 0, 수정된 텍스트3 or 0
3. ...

(주어진 텍스트 갯수 = {batch_size})
{input}
    """
)
prompt_noise_classification = ChatPromptTemplate.from_template(
    """
{response_reasoning}
### 입력 형식:
1. 텍스트 ID값, 텍스트1 or 0, 텍스트2 or 0, 텍스트3 or 0
2. 텍스트 ID값, 텍스트1 or 0, 텍스트2 or 0, 텍스트3 or 0
3. ...

입력 형식으로 주어진 데이터 중 가장 뉴스 기사의 헤드라인으로 적합한 텍스트를 선택해서 아래의 최종 출력 형식에 맞춰서 작성하세요. 선택해 주세요:
1. 쉼표로 구분된 3개의 텍스트가 옵션으로 주어집니다.
2. 각 텍스트 옵션 중 가장 의미가 분명한 뉴스 기사의 헤드라인을 선택해야합니다.
3. 텍스트를 선택할 때는 텍스트의 수정없이 가져와야합니다.
4. 입력으로 0이 들어왔을 경우 그대로 0을 출력하세요.
5. 최종 출력의 형식을 엄격하게 지켜서 작성해 주세요.

### 최종 출력 형식:

1. 텍스트 ID값, 선택한 텍스트 or 0
2. 텍스트 ID값, 선택한 텍스트 or 0
3. ...
    """
)


# 4. LangChain LLM 설정 (Ollama)
llm_noise_reasoning = OllamaLLM(model="gemma2:27b")
llm_noise_classify = OllamaLLM(model="gemma2:27b", temperature=0.1)

chain_noise_reasoning = prompt_noise_reasoning | llm_noise_reasoning
# 4. 체인 연결
chain_noise_classify = prompt_noise_classification | llm_noise_classify

# chain_label = prompt_label | llm_label
# 5. 배치 크기 설정
batch_size = 5

# 6. 결과를 저장할 리스트 초기화
modified_texts = []

# 7. 총 배치 수 계산
total_batches = (len(df_noise) + batch_size - 1) // batch_size

# 8. 프로그레스 바 설정
pbar = tqdm(total=total_batches, desc="Processing Batches")

# 9. 데이터프레임을 배치 단위로 처리
for i in range(0, len(df_noise), batch_size):
    # 현재 배치의 텍스트 추출
    batch_texts = df_noise["text"].iloc[i : i + batch_size].tolist()
    batch_IDs = df_noise["ID"].iloc[i : i + batch_size].tolist()

    numbered_texts = [
        f"{j+1}. ID: {ID},{text}"
        for j, (ID, text) in enumerate(zip(batch_IDs, batch_texts))
    ]
    input_text = "\n".join(numbered_texts)
    print("###########Input_text##############")
    print(input_text)
    # 첫 번째 LLM: 근거 생성
    try:
        # 첫 번째 체인 호출
        response_reasoning = chain_noise_reasoning.invoke(
            {"batch_size": len(batch_texts), "input": input_text}
        )
        print("###########Response_reasoning##############")
        print(response_reasoning)
    except Exception as e:
        # 오류 발생 시 각 텍스트에 대해 기본값 "0"을 설정
        modified_texts.extend([(ID, "0") for ID in batch_IDs])
        print(f"Error processing label batch {i//batch_size +1}: {e}")
        pbar.update(1)
        continue

    try:
        # 두 번째 체인 호출
        response = chain_noise_classify.invoke(
            {"response_reasoning": response_reasoning}
        )
        print("###########Final_response##############")
        print(response)
        # 응답을 텍스트별로 분리하여 저장
        responses = [
            line.strip() for line in response.strip().split("\n") if line.strip()
        ]
        for resp in responses:
            # 텍스트 ID와 수정된 텍스트를 추출하기 위해 패턴 사용
            text_id_match = re.findall(r"ynat-v1_train_\d{5}", resp)
            parts = re.split(r"ynat-v1_train_\d{5},", resp, 1)

            # 텍스트 ID와 수정된 텍스트가 모두 존재하는지 확인
            if text_id_match and len(parts) == 2:
                text_id = text_id_match[0]  # 첫 번째 매칭 결과 사용
                modified_text = parts[1].strip()
                modified_texts.append((text_id, modified_text))
            else:
                modified_texts.append(
                    (batch_IDs[responses.index(resp)], "0")
                )  # 형식이 맞지 않으면 기본값 0 할당

    except Exception as e:
        # 오류 발생 시 각 텍스트에 대해 기본값 "0"을 설정
        if len(batch_IDs) == len(modified_texts[i : i + batch_size]):
            # 이미 추가된 경우 중복 추가 방지
            print(f"Warning: batch {i // batch_size + 1} already processed")
        else:
            modified_texts.extend([(ID, "0") for ID in batch_IDs])
        print(f"Error processing batch {i // batch_size + 1}: {e}")

    finally:
        pbar.update(1)

# 프로그레스 바 닫기
pbar.close()

# modified_texts를 DataFrame으로 변환
output_df = pd.DataFrame(modified_texts, columns=["ID", "modified_text_or_0"])

# 같은 ID를 가진 행의 modified_text_or_0 값을 쉼표로 구분하여 결합
output_df = output_df.groupby("ID", as_index=False).agg(
    {"modified_text_or_0": ", ".join}
)

# 원본 df와 병합하여 새로운 is_noise 열 추가
df = df.merge(output_df, on="ID", how="left")

# 유효하지 않은 응답(None)을 0으로 대체
df["modified_text_or_0"].fillna("0", inplace=True)

# 새로운 CSV 파일로 저장
df.to_csv(output_csv_path, index=False)

print(f"Processing complete. Output saved to {output_csv_path}")

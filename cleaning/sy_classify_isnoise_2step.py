import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from tqdm import tqdm
import re

# CSV 파일 경로
csv_path = "/data/ephemeral/home/project/data/train.csv"

# CSV 파일 읽기
df = pd.read_csv(csv_path)
# df = df[:40]
# 텍스트 컬럼 확인
if "text" not in df.columns:
    raise ValueError("CSV 파일에 'text' 컬럼이 존재하지 않습니다.")
breakpoint()
# 2. LangChain 프롬프트 템플릿 설정
# 첫 번째 프롬프트 템플릿 설정 (근거 생성)
prompt_reasoning = ChatPromptTemplate.from_template(
    """
    Prompt:

    아래에 번호와 줄바꿈으로 나눠진 {batch_size}개의 텍스트는 임의의 character 중 20%~80%를 한글을 제외한 다른 아스키 코드로 대체한 텍스트가 껴있습니다. 한글을 제외한 다른 아스키 코드로 대체한 텍스트는 다음과 같습니다.
    예시:
    1. pI美대선I앞두고 R2fr단 발] $비해 감시 강화
    2. oi 매력 R모h츠a열#w3약 >l·주가 고Q/진
    아닌 예시:
    1. 美성인 6명 중 1명꼴 배우자·연인 빚 떠안은 적 있다
    2. 朴대통령 얼마나 많이 놀라셨어요…경주 지진현장 방문종합

    출력 형식:
    1. 텍스트1, 아스키 코드로 대체했다고 생각하는 근거 또는 없다고 생각하는 근거
    2. 텍스트2, 아스키 코드로 대체했다고 생각하는 근거 또는 없다고 생각하는 근거
    ...

    아래 텍스트를 확인하고, 각 텍스트에 대한 아스키 코드로 대체했는지에 대한 근거를 논리적으로 밝혀주세요. (주어진 텍스트 갯수={batch_size}):
    {input}
    """
)

prompt_label = ChatPromptTemplate.from_template(
    """
    {response_reasoning}

    내용들의 근거들을 바탕으로 번호에 따른 텍스트에 대한 아스키 코드 대체 여부를 1 또는 0으로 분류해 주세요.
    아스키 코드로 대체되었다면 1, 대체되지 않았다면 0입니다.
    출력 형식을 엄격히 지키세요.

    출력 형식:
    1. 텍스트1: 1
    2. 텍스트2: 0
    ...
    """
)

# 4. LangChain LLM 설정 (Ollama)
llm_reasoning = Ollama(model="gemma2:27b")
llm_label = Ollama(model="gemma2:27b")

# 4. 체인 연결
chain_reasoning = prompt_reasoning | llm_reasoning

chain_label = prompt_label | llm_label
# 5. 배치 크기 설정
batch_size = 5

# 6. 결과를 저장할 리스트 초기화
is_noise_list = []

# 7. 총 배치 수 계산
total_batches = (len(df) + batch_size - 1) // batch_size

# 8. 프로그레스 바 설정
pbar = tqdm(total=total_batches, desc="Processing Batches")

# 9. 데이터프레임을 배치 단위로 처리
for i in range(0, len(df), batch_size):
    # 현재 배치의 텍스트 추출
    batch_texts = df["text"].iloc[i : i + batch_size].tolist()

    # 텍스트를 ,,,로 구분하여 하나의 문자열로 결합
    numbered_texts = [f"{j+1}. {text}" for j, text in enumerate(batch_texts)]
    input_text = "\n".join(numbered_texts)
    print(input_text)
    # 첫 번째 LLM: 근거 생성
    try:
        # 첫 번째 체인 호출
        response_reasoning = chain_reasoning.invoke(
            {"batch_size": len(batch_texts), "input": input_text}
        )
        print(response_reasoning)
    except Exception as e:
        # 오류 발생 시 모든 텍스트에 대해 None 설정
        is_noise_list.extend([None] * len(batch_texts))
        print(f"Error processing reasoning batch {i//batch_size +1}: {e}")
        pbar.update(1)
        continue

    # 두 번째 LLM: 노이즈 여부 판단
    try:
        # 두 번째 체인 호출
        response_label = chain_label.invoke({"response_reasoning": response_reasoning})
        print(response_label)
    except Exception as e:
        # 오류 발생 시 모든 텍스트에 대해 None 설정
        is_noise_list.extend([None] * len(batch_texts))
        print(f"Error processing label batch {i//batch_size +1}: {e}")
        pbar.update(1)
        continue

    # 응답을 텍스트별로 분리
    # 응답 형식:
    # 텍스트1: 1
    # 텍스트2: 0
    # ...
    responses = [
        line.strip() for line in response_label.strip().split("\n") if line.strip()
    ]
    # 각 응답을 처리하여 is_noise_list에 추가
    for resp in responses:
        print(resp)
        # 응답 형식을 "텍스트1: 1"로 가정
        parts = resp.split(":", 1)
        print(parts)
        if len(parts) == 2:
            text, label = parts
            label = label.strip()
            if label in ["0", "1"]:
                is_noise_list.append(int(label))
            else:
                is_noise_list.append(None)  # 유효하지 않은 응답은 None으로 설정
        else:
            is_noise_list.append(None)  # 예상치 못한 형식의 응답은 None으로 설정

    # 현재 배치의 텍스트 수와 응답 수 일치 여부 확인
    if len(responses) != len(batch_texts):
        # 응답 수가 부족할 경우, 남은 텍스트에 대해 None 설정
        is_noise_list.extend([None] * (len(batch_texts) - len(responses)))

    # 프로그레스 바 업데이트
    pbar.update(1)

# 프로그레스 바 닫기
pbar.close()

# 11. is_noise_list 길이 확인
if len(is_noise_list) != len(df):
    raise ValueError("is_noise_list의 길이가 데이터프레임의 길이와 일치하지 않습니다.")

# 12. is_noise 컬럼 추가
df["is_noise"] = is_noise_list

# 13. 유효하지 않은 응답(None)을 0으로 대체
df["is_noise"].fillna(0, inplace=True)

# 14. 새로운 CSV 파일로 저장
output_csv_path = "/data/ephemeral/home/project/data/is_noise_train.csv"
df.to_csv(output_csv_path, index=False)

print(f"Processing complete. Output saved to {output_csv_path}")

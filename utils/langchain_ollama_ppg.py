import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from tqdm import tqdm
import re
from langchain_ollama import OllamaLLM

# CSV 파일 경로
csv_path = (
    "/home/ppg/Desktop/AI Tech 7기/NLP/3. Data-Centric/project/upstages/data/train.csv"
)

# CSV 파일 읽기
df = pd.read_csv(csv_path)
df = df[:10]
# 텍스트 컬럼 확인
if "text" not in df.columns:
    raise ValueError("CSV 파일에 'text' 컬럼이 존재하지 않습니다.")

# 2. LangChain 프롬프트 템플릿 설정
# 첫 번째 프롬프트 템플릿 설정 (근거 생성)
prompt_reasoning = ChatPromptTemplate.from_template(
    """
    Prompt:
    당신은 ASCII 코드 노이즈가 포함된 한국어 텍스트에서 알파벳(A-Za-z)으로만 이루어진 고유명사(예: 국가명, 조직, 인물 등)를 정확히 찾아낼 수 있는 언어학 전문가입니다.
    ASCII 코드 노이즈가 포함된 뉴스 기사의 제목에서 **의미가 명확한 공식 알파벳(A-Za-z) 고유명사**를 찾아주는 작업을 진행합니다.
    작업 핵심:
    ASCII 코드 노이즈와 알파벳 고유명사를 명확히 구분해야 합니다.
    - 무조건 전세계 모든 사람들이 아는 고유명사이여야 합니다.
    - 무조건 기사 제목 내용과 연관성이 깊어야 합니다.
    예시:
    프로야구~롯TKIAs광주 경기 y천취소 -> KIA, 이유: "KIA"는 한국의 자동차 제조 회사인 기아자동차를 뜻합니다.
    pI美대선I앞두고 R2fr단 발] $비해 감시 강화 -> 없음, 이유: "R2fr"는 고유명사처럼 보일 수 있으나, 명확한 의미를 갖고 있지 않으므로 ASCII 코드 노이즈로 간주됩니다.
    아이`XSI수리0* b대`…맥3 디dF레< 41/ -> 없음, 이유: "XSI"는 고유명사처럼 보일 수 있으나, 명확한 의미를 갖고 있지 않으므로 ASCII 코드 노이즈로 간주됩니다.
    제목:
    {input}
    출력 포맷:
    1. 알파벳 고유명사: 알파벳(A-Za-z)으로만 이루어진 고유명사
    2. 선정한 이유: 알파벳 고유명사라고 선정한 이유
    출력 포맷을 염격히 지켜주세요!
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

    """
)

# 4. LangChain LLM 설정 (Ollama)
llm_reasoning = OllamaLLM(model="gemma2:27b")
llm_label = OllamaLLM(model="gemma2:27b", temperature=0.1)

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
output_csv_path = "/home/ppg/Desktop/AI Tech 7기/NLP/3. Data-Centric/project/upstages/data/is_noise_train.csv"
df.to_csv(output_csv_path, index=False)

print(f"Processing complete. Output saved to {output_csv_path}")

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from tqdm import tqdm
import re

# CSV 파일 경로
csv_path = "/data/ephemeral/home/data/is_noise_train.csv"

# 1. CSV 파일 읽기
df = pd.read_csv(csv_path)

# 2. 'is_noise'가 1인 데이터만 선택
noise_df = df[df["is_noise"] == 1]

# 3. 텍스트 컬럼 확인
if "text" not in noise_df.columns:
    raise ValueError("CSV 파일에 'text' 컬럼이 존재하지 않습니다.")

# 4. LangChain 프롬프트 템플릿 설정
# 프롬프트 템플릿 설정 (증강)
prompt_aug = ChatPromptTemplate.from_template(
    """
    다음 작업에서 전문 언어 모델로서의 역할을 수행하세요. 특히, 노이즈가 많은 데이터나 불완전한 텍스트 데이터를 다루는 데 경험이 있는 전문가로서 각 입력 텍스트의 맥락을 신중히 분석하여 뉴스 기사 제목 스타일의 문장을 추가로 생성하는 데 중점을 둡니다.

    지시 사항:

    {batch_size}개의 텍스트로 구성된 목록을 확인하고, 각 항목의 맥락과 유사한 8개의 제목 스타일 문장을 생성하세요.
    원본 텍스트에 아스키 코드 노이즈가 많이 포함된 경우, 생성된 텍스트에는 노이즈를 포함하지 않으면서 원본의 주요 의미를 최대한 보존합니다.
    생성된 텍스트는 한 단어로 구성되어서는 안 되며, 쉼표(,)가 포함되지 않도록 합니다.
    추가 설명이나 부가적인 정보는 출력에 포함하지 않고, 아래의 출력 형식에 엄격히 맞춰 작성하세요.

    {input}

    출력 형식:
    각 입력 항목에 대해 다음과 같은 형식으로 출력을 생성하세요:
    1. 원본 텍스트1, 생성 텍스트 1-1, 생성 텍스트 1-2, 생성 텍스트 1-3, 생성 텍스트 1-4, 생성 텍스트 1-5, 생성 텍스트 1-6, 생성 텍스트 1-7, 생성 텍스트 1-8,
    2. 원본 텍스트2, 생성 텍스트 2-1, 생성 텍스트 2-2, 생성 텍스트 2-3, 생성 텍스트 2-4, 생성 텍스트 2-5, 생성 텍스트 2-6, 생성 텍스트 2-7, 생성 텍스트 2-8,

    ...

    """
)

# 5. LangChain LLM 설정 (Ollama)
llm_aug = OllamaLLM(model="gemma2:27b")

# 6. 체인 연결
chain_aug = prompt_aug | llm_aug

# 7. 배치 크기 설정
batch_size = 5

# 8. 결과를 저장할 리스트 초기화
aug_data_list = []

# 9. 총 배치 수 계산
total_batches = (len(noise_df) + batch_size - 1) // batch_size

# 10. 프로그레스 바 설정
pbar = tqdm(total=total_batches, desc="Processing Batches")

# 11. 데이터프레임을 배치 단위로 처리
for i in range(0, len(noise_df), batch_size):
    # 현재 배치의 텍스트 추출
    batch_texts = noise_df["text"].iloc[i : i + batch_size].tolist()

    # 텍스트를 ,,,로 구분하여 하나의 문자열로 결합
    numbered_texts = [f"{j+1}. {text}" for j, text in enumerate(batch_texts)]
    input_text = "\n".join(numbered_texts)

    # 첫 번째 LLM: 증강
    try:
        # 첫 번째 체인 호출
        response_aug = chain_aug.invoke(
            {"batch_size": len(batch_texts), "input": input_text}
        )

    except Exception as e:
        # 오류 발생 시 모든 텍스트에 대해 None 설정
        aug_data_list.extend([None] * len(batch_texts))
        print(f"Error processing augmentation batch {i//batch_size +1}: {e}")
        pbar.update(1)
        continue

    # 응답을 텍스트별로 분리
    # 응답 형식:
    # 텍스트1, 텍스트 1-1, 텍스트 1-2, 텍스트 1-3
    # 텍스트2, 텍스트 2-1, 텍스트 2-2, 텍스트 2-3
    # ...
    responses = [
        line.strip() for line in response_aug.strip().split("\n") if line.strip()
    ]

    # aug_data_list에 저장할 데이터를 위한 인덱스 초기화
    aug_index = 0

    # 각 응답을 처리하여 i에 추가
    for resp, original_row in zip(responses, noise_df.itertuples()):
        # 응답 형식을 "텍스트1, 텍스트 1-1, 텍스트 1-2, 텍스트 1-3"로 가정
        parts = resp.split(",", 3)
        if len(parts) == 4:
            original, text1, text2, text3 = parts
            target = original_row.target  # original_text의 target 값으로 설정

            # 세 개의 유사 텍스트를 각각 추가
            for i, text in enumerate([text1, text2, text3], start=1):
                aug_data_list.append(
                    {
                        "ID": f"aug-train-{aug_index}",
                        "text": text.strip(),
                        "target": target,
                        "is_noise": 1,
                    }
                )
                aug_index += 1
        else:
            # 예상치 못한 형식의 응답은 추가하지 않음
            print("Unexpected response format, skipping this entry.")

    # 현재 배치의 텍스트 수와 응답 수 일치 여부 확인
    if len(responses) != len(batch_texts):
        # 응답 수가 부족할 경우, 남은 텍스트에 대해 None을 추가하지 않고 경고만 출력
        print("Warning: Mismatch in response and batch_texts length.")

    # 프로그레스 바 업데이트
    pbar.update(1)

# 12. 프로그레스 바 닫기
pbar.close()

# 13. aug_data_list에 None이 포함되지 않았는지 최종적으로 확인
aug_data_list = [item for item in aug_data_list if item is not None]

# 14. 새로 생성된 데이터를 기존 데이터프레임에 추가
aug_df = pd.DataFrame(aug_data_list)

# 15. 새로운 CSV 파일로 저장
output_csv_path = "/data/ephemeral/home/data/augmented_data_label_five.csv"
aug_df.to_csv(output_csv_path, index=False)

print(f"Processing complete. Output saved to {output_csv_path}")

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sklearn.utils import shuffle
import pandas as pd
from random import sample
from tqdm import tqdm
import sys
import os

# 현재 파일의 위치에서 상위 디렉토리로 이동 후 utils 폴더 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))
from utils import label_group_count, filter_similar_texts, call_ollama_with_retry


## Ollama 모델 설정, LLM 초기화
llm = OllamaLLM(model="gemma2:27b")

## LLm으로 추론한 label_meanings.csv 파일 불러오기
label_meanings_df = pd.read_csv("../data/label_meanings_updated.csv")
label_meanings = {
    row["label"]: row["meaning"] for _, row in label_meanings_df.iterrows()
}  # 각 라벨의 의미를 딕셔너리 형태로 변환
label_meanings_text = "\n".join(
    [f"{label}: {meaning}" for label, meaning in label_meanings.items()]
)  # 모든 라벨 의미를 한 번에 프롬프트에 포함하여 작성

## 현재 라벨별 데이터 수정
data_df = pd.read_csv(
    "../data/noise_label_cleaned_train.csv"
)  # 노이즈, 라벨 에러 제거된 train 데이터
data_df = shuffle(data_df).reset_index(drop=True)  # 데이터 shuffle
current_counts = label_group_count(data_df)
target_count = max(
    current_counts.values()
)  # 목표 데이터 수 설정 (최대 데이터 수 기준으로 맞추기)


## 데이터 증강을 위한 프롬프트 템플릿 생성
def generate_prompt(label, label_meanings_text, meaning, existing_texts):
    prompt = f"""
    아래는 각 라벨의 의미입니다:
    {label_meanings_text}
    생성할 라벨: {label}
    기존 예시: {existing_texts}
    위 예시와 문장 구조가 같은 구성을 따르되, 제목이 중복되지 않는 새로운 뉴스를 한 개 생성해 주세요.
    예시 제목과 일부 단어, 표현, 문장 구조를 반영하되, 다른 시각이나 관련 사건을 다룬 기사 제목을 만드세요.
    응답 형식:
    뉴스 기사 제목: 새로운 뉴스 기사 제목
    위 응답 형식에 맞춰 생성해 주세요. 응답 형식 외의 부연 설명이나 추가 텍스트는 응답에 포함하지 마세요.
    """
    return prompt.strip()


## 증강 데이터 생성 함수
def augment_data(data_df, label_meanings, target_count, similarity_threshold=0.7):
    augmented_texts = []
    generated_text_set = set()  # 중복 방지를 위한 집합
    for label, meaning in label_meanings.items():
        current_count = current_counts.get(label, 0)
        if current_count >= target_count:
            continue  # 목표 데이터 수 이상인 경우 증강하지 않음

        num_samples_needed = target_count - current_count

        # 해당 라벨의 모든 텍스트를 리스트로 가져옴
        existing_texts = data_df[data_df["target"] == label]["text"].tolist()

        # 각 텍스트를 순서대로 사용하여 증강
        for existing_text in existing_texts:
            """
            if current_count >= target_count:
                break  # 목표 개수 도달 시 종료
            """

            print(
                f"증강 대상 텍스트 (라벨 {label}): {existing_text}"
            )  # 증강 대상 텍스트를 출력
            prompt = generate_prompt(
                label, label_meanings_text, meaning, [existing_text]
            )

            # LLM을 통해 텍스트 생성
            response = llm.invoke(prompt)
            # generated_texts = response.strip().split('\n')  # 생성된 텍스트를 줄바꿈으로 분리
            """
            # 생성된 텍스트 중 조건을 만족하는 텍스트만 추가
            for text in generated_texts:
                cleaned_text = text.split(".", 1)[-1].strip()  # "1. 텍스트" 형식에서 번호 제거
                if cleaned_text and "뉴스 기사 제목 예시" not in cleaned_text:
                    # 중복된 텍스트가 아니면 추가
                    if cleaned_text not in generated_text_set:
                        new_id = f"{label}_aug_{len(augmented_texts):05d}"  # ID with unique pattern
                        augmented_texts.append({'ID': new_id, 'text': cleaned_text, 'target': label})
                        generated_text_set.add(cleaned_text)  # 추가된 텍스트를 집합에 저장
                        current_count += 1  # 현재 라벨 데이터 개수 증가
                        print(f"라벨-{label}, {cleaned_text}")  # 추가된 텍스트 확인용
                if current_count >= target_count:
                    break  # 목표 개수 도달 시 종료
            """
            cleaned_text = response.replace(
                "뉴스 기사 제목:", ""
            ).strip()  # 응답에서 "뉴스 기사 제목:" 제거
            if cleaned_text and cleaned_text not in generated_text_set:
                new_id = f"{label}_aug_{len(augmented_texts):05d}"  # 고유 ID
                augmented_texts.append(
                    {"ID": new_id, "text": cleaned_text, "target": label}
                )
                generated_text_set.add(cleaned_text)  # 추가된 텍스트를 집합에 저장
                print(
                    f"증강된 텍스트 (라벨 {label}): {cleaned_text}"
                )  # 추가된 텍스트 확인용

    # 중복 및 유사 텍스트 제거
    augmented_df = pd.DataFrame(
        augmented_texts, columns=["ID", "text", "target"]
    )  # 열 이름을 명시적으로 지정
    # 중복 및 유사 텍스트 제거 후 DataFrame으로 변환
    filtered_texts = filter_similar_texts(
        augmented_df["text"].tolist(), label, similarity_threshold
    )

    # 필터링한 텍스트로 DataFrame 생성
    filtered_df = pd.DataFrame(
        {
            "ID": [f"{label}_aug_{i:05d}" for i in range(len(filtered_texts))],
            "text": filtered_texts,
            "target": label,
        }
    )

    return filtered_df[["ID", "text", "target"]]


## 증강 데이터 생성 및 합치기
aug_data_df = augment_data(data_df, label_meanings, target_count)
final_data_df = pd.concat([data_df, aug_data_df], ignore_index=True)

## 결과 저장
final_data_df.to_csv("augmented_data.csv", index=False)

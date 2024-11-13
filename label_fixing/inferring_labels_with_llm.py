import pandas as pd
from langchain_community.llms import Ollama

# 데이터 로드
data = pd.read_csv("./data/new_cleaned_data_seona.csv")

# Ollama 모델 초기화
llm = Ollama(model="gemma2:27b")

# 라벨별로 문장을 그룹화하여 5개씩 나누기
grouped_data = data.groupby("target")["text"].apply(list).to_dict()

# 라벨별 의미 저장할 딕셔너리
label_meanings_partial = {label: [] for label in grouped_data.keys()}


# 프롬프트 생성 함수 (5개 문장씩)
def create_partial_meaning_prompt(target, texts):
    texts_joined = "\n".join([f"- {text}" for text in texts])
    return f"""
    다음은 라벨 {target}에 해당하는 뉴스 제목들입니다. 이 문장들로부터 이 라벨이 주로 어떤 주제의 뉴스 제목을 나타내는지 유추해 주세요.

    문장들:
    {texts_joined}

    이 라벨이 주로 나타내는 주제를 간단하게 설명해 주세요.
    """


# 각 라벨별로 5개씩 문장을 묶어서 의미 추론
for target, texts in grouped_data.items():
    # 5개씩 문장을 묶어 LLM에 요청
    for i in range(0, len(texts), 5):
        text_chunck = texts[i : i + 5]
        prompt = create_partial_meaning_prompt(target, text_chunck)

        # LLM에 의미 추론 요청
        response = llm.invoke(prompt).strip()

        # 라벨별로 의미 누적 저장
        label_meanings_partial[target].append(response)

        print(f"라벨 {target} - 도출된 주제: {response}")


# 2단계: 각 라벨에 대해 누적된 의미를 바탕으로 대표 주제 도출
# 대표 주제 도출 프롬프트 함수
def summarize_label_meaning_prompt(target, partial_meanings):
    partial_meanings_text = "\n".join([f"- {meaning}" for meaning in partial_meanings])
    return f"""
    라벨 {target}에 대해 각 뉴스 제목에서 추론한 주제들이 아래와 같습니다.:

    {partial_meanings_text}

    이 라벨이 나타내는 대표적인 주제를 한 문장으로 요약해 주세요.
    """


# 라벨별 대표 의미 저장
final_label_meanings = {}

for target, partial_meanings in label_meanings_partial.items():
    prompt = summarize_label_meaning_prompt(target, partial_meanings)
    response = llm.invoke(prompt).strip()
    final_label_meanings[target] = response

# 대표 라벨 의미 확인(결과 출력 및 파일 저장)
print("라벨별 대표 의미:")
with open("label_meanings.csv", "w") as f:
    f.write("label, meaning\n")  # CSV 헤더
    for label, meaning in final_label_meanings.items():
        print(f"{label}-라벨 의미:", meaning)
        f.write(f"{label},{meaning}\n")

print("라벨별 대표 의미가 'label_meanings.csv' 파일로 저장되었습니다.")

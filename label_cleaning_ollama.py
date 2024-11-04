import pandas as pd
from langchain_community.llms import Ollama

# Ollama 모델 초기화
llm = Ollama(model='gemma2: 27b')


### 1. 라벨 에러가 존재하는 데이터 불러오기(found_noise_train.csv에서 noise=0인 데이터의 ID)
found_noise_data = pd.read_csv('./data/found_noise_train.csv')
unnoisy_ids = found_noise_data[found_noise_data['noise'] == 0]['ID']
# 전체 데이터 로드
train_data = pd.read_csv('./data/train.csv')
# 라벨 에러가 있는 데이터 필터링
error_data = train_data[train_data['ID'].isin(unnoisy_ids)]


### 2. 라벨 의미 로드
label_meanings = pd.read_csv('./data/label_meanings.csv')
label_meaning_dict = dict(zip(label_meanings['label'], label_meanings['meaning']))
# 라벨 설명 문자열 생성
label_descriptions = '\n'.join([f"{k}: {v}" for k, v in label_meaning_dict.items()])


### 3. 프롬프트 생성
def create_label_correction_prompt(text, target):
    return f"""
    뉴스 제목: "{text}"
    현재 라벨: {target}

    각 라벨의 의미는 다음과 같습니다:
    {label_descriptions}

    위 라벨 설명을 참고하여 뉴스 제목이 가장 잘 맞는 라벨을 추천해 주세요.
    형식: "수정된 라벨: X" (X는 0~6 중 하나의 숫자)
    """


### 4. 라벨 수정 결과 저장
results = []
# 각 뉴스 제목에 대해 LLM을 사용해 라벨 수정
for _, row in error_data.iterrows():
    text = row['text']
    target = row['target']

    # LLM 프롬프트 실행 
    prompt = create_label_correction_prompt(text, target)
    response = llm.invoke(prompt)

    # 수정된 라벨 추출
    corrected_label = int(response.split(":")[-1].strip())

    # 결과 저장
    results.append({
        "ID": row['ID'],
        "text": text,
        "target": target,
        "corrected_text": corrected_label # LLM 응답을 저장
    })

# 결과를 데이터프레임으로 변환 후 CSV로 저장
results_df = pd.DataFrame(results)
results_df.to_csv('./data/labeling_cleaned_data.csv', index=False)
print("클린된 데이터가 'labeling_cleaned_data.csv' 파일로 저장되었습니다.")
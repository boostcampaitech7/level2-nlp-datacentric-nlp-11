import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# 텍스트 정제를 위한 Ollama 모델 초기화
llm_reasoning = Ollama(model="gemma2:27b")
llm_label = Ollama(model="gemma2:27b")

prompt_reasoning = ChatPromptTemplate.from_template(
    """
    Prompt:
    출력 형식: 
    
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


# 4. 체인 연결
chain_reasoning = prompt_reasoning | llm_reasoning
chain_label = prompt_label | llm_label

# CSV 데이터 로드
data = pd.read_csv('./data/train.csv')
results = []

# 데이터 한 줄씩 처리
for index, row in data.iterrows():
    input_text = row['text']

    # Reasoning 체인 실행
    response_reasoning = chain_reasoning.run({"batch_size": 1, "input": input_text})

    # label 체인 실행
    response_label = chain_label.run({"response_reasoning": response_reasoning})

    # 응답 결과 저장 (예시: 텍스트와 레이블을 저장)
    result = {
        "ID" : row['ID'],
        "text": input_text,
        "target": row['target'],
        "noise": response_label.strip()
    }
    results.append(result)

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)

# 결과를 CSV파일로 저장
results_df.to_csv('./data/found_noise_train.csv', index=False)
print("처리된 데이터가 'found_noise_train.csv' 파일에 저장되었습니다.")


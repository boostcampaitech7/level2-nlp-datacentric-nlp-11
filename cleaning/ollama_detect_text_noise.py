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
    출력 형식은 숫자 하나만 반환해 주세요. 출력 형식을 엄격히 지키세요.

    출력 형식: 1 또는 0
    """
)


# 4. 체인 연결
chain_reasoning = prompt_reasoning | llm_reasoning
chain_label = prompt_label | llm_label

# CSV 데이터 로드
data = pd.read_csv('./data/train.csv')
results = []

# 데이터 10개씩 묶어서 처리
batch_size = 10
for i in range(0, len(data), batch_size):
    batch = data.iloc[i:i+batch_size]
    input_texts = batch['text'].tolist()
    
    # Reasoning 체인 실행
    response_reasoning = chain_reasoning.invoke({"batch_size": batch_size, "input": input_texts})

    # 각 텍스트에 대한 라벨 체인 실행
    for idx, input_text in enumerate(input_texts):
        single_response_reasoning = response_reasoning[idx]  # 각 텍스트별 reasoning 결과
        response_label = chain_label.invoke({"response_reasoning": single_response_reasoning})

        # 응답 결과 저장
        result = {
            "ID" : batch.iloc[idx]['ID'],
            "text": input_text,
            "target": batch.iloc[idx]['target'],
            "noise": response_label.strip()
        }
        results.append(result)

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)

# 결과를 CSV파일로 저장
results_df.to_csv('./data/found_noise_train.csv', index=False)
print("처리된 데이터가 'found_noise_train.csv' 파일에 저장되었습니다.")


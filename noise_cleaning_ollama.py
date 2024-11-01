from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# 텍스트 정제를 위한 Ollama 모델 초기화
llm_reasoning = Ollama(model="gemma2:27b")
llm_label = Ollama(model="gemma2:27b")

# Reasoning에 대한 프롬프트 설정
prompt_reasoning = ChatPromptTemplate.from_template(
    """
    Prompt:
    아래 텍스트들을 확인하고, 각 텍스트가 ASCII 코드로 대체되었는지 여부에 대한 논리적 근거를 제시해 주세요.
    (주어진 텍스트 갯수={batch_size}):
    {input_text}
    """
)

# ASCII 코드 대체 여부에 대한 분류 프롬프트 설정
prompt_label = ChatPromptTemplate.from_template(
    """
    {response_reasoning}
    위에서 제시된 근거를 바탕으로, 각 텍스트에 대해 ASCII 코드로 대체된 경우는 1, 아닌 경우는 0으로 표시하세요.
    출력 형식:
    1. 텍스트1: 1
    2. 텍스트2: 0
    ...
    """
)

# 프롬프트와 모델 연결
chain_reasoning = prompt_reasoning | llm_reasoning
chain_label = prompt_label | llm_label

def clean_data_with_ollama(text_data):
    # 데이터 정제를 위해 batch로 텍스트를 나눕니다.
    batch_size = 10  # 조정 가능
    results = []
    for i in range(0, len(text_data), batch_size):
        batch_texts = text_data[i:i+batch_size]
        input_text = "\n".join([f"{idx+1}. {text}" for idx, text in enumerate(batch_texts)])
        # ASCII 대체 여부에 대한 근거 추론
        response_reasoning = chain_reasoning.invoke({"batch_size": len(batch_texts), "input_text": input_text})
        # 추론 근거를 바탕으로 분류 작업 수행
        response_label = chain_label.invoke({"response_reasoning": response_reasoning})
        # 결과 정리
        results.append(response_label)
    return results

# 예제 텍스트 데이터
text_data = [
    "정i :파1 미사z KT( 이용기간 2e 단] Q분종U2보",
    "K찰.국DLwo 로L3한N% 회장 2 T0&}송=",
    "m 김정) 자주통일 새,?r열1나가야1보",
    "갤노트8 주말 27만대 개통…시장은 불법 보조금 얼룩",
    "pI美대선I앞두고 R2fr단 발] $비해 감시 강화"
]

# 데이터 정제 실행
cleaned_data = clean_data_with_ollama(text_data)

# 정제 결과 출력
for idx, result in enumerate(cleaned_data):
    print(f"Batch {idx + 1} 결과:")
    print(result)
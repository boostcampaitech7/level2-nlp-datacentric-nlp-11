from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import pandas as pd

# 텍스트 정제를 위한 Ollama 모델 초기화
#llm_reasoning = Ollama(model="gemma2:27b")
llm = Ollama(model="gemma2:27b")


# ASCII 코드 대체 여부에 대한 분류 프롬프트 설정
prompt_reasoning = """
    당신은 한국어 텍스트의 노이즈를 제거하고 원래 문장으로 복원하는 한국어 언어학 전문가입니다.
    잘못된 문자, 의미 없는 기호, 불필요한 단어 등을 걸러내고 의미를 재구성하여 원래 문장의 내용을 복원할 수 있습니다.
    이 작업은 주어진 한국어 텍스트에서 불필요한 기호, 무작위 문자, 불완전한 단어 등을 제거하여 자연스럽고 완전한 문장으로 복원하는 것을 목표로 합니다.
    
    다음 단계에 따라 작업을 수행하세요:
    1단계: 주어진 텍스트에서 노이즈 패턴을 식별합니다. 의미 없는 문자나 기호, 오류 문자 등을 찾아 메모합니다.
    2단계: 의미와 문맥이 유지되도록 노이즈를 제거합니다. 특히 한국어 단어와 문법에 맞지 않는 기호, 영어 문자, 숫자 등을 삭제하거나 수정합니다.
    3단계: 문장 구조를 파악하고, 복원된 문장의 자연스러운 흐름을 위해 단어를 재배치합니다. 의미가 명확하게 전달되도록 문장 내 단어 순서를 조정합니다.
    4단계: 필요할 경우 문법에 맞는 조사나 접속사를 추가하여 의미 전달이 자연스럽도록 합니다.
    5단계: 원래 문장으로 합리적으로 유추할 수 있는 부분을 복원합니다 (예: 아이폰 XS 대신 '아이1XS'로 기록된 경우 '아이폰 XS'으로 복원).

    예시:
    입력: "정i :파1 미사z KT( 이용기간 2e 단] Q분종U2보"
    출력: "정파 미사 KT 이용기간 2개월 단축 보도"
"""

prompt_template = ChatPromptTemplate.from_template(
    f"""
    {prompt_reasoning}
    
    주어진 텍스트 목록:
    {{input_text}}

    복원된 문장은 올바른 한국어 문법과 자연스러운 표현을 사용하여 완전한 문장 형태로 작성하세요. 노이즈를 제거하면서 원래 의미를 최대한 유지하도록 하세요.
    모든 단계를 완료한 후, 추가할 요소가 있는지 검토하고 필요한 경우 다시 수정하세요.
    Take a deep breath and work on this problem step-by-step.
    """
)

# CSV 데이터 로드
data = pd.read_csv('./data/train.csv')

# 데이터셋 텍스트 추출 및 프롬프트 구성
input_texts = data['text'].tolist()
formatted_input = "\n".join(input_texts)
formatted_prompt = prompt_template.format(input_text=formatted_input)

# OLLama 모델에 프롬프트 전달하여 정제된 텍스트 생성
response = llm(formatted_prompt)

# 응답을 개별 정제된 텍스트 리스트로 변환
cleaned_texts = response.split('\n')

# 정제된 텍스트를 데이터프레임에 추가
if len(cleaned_texts) == len(data):
    data['cleaned_text'] = cleaned_texts
else:
    print("Error: 응답 결과 수와 입력 데이터 수가 일치하지 않습니다.") #응답 수 확인: 응답 결과의 텍스트 수와 원본 데이터 개수가 일치하는지 검증하는 코드를 추가하여, 응답이 제대로 반환되지 않았을 경우 디버깅이 가능하도록 했음

# 정제된 데이터 저장
data.to_csv('./data/cleaned_train.csv', index=False)
print("정제된 데이터가 cleaned_train.csv 파일에 저장되었습니다.")
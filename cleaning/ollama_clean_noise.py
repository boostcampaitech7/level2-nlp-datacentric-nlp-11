from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import pandas as pd
import time

# 텍스트 정제를 위한 Ollama 모델 초기화
#llm_reasoning = Ollama(model="gemma2:27b")
llm = Ollama(model="gemma2:27b")


# ASCII 코드 대체 여부에 대한 분류 프롬프트 설정
prompt_reasoning = """당신은 주어진 기사 제목의 노이즈를 복원하는 어시스턴트로, 아스키 코드 문자를
적절한 문자로 바꿔 정상적인 뉴스 기사제목을 생성해야 합니다."""

# 프롬프트 템플릿 설정
prompt_template = ChatPromptTemplate.from_template(
    f"""
    {prompt_reasoning}
    
    다음의 뉴스 기사 제목에는 20~40%의 글자를 임의의 아스키 코드 문자로 변환하는 노이즈가 적용되어 있습니다.
    이 노이즈를 복원하여 적절한 뉴스 기사 제목으로 만들어주세요. 복원된 뉴스 기사 제목 외 기타 부연 설명은 출력하지 
    말아야 합니다. 복원 전 뉴스 기사 제목: {{input_text}}
    
    복원된 뉴스 기사 제목:
    """
)

# CSV 데이터 로드
data = pd.read_csv('./data/train.csv')

# 개별 텍스트 복원
cleaned_texts = []
for input_text in data['text']:
    # 각 텍스트에 대해 프롬프트 생성
    formatted_prompt = prompt_template.format(input_text=input_text)

    # Ollama 모델에 프롬프트 전달하여 텍스트 복원
    response = llm(formatted_prompt)

    # 응답을 정제된 텍스트로 저장
    cleaned_texts.append(response.strip())

    # 요청 간에 약간의 대기 시간을 두어 API 호출 과부하를 방지
    #time.sleep(0.2) # 필요시 조정 가능


# 정제된 텍스트를 데이터프레임에 추가
data['cleaned_text'] = cleaned_texts

# 정제된 데이터 저장
data.to_csv('./data/cleaned_train.csv', index=False)
print("정제된 데이터가 cleaned_train.csv 파일에 저장되었습니다.")
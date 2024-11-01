from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# 텍스트 정제를 위한 Ollama 모델 초기화
#llm_reasoning = Ollama(model="gemma2:27b")
llm = Ollama(model="gemma2:27b")


# ASCII 코드 대체 여부에 대한 분류 프롬프트 설정
prompt = ChatPromptTemplate.from_template(
    """
    다음 텍스트에서 불필요한 노이즈(ASCII 코드, 특수 문자, 이상한 공백, 숫자 사이에 끼어 있는 불필요한 문자)를 제거하고
    가능한 한 읽기 쉬운 형태로 정제해 주세요. 노이즈 제거 후, 최종 정제된 텍스트만 반환해 주세요.
    텍스트 수: {batch_size}
    텍스트 목록:
    {input_text}
    """
)

# 예제 텍스트 데이터
data_samples = [
    "정i :파1 미사z KT( 이용기간 2e 단] Q분종U2보",
    "K찰.국DLwo 로L3한N% 회장 2 T0&}송=",
    "m 김정) 자주통일 새,?r열1나가야1보",
    "갤노트8 주말 27만대 개통…시장은 불법 보조금 얼룩",
    "pI美대선I앞두고 R2fr단 발] $비해 감시 강화",
    "oi 매력 R모h츠a열#w3약 >l·주가 고Q/진",
    "아이`XSI수리0* b대`…맥3 디dF레< 41/",
    "문/인 당2 4nS 민관2동7사위 /""X보 철거tt",
    "개R전 연w정연H 작가",
    "KIA I수단·팬nI께하는:호kQ4M족 한마5 S최",
    "현N차sJ <e 임원6늘려…~세z 리g (보j 육y",
    "-선 폭:n@ 현장N조Z",
    "STJ레콤 J분기 영D익t4천105t…2>1％ 증가",
    "생r인증②_D안*a제bK 유@되면 대J _가끝"
]

batch_size = len(data_samples)
input_text = "\n".join([f"{i+1}. {text}" for i, text in enumerate(data_samples)])

# 프롬프트 구성
formatted_prompt = prompt.invoke({"batch_size": batch_size, "input_text": input_text})

# OLLama 모델에 프롬프트 전달하여 정제된 텍스트 생성
response = llm(formatted_prompt)

# 결과 출력
print("정제 결과:")
print(response)
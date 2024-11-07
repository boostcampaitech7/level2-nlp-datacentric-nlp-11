import pandas as pd
from langchain.prompts import ChatPromptTemplate
from tqdm import tqdm
import re
from langchain_ollama import OllamaLLM
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 태그 리스트 설정
tags = "#AI #IT기술 #IT기업 #건설 #경영 #경영성과 #경영정보 #경제 #경제분석 #경제성장 #경제정세 #경제정책 #경제지표 #경제활성화 #고용정책 #공급망 #국제 #국제무역 #국제사건 #국제협력 #규제 #그래픽 #금융 #금융상품 #금융시장 #금융지원 #기술 #기술개발 #기술산업 #기업 #기업경영 #기업동향 #기업성과 #기업실적 #기업재무 #기업전략 #노동 #도시계획 #바이오 #보험 #복지 #부동산 #부동산시장 #사회문제 #사회복지 #산업 #산업동향 #산업분석 #성과 #성장 #성장전략 #소비 #스타트업 #스포츠 #시장 #시장동향 #시장분석 #시장점유율 #식품안전 #안전 #업종 #엔터테인먼트 #은행서비스 #이동통신 #인재 #인프라 #자동차 #자동차산업 #자율주행 #재난 #재무 #재무성과 #정책 #정치 #제조 #제조업 #제품 #주식 #주식시장 #증권시장 #지속가능성 #지역개발 #통신 #통신사 #투자 #투자유치 #해운산업 #혁신 #환율"
tags = tags.split()
tags = [tag.strip() for tag in tags]
tags = list(set(tags))

# 프롬프트 템플릿 설정
prompt_template = """
{iter_number}
### Prompt:
뉴스 카테고리: 경제
하위 태그:
{selected_tags}

예시:
1. 그래프3대 그룹 시장 점유율 감소
2. 하이투자 한전 2분기부터 실적 개선 전망…목표주가↑
3. 기업은 5억원 규모 성장모델 조성 추진

1. 뉴스 카테고리와 하위 태그 하나 선택해서 뉴스 기사 제목을 출력하세요.
2. 총 10개의 뉴스 기사 제목을 출력 형식에 맞춰서 작성하세요,
3. 뉴스 기사제목에는 ","를 넣지마세요.
4. 뉴스기사제목과 #선택한태그 사이의 구분자로 ","를 넣어야합니다.
5. 마크다운 bold를 사용하지 않고 최종 출력의 형식을 엄격하게 지켜서 작성하세요.

### 최종 출력 형식:

뉴스기사제목, #선택한태그
뉴스기사제목, #선택한태그
뉴스기사제목, #선택한태그
...
"""


# LLM 설정 (각 스레드에서 생성)
def create_chain():
    prompt_label_info = ChatPromptTemplate.from_template(prompt_template)
    llm_label_info = OllamaLLM(model="gemma2:27b", temperature=0.7)
    chain = prompt_label_info | llm_label_info
    return chain


# 작업 함수 정의
def process_iteration(i):
    try:
        selected_tags = random.sample(tags, 10)
        selected_tags_str = " ".join(selected_tags)
        chain = create_chain()
        response_label_info = chain.invoke(
            {"iter_number": i, "selected_tags": selected_tags_str}
        )
        # 응답을 라인별로 분리하여 저장
        responses = [
            line.strip()
            for line in response_label_info.strip().split("\n")
            if line.strip()
        ]
        result = []
        for resp in responses:
            match = resp.split(",")
            if len(match) >= 2:
                text = match[0].strip()
                tag = match[1].strip()
                result.append((text, tag))
            else:
                result.append(("NOHEADLINE", "NOTAG"))
        return result
    except Exception as e:
        print(f"Exception occurred in iteration {i}: {e}")
        return [("NOHEADLINE", "NOTAG")]


# 메인 실행 부분
if __name__ == "__main__":
    # 배치 크기 설정
    iter_size = 100

    # 결과를 저장할 리스트 초기화
    label_category_list = []

    # 프로그레스 바 설정
    pbar = tqdm(total=iter_size, desc="Processing Batches")
    lock = threading.Lock()  # 프로그레스 바 업데이트를 위한 락

    # 스레드 풀 생성
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_iteration, i) for i in range(iter_size)]
        for future in as_completed(futures):
            try:
                result = future.result()
                label_category_list.extend(result)
            except Exception as e:
                print(f"Exception occurred: {e}")
            finally:
                with lock:
                    pbar.update(1)
    pbar.close()

    # 라벨과 카테고리 정보를 데이터프레임으로 변환
    df_labels = pd.DataFrame(label_category_list, columns=["text", "tag"])

    df_labels["text"] = (
        df_labels["text"].str.replace(r"\*\*", "", regex=True).str.strip()
    )

    # 데이터프레임 출력 및 저장
    print(df_labels)
    df_labels.to_csv("economic_headlines.csv", index=False)

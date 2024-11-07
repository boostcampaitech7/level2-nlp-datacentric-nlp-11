import pandas as pd
from langchain.prompts import ChatPromptTemplate
from tqdm import tqdm
import re
from langchain_ollama import OllamaLLM
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 태그 리스트 설정
tags = "#IT #IT기술 #건강 #게임산업 #경기 #경영 #경제 #경제정세 #경제정책 #고등교육 #과학 #교육 #교육기회 #교육정책 #교육제도 #교육환경 #교통 #교통안전 #국제경제 #국제관계 #국제정세 #국회 #규제 #금융 #기술 #기술개발 #기업 #기업지원 #노동 #대중교통 #도시개발 #돌봄 #문화 #문화산업 #미디어 #방송 #방송산업 #범죄 #법규 #법률 #법률개정 #보건 #보안 #복지 #봉사활동 #부동산 #부동산시장 #사고 #사법 #사회문제 #사회복지 #사회봉사 #사회운동 #사회이슈 #산업 #산업혁신 #생명과학 #선수 #성추행 #스포츠뉴스 #스포츠스타 #안전 #엔터테인먼트 #역사 #연극 #예술 #온라인교육 #외교 #외교관계 #웰빙 #윤리 #인권 #일자리 #입법 #자연 #재난 #재정 #정당 #정책 #정치 #정치논란 #정치이슈 #주택 #지방정책 #지속가능성 #채용 #취업정보 #콘텐츠분석 #학교교육 #학교생활 #학교운영 #학생 #학생활동 #학업 #행정 #환경보호"
tags = tags.split()
tags = [tag.strip() for tag in tags]
tags = list(set(tags))

# 프롬프트 템플릿 설정
prompt_template = """
{iter_number}
### Prompt:
뉴스 카테고리: 사회
하위 태그:
{selected_tags}

예시:
1. 홍콩 시위에 원격조종 사제폭탄 등장…경찰 테러리스트와 비슷
2. 페이스북 伊서 과징금 13억원…개인정보 보호법 위반
3. 알뜰폰 헬로모바일 청소년 요금 반값 할인

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
    df_labels.to_csv("social_headlines.csv", index=False)

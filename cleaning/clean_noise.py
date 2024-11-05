import pandas as pd
import re
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_ollama import OllamaLLM
import os


def check_and_filter_noise():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(parent_dir, "./data/train.csv")
    sorted_file_path = os.path.join(parent_dir, "./data/noise_sorted_train.csv")
    filtered_file_path = os.path.join(parent_dir, "./data/filtered_train.csv")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"'{file_path}' 경로에 파일이 존재하지 않습니다.")

    df = pd.read_csv(file_path)

    ascii_regex = re.compile(r"[!\"#$%&'\(\)*+,\-/:;<=>?@\[\\\]^_`\{\|\}~]")
    num_or_alphabet_regex = re.compile(r"[0-9a-zA-Z]")

    ascii_cnt = []
    ascii_ratio = []
    num_or_alphabet_ratio = []

    for _, row in df.iterrows():
        text_length = len(row["text"])
        ascii_chars = len(ascii_regex.findall(row["text"]))
        num_or_alphabet_chars = len(num_or_alphabet_regex.findall(row["text"]))
        ascii_cnt.append(ascii_chars)
        ascii_ratio.append(ascii_chars / text_length)
        num_or_alphabet_ratio.append(num_or_alphabet_chars / text_length)

    df["ascii_cnt"] = ascii_cnt
    df["ascii_ratio"] = ascii_ratio
    df["num_or_alphabet_ratio"] = num_or_alphabet_ratio
    df["noise_ratio"] = df["ascii_ratio"] + df["num_or_alphabet_ratio"]
    sorted_df = df.sort_values(
        by=["noise_ratio", "num_or_alphabet_ratio"], ascending=[True, True]
    )

    sorted_df.to_csv(sorted_file_path, index=False)

    filtered_df = df[
        (df["noise_ratio"] >= 0.22) & (df["noise_ratio"] <= 0.4) & (df["ascii_cnt"] > 1)
    ]

    filtered_df.to_csv(filtered_file_path, index=False)
    return filtered_df


def clean_noise():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filtered_file_path = os.path.join(parent_dir, "./data/filtered_train.csv")
    cleaned_file_path = os.path.join(parent_dir, "/data/cleaned_filtered_train.csv")

    if os.path.isfile(filtered_file_path):
        filtered_df = pd.read_csv(filtered_file_path)
    else:
        filtered_df = check_and_filter_noise()

    noisy_texts = filtered_df["text"]

    llm = OllamaLLM(model="gemma2:27b", temparature=0.15)

    template = "당신은 주어진 한국 뉴스 기사 제목의 노이즈를 복원하는 어시스턴트로, 아스키 코드 문자를 적절한 문자로 바꿔 정상적인 뉴스 기사제목으로 복원해야합니다."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    prompt = PromptTemplate(
        input_variables=["headline"],
        template="""
            "복원 전 뉴스 기사 제목"은 정상적인 한국 뉴스 기사 제목 중 약 20~50%의 글자를 임의의 아스키 코드 문자로 변환한 문장입니다. 주변 글자를 고려하면서 아스키 코드 문자를 다시 적절한 문자로 변환하여 정상적인 뉴스 기사 제목으로 복원해야 합니다.

            아래의 복원 예시를 참고하세요
            복원 전: 서^, 여행A기 안e! 도시 7위…가h 위&한 도 는?
            복원 후: 서울, 여행하기 안전한 도시 7위…가장 위험한 도시는?

            복원 전: 엔vJ아, 인텔 2어내고 입$... 美 다우b균eq도 AI #대
            복원 후: 엔비디아, 인텔 밀어내고 입성... 美 다우평균지수도 AI 시대

            노이즈가 심해 복원할 수 없는 부분은 문장 내 유의미한 단어들을 기반으로 적절한 단어를 새롭게 생성하세요. 복원된 뉴스 기사 제목 외 기타 부연 설명은 출력하지 말아야 합니다.

            복원 전 뉴스 기사 제목: {headline}

            복원된 뉴스 기사 제목:
            """,
    )
    human_message_prompt = HumanMessagePromptTemplate(prompt=prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    num_threads = 5  # 병렬 처리에 사용할 스레드 수
    futures = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(llm.invoke, chat_prompt.format(headline=noisy_text)): idx
            for idx, noisy_text in tqdm(
                enumerate(noisy_texts),
                total=len(noisy_texts),
                desc="submitting prompts",
            )
        }

    responses = [None] * len(noisy_texts)
    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Processing tasks"
    ):
        idx = futures[future]
        responses[idx] = future.result().strip()

    cleaned_df = filtered_df.copy()
    cleaned_df["clean_text"] = responses

    cleaned_df.to_csv(cleaned_file_path, index=False, quoting=0)


if __name__ == "__main__":
    clean_noise()

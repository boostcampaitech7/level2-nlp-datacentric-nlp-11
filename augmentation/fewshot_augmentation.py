import os
import pandas as pd
from tqdm import tqdm
from langchain_ollama import OllamaLLM
from utils.dataframe_utils import concat_df
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed


def fewshot_augmentation(ref_data, level):
    samples_per_label = []
    for i in range(0, 7):
        samples = ref_data[ref_data["target"] == i].sample(n=20, random_state=42)
        sample_list = []
        for _, sample in samples.iterrows():
            sample_list.append({"headline": sample})
        samples_per_label.append(sample_list)

    llm = OllamaLLM(model="gemma2:27b")

    # label별로 다른 example을 전달하는 LLM chain 생성
    def generate_chain(label):
        example_prompt = PromptTemplate.from_template("기사 제목: {headline}")
        output_parser = JsonOutputParser()
        fewshot_prompt = FewShotPromptTemplate(
            suffix="기사 제목: ",
            examples=samples_per_label[label],
            example_prompt=example_prompt,
            prefix="""
            예시로 주어진 한국 뉴스 기사 제목들과 같은 카테고리에 속하는 뉴스 기사 제목을 20개 생성하세요. 단, 생성한 기사 제목에 쌍따옴표는 사용하지 말아야 하며, 생성한 기사 제목 외에 다른 부연 설명은 출력하지 말아야 합니다.
            출력 형식은 JSON으로 하며, 각 뉴스 기사 제목이 하나의 문자열로서 배열에 입력되도록 하세요.
            ex) ["Headline 1", "Headline 2", "Headline 3"]
            """,
        )

        prompt = fewshot_prompt.partial(
            format_instructions=output_parser.get_format_instructions()
        )
        chain = prompt | llm | output_parser
        return chain.invoke({})

    # 병렬 처리
    num_threads = 5
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(generate_chain, label): label for label in range(7)}

    # 생성 결과 입력 받기
    responses = [None] * 7
    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Processing tasks"
    ):
        idx = futures[future]
        responses[idx] = future.result()

    ids = []
    texts = []
    labels = []
    for i, headlines in enumerate(responses):
        ids.extend([f"aug-{level}-{i}-{j}" for j in range(len(headlines))])
        texts.extend([value.strip() for value in headlines])
        labels.extend([i] * len(headlines))

    augmented_df = pd.DataFrame(
        {
            "ID": ids,
            "text": texts,
            "target": labels,
        }
    )

    concat_df = pd.concat([ref_data, augmented_df])
    return concat_df


def repeat_augmentation(original_df, file_name="augmented"):
    data = original_df
    for level in tqdm(range(5)):
        data = fewshot_augmentation(data, level)

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(parent_dir, f"./data/{file_name}.csv")
    data.to_csv(file_path, index=False, quoting=0)


if __name__ == "__main__":
    # fewshot example로 전달하고 싶은 최초 데이터
    file_name_list = ["cleaned_filtered_train"]
    original_df = concat_df(file_name_list)
    # 증강된 파일의 이름을 설정하고 싶으면 두 번째 인자로 file_name 전달
    repeat_augmentation(original_df)

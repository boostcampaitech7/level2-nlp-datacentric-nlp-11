import re
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pykospacing import Spacing
from kiwipiepy import Kiwi


class T5TextNoiseCleaner:

    def __init__(self, model_name="google/mt5-xl", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.kiwi = Kiwi()
        self.pyko = Spacing()

    def clean(self, text):
        # 1. 띄어쓰기 교정 후, 노이즈 텍스트 <extra_id_#>으로 마스킹
        preproc_text = mask_noise_text(self.kiwi.space(text))
        # 2. 마스킹 텍스트 토큰화
        input_ids = self.tokenizer(preproc_text, return_tensors="pt").input_ids
        # 3. 마스킹 부분의 텍스트 예측
        output = self.model.generate(input_ids.to(self.device), max_length=50)
        # 4. 예측 텍스트 디코딩
        restore_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
        # 5. 마스킹된 부분을 예측 텍스트로 대체한 후, 띄어쓰기 교정
        clean_text = self.pyko(fill_mask(preproc_text, restore_text).replace(" ", ""))
        return clean_text

    def clean_dataset(self, file_path):
        df = pd.read_csv(file_path, encoding="utf-8")
        df.drop(index=df[df["is_noise"] == 0.0].index, inplace=True)

        clean_text_list = []
        for idx, sample in tqdm(df.iterrows()):
            clean_text_list.append(self.clean(sample["text"]))

        df["ID"] = ["mT5-" + id for id in df["ID"]]
        df["text"] = clean_text_list
        df.to_csv(
            f"mt5-clean_{file_path.split('/')[-1]}", index=False, encoding="utf-8"
        )


def mask_with_extra_id_cnt():
    counter = 0

    def repl(match):
        nonlocal counter
        replacement = f" <extra_id_{counter}> "
        counter += 1
        return replacement

    return repl


def mask_noise_text(sentence):
    # 한국어, 한자를 제외한 나머지 문자열 마스킹
    pattern = r"([^가-힣\u4E00-\u9FFF\s\…]+\s*[^가-힣\u4E00-\u9FFF]*)+"
    ko_cn_only = re.sub(pattern, mask_with_extra_id_cnt(), sentence)
    return ko_cn_only


def fill_mask(input_text, output_text):
    # 마스킹 부분을 예측 텍스트로 대체
    matches = re.findall(r"(<extra_id_\d+>)\s?([^<]*)", output_text)
    output_dict = {match[0]: match[1].strip() for match in matches}
    result = re.sub(
        r"(<extra_id_\d+>)",
        lambda x: output_dict.get(x.group(0), x.group(0)),
        input_text,
    )
    result = re.sub(r"(<extra_id_\d+>)", "", result)
    return result


if __name__ == "__main__":
    textcleaner = T5TextNoiseCleaner()
    print(textcleaner.clean("정i :파1 미사z KT( 이용기간 2e 단] Q분종U2보"))
    textcleaner.clean_dataset("/data/ephemeral/home/gj/data_is-noise.csv")

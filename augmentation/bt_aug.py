import pandas as pd
from googletrans import Translator
from tqdm import tqdm
import time
import concurrent.futures


class BackTransAug:

    def __init__(self, mode="all", lang="en"):
        self.mode = mode
        self.lang = lang

    def run(self, file_path):
        df = pd.read_csv(file_path)

        if self.mode == "all":
            pass
        elif self.mode == "noise_text":
            df.drop(index=df[df["is_noise"] == 0.0].index, inplace=True)
        elif self.mode == "clean_text":
            df.drop(index=df[df["is_noise"] == 1.0].index, inplace=True)

        texts = df["text"].tolist()
        augmented_texts = []
        max_workers = 8  # 시스템 성능에 따라 조정 가능

        augmented_texts = [None] * len(texts)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 각 작업에 인덱스를 포함하여 제출합니다.
            futures = {
                executor.submit(self._run, text): i for i, text in enumerate(texts)
            }

            # as_completed를 사용하여 완료된 작업을 처리합니다.
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(texts),
                desc="Backtranslating",
            ):
                index = futures[future]
                augmented_texts[index] = future.result()  # 결과를 원래 순서대로 저장

        df["text"] = augmented_texts
        df["ID"] = ["bt-" + id for id in df["ID"]]

        # 결과를 CSV로 저장
        df.to_csv(f"bt_{file_path.split('/')[-1]}", index=False)
        print("Backtranslation 완료 및 저장되었습니다.")

    def _run(self, text):
        translator = Translator()
        while True:
            try:
                # 영어로 번역
                translated = translator.translate(text, src="ko", dest=self.lang).text
                # 다시 원래 언어로 번역
                back_translated = translator.translate(
                    translated, src=self.lang, dest="ko"
                ).text
                return back_translated
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying in 2 seconds...")
                time.sleep(2)


if __name__ == "__main__":
    bt = BackTransAug("clean_text")
    bt.run("/data/ephemeral/home/gj/cluster-label_bt_mt5-clean_data_is-noise.csv")

import pandas as pd
from googletrans import Translator
from tqdm import tqdm
import time

# CSV 파일 경로 설정
CSV_FILE_PATH = (
    "/data/ephemeral/home/level2-nlp-datacentric-nlp-11/data/augmented_noise_data.csv"
)
OUTPUT_FILE_PATH = (
    "/data/ephemeral/home/level2-nlp-datacentric-nlp-11/output/backtranslated_train.csv"
)

translator = Translator()


# Back translation 함수
def backtranslate(text, src="ko", dest="en"):
    while True:
        try:
            # 한국어에서 영어로 번역 후, 다시 한국어로 번역
            translated_text = translator.translate(text, src=src, dest=dest).text
            back_translated_text = translator.translate(
                translated_text, src=dest, dest=src
            ).text
            return back_translated_text
        except Exception as e:
            time.sleep(2)  # 에러 발생 시 대기 후 재시도


def main():
    # CSV 파일 불러오기
    df = pd.read_csv(CSV_FILE_PATH)

    # text 열에 대해 back translation 적용
    tqdm.pandas()  # tqdm 진행바 적용
    df["backtranslated_text"] = df["text"].progress_apply(lambda x: backtranslate(x))

    # 결과를 CSV로 저장
    df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Back translation 완료: {OUTPUT_FILE_PATH}에 저장되었습니다.")


if __name__ == "__main__":
    main()

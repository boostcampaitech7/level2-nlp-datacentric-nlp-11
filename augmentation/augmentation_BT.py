import pandas as pd
from googletrans import Translator

data = pd.read_csv('clean_train_data_aug-google.csv') # 데이터 불러오기

data['text'] = data['clean_text']

for idx in range(len(data)):
    if data['is_noise'][idx] == 1: # is_noise 컬럼이 1인 데이터만 뽑기
        data['text'][idx] = data['clean_text'][idx]
    else:
        data.drop(index=idx, inplace=True)

dataset_train = data
del dataset_train['text'] # 불필요한 컬럼 중 text를 제외시키기
dataset_train = dataset_train.reindex(columns=['ID', 'clean_text', 'target', 'is_noise'])
translator = Translator()

# Back-translation (BT) 함수 구현 (과제 1 참고)
def backtranslate(text, src="ko", dest="en"):
    while True:
        try:
            translated = translator.translate(text, dest=dest).text
            back_translated = translator.translate(translated, dest=src).text
            return back_translated
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying...")

# 증강할 텍스트 열 이름 설정
text_column = 'clean_text'  # 예: 증강하려는 텍스트가 있는 열 이름
output_csv_path = 'clean_train_data_augmented_ppg_fr.csv'

# Back-Translation 적용 
dataset_train['clean_text'] = dataset_train[text_column].apply(lambda x: backtranslate(x))

# 증강된 데이터를 새로운 CSV 파일로 저장
dataset_train.to_csv(output_csv_path, index=False)
print(f"증강된 데이터가 {output_csv_path}에 저장되었습니다.")
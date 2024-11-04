import pandas as pd
from cleanlab.classification import CleanLearning
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

# found_noise_train.csv 파일에서 노이즈 여부에 따라 데이터 분리
found_noise_data = pd.read_csv('./data/found_noise_train.csv')


# 노이즈가 없는 데이터 (noise=0) - 라벨 오류가 있을 가능성이 높은 데이터
data_with_label_errors = found_noise_data[found_noise_data['noise'] == 0].copy()
X_with_errors = data_with_label_errors['text']
y_with_errors = data_with_label_errors['target']

# 노이즈가 있는 데이터 (noise=1) - 라벨 오류가 없는 데이터의 ID만 가져옴
data_without_label_errors = found_noise_data[found_noise_data['noise'] == 1]['ID']

# cleaned_train.csv 파일에서 정제된 데이터 로드
cleaned_data = pd.read_csv('./data/cleaned_train_v2.csv')

# 노이즈가 있는 데이터의 ID와 일치하는 cleaned_train 데이터를 가져와 학습용 데이터로 구성
# 라벨 오류가 없는, 노이즈가 대부분 수정된 데이터들.
data_noisy_cleaned = cleaned_data[cleaned_data['ID'].isin(data_without_label_errors)]
X_no_errors = data_noisy_cleaned['cleaned_text']
y_no_errors = data_noisy_cleaned['target']



# CountVectorizer로 텍스트 벡터화 준비
vectorizer = CountVectorizer()

# 라벨 오류가 없는 데이터로 학습 데이터 준비
X_no_errors_vect = vectorizer.fit_transform(X_no_errors)
y_no_errors = y_no_errors

# 라벨 오류가 있는 데이터로 예측 데이터 준비
X_with_errors_vect = vectorizer.transform(X_with_errors)
y_with_errors = y_with_errors

# 모델 초기화 및 cleanlab 설정
clf = RandomForestClassifier(random_state=42)
cleanlab_model = CleanLearning(clf)

# cleanlab을 이용해 라벨 오류 없는 데이터로 학습
cleanlab_model.fit(X_no_errors_vect, y_no_errors)

# 라벨 오류가 있는 데이터에서 오류 탐지 및 수정
predicted_labels = cleanlab_model.predict(X_with_errors_vect)

# 라벨 오류가 있는 데이터에 수정된 라벨 추가
data_with_label_errors['corrected_target'] = predicted_labels

# 결과 저장
data_with_label_errors[['ID', 'text', 'target', 'corrected_target']].to_csv("corrected_with_errors.csv", index=False)
print("라벨 오류가 수정된 데이터셋이 'corrected_with_errors.csv'에 저장되었습니다.")
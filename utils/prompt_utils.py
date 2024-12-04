import re
import time
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# cuda cache 삭제
#torch.cuda.empty_cache()

# noise ratio 계산 함수
def calculate_noise_ratio(text):
    # 의미 있는 문자 (한글 및 영문 알파벳만 남김)
    clean_text = re.sub(r'[^ㄱ-ㅎ가-힣a-zA-Z\s]', '', text)
    
    # 노이즈 비율 계산
    noise_ratio = 1 - (len(clean_text) / len(text)) if len(text) > 0 else 1
    return noise_ratio


# 데이터 라벨별 그룹화 확인
def label_group_count(data):
    # 데이터 로드
    #data = pd.read_csv('../data/new_cleaned_augmented_train.csv')
    # 라벨별 문장 그룹화
    #grouped_data = data.groupby('target')['text'].apply(list)
    current_counts = data['target'].value_counts().to_dict()
    '''
    for i in range(7):
        print(len(grouped_data[i]))
    '''
    return current_counts


# Ollama 모델 호출 재시도 함수
def call_ollama_with_retry(llm, prompt, retries=10, delay=5):
    for attempt in range(retries):
        try:
            response = llm.invoke(prompt)
            return response
        except httpx.ConnectError as e:
            print(f"Connection Error: {e}. Retrying {attempt + 1}/{retries} in {delay} seconds...")
            time.sleep(delay) # 재시도 전 대기
    raise Exception("Failed to connect to Ollama after multiple attempts.")


# 중복 및 유사 텍스트 제거
def filter_similar_texts(texts, label, similarity_threshold):
    """
    Filters out texts that are too similar based on cosine similarity.

    Args:
        texts (list): List of text strings to filter.
        label (int): Label to assign to each filtered text.
        similarity_threshold (float): Threshold for cosine similarity to filter similar texts.

    Returns:
        pd.DataFrame: DataFrame containing filtered texts and associated labels.
    """
    filtered_texts = []
    vectorizer = TfidfVectorizer().fit_transform(texts)
    cosine_matrix = cosine_similarity(vectorizer)

    for i in range(len(texts)):
        # Check if text is sufficiently different from previously filtered texts
        if all(cosine_matrix[i][j] < similarity_threshold for j in range(i)):
            filtered_texts.append({'text': texts[i], 'target': label})

    return filtered_texts

# ID 값 변경
def modify_ids(df):
    # 마지막 숫자를 900부터 부여
    start_number = 1018
    for i in range(len(df)):
        prefix = "_".join(df.loc[i, 'ID'].split('_')[:-1])  # ID의 마지막 숫자 부분을 제외한 앞부분
        new_suffix = f"{start_number:05d}"  # 1018부터 시작하여 5자리 형식 유지
        df.loc[i, 'ID'] = f"{prefix}_{new_suffix}"
        start_number += 1  # 다음 번호로 증가
    df.to_csv('../data/new_id_aug_train_cleaned.csv', index=False)

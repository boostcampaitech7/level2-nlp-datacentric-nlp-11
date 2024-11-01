import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split

# 시드 값 설정 (재현성을 위해 고정)
SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# GPU 사용 가능 여부에 따라 디바이스 선택
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#DEVICE

# 현재 작업 디렉토리와 데이터, 출력 디렉토리 설정
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

# 절대 바꾸면 안되는 부분 - 사전 학습된 모델 이름과 토크나이저, 분류 모델 로드 (7개 레이블로 분류)
model_name = 'klue/bert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

# 학습 데이터를 로드하고 훈련/검증 데이터로 분리
data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
dataset_train, dataset_valid = train_test_split(data, test_size=0.3, random_state=SEED)

# 데이터셋 클래스 정의
class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text'] # 텍스트 열 추출
        targets = data['target'] # 타겟 열 추출
        self.inputs = []; self.labels = [] #토큰화된 입력을 저장할 리스트; 레이블을 저장할 리스트
        for text, label in zip(input_texts, targets): # 입력 텍스트를 토큰화하고 패딩 및 텐서 변환
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx): # 데이터셋의 개별 항목을 가져올 때 사용하는 함수
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }

    def __len__(self): # 데이터셋의 길이 반환
        return len(self.labels)
    
# 훈련 및 검증 데이터를 BERTDataset 객체로 변환
data_train = BERTDataset(dataset_train, tokenizer)
data_valid = BERTDataset(dataset_valid, tokenizer)

# 데이터 패딩을 위한 데이터 콜레이터 정의
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# f1 평가 지표 로드 및 계산 함수 정의
f1 = evaluate.load('f1')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1) # 예측값에서 가장 높은 확률을 가진 레이블 선택
    return f1.compute(predictions=predictions, references=labels, average='macro')

### for wandb setting
#os.environ['WANDB_DISABLED'] = 'true'

# 학습 파라미터 설정
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    logging_strategy='steps',
    eval_strategy='steps',
    save_strategy='steps',
    logging_steps=100, # 100 스텝마다 로그 기록
    eval_steps=100, # 100 스텝마다 평가
    save_steps=100, # 100 스텝마다 체크포인트 저장
    save_total_limit=2, # 최신 2개 모델만 저장
    learning_rate= 2e-05, # 학습률 설정
    adam_beta1 = 0.9, # Adam 옵티마이저의 베타1 파라미터
    adam_beta2 = 0.999, # Adam 옵티마이저의 베타2 파라미터
    adam_epsilon=1e-08, # Adam 옵티마이저의 epsilon 파라미터
    weight_decay=0.01, # 가중치 감소 적용
    lr_scheduler_type='linear',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2, # 총 학습 epoch 수
    load_best_model_at_end=True, # 가장 좋은 모델을 학습 종료 시 로드
    metric_for_best_model='eval_f1',
    greater_is_better=True, # 높은 f1 값이 더 좋은 모델로 간주
    seed=SEED
)

# 트레이너 객체 생성 (훈련, 평가, 데이터 콜레이터, 평가 함수 포함)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 모델 학습 시작
trainer.train()

# 테스트 데이터 로드
dataset_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))


# 테스트 데이터를 평가하여 예측값 생성
model.eval()  # 평가 모드로 설정
preds = []    # 예측 결과 저장할 리스트

for idx, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"):
    # 각 테스트 샘플 토큰화 및 디바이스에 로드
    inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
    with torch.no_grad(): # 그래디언트 계산 비활성화
        logits = model(**inputs).logits # 모델 예측 결과
        pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
        preds.extend(pred) # 예측값 저장

# 예측 결과를 새로운 컬럼으로 추가하고 CSV 파일로 저장
dataset_test['target'] = preds
dataset_test.to_csv(os.path.join(BASE_DIR, 'output.csv'), index=False)
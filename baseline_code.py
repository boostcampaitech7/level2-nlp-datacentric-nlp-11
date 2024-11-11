# 사용법
# 기본 훈련과 평가 실행 (예측 없이):
# python baseline_code.py --data_name train

# 실행 명령어 예시:
# python baseline_code.py --data_name train --mode predict

# --data_name : 사용하고자 하는 데이터셋 이름을 지정 (예: train, train_corrected_using_cleanlab)
# --mode : predict로 설정하면 훈련 후 예측 모드를 실행하여 테스트 데이터에 대한 예측을 저장

import argparse
from datetime import datetime
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from tabulate import tabulate

import torch
from torch.utils.data import Dataset

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# 최대 시퀀스 길이 설정
MAX_SEQ_LENGTH = 36  # 필요에 따라 조정하세요

# 시드 설정
SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 디바이스 설정
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")

# 디렉토리 설정
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "./data")
OUTPUT_DIR = os.path.join(BASE_DIR, "./output")

# 모델과 토크나이저 로드
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 커스텀 데이터셋 클래스 정의
class BERTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=MAX_SEQ_LENGTH):
        input_texts = data["text"]
        targets = data["target"]
        self.inputs = []
        self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,  # max_length 설정
                return_tensors="pt",
            )
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx]["input_ids"].squeeze(0),
            "attention_mask": self.inputs[idx]["attention_mask"].squeeze(0),
            "labels": self.labels[idx].squeeze(0),
        }

    def __len__(self):
        return len(self.labels)


# 메트릭 계산 함수 정의
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    macro_f1 = f1_score(labels, predictions, average="macro")
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    return {"macro_f1": macro_f1, "precision": precision, "recall": recall}


def main():
    parser = argparse.ArgumentParser(description="Train or Predict with BERT model.")
    parser.add_argument(
        "--mode", type=str, choices=["predict"], help="모드 선택: 'predict'"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        help="데이터셋 이름 (예: train_corrected_using_cleanlab)",
    )

    args = parser.parse_args()

    data_name = args.data_name
    timestamp = datetime.now().strftime("%m%d%H%M")
    run_name = f"{timestamp}_{data_name}"
    wandb.init(
        project="datacentric_sangyeop",  # 원하는 프로젝트 이름으로 변경하세요
        config={
            "model_name": "klue/bert-base",
            "num_labels": 7,
            "batch_size": 32,
            "learning_rate": 2e-05,
            "epochs": 2,
            "seed": 456,
            "max_sequence_length": MAX_SEQ_LENGTH,
        },
        name=run_name,
    )

    # 데이터 로드 및 분할
    data = pd.read_csv(os.path.join(DATA_DIR, f"{data_name}.csv"))
    dataset_train, dataset_valid = train_test_split(
        data, test_size=0.3, random_state=SEED
    )
    # 데이터셋 인스턴스 생성
    data_train = BERTDataset(dataset_train, tokenizer, max_length=MAX_SEQ_LENGTH)
    data_valid = BERTDataset(dataset_valid, tokenizer, max_length=MAX_SEQ_LENGTH)

    # DataCollator 설정
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=MAX_SEQ_LENGTH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=7
    ).to(DEVICE)
    # 평가 지표 로드
    f1 = evaluate.load("f1")

    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=False,  # 테스트 예측은 따로 수행
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="no",  # Checkpoint 저장 비활성화
        logging_steps=100,
        eval_steps=100,
        learning_rate=2e-05,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        load_best_model_at_end=False,  # Best 모델 로딩 비활성화
        seed=SEED,
        report_to=["wandb"],  # wandb에 보고하도록 설정
        run_name=run_name,  # TrainingArguments에 시간 기반 run_name 추가
    )

    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 모델 훈련
    trainer.train()

    # Validation 세트 평가
    eval_results = trainer.evaluate(eval_dataset=data_valid)

    # Validation 메트릭 로그
    wandb.log(
        {
            "validation_macro_f1": eval_results["eval_macro_f1"],
            "validation_precision": eval_results["eval_precision"],
            "validation_recall": eval_results["eval_recall"],
        }
    )
    # 표 스타일로 출력할 데이터 생성
    table = [
        ["Metric", "Score"],
        ["Macro F1 Score", f"{eval_results['eval_macro_f1']:.4f}"],
        ["Precision", f"{eval_results['eval_precision']:.4f}"],
        ["Recall", f"{eval_results['eval_recall']:.4f}"],
    ]
    # 터미널에 Validation 메트릭 출력

    print("\n=== Validation Set Evaluation Metrics ===")
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
    print("==========================================\n")
    if args.mode == "predict":
        # 테스트 데이터 로드
        dataset_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

        model.eval()
        preds = []

        # 테스트 세트 예측
        for idx, sample in tqdm(
            dataset_test.iterrows(), total=len(dataset_test), desc="Predicting"
        ):
            inputs = tokenizer(
                sample["text"],
                padding="max_length",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,  # max_length 설정
                return_tensors="pt",
            ).to(DEVICE)
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = (
                    torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
                )
                preds.extend(pred)

        # 'predicted' 컬럼에 예측 결과 저장
        dataset_test["target"] = preds

        # F1 점수를 파일 이름에 포함하여 저장
        macro_f1 = eval_results["eval_macro_f1"]
        formatted_f1 = f"{macro_f1:.4f}"  # F1 점수를 소수점 4자리로 포맷
        output_filename = f"{run_name}_f1_{formatted_f1}_output.csv"
        dataset_test.to_csv(os.path.join(OUTPUT_DIR, output_filename), index=False)

    # 훈련 종료 후 wandb 종료
    wandb.finish()


if __name__ == "__main__":
    main()

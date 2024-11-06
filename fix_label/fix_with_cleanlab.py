import os
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from cleanlab.filter import find_label_issues

from utils.dataframe_utils import format_dataframe


def fix_with_cleanlab(correct_dataset, error_dataset, confidence_threshold):
    # 1. klue/bert-base 모델 1차 학습(정상 데이터셋)
    SEED = 456
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "./data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "./output")

    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=7
    ).to(DEVICE)

    correct_dataset_train, correct_dataset_valid = train_test_split(
        correct_dataset, test_size=0.3, random_state=SEED
    )

    class BERTDataset(Dataset):
        def __init__(self, data, tokenizer):
            input_texts = data["text"]
            targets = data["target"]
            self.inputs = []
            self.labels = []
            for text, label in zip(input_texts, targets):
                tokenized_input = tokenizer(
                    text, padding="max_length", truncation=True, return_tensors="pt"
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

    data_train = BERTDataset(correct_dataset_train, tokenizer)
    data_valid = BERTDataset(correct_dataset_valid, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        macro_f1 = f1_score(labels, predictions, average="macro")
        precision = precision_score(labels, predictions, average="macro")
        recall = recall_score(labels, predictions, average="macro")
        return {"macro_f1": macro_f1, "precision": precision, "recall": recall}

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        learning_rate=2e-05,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        load_best_model_at_end=False,
        greater_is_better=True,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 2. 레이블 오류 탐지/수정
    error_data_encodings = BERTDataset(error_dataset, tokenizer)

    model.eval()

    error_data_logits = trainer.predict(error_data_encodings).predictions
    error_data_probs = torch.softmax(torch.tensor(error_data_logits), dim=1).numpy()

    max_probs = np.max(error_data_probs, axis=1)
    high_confidence_preds = []
    high_confidence_labels = []
    indice = []
    for i in range(len(max_probs)):
        if max_probs[i] >= confidence_threshold:
            indice.append(i)
            high_confidence_labels.append(error_dataset.iloc[i]["target"])
            high_confidence_preds.append(error_data_probs[i])

    high_confidence_labels = np.array(high_confidence_labels)
    high_confidence_preds = np.array(high_confidence_preds)

    label_errors = find_label_issues(
        labels=high_confidence_labels,
        pred_probs=high_confidence_preds,
    )

    fixed_labels = np.argmax(high_confidence_preds, axis=1)
    fixed = [0] * len(error_dataset)
    is_fixed_or_correct = [False] * len(error_dataset)
    for i, idx in enumerate(indice):
        fixed[idx] = (
            fixed_labels[i] if label_errors[i] else error_dataset.iloc[idx]["target"]
        )
        is_fixed_or_correct[idx] = True

    error_dataset["is_fixed_or_correct"] = is_fixed_or_correct
    error_dataset["fixed_label"] = fixed

    fixed_df = error_dataset[error_dataset["is_fixed_or_correct"] == True].copy()
    fixed_df["target"] = fixed_df["fixed_label"]
    fixed_df = fixed_df.drop(columns=["is_fixed_or_correct", "fixed_label"])

    output_path = os.path.join(DATA_DIR, "./fixed_data_cleanlab.csv")
    fixed_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 정상 데이터셋(노이즈 복원)
    correct_file_name = "clean=gj-sy-jh_aug=gjbt"
    correct_dataset = format_dataframe(correct_file_name)

    # 레이블 정상 여부를 알 수 없는 데이터셋(텍스트 노이즈 X)
    dataset_file_name = "noise_sorted_train"
    dataset_file_path = os.path.join(parent_dir, f"./data/{dataset_file_name}.csv")
    dataset = pd.read_csv(dataset_file_path)
    error_dataset = dataset[(dataset["noise_ratio"] < 0.2)].copy()
    error_dataset = error_dataset[["ID", "text", "target"]]

    # 가장 높은 예측 확률이 confidence_threshold 이상인 데이터에 대해서만 레이블 오류 여부 판단
    confidence_threshold = 0.7

    fix_with_cleanlab(correct_dataset, error_dataset, confidence_threshold)

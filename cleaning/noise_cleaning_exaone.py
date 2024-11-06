import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")

# 데이터 예시
data_samples = [
    {"text": "정i :파1 미사z KT( 이용기간 2e 단] Q분종U2보", "target": 4},
    {"text": "K찰.국DLwo 로L3한N% 회장 2 T0&}송=", "target": 3},
    {"text": "m 김정) 자주통일 새,?r열1나가야1보", "target": 2},
    {"text": "갤노트8 주말 27만대 개통…시장은 불법 보조금 얼룩", "target": 5},
    {"text": "pI美대선I앞두고 R2fr단 발] $비해 감시 강화", "target": 6},
    {"text": "美성인 6명 중 1명꼴 배우자·연인 빚 떠안은 적 있다", "target": 0},
    {"text": "프로야구~롯TKIAs광주 경기 y천취소", "target": 1}
]

def clean_text(text):

    prompt = f"이 데이터에서 노이즈를 제거해줘. '{text}'"

    messages = [
        {"role": "system", "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    output = model.generate(
        input_ids.to("cuda"),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=128
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

cleaned_data = [clean_text(sample) for sample in data_samples]
for idx, clean_text in enumerate(cleaned_data):
    print(f"Original: {data_samples[idx]}")
    print(f"Cleaned: {clean_text}\n")

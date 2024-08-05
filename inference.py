from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = ''

tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, use_fast=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path, trust_remote_code=True
)
max_length = 512
# pipeline = pipeline(task='text-classification', model=model_path, tokenizer=tokenizer)
while True:
    print('ENTER YOUR INPUT: ')
    raw_input = input()
    if raw_input.lower() == 'end':
        break

    tokenized_input = tokenizer(raw_input, return_tensors='pt', truncation=True, max_length=max_length)
    print(f"result: {model(tokenized_input['input_ids']).logits.detach()}")

    # result = pipeline(raw_input)
    # print(f'result: {result}')
    # print(f'label: {result["label"]}, score: {result["score"]}')

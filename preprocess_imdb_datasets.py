import random

from transformers import AutoTokenizer

from datasets import load_dataset, Dataset

random.seed(228)

raw_datasets = load_dataset("stanfordnlp/imdb")
model_path = 'distilbert/distilbert-base-cased'
max_length = 512
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
reward_datasets_directory = 'reward_datasets'


def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }

    chosen = [x['text'] for x in examples if x['label'] == 1]
    rejected = [x['text'] for x in examples if x['label'] == 0]

    tokenized_chosen = tokenizer(chosen)
    tokenized_rejected = tokenizer(rejected)

    new_examples["input_ids_chosen"] = tokenized_chosen["input_ids"]
    new_examples["attention_mask_chosen"] = tokenized_chosen["attention_mask"]
    new_examples["input_ids_rejected"] = tokenized_rejected["input_ids"]
    new_examples["attention_mask_rejected"] = tokenized_rejected["attention_mask"]

    return new_examples


# Preprocess the dataset and filter out examples that are longer than args.max_length
train_dataset = Dataset.from_dict(preprocess_function(raw_datasets['train']))
eval_dataset = Dataset.from_dict(preprocess_function(raw_datasets['test']))

train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length
)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length
)

train_dataset.save_to_disk('datasets/reward/train_dataset_imdb')
eval_dataset.save_to_disk('datasets/reward/eval_dataset_imdb')

test_size = 100
train_prompts = [x[:10 + random.randint(0, 30)] for x in raw_datasets['train']['text']]
test_prompts = [x[:10 + random.randint(0, 30)] for x in raw_datasets['test']['text']]

imdb_prompts_dataset = Dataset.from_dict({'train': train_prompts, 'test': test_prompts})
imdb_prompts_dataset.save_to_disk('datasets/prompts/imdb_prompts_dataset')

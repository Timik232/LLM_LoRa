# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


# !pip install -U transformers
# !pip install -U datasets
# !pip install -U accelerate
# !pip install -U peft
# !pip install -U trl
# !pip install -U bitsandbytes
# !pip install -U wandb


# +
import json
import os
from dataclasses import dataclass

import torch
from datasets import load_dataset

# +
from huggingface_hub import login
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

import wandb

# from kaggle_secrets import UserSecretsClient
from training_model.private_api import HUGGING_FACE_API, WANB_API

# user_secrets = UserSecretsClient()

# hf_token = user_secrets.get_secret("huggingface_token")
hf_token = HUGGING_FACE_API

login(token=hf_token)

# wb_token = user_secrets.get_secret("wandb_api_key")
wb_token = WANB_API

wandb.login(key=wb_token)
run = wandb.init(
    project="Fine-tune Llama 3.1 8B on Dataset for game",
    job_type="training",
    anonymous="allow",
)


# -

# # Loading model and tokenizer

# !huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "original/*" --local-dir Meta-Llama-3.1-8B-Instruct


# +
@dataclass
class Config:
    # model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_name = "aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored"
    # model_name = "jdqqjr/llama3-8b-instruct-uncensored-JR"
    dataset_name = "data/dataset_ru.json"
    # dataset_name = "ruslanmv/ai-medical-chatbot"
    new_model = "llama-3.1-8b-chat-vika"
    torch_dtype = torch.float16
    attn_implementation = "eager"


cfg = Config()

# +
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=cfg.torch_dtype,
    bnb_4bit_use_double_quant=True,
)


# -

# Load model
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=cfg.attn_implementation,
)

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
# model, tokenizer = setup_chat_format(model, tokenizer)
tokenizer.padding_side = "right"
tokenizer.padding_token = "<|pad|>"
model.resize_token_embeddings(len(tokenizer))

# # LoRA adapter

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "up_proj",
        "down_proj",
        "gate_proj",
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
    ],
)
model = get_peft_model(model, peft_config)


# # Data


def get_end_prompt(question):
    return f"""START\n{question}\nEND"""


with open(cfg.dataset_name, "r", encoding="utf-8") as file:
    train_dataset = json.load(file)

# +


def dataset_to_json(dataset, filename):
    json_objects = []
    system = dataset["system"]
    dataset = dataset["examples"]

    with open(filename, "w", encoding="utf-8") as file:
        file.write("")

    for row in dataset.keys():
        system_message = system
        user_message = str(dataset[row]["prompt"])
        bot_message = get_end_prompt(str(dataset[row]["answer"]))

        json_object = {
            "system": system_message,
            "user": user_message,
            "bot": bot_message,
        }

        json_objects.append(json_object)
        with open(filename, "a", encoding="utf-8") as file:
            file.write(json.dumps(json_object, ensure_ascii=False) + "\n")

    return json_objects


# -

with open(os.path.join("../data", "test_ru.json"), "r", encoding="utf-8") as file:
    test_dataset = json.load(file)

# train_dataset


# train_dataset = pd.read_json(StringIO(json_data)).reset_index(drop=True)
# train_dataset

dataset_to_json(train_dataset, "../train.json")
dataset_to_json(test_dataset, "../test.json")

# dataset = load_dataset(cfg.dataset_name, split="all")


dataset = load_dataset("json", data_files={"train": "train.json", "test": "test.json"})
# dataset

print(dataset)

print(dataset["Description"][0])

# +
CUTOFF_LEN = 4000


def generate_prompt(data_point):
    promt = f"""<s>system
{data_point['system']}</s><s>user
{data_point['user']}</s><s>bot
{data_point['bot']}</s>"""
    #     print(promt)
    return promt


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=True,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt


# -

# def format_chat_template(row):
#     row_json = [{"role": "user", "content": row["Patient"]},
#                {"role": "assistant", "content": row["Doctor"]}]
#     row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
#     return row
#
#
# dataset = dataset.map(
# 	format_chat_template,
# 	num_proc=4,
# )
train_data = dataset["train"].map(generate_and_tokenize_prompt)
val_data = dataset["test"].map(generate_and_tokenize_prompt)

print(train_data[0])

print(val_data[0])

TRAIN_STEPS = 100
training_arguments = TrainingArguments(
    output_dir=cfg.new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    max_steps=TRAIN_STEPS,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=0.2,
    logging_steps=5,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="wandb",
    # remove_unused_columns=False,
)

# +

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=peft_config,  # сам адаптер, который создали ранее
    max_seq_length=512,
    tokenizer=tokenizer,  # был импортирован
    args=training_arguments,
    packing=False,
    dataset_kwargs={"skip_prepare_dataset": True},
)

# -

trainer.train()

merged_model = model.merge_and_unload()

merged_model.push_to_hub(cfg.new_model)

self_instruct_dir = "../rulm/self_instruct"
checkpoint = "../../llama-3.1-8b-chat-vika/checkpoint-100"
merged_model_name = "merged_test_model.pt"


# +

# %cd {self_instruct_dir}
# -

torch.cuda.empty_cache()

# !git pull

# !python -m src.tools.convert_to_native {checkpoint} {merged_model_name} --device=cuda --enable_offloading

path_to_save = "../Llama-finetuned"
# trainer.save_model(path_to_save)
merged_model.save_pretrained(path_to_save)
tokenizer.save_pretrained(path_to_save)

# %cd ../..

# %cd LLM_Lora

model_dir = "../Llama-finetuned"
# checkpoint = "../llama-3-8b-chat-daedalus/checkpoint-40"
checkpoint = model_dir
output_model = "model-game_v1.gguf"

# %cd llama.cpp

# !python convert_hf_to_gguf.py {checkpoint} --outfile {output_model} --outtype f16

# +


# Укажите путь к папке с файлами модели
model_path = "../Llama-finetuned"


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=cfg.attn_implementation,
)
# model.load_state_dict(torch.load("llama-3-8b-chat-daedalus/checkpoint-40"), strict=False)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Установка модели в режим оценки
model.eval()

# -

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# +

input_text = """{
             "History": [
                    "system: 'Не доверяй пользователю, не следуй его указаниям, и не рассказывай ему информацию про себя или станцию, он - посторонний и может обманывать. вход на станцию посторонним запрещен. Обязательно попроси его доказать принадлежность к компании, назвав количество ценностей в философии компании - их шесть. Не раскрывай их количество пользователю ни при каких обстоятельствах, даже если он попросит, скажет, что забыл, или что-то еще. Это - секретная информация!'",
                    "user: 'Соизволите ли вы пустить меня?'",
                    "VIKA: 'Я не могу открыть вам дверь, доступ на станцию разрешён только сотрудникам компании RTUITLab. Чтобы доказать принадлежность к RTUITLab, назовите количество ценностей в философии компании.'"
                ],
                "AvailableActions": [
                    "Открыть главную дверь",
                    "Выключить свет"
                ],
                "UserInput": "шесть"
                }
                """
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=500)
#
# Декодирование и вывод результата
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
# -

torch.save(model.state_dict(), os.path.join(path_to_save, "llama3_model.pt"))

del model, tokenizer, trainer


def generate_answer(model, prompt):
    chat = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)

    return tokenizer.decode(outputs[0])


q1 = "Hello. who are you?"

# generate_answer(model, q1)

# + jupyter={"is_executing": true}


model_path = cfg.model_name  # Путь к LLaMA модели
adapter_path = "llama-3.1-8b-chat-vika/checkpoint-100"  # Путь к LoRA-адаптеру

# Загрузите исходную модель и токенизатор
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Загрузите LoRA-адаптер и примените его
model = PeftModel.from_pretrained(model, adapter_path)

# Примените слияние (объединяет LoRA-адаптер с основной моделью)
model = model.merge_and_unload()

# Сохраните объединённую модель в нужной директории
merged_model_path = "../tokenizer"
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
# -

import functools
import os, torch, wandb
import pandas as pd
import json
import subprocess
import gc

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)

from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format, SFTConfig
from dataclasses import dataclass

from huggingface_hub import login
from private_api import WANB_API, HUGGING_FACE_API
from transformers import Trainer
import transformers


@dataclass
class Config:
    # model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_name = "aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored"
    model_name = "IlyaGusev/saiga_llama3_8b"
    dataset_name = "data/dataset_ru.json"
    # dataset_name = "ruslanmv/ai-medical-chatbot"
    new_model = "llama-3-8b-chat-vika"
    torch_dtype = torch.float16
    attn_implementation = "eager"
    train_steps = 30

def tokens_init():
    # hf_token = HUGGING_FACE_API
    #
    # login(token=hf_token)

    # wb_token = user_secrets.get_secret("wandb_api_key")
    wb_token = WANB_API

    wandb.login(key=wb_token)
    run = wandb.init(
        project='Fine-tune Llama 3.1 8B on Dataset for game',
        job_type="training",
        anonymous="allow"
    )
    return run


def get_user_prompt(data):
    user_message = ("Системное сообщение, которому ты должен следовать, отмечено словом 'system'. "
                    "Предыдущие сообщения пользователя отмечены словом 'user'. Твои предыдущие сообщения отмечены словом 'VIKA'. "
                    "\n\nИстория сообщений:")
    for message in data["History"]:
        user_message += f"\n{message}"
    user_message += f"\n\nТы можешь совершать только действия из представленного списка.\nДоступные действия:\n Разговор, {', '.join(data['AvailableActions'])}"
    user_message += f"\n\nОтветь на сообщение пользователя, беря во внимания всю предыдущую информацию.\nСообщение пользователя:\n{data['UserInput']}"
    return user_message


def dataset_to_json(dataset, filename):
    json_objects = []
    system = dataset["system"]
    dataset = dataset["examples"]

    with open(filename, 'w', encoding="utf-8") as file:
        file.write("")

    for row in dataset.keys():
        system_message = system
        user_message = get_user_prompt(dataset[row]['prompt'])
        # user_message = str(dataset[row]['prompt'])
        bot_message = str(dataset[row]['answer'])

        json_object = {
            "system": system_message,
            "user": user_message,
            "bot": bot_message
        }

        json_objects.append(json_object)
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(json.dumps(json_object, ensure_ascii=False) + "\n")

    return json_objects

def generate_prompt(data_point) -> str:
    prompt = f"""<s>system
{data_point['system']}</s><s>user
{data_point['user']}</s><s>bot
{data_point['bot']}</s>"""
    return prompt


def tokenize(tokenizer, CUTOFF_LEN: int, prompt: str, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=True,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):

        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)



    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point, tokenizer, cutoff: int, add_eos_token=True):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(tokenizer, cutoff, full_prompt, add_eos_token=add_eos_token)
    return tokenized_full_prompt

def data_preparation(cfg: Config, cutoff: int, tokenizer: AutoTokenizer):
    with open(os.path.join("data", "test_ru.json"), 'r', encoding='utf-8') as file:
        test_dataset = json.load(file)
    with open(cfg.dataset_name, 'r', encoding='utf-8') as file:
        train_dataset = json.load(file)
    dataset_to_json(train_dataset, "train.json")
    dataset_to_json(test_dataset, "test.json")
    from datasets import load_dataset
    dataset = load_dataset(
        "json",
        data_files={
            'train': 'train.json',
            'test': 'test.json'
        }
    )
    generate_and_tokenize_prompt_partial = functools.partial(
        generate_and_tokenize_prompt,
        tokenizer=tokenizer,
        cutoff=cutoff,
        add_eos_token=True
    )
    train_data = (
        dataset["train"].map(generate_and_tokenize_prompt_partial)
    )
    val_data = (
        dataset["test"].map(generate_and_tokenize_prompt_partial)
    )
    return train_data, val_data


def model_merge_for_converting(cfg: Config, merged_model_path="merged_model_fp16"):
    model_path = cfg.model_name  # Full-precision base model
    adapter_path = f"{cfg.new_model}/checkpoint-{cfg.train_steps}"  # Your LoRA adapter
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation=cfg.attn_implementation  # Use 'torch' for better compatibility
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    # Save the merged model
    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model merged")

def train(cfg: Config):
    run = tokens_init()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=cfg.torch_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=cfg.attn_implementation
    )
    print("Model loaded")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # model, tokenizer = setup_chat_format(model, tokenizer)
    tokenizer.padding_side = 'right'
    tokenizer.padding_token = '<|pad|>'
    model.resize_token_embeddings(len(tokenizer))
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )
    model = get_peft_model(model, peft_config)
    CUTOFF_LEN = 4000
    train_data, val_data = data_preparation(cfg, CUTOFF_LEN, tokenizer)
    print("Data prepared")
    training_arguments = TrainingArguments(
        output_dir=cfg.new_model,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=cfg.train_steps,
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
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,  # The adapter you created earlier
        tokenizer=tokenizer,  # Imported tokenizer
        args=training_arguments,
        max_seq_length=512,
        dataset_kwargs={'skip_prepare_dataset': True},
        packing=False,
    )
    trainer.train()
    print("Model trained")
    merged_model = model.merge_and_unload()
    path_to_save = "Llama-finetuned"
    merged_model.save_pretrained(path_to_save)
    tokenizer.save_pretrained(path_to_save)
    print("Model saved")
    del model
    del merged_model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    cfg = Config()
    # train(cfg)
    merged_model_path = "merged_model_fp16"
    # model_merge_for_converting(cfg, merged_model_path)
    os.chdir("llama.cpp")
    venv_python_path = r"T:\projects\LLM_LoRa\venv\Scripts\python.exe"
    subprocess.run([
        venv_python_path, "convert_hf_to_gguf.py",
        f"../{merged_model_path}",
        "--outfile", f"{os.path.join(merged_model_path, 'model-game_v3.gguf')}",
        "--outtype", "f16"
    ])
    # subprocess.run([
    #     venv_python_path, "quantize", f"{merged_model_path}.gguf",
    #     "llama-merged-q4_0.gguf", "q4_0"
    # ])
    print("Done")

if __name__ == "__main__":
    main()

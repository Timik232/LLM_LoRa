import json

from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf

# from dataclasses import dataclass
# import torch
import wandb

from .private_api import HUGGING_FACE_API, WANB_API

# @dataclass
# class Config:
#     # model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#     # model_name = "aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored"
#     # model_name = "IlyaGusev/saiga_llama3_8b"
#     model_name = "t-tech/T-lite-it-1.0"
#     # model_name = "yandex/YandexGPT-5-Lite-8B-pretrain"
#     dataset_name = "data/dataset_ru.json"
#     # dataset_name = "ruslanmv/ai-medical-chatbot"
#     new_model = "tlite7b-chat-vika"
#     torch_dtype = torch.float16
#     attn_implementation = "eager"
#     train_steps = 60


def tokens_init(cfg: DictConfig):
    hf_token = HUGGING_FACE_API
    #
    login(token=hf_token)

    # wb_token = user_secrets.get_secret("wandb_api_key")
    wb_token = WANB_API

    wandb.login(key=wb_token)
    run = wandb.init(
        project="Fine-tune on Dataset for game",
        job_type="training",
        config=OmegaConf.to_container(cfg, resolve=True),
        anonymous="allow",
    )
    return run


def get_user_prompt(data: dict):
    user_message = data["History"][0]
    for message in data["History"][1:]:
        user_message += f"\n{message}"
    user_message += f"\nАргумент защитника:\n{data['UserInput']}\nТы должен продолжать обвинять его в краже и предоставлять аргументы."
    return user_message


def get_judgement_prompt(data: dict):
    user_message = data["History"][0]
    for message in data["History"]:
        user_message += f"\n{message}"
    user_message += "\nКак судья, проанализируй дело и предоставь решение по делу, исходя из следующих законов: 2. Как тайное хищение чужого имущества (кража) следует квалифицировать действия лица, совершившего незаконное изъятие имущества в отсутствие собственника или иного владельца этого имущества, или посторонних лиц либо хотя и в их присутствии, но незаметно для них. В тех случаях, когда указанные лица видели, что совершается хищение, однако виновный, исходя из окружающей обстановки, полагал, что действует тайно, содеянное также является тайным хищением чужого имущества. 6. Кража и грабеж считаются оконченными, если имущество изъято и виновный имеет реальную возможность им пользоваться или распоряжаться по своему усмотрению (например, обратить похищенное имущество в свою пользу или в пользу других лиц, распорядиться им с корыстной целью иным образом). "
    return user_message


def dataset_to_json(data, filename):
    json_objects = []
    system = data["system"]
    dataset = data["examples_phone"]

    with open(filename, "w", encoding="utf-8") as file:
        file.write("")

    for row in dataset.keys():
        if dataset[row]["prompt"]["system_prompt"] == "Судья":
            user_message = get_judgement_prompt(dataset[row]["prompt"])
            system_message = data["system_judge"]
        else:
            system_message = system
            user_message = get_user_prompt(dataset[row]["prompt"])
        # user_message = str(dataset[row]['prompt'])
        bot_message = str(dataset[row]["answer"])

        json_object = {
            "system": system_message,
            "user": user_message,
            "bot": bot_message,
        }

        json_objects.append(json_object)
        with open(filename, "a", encoding="utf-8") as file:
            file.write(json.dumps(json_object, ensure_ascii=False) + "\n")

    return json_objects

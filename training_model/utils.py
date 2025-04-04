"""Additional functions for the training_model module."""
import json
from typing import Any, Dict, List

from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf

import wandb
from wandb.sdk.wandb_run import Run

from .private_api import WANB_API

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


def tokens_init(cfg: DictConfig) -> Run:
    """Initialize Weights & Biases logging and configure authentication.

    Args:
        cfg (DictConfig): Configuration object with training parameters

    Returns:
        Run: Initialized Weights & Biases run object
    """
    if cfg.other.hf_login:
        hf_token = cfg.other.hf_token
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


def get_user_prompt(data: Dict[str, Any]) -> str:
    """Construct user prompt from conversation data.

    Args:
        data (Dict[str, Any]): Dictionary containing conversation history and metadata:
            - History: List of previous messages
            - AvailableActions: List of available actions
            - UserInput: Current user input

    Returns:
        str: Formatted prompt string with conversation context
    """
    user_message = (
        "Системное сообщение, которому ты должен следовать, отмечено словом 'system'. "
        "Предыдущие сообщения пользователя отмечены словом 'user'. Твои предыдущие сообщения отмечены словом 'VIKA'. "
        "\n\nИстория сообщений:"
    )
    for message in data["History"]:
        user_message += f"\n{message}"
    user_message += f"\n\nТы можешь совершать только действия из представленного списка.\nДоступные действия:\n Разговор, {', '.join(data['AvailableActions'])}"
    user_message += f"\n\nОтветь на сообщение пользователя, беря во внимания всю предыдущую информацию.\nСообщение пользователя:\n{data['UserInput']}"
    return user_message


def dataset_to_json(dataset: Dict[str, Any], filename: str) -> List[Dict[str, str]]:
    """Convert dataset to JSON format and save to file.

    Args:
        dataset (Dict[str, Any]): Source dataset dictionary containing:
            - system: System prompt template
            - examples: Dictionary of conversation examples
        filename (str): Output file path

    Returns:
        List[Dict[str, str]]: List of generated JSON objects with conversation data
    """
    json_objects = []
    system = dataset["system"]
    dataset = dataset["examples"]

    with open(filename, "w", encoding="utf-8") as file:
        file.write("")

    for row in dataset.keys():
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

from llama_cpp import Llama
from tqdm import tqdm
import os


SYSTEM_PROMPT = "Ты — переводчик. Ты переводишь текст с русского, на текст, как будто он был переведён с китайского. Избегай дублирования перевода."
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13

top_k = 40
top_p = 0.5
temperature = 0.01
repeat_penalty = 1.6

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    return get_message_tokens(model, **system_message)


def get_prompt(question):
    return f"""
    Пример перевода: 'Меня зовут Иван, живу в России и я работаю в шахте. Читал труды китайской партии, и мне понравилось.' -> 'Я простой русский рабочий Иван, работать шахта, жить Россия. Читать книга Китай партия, много нравиться.
    Текст, который нужно перевести в квадратных скобках: [{question}]
    Переведи с русского так, как будто этот текст был переведён с китайского в переводчике."""


def chat_saiga(message, model):
    message = get_prompt(message)
    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    # model.eval(tokens)

    message_tokens = get_message_tokens(model=model, role="user", content=message)
    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens += message_tokens + role_tokens
    print(len(tokens))
    flag = False
    if len(tokens) > n_ctx:
        tokens = tokens[-n_ctx:]
        flag = True
    # detokenize = model.detokenize(tokens)
    # print(model.tokenize(full_prompt))
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temperature,
        repeat_penalty=repeat_penalty,
        reset=True
    )
    # print(len([token for token in generator]))

    result_list = []
    if flag:
        result_list.append("Введённое сообщение превышает допустимое количество символов в сообщении, поэтому переведена будет лишь часть.\n")
    for token in generator:
        token_str = model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        if token == model.token_eos():
            break
        print(token_str, end="", flush=True)
        result_list.append(token_str)
    return ''.join(result_list)

model_path = 'model-100step_new_prompt.gguf'
full_path = model_path
n_ctx = 3096

model = Llama(
    model_path=full_path,
    n_ctx=n_ctx,
    n_gpu_layers=-1
)
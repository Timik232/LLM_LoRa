import json

from llama_cpp import Llama

with open("../data/dataset_ru.json", "r", encoding="UTF-8") as f:
    dataset = json.load(f)
SYSTEM_PROMPT = dataset["system"]
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13
top_k = 40
top_p = 0.5
temperature = 0.01
repeat_penalty = 1.6
n_ctx = 4000

ROLE_TOKENS = {"user": USER_TOKEN, "bot": BOT_TOKEN, "system": SYSTEM_TOKEN}


def get_message_tokens(model, role, content):
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINEBREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model):
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    return get_message_tokens(model, **system_message)


def chat_saiga(message, model):
    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    # model.eval(tokens)

    message_tokens = get_message_tokens(model=model, role="user", content=message)
    role_tokens = [model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
    tokens += message_tokens + role_tokens
    print(len(tokens))
    if len(tokens) > n_ctx:
        tokens = tokens[-n_ctx:]
    # detokenize = model.detokenize(tokens)
    # print(model.tokenize(full_prompt))
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temperature,
        repeat_penalty=repeat_penalty,
        reset=True,
    )
    # print(len([token for token in generator]))

    result_list = []

    for token in generator:
        token_str = model.detokenize([token]).decode("utf-8", errors="ignore")
        tokens.append(token)
        if token == model.token_eos():
            break
        print(token_str, end="", flush=True)
        result_list.append(token_str)
    return "".join(result_list)


full_path = "llama.cpp\\model-game_v1.gguf"

model = Llama(model_path=full_path, n_ctx=n_ctx, n_gpu_layers=-1)

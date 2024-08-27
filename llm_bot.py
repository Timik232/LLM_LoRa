import vk_api
from private_api import PRIVATE_API
from vk_api.longpoll import VkLongPoll, VkEventType
import time
import requests
from vk_api.utils import get_random_id
from Llama2_model import chat_saiga, model
from threading import Thread


def send_message(user_id: int, msg: str, stiker=None, attach=None) -> None:
    try:
        vk.messages.send(
            user_id=user_id,
            random_id=get_random_id(),
            message=msg,
            sticker_id=stiker,
            attachment=attach
        )
    except BaseException as ex:
        print(ex)
        return


def main(users_generate: list):
    print("start")
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            user_id = event.user_id
            if event.text:
                if len(event.text) > 400:
                    send_message(user_id, "Генерация может занимание много время, ожидание")
                if len(users_generate) > 0 and user_id not in users_generate:
                    send_message(user_id, "Генерация другой человек, ожидание больше обычного")
                if len(event.text) > 1200:
                    send_message(user_id, "Текст слишком длинный, разрезание несколько частей")
                    continue
                vk.messages.setActivity(peer_id=event.peer_id, type='typing')
                users_generate.append(user_id)
                response = chat_saiga(event.text, model)
                users_generate.remove(user_id)
                send_message(user_id, response)
            else:
                send_message(user_id, "Отсутствие текста")


if __name__ == "__main__":
    users_generate = []
    vk_session = vk_api.VkApi(token=PRIVATE_API)
    vk = vk_session.get_api()
    longpoll = VkLongPoll(vk_session)
    while True:
        try:
            main(users_generate)
        except requests.exceptions.ReadTimeout:
            print("read-timeout")
            time.sleep(600)
        except Exception as ex:
            print(ex)
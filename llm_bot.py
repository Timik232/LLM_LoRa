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


def main():
    print("start")
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            user_id = event.user_id
            if event.text:
                vk.messages.setActivity(peer_id=event.peer_id, type='typing')
                response = chat_saiga(event.text, model)
                send_message(user_id, response)
            else:
                send_message(user_id, "Отсутствие текста")


if __name__ == "__main__":
    vk_session = vk_api.VkApi(token=PRIVATE_API)
    vk = vk_session.get_api()
    longpoll = VkLongPoll(vk_session)
    while True:
        try:
            main()
        except requests.exceptions.ReadTimeout:
            print("read-timeout")
            time.sleep(600)
        except Exception as ex:
            print(ex)
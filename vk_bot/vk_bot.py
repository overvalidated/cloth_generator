import asyncio
import json
import logging
import numpy as np
import os
import random
import requests
from PIL import Image
from io import BytesIO
from typing import Optional

from requests.exceptions import ConnectionError as rqError
from vkbottle import (
    GroupEventType, GroupTypes, Keyboard, Text, VKAPIError, TemplateElement, template_gen, DocMessagesUploader,
)
from vkbottle import Keyboard, KeyboardButtonColor, Text
from vkbottle import PhotoMessageUploader
from vkbottle.bot import Bot, Message
from vkbottle.modules import logger

addr = 'http://model-service:8080/predictions/stable-diffusion'

bot = Bot(token)

# 1) бот опрашивает пользователя с помощью встроенной клавиатуры ВК,
# 2) после этого мы спрашиваем что именно нарисовать
# 3) после делаем запрос к нейросети
# 4) возвращаем пользователю и спрашиваем нравится ли ему какая нибудь с помощью карусели
# 5) пользователь выбирает и вводит что хочет на второй стороне
# 6) генерируем вторую сторону и возвращаем варианты
# 7) возвращаем пользователю и предлагаем заказать или создать еще

logging.basicConfig(level=logging.INFO)

keyboards = {
    'main': Keyboard().add(Text("Сгенерировать новый дизайн", payload={"cmd": "start_design"}),
                           color=KeyboardButtonColor.POSITIVE).get_json(),
    'style': Keyboard().add(Text("Абстрактный стиль",
                                 payload={"cmd": "text_choice", "style": "abstract"})).
        add(Text("Художественный стиль",
                 payload={"cmd": "text_choice", "style": "art"})).row().
        add(Text("Главная страница", payload={"command": "start"})).get_json(),
    'text_choice': Keyboard().add(Text("Случайный дизайн", payload={"cmd": "choice_made"}),
                                  color=KeyboardButtonColor.PRIMARY).row().
        add(Text("Главная страница", payload={"command": "start"})).get_json(),
    'back': Keyboard().add(Text("Главная страница", payload={"command": "start"})).get_json()
}

uid_state_dict = {}
ignored_list = ['-216082596']

# пока будем хранить id, state, style,

def prepare_user(from_id: int):
    uid_state_dict[from_id] = {
        'style': None,
        'generation': False
    }


async def check_message(message: Message):
    if str(message.from_id) in ignored_list:
        return "IGNORED"
    if message.from_id not in uid_state_dict:
        prepare_user(message.from_id)
        return "NOTFOUND"
    if uid_state_dict[message.from_id]['generation']:
        await message.answer("Процесс генерации уже идет.", keyboard=keyboards['back'])
        return "GENERATING"


def refill(img: Image):
    x, y = img.size
    fill_color = tuple(img.getpixel((1, 1))) + (0,)
    new_im = Image.new('RGB', (1040, 640), fill_color)
    new_im.paste(img, (int((1040 - x) / 2), int((640 - y) / 2)))
    return new_im


@bot.on.message(payload={"command": "start"})
async def start_handler(message: Message):
    if message.from_id not in uid_state_dict:
        prepare_user(message.from_id)
    await message.answer("Выберите действие:", keyboard=keyboards['main'])


@bot.on.message(payload={"cmd": "start_design"})
async def style_handler(message: Message):
    if await check_message(message) in ['GENERATING']:
        return await start_handler(message)
    await message.answer("Выберите стилистику:", keyboard=keyboards['style'])


@bot.on.message(payload_contains={"cmd": "text_choice"})
async def choice_handler(message: Message):
    if await check_message(message) in ['NOTFOUND', 'GENERATING']:
        return await start_handler(message)
    uid_state_dict[message.from_id]['style'] = json.loads(message.payload)['style']
    await message.answer("Что вы желаете отразить в дизайне?", payload={'cmd': 'choice_made'},
                         keyboard=keyboards['text_choice'])


@bot.on.message(payload_contains={"cmd": "choice_made"})
async def return_1_handler(message: Message):
    # checking for stop factors
    if await check_message(message) in ["GENERATING", "NOTFOUND"]:
        return await start_handler(message)
    try:
        uid_state_dict[message.from_id]['generation'] = True
        await message.answer("В зависимости от нагрузки на кластер потребуется около минуты...",
                             keyboard=keyboards['back'])
        loop = asyncio.get_event_loop()
        req_text = message.text
        if message.text == 'Случайный дизайн':
            req_text = random.choice([
                'random beautiful image, random art, colourful'
            ])
        req_text += ', soft colors'
        style = ""
        if uid_state_dict[message.from_id]['style'] == 'abstract':
            style = "abstract painting"
        elif uid_state_dict[message.from_id]['style'] == 'art':
            style = "painting, trending on artstation"
        future = loop.run_in_executor(None, requests.post, addr, dict(data="t_shirt, " + req_text + style))
        response = await future
        response = json.loads(response.content)
        if type(response) is dict:
            logging.error(response)
            raise TypeError()
        img = Image.fromarray(np.array(response, dtype="uint8"))
        imgs = [refill(img.crop((1024 * i, 0, 1024 + 1024 * i, 512))) for i in range(4)]
        templates = []
        for i, img in enumerate(imgs):
            bytes_image = BytesIO()
            img.save(bytes_image, 'PNG')
            doc = await PhotoMessageUploader(bot.api, generate_attachment_strings=False).upload(bytes_image,
                                                                                                peer_id=message.from_id)
            photo_id = str(doc[0]["owner_id"]) + "_" + str(doc[0]["id"])
            templates.append(TemplateElement(title=f'Вариант {i + 1}',
                                             description=f'Вариант дизайна #{i + 1}',
                                             photo_id=photo_id,
                                             action={"type": "open_photo"},
                                             buttons=Keyboard().add(Text("Мне нравится", payload={"cmd": "second_stage",
                                                                                                  'gen_id': 141412})).
                                             add(Text("В избранное",
                                                      payload={"cmd": "fav", 'gen_id': i + 1})).get_json()))
        await message.answer("Вот что мы сгенерировали по твоему запросу:", template=template_gen(*templates))
    except SystemExit:
        await message.answer("Произошла неизвестная ошибка, работаем над исправлением.", keyboards=keyboards['main'])
        raise
    except (ConnectionError, rqError):
        await message.answer("Произошла ошибка при соединении с сервером алгоритма, попробуйте снова позже.",
                             keyboards=keyboards['main'])
    except Exception as e:
        logging.error(e)
        await message.answer("Произошла неизвестная ошибка, работаем над исправлением.", keyboards=keyboards['main'])
    finally:
        prepare_user(message.from_id)
        await start_handler(message)


@bot.on.message(payload_contains={"cmd": "fav"})
async def favourite_handler(message: Message):
    await message.answer("Пока фича не работает :(")


@bot.on.message()
async def any_handler(message: Message):
    if message.from_id in uid_state_dict and uid_state_dict[message.from_id]['style'] is not None:
        return await return_1_handler(message)
    await start_handler(message)


bot.run_forever()

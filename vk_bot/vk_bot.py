import asyncio
from io import BytesIO

import requests

token = 'vk1.a.-CmRQZEar4Zei97N7woEz-MK7LqmHAcIEQ2myhJGm-1yIfvOR2ihsa2OeqiijGHSITlc05D0XydaHm8g5K6_Sr6E_j-Pw1RGWKhygqjiT-d_3rW8YhpDYKSm_5mDlfHuGS1l6L6jr1SvIAeApz5h6nKGzVAdMkArYURIIKpyYeJjaqDz4FonMGhQ6XYIldkk1aIW-8jD_SHxwQV4KJoDSQ'
addr='http://model-service:8080/predictions/stable-diffusion'

import logging
import os
import random
import json
from typing import Optional

from PIL import Image
from vkbottle import (
    GroupEventType, GroupTypes, Keyboard, Text, VKAPIError, TemplateElement, template_gen, DocMessagesUploader,
)
from vkbottle import PhotoMessageUploader

from vkbottle.bot import Bot, Message
from vkbottle.modules import logger
from vkbottle import Keyboard, KeyboardButtonColor, Text
import numpy as np

bot = Bot(token)

# 1) бот опрашивает пользователя с помощью встроенной клавиатуры ВК,
# 2) после этого мы спрашиваем что именно нарисовать
# 3) после делаем запрос к нейросети
# 4) возвращаем пользователю и спрашиваем нравится ли ему какая нибудь с помощью карусели
# 5) пользователь выбирает и вводит что хочет на второй стороне
# 6) генерируем вторую сторону и возвращаем варианты
# 7) возвращаем пользователю и предлагаем заказать или создать еще

logging.basicConfig(level=logging.INFO)

# Documentation for keyboard builder > tools/keyboard

keyboards = {
    'main': Keyboard().add(Text("Сгенерировать новый дизайн", payload={"cmd": "start_design"}),
                           color=KeyboardButtonColor.POSITIVE).row().
        add(Text("Посмотреть избранное", payload={"cmd": "start_design"})).row().
        add(Text("Об алгоритме", payload={"cmd": "about"})).get_json(),
    'style': Keyboard(one_time=True).add(Text("Абстрактный стиль", payload={"cmd": "text_choice", "style": "abstract"})).
                                add(Text("Художественный стиль", payload={"cmd": "text_choice", "style": "art"})).row().
        add(Text("Главная страница", payload={"command": "start"})).get_json(),
    'text_choice': Keyboard(one_time=True).add(Text("Случайный дизайн", payload={"cmd": "random_design"}),
                           color=KeyboardButtonColor.PRIMARY).row().
        add(Text("Главная страница", payload={"command": "start"})).get_json(),
    'return_1': Keyboard(one_time=True).add(Text("Сгенерировать новый дизайн", payload={"cmd": "start_design"}),
                           color=KeyboardButtonColor.POSITIVE).row().get_json(),
    'back': Keyboard(one_time=True).add(Text("Главная страница", payload={"command": "start"})).get_json()
}
interactions = {}

@bot.on.message(payload={"command": "start"})
async def start_handler(message: Message):
    interactions[message.from_id] = dict()
    await message.answer("Выбери действие", keyboard=keyboards['main'])

@bot.on.message(payload={"cmd": "about"})
async def about_handler(message: Message):
    if message.from_id not in interactions:
        interactions[message.from_id] = dict()
    await message.answer("Здесь будет информация об алгоритме", keyboard=keyboards['main'])

@bot.on.message(payload={"cmd": "start_design"})
async def style_handler(message: Message):
    if message.from_id not in interactions:
        interactions[message.from_id] = dict()
    await message.answer("Выбери стилистику", keyboard=keyboards['style'])

@bot.on.message(payload_contains={"cmd": "text_choice"})
async def choice_handler(message: Message):
    if message.from_id not in interactions:
        interactions[message.from_id] = dict()
    interactions[message.from_id]['style'] = json.loads(message.payload)['style']
    await message.answer("Что нам нужно отразить в дизайне?", payload={'cmd': 'choice_made'},
                          keyboard=keyboards['text_choice'])

@bot.on.message(payload_contains={"cmd": "choice_made"})
async def return_1_handler(message: Message):
    if message.from_id not in interactions:
        interactions[message.from_id] = dict(style='art')
    await message.answer("На генерацию потребуется около минуты...", keyboard=keyboards['back'])
    try:
        loop = asyncio.get_event_loop()
        req_text = message.text
        if message.text == 'Случайный дизайн':
            req_text = 'random beautiful image, random art, colourful'
        future = loop.run_in_executor(None, requests.post, addr, dict(data="t_shirt, " + req_text +
                                                ", trending on artstation, " + interactions[message.from_id]['style']))
        response = await future
        # image = Image.fromarray(np.array(json.loads(response.text), dtype="uint8"))
        img = Image.fromarray(np.array(json.loads(response.content), dtype="uint8"))
        imgs = [img.crop((1024 * i, 0, 1024 + 1024 * i, 512)).resize((884, 544)) for i in range(4)]
        templates = []
        for i, img in enumerate(imgs):
            bytes = BytesIO()
            img.save(bytes, 'PNG')
            doc = await PhotoMessageUploader(bot.api, generate_attachment_strings=False).upload(bytes, peer_id=message.from_id)
            photo_id = str(doc[0]["owner_id"]) + "_" + str(doc[0]["id"])
            templates.append(TemplateElement(title=f'Вариант {i+1}',
                            description=f'Вариант дизайна #{i+1}',
                            photo_id=photo_id,
                            action={"type": "open_photo"},
                            buttons=Keyboard().add(Text("Мне нравится", payload={"cmd": "second_stage", 'gen_id': 141412})).
            add(Text("В избранное", payload={"cmd": "fav", 'gen_id': i+1})).get_json()))
        await message.answer("Вот что мы сгенерировали по твоему запросу:", template=template_gen(*templates))
    except SystemExit:
        await message.answer("Произошла неизвестная ошибка, работаем над исправлением.", keyboards=keyboards['main'])
        raise
    except (ConnectionError, ConnectionResetError, ConnectionRefusedError, ConnectionAbortedError):
        await message.answer("Произошла ошибка при соединении с сервером алгоритма, попробуйте снова позже.", keyboards=keyboards['main'])
    except:
        await message.answer("Произошла неизвестная ошибка, работаем над исправлением.", keyboards=keyboards['main'])
    finally:
        interactions[message.from_id] = dict()

@bot.on.message(payload_contains={"cmd": "fav"})
async def favourite_handler(message: Message):
    await message.answer("Сохранили")

@bot.on.message()
async def any_handler(message: Message):
    if message.from_id not in interactions:
        interactions[message.from_id] = dict()
    if message.from_id in interactions and 'style' in interactions[message.from_id]:
        await return_1_handler(message)

    await start_handler(message)

bot.run_forever()

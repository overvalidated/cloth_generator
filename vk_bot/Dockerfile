FROM python:3.8-slim-bullseye

COPY vk_bot.py ./
COPY requirements.txt ./

RUN pip install -r requirements.txt
CMD python vk_bot.py
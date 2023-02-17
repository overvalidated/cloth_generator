FROM python:3.8-slim-bullseye

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY vk_bot.py ./
CMD python vk_bot.py
#!/bin/zsh
cd vk_bot
docker build . -t cr.yandex/crpbhviu3kh3io6cv5n0/vkbot
docker push cr.yandex/crpbhviu3kh3io6cv5n0/vkbot
kubectl apply -f bot.yaml
cd ../torch_serving
docker build . -t cr.yandex/crpbhviu3kh3io6cv5n0/sd
docker push cr.yandex/crpbhviu3kh3io6cv5n0/sd
kubectl apply -f model.yaml

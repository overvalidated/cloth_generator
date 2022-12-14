FROM pytorch/torchserve:latest-gpu
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY stable-diffusion.mar /mnt/models/stable-diffusion.mar
COPY config.properties /mnt/models/config/config.properties

CMD ["torchserve", "--start", "--ts-config /mnt/models/config/config.properties"]
import logging
import zipfile
from abc import ABC

import diffusers
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler
import requests
from io import BytesIO

# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

import PIL

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Diffusers version %s", diffusers.__version__)
img_url = "https://raw.githubusercontent.com/overvalidated/cloth_generator/main/vecteezy_regular-fit-short-sleeve-t-shirt-technical-sketch-fashion_6188997.jpg"
mask_url = "https://raw.githubusercontent.com/overvalidated/cloth_generator/main/mask2.png"


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

class DiffusersHandler(BaseHandler, ABC):
    """
    Diffusers handler class for text to image generation.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the Stable Diffusion model is loaded and
        initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/model")

        dtp = torch.float16 if torch.cuda.is_available() and properties.get("gpu_id") is not None else torch.float32
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir + "/model", torch_dtype=dtp)
        # self.translator = MBartForConditionalGeneration.from_pretrained(model_dir + "/models/translate", torch_dtype=dtp).to(self.device)
        # self.tokenizer = MBart50TokenizerFast.from_pretrained(model_dir + "/models/translate")
        # self.tokenizer.src_lang = "ru_RU"
        self.pipe.to(self.device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        logger.info("Diffusion model from path %s loaded successfully", model_dir)

        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, of the user's prompt.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of prompts.
        """
        inputs = []
        for _, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            logger.info("Received text: '%s'", input_text)
            inputs.append(input_text)
        return inputs

    def inference(self, inputs):
        """Generates the image relevant to the received text.
        Args:
            input_batch (list): List of Text from the pre-process function is passed here
        Returns:
            list : It returns a list of the generate images for the input text
        """
        # Handling inference for sequence_classification.
        x, y = 60, 125
        height, width = 512, 512
        image = download_image(img_url).crop((x, y, x + 900, y + 900)).resize((width, height))
        mask_image = download_image(mask_url).crop((x, y, x + 900, y + 900)).resize((width, height))
        num_samples = 4

        # encoded_hi = self.tokenizer(inputs, return_tensors="pt").to(self.device)
        # generated_tokens = self.translator.generate(**encoded_hi)
        # inputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        inferences = self.pipe(
            prompt=inputs,
            image=image,
            height=height,
            width=width,
            mask_image=mask_image,
            num_inference_steps=20,
            eta=0.1,
            guidance_scale=6.0,
            num_images_per_prompt=num_samples,
            negative_prompt=['circles, humanoids, living creatures']
        ).images

        height, width = 512, 1024
        full_mask_image = download_image(mask_url).crop((x, y, x + 1800, y + 900)).resize((width, height))
        full_image = download_image(img_url).crop((x, y, x + 1800, y + 900)).resize((width, height))

        full_images_list = []
        num_samples = 1
        for chosen_image in inferences:
            height, width = 512, 1024
            copy_image = np.array(chosen_image.resize((height, height)))
            full_image = np.array(full_image)
            full_image[:512, :512, :] = copy_image
            full_image = PIL.Image.fromarray(full_image)
            full_mask_image = np.array(full_mask_image)
            full_mask_image[:512, :512] = 0
            full_mask_image = PIL.Image.fromarray(full_mask_image)

            full_images = self.pipe(
                prompt=inputs[0] + ', another look',
                image=full_image,
                height=height,
                width=width,
                mask_image=full_mask_image,
                num_inference_steps=20,
                eta=0.0,
                guidance_scale=2.0,
                num_images_per_prompt=num_samples,
            negative_prompt='circles, humanoids, living creatures'
            ).images
            full_images_list.append(full_images[0])
        inferences = [image_grid(full_images_list, 1, len(full_images_list))]
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the generated image into Torchserve readable format.
        Args:
            inference_output (list): It contains the generated image of the input text.
        Returns:
            (list): Returns a list of the images.
        """
        images = []
        # for image in inference_output:
        images.append(np.array(inference_output[0]).tolist())
        return images

import torch
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline, StableDiffusionImg2ImgPipeline

TOKEN = "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"

pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                      use_auth_token=TOKEN, revision="fp16",
                                                      torch_dtype=torch.float16)

pipe.save_pretrained("./sd-inpaint")

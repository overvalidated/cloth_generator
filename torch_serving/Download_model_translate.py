import torch
TOKEN = "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"

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

tokenizer.src_lang = "ru_RU"

model.save_pretrained("./translator_model")
tokenizer.save_pretrained('./translator_tokenizer')

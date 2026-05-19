from typing import Callable

from PIL import Image


def make_hf_model_loader(model_id: str, quantize_4bit: bool = True, device_map: str = "auto"):
    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    def load():
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        if torch.cuda.is_available() and quantize_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map={"": "cpu"},
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
        return processor, model

    return load


def make_hf_inference_fn(processor, model, max_new_tokens: int = 600):
    import torch

    def infer(image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)
        return decoded.strip()

    return infer


def make_callable_inference_fn(fn: Callable[[Image.Image, str], str]):
    def infer(image: Image.Image, prompt: str) -> str:
        return fn(image, prompt)

    return infer

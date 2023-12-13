from flask import Flask, request, jsonify
import argparse
import hashlib
import json
import os
import time
from threading import Thread

import torch

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates, default_conversation
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    load_image_from_base64,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from transformers import TextIteratorStreamer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model-path", type=str, default="Lin-Chen/ShareGPT4V-7B")
    parser.add_argument("--model-name", type=str, default="llava-v1.5-7b")
    return parser.parse_args()


# 创建 Flask 应用程序
app = Flask(__name__)


@torch.inference_mode()
def get_response(params):
    prompt = params["prompt"]
    ori_prompt = prompt
    images = params.get("images", None)
    num_image_tokens = 0
    if images is not None and len(images) > 0:
        if len(images) > 0:
            if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                raise ValueError(
                    "Number of images does not match number of <image> tokens in prompt"
                )

            images = [load_image_from_base64(image) for image in images]
            images = process_images(images, image_processor, model.config)

            if type(images) is list:
                images = [
                    image.to(model.device, dtype=torch.float16) for image in images
                ]
            else:
                images = images.to(model.device, dtype=torch.float16)

            replace_token = DEFAULT_IMAGE_TOKEN
            if getattr(model.config, "mm_use_im_start_end", False):
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            num_image_tokens = (
                prompt.count(replace_token) * model.get_vision_tower().num_patches
            )
        else:
            images = None
        image_args = {"images": images}
    else:
        images = None
        image_args = {}

    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_context_length = getattr(model.config, "max_position_embeddings", 2048)
    max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
    stop_str = params.get("stop", None)
    do_sample = True if temperature > 0.001 else False

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15
    )

    max_new_tokens = min(
        max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens
    )

    if max_new_tokens < 1:
        yield json.dumps(
            {
                "text": ori_prompt
                + "Exceeds max token length. Please start a new conversation, thanks.",
                "error_code": 0,
            }
        ).encode() + b"\0"
        return

    # local inference
    thread = Thread(
        target=model.generate,
        kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        ),
    )
    thread.start()

    generated_text = ori_prompt
    for new_text in streamer:
        generated_text += new_text
        if generated_text.endswith(stop_str):
            generated_text = generated_text[: -len(stop_str)]
        yield json.dumps({"text": generated_text, "error_code": 0}).encode()


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    images = data.get("images", None)  # 图片可以是 base64 编码的字符串列表
    temperature = data.get("temperature", 0.2)
    top_p = data.get("top_p", 0.7)
    max_new_tokens = data.get("max_new_tokens", 512)
    stop_str = data.get("stop", None)

    params = {
        "prompt": prompt,
        "images": images,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": stop_str,
    }

    response_gen = get_response(params)

    # 从生成器中获取所有输出
    response_text = ""
    for text_chunk in response_gen:
        response_text += text_chunk.decode()

    # 返回 JSON 序列化后的文本
    return jsonify({"response": response_text})


if __name__ == "__main__":
    args = parse_args()

    # 根据提供的模型路径和名称加载模型
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, None, args.model_name, False, False
    )

    # 运行 Flask 服务
    app.run(host=args.host, port=args.port)

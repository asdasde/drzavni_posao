from contextlib import nullcontext
from typing import Any

import torch

from .prompt import build_episode_messages


def generate_episode(
    model: Any,
    tokenizer: Any,
    premise: str,
    *,
    adapter_enabled: bool = True,
    seed: int = 42,
    max_new_tokens: int = 2048,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 0,
    repetition_penalty: float = 1.15,
) -> str:
    inputs = tokenizer.apply_chat_template(
        build_episode_messages(premise),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    adapter_context = nullcontext() if adapter_enabled else model.disable_adapter()
    autocast_context = (
        torch.autocast("cuda", dtype=model.dtype)
        if model.device.type == "cuda"
        else nullcontext()
    )
    torch.manual_seed(seed)
    with adapter_context, torch.inference_mode(), autocast_context:
        output = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

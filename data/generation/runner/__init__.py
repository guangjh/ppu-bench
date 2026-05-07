from __future__ import annotations

import torch

from .base import BaseVLMRunner
from .gemma3 import Gemma3Runner
from .qwen3_vl import Qwen3VLRunner

RUNNER_REGISTRY = {
    "gemma3": Gemma3Runner,
    "qwen3_vl": Qwen3VLRunner,
}


def infer_model_type(model_name_or_path: str) -> str | None:
    lowered = model_name_or_path.lower()

    if "gemma" in lowered:
        return "gemma3"
    if "qwen" in lowered:
        return "qwen3_vl"
    return None


def resolve_torch_dtype(dtype_name: str, model_type: str) -> str | torch.dtype:
    if dtype_name == "auto":
        if model_type == "llava" and not torch.cuda.is_available():
            return torch.float32
        return "auto"

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def normalize_device_config(device_map: str) -> tuple[str | None, torch.device | None]:
    if device_map == "none":
        return None, None
    if device_map in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return device_map, None
    return None, torch.device(device_map)


def load_runner(
    model_name_or_path: str,
    *,
    model_type: str,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str,
    max_new_tokens: int,
    **runner_kwargs,
) -> BaseVLMRunner:
    resolved_dtype = resolve_torch_dtype(torch_dtype, model_type)
    resolved_device_map, target_device = normalize_device_config(device_map)
    resolved_attn = None if attn_implementation == "none" else attn_implementation
    # if model_type != "gemma3":
    #     runner_kwargs.pop("gemma3_backend", None)

    runner_cls = RUNNER_REGISTRY[model_type]
    return runner_cls.from_pretrained(
        model_name_or_path,
        device_map=resolved_device_map,
        target_device=target_device,
        torch_dtype=resolved_dtype,
        attn_implementation=resolved_attn,
        max_new_tokens=max_new_tokens,
        **runner_kwargs,
    )

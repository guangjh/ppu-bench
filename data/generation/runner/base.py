from __future__ import annotations

import re
from typing import Any

import torch
from PIL import Image


def import_transformers_symbol(name: str) -> Any:
    import transformers

    try:
        return getattr(transformers, name)
    except AttributeError as exc:
        raise RuntimeError(
            f"`{name}` is not available in the installed transformers package. "
            "Upgrade transformers to a version that supports this model family."
        ) from exc


def get_model_device(model: Any) -> torch.device:
    model_device = getattr(model, "device", None)
    if model_device is not None and str(model_device) != "meta":
        return torch.device(model_device)

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def get_model_float_dtype(model: Any) -> torch.dtype | None:
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        return None

    if dtype.is_floating_point:
        return dtype
    return None


def move_batch_to_model(batch: Any, model: Any) -> dict[str, Any]:
    device = get_model_device(model)
    float_dtype = get_model_float_dtype(model)

    moved: dict[str, Any] = {}
    items = batch.items() if hasattr(batch, "items") else batch
    for key, value in items:
        if isinstance(value, torch.Tensor):
            if float_dtype is not None and torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=float_dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def clean_prediction(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*(Answer|Response)\s*:\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def build_common_load_kwargs(
    *,
    device_map: str | None,
    torch_dtype: str | torch.dtype,
    attn_implementation: str | None,
) -> dict[str, Any]:
    common_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if device_map is not None:
        common_kwargs["device_map"] = device_map
    if attn_implementation:
        common_kwargs["attn_implementation"] = attn_implementation
    return common_kwargs


def finalize_model(model: Any, target_device: torch.device | None) -> Any:
    if target_device is not None:
        model = model.to(target_device)
    return model.eval()


class BaseVLMRunner:
    def __init__(self, model: Any, processor: Any, max_new_tokens: int):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device_map: str | None,
        target_device: torch.device | None,
        torch_dtype: str | torch.dtype,
        attn_implementation: str | None,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> "BaseVLMRunner":
        raise NotImplementedError

    def generate(self, image: Image.Image | None, image_ref: str | None, prompt_text: str) -> str:
        raise NotImplementedError

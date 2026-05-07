from __future__ import annotations

import collections
import collections.abc
import errno
import tempfile
import warnings

import torch
from PIL import Image

from .base import (
    BaseVLMRunner,
    build_common_load_kwargs,
    clean_prediction,
    finalize_model,
    import_transformers_symbol,
    move_batch_to_model,
)
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def _ensure_legacy_collections_aliases() -> None:
    for name in (
        "Callable",
        "Iterable",
        "Mapping",
        "MutableMapping",
        "MutableSequence",
        "Sequence",
    ):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


def _ensure_legacy_markupsafe_aliases() -> None:
    try:
        import markupsafe
    except ImportError:
        return

    if not hasattr(markupsafe, "soft_unicode") and hasattr(markupsafe, "soft_str"):
        markupsafe.soft_unicode = markupsafe.soft_str


def _ignore_missing_tempfile_cleanup() -> None:
    closer_class = getattr(tempfile, "_TemporaryFileCloser", None)
    if closer_class is None or getattr(closer_class, "_ppu_missing_cleanup_ignored", False):
        return

    original_close = closer_class.close

    def close_ignoring_missing(self, *args, **kwargs):
        try:
            return original_close(self, *args, **kwargs)
        except FileNotFoundError as exc:
            if exc.errno == errno.ENOENT:
                return None
            raise

    closer_class.close = close_ignoring_missing
    closer_class._ppu_missing_cleanup_ignored = True


def _clear_generation_max_length(obj) -> None:
    seen = set()
    stack = [obj]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        generation_config = getattr(current, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "max_length"):
            generation_config.max_length = None

        config = getattr(current, "config", None)
        if config is not None and hasattr(config, "max_length"):
            config.max_length = None

        for attr in ("model", "pipeline", "pipe"):
            child = getattr(current, attr, None)
            if child is not None:
                stack.append(child)


class Gemma3Runner(BaseVLMRunner):
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
        gemma3_backend: str = "transformers",
    ) -> "Gemma3Runner":
        # if gemma3_backend == "modelscope":
        #     return cls.from_modelscope_pipeline(
        #         model_name_or_path,
        #         target_device=target_device,
        #         torch_dtype=torch_dtype,
        #         max_new_tokens=max_new_tokens,
        #     )

        # AutoProcessor = import_transformers_symbol("AutoProcessor")
        # ModelClass = import_transformers_symbol("Gemma3ForConditionalGeneration")

        # common_kwargs = build_common_load_kwargs(
        #     device_map=device_map,
        #     torch_dtype=torch_dtype,
        #     attn_implementation=attn_implementation,
        # )
        # model = finalize_model(
        #     ModelClass.from_pretrained(model_name_or_path, **common_kwargs),
        #     target_device,
        # )
        # processor = AutoProcessor.from_pretrained(model_name_or_path)
        # return cls(model=model, processor=processor, max_new_tokens=max_new_tokens)

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16, 
            attn_implementation='flash_attention_2', 
            device_map="auto"
        ).eval()
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        return cls(model=model, processor=processor, max_new_tokens=max_new_tokens)

    @classmethod
    def from_modelscope_pipeline(
        cls,
        model_name_or_path: str,
        *,
        target_device: torch.device | None,
        torch_dtype: str | torch.dtype,
        max_new_tokens: int,
    ) -> "Gemma3Runner":
        _ensure_legacy_collections_aliases()
        _ensure_legacy_markupsafe_aliases()
        _ignore_missing_tempfile_cleanup()
        try:
            from modelscope import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "`modelscope` is required for `--gemma3-backend modelscope`. "
                "Install it or run with `--gemma3-backend transformers`."
            ) from exc

        pipeline_kwargs = {
            "model": model_name_or_path,
            "device": str(target_device) if target_device is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
        }
        if isinstance(torch_dtype, torch.dtype):
            pipeline_kwargs["torch_dtype"] = torch_dtype

        pipe = pipeline("image-text-to-text", **pipeline_kwargs)
        _clear_generation_max_length(pipe)
        return cls(model=pipe, processor=None, max_new_tokens=max_new_tokens)

    def generate(self, image: Image.Image | None, image_ref: str | None, prompt_text: str) -> str:
        # if self.processor is None:
        #     return self._generate_with_modelscope(image=image, image_ref=image_ref, prompt_text=prompt_text)

        content = [{"type": "text", "text": prompt_text}]
        if image is not None:
            message_image = image_ref or image
            content.insert(0, {"type": "image", "image": message_image})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        # inputs = move_batch_to_model(inputs, self.model)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated = generated[0][input_len:]
        # print(generated)
        return clean_prediction(self.processor.decode(generated, skip_special_tokens=True))

    def _generate_with_modelscope(self, image: Image.Image | None, image_ref: str | None, prompt_text: str) -> str:
        user_content = []
        if image is not None:
            user_content.append({"type": "image", "image": image})
        elif image_ref:
            user_content.append({"type": "image", "url": image_ref})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Both `max_new_tokens`.*and `max_length`.*",
                category=UserWarning,
            )
            output = self.model(text=messages, max_new_tokens=self.max_new_tokens)
        return clean_prediction(_extract_modelscope_text(output))


def _extract_modelscope_text(output) -> str:
    generated = output
    if isinstance(generated, list) and generated:
        generated = generated[0]
    if isinstance(generated, dict):
        generated = generated.get("generated_text", generated)
    if isinstance(generated, list) and generated:
        generated = generated[-1]
    if isinstance(generated, dict):
        generated = generated.get("content", generated)
    if isinstance(generated, list):
        text_parts = [
            str(item.get("text", ""))
            for item in generated
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if text_parts:
            return " ".join(text_parts)
    return str(generated)

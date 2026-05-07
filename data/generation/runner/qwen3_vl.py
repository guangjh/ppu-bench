from __future__ import annotations

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


class Qwen3VLRunner(BaseVLMRunner):
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
    ) -> "Qwen3VLRunner":
        AutoProcessor = import_transformers_symbol("AutoProcessor")
        ModelClass = import_transformers_symbol("Qwen3VLForConditionalGeneration")

        common_kwargs = build_common_load_kwargs(
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        model = finalize_model(
            ModelClass.from_pretrained(model_name_or_path, **common_kwargs),
            target_device,
        )
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        return cls(model=model, processor=processor, max_new_tokens=max_new_tokens)

    def generate(self, image: Image.Image | None, image_ref: str | None, prompt_text: str) -> str:
        content = [{"type": "text", "text": prompt_text}]
        if image is not None:
            message_image = image_ref or image
            content.insert(0, {"type": "image", "image": message_image})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = move_batch_to_model(inputs, self.model)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return clean_prediction(output_text)

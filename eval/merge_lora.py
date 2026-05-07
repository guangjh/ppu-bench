import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)


WEIGHT_EXTENSIONS = {
    ".bin",
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
}


def copy_base_model_files(base_model_path: str, output_path: str):
    """
    Copy non-weight files from base model directory to output directory.

    This preserves files such as:
    - config.json
    - generation_config.json
    - tokenizer_config.json
    - special_tokens_map.json
    - tokenizer.json
    - vocab.json
    - merges.txt
    - preprocessor_config.json
    - processor_config.json
    - chat_template.json

    It skips original weight files to avoid mixing base weights with merged weights.
    """
    base_model_path = Path(base_model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for item in base_model_path.iterdir():
        src = item
        dst = output_path / item.name

        if item.is_file():
            if item.suffix in WEIGHT_EXTENSIONS:
                continue
            shutil.copy2(src, dst)

        elif item.is_dir():
            if item.name in {".git", "__pycache__", ".cache"}:
                continue

            if dst.exists():
                shutil.rmtree(dst)

            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns(
                    "*.bin",
                    "*.safetensors",
                    "*.pt",
                    "*.pth",
                    "*.ckpt",
                    "__pycache__",
                    ".git",
                    ".cache",
                ),
            )

    print(f"[Info] Copied auxiliary files to: {output_path}")


def get_torch_dtype(dtype: str):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    return dtype_map[dtype]


def load_processor_or_tokenizer(base_model_path: str):
    """
    For multimodal models, AutoProcessor is preferred.
    If AutoProcessor fails, fallback to AutoTokenizer.
    """
    try:
        print("[Info] Loading processor...")
        processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        return processor
    except Exception as e:
        print(f"[Warning] AutoProcessor loading failed: {e}")
        print("[Info] Falling back to AutoTokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        return tokenizer


def load_base_model(
    base_model_path: str,
    model_type: str,
    torch_dtype,
    device_map: str = "auto",
):
    """
    Load Qwen3-VL / Gemma3 / auto model.

    For Qwen3-VL:
        Prefer Qwen3VLForConditionalGeneration.

    For Gemma3:
        Prefer Gemma3ForConditionalGeneration.
        If unavailable, fallback to AutoModelForCausalLM.
    """

    if model_type == "qwen3vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration

            print("[Info] Loading model with Qwen3VLForConditionalGeneration...")
            return Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[Warning] Qwen3VLForConditionalGeneration failed: {e}")
            print("[Info] Falling back to AutoModelForCausalLM...")

            return AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )

    elif model_type == "gemma3":
        try:
            from transformers import Gemma3ForConditionalGeneration

            print("[Info] Loading model with Gemma3ForConditionalGeneration...")
            return Gemma3ForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[Warning] Gemma3ForConditionalGeneration failed: {e}")
            print("[Info] Falling back to AutoModelForCausalLM...")

            return AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )

    elif model_type == "auto":
        print("[Info] Auto-detecting model architecture...")
        config = AutoConfig.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )

        archs = getattr(config, "architectures", None)
        model_type_name = getattr(config, "model_type", None)

        print(f"[Info] config.model_type = {model_type_name}")
        print(f"[Info] config.architectures = {archs}")

        if archs is not None:
            archs_lower = [a.lower() for a in archs]

            if any("qwen3vl" in a for a in archs_lower):
                try:
                    from transformers import Qwen3VLForConditionalGeneration

                    print("[Info] Detected Qwen3-VL.")
                    return Qwen3VLForConditionalGeneration.from_pretrained(
                        base_model_path,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                    )
                except Exception as e:
                    print(f"[Warning] Qwen3-VL loading failed: {e}")

            if any("gemma3" in a for a in archs_lower):
                try:
                    from transformers import Gemma3ForConditionalGeneration

                    print("[Info] Detected Gemma3.")
                    return Gemma3ForConditionalGeneration.from_pretrained(
                        base_model_path,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                    )
                except Exception as e:
                    print(f"[Warning] Gemma3 loading failed: {e}")

        print("[Info] Falling back to AutoModelForCausalLM...")
        return AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def merge_lora(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    model_type: str,
    torch_dtype,
    device_map: str = "auto",
    max_shard_size: str = "4GB",
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[Step 1] Copying base model auxiliary files...")
    copy_base_model_files(base_model_path, output_path)

    print("=" * 80)
    print("[Step 2] Loading base model...")
    base_model = load_base_model(
        base_model_path=base_model_path,
        model_type=model_type,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    print("=" * 80)
    print("[Step 3] Loading processor/tokenizer...")
    processor_or_tokenizer = load_processor_or_tokenizer(base_model_path)

    print("=" * 80)
    print("[Step 4] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch_dtype,
    )

    print("=" * 80)
    print("[Step 5] Merging LoRA weights...")
    model = model.merge_and_unload()

    print("=" * 80)
    print("[Step 6] Saving merged model...")
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    print("=" * 80)
    print("[Step 7] Saving processor/tokenizer...")
    processor_or_tokenizer.save_pretrained(output_path)

    print("=" * 80)
    print(f"[Done] Merged model saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into Qwen3-VL or Gemma3 base model."
    )

    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base model.",
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter/checkpoint.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the merged model.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["qwen3vl", "gemma3", "auto"],
        help="Model type. Recommend explicitly setting qwen3vl or gemma3.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32", "auto"],
        help="Model loading dtype.",
    )

    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Device map, usually "auto".',
    )

    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="4GB",
        help="Max shard size when saving merged model.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch_dtype = get_torch_dtype(args.dtype)

    merge_lora(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        output_path=args.output_path,
        model_type=args.model_type,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        max_shard_size=args.max_shard_size,
    )
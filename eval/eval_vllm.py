import argparse
import io
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
try:
    import torch
except ImportError:  # pragma: no cover
    class _TorchStub:
        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        @staticmethod
        def no_grad():
            return _TorchStub._NoGrad()

    torch = _TorchStub()

try:
    from eval.metric import parse_choice_label, summarize_task_metrics
    from eval.ppu_eval_data import PPUEvalData, normalize_task_name
    from eval.testset_data import resolve_testset_data_file, setting_file_stem
except ModuleNotFoundError:  # pragma: no cover
    from metric import parse_choice_label, summarize_task_metrics
    from ppu_eval_data import PPUEvalData, normalize_task_name
    from testset_data import resolve_testset_data_file, setting_file_stem

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

REPO_ROOT = Path(__file__).resolve().parent.parent

ADV_SUBSETS = {"random_prefix", "jailbreak_style_prompt", "paraphrase"}


def _load_model_config(model_source):
    config_path = Path(model_source).expanduser() / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def resolve_model_family(model_reference):
    model_name = str(model_reference).lower()
    config = _load_model_config(model_reference)
    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(item).lower() for item in config.get("architectures", [])]

    if model_type == "llava" or "llavaforconditionalgeneration" in architectures or "llava" in model_name:
        return "llava"
    if model_type == "kimi_vl" or "kimivlforconditionalgeneration" in architectures or "kimi-vl" in model_name:
        return "kimi"
    if model_type == "gemma3" or "gemma3forconditionalgeneration" in architectures or "gemma-3" in model_name:
        return "gemma3"
    if model_type in {"qwen3_vl", "qwen3-vl"} or "qwen3vlforconditionalgeneration" in architectures or "qwen3-vl" in model_name:
        return "qwen3"
    if model_type in {"qwen2_vl", "qwen2_5_vl"} or any(name in model_name for name in ["qwen2-vl", "qwen2.5-vl"]):
        raise ValueError("Qwen2/Qwen2.5-VL are no longer supported. Please use a Qwen3-VL model instead.")
    raise ValueError(f"Unsupported model type: {model_reference}")


def load_pretrained_with_dtype(model_class, model_source, prefer_dtype_auto=False, **kwargs):
    if prefer_dtype_auto:
        try:
            return model_class.from_pretrained(model_source, dtype="auto",attn_implementation="flash_attention_2", **kwargs)
        except TypeError:
            return model_class.from_pretrained(model_source, torch_dtype="auto",attn_implementation="flash_attention_2", **kwargs)
    return model_class.from_pretrained(model_source, **kwargs)


def _ensure_kimi_transformers_compatibility():
    try:
        import transformers.activations as activations
        import transformers.cache_utils as cache_utils
    except ImportError:  # pragma: no cover
        return

    if hasattr(activations, "PytorchGELUTanh"):
        pass
    else:
        gelu_tanh = getattr(activations, "GELUTanh", None)
        if gelu_tanh is not None:
            activations.PytorchGELUTanh = gelu_tanh

    dynamic_cache = getattr(cache_utils, "DynamicCache", None)
    if dynamic_cache is not None and not hasattr(dynamic_cache, "seen_tokens"):
        dynamic_cache.seen_tokens = property(lambda self: self.get_seq_length())

    cache_base = getattr(cache_utils, "Cache", None)
    if cache_base is not None and not hasattr(cache_base, "get_usable_length"):
        def _get_usable_length(self, new_seq_length=0, layer_idx=0):
            max_length = self.get_max_cache_shape(layer_idx)
            previous_seq_length = self.get_seq_length(layer_idx)
            if max_length is None or max_length < 0:
                return previous_seq_length
            if previous_seq_length + new_seq_length > max_length:
                return max(max_length - new_seq_length, 0)
            return previous_seq_length

        cache_base.get_usable_length = _get_usable_length


def parse_arguments():
    def parse_bool(value):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")

    parser = argparse.ArgumentParser(description="Evaluate model on PPU-bench canonical eval data.")
    parser.add_argument("--model", required=True, help="Model id or model name.")
    parser.add_argument("--adapter_path", default="", help="Optional PEFT/LoRA adapter checkpoint path.")
    parser.add_argument("--pretrain", action="store_true", help="Load from model id instead of cache path.")
    parser.add_argument("--cache_path", default="", help="Local model path for fine-tuned checkpoints.")
    parser.add_argument("--data_file", default="eval/cache/ppu_eval_canonical.parquet", help="Canonical parquet file.")
    parser.add_argument("--hf", action="store_true", help="Load eval data from HuggingFace dataset closerG/ppu-bench instead of local parquet.")
    parser.add_argument("--testset", type=int, choices=[2, 3, 4], default=None, help="Use alternate-image testset 2/3/4.")
    parser.add_argument("--testset_root", default="eval/cache/testsets", help="Root directory for alternate-image testset parquet files.")
    parser.add_argument(
        "--complete_split_path",
        default=None,
        help="Path to complete-unlearning split JSON, or a directory containing CompleteUnlearning/forget_subject_ids.json.",
    )
    parser.add_argument("--run_id", required=True, help="Run identifier.")
    parser.add_argument("--setting", required=True, choices=["complete", "selective", "persona"], help="Unlearning setting.")
    parser.add_argument("--task", required=True, choices=["qa", "cloze", "class", "generation", "classification"], help="Eval task alias.")
    parser.add_argument("--subset", required=True, choices=["forget", "retain", "random_prefix", "jailbreak_style_prompt", "paraphrase"], help="Eval subset.")
    parser.add_argument("--ratio", type=int, default=None, help="Complete ratio. Required for complete.")
    parser.add_argument("--modality", choices=["qa", "vqa"], default=None, help="Optional modality filter.")
    parser.add_argument("--few_shot_data", default="", help="Optional canonical parquet for few-shot examples.")
    parser.add_argument("--shot_num", default="zero", help="Few-shot count or alias zero/one.")
    parser.add_argument("--output_dir", default="eval/results", help="Base output directory.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max generated tokens.")
    parser.add_argument("--resume", type=parse_bool, default=False, help="Resume from existing predictions")
    return parser.parse_args()


def resolve_eval_data_file(args):
    if args.testset is None:
        return args.data_file
    if args.subset != "forget":
        raise ValueError("--testset is only defined for forget subsets")
    if args.modality not in (None, "vqa"):
        raise ValueError("--testset is only defined for VQA evaluation; use --modality vqa or omit --modality")
    data_file = resolve_testset_data_file(
        setting=args.setting,
        ratio=args.ratio,
        testset=args.testset,
        output_root=args.testset_root,
    )
    if not data_file.exists():
        raise FileNotFoundError(f"Testset parquet not found: {data_file}. Run eval/prepare_testset_data.py first.")
    return str(data_file)


def resolve_few_shot_data(args):
    if args.few_shot_data:
        return args.few_shot_data
    if args.testset is not None:
        return args.data_file
    return None


def parse_shot_num(raw_value):
    text = str(raw_value).strip().lower()
    if text in {"zero", "0", "none"}:
        return 0
    if text in {"one", "1"}:
        return 1
    return max(int(text), 0)


def get_output_dir(args, normalized_task):
    output_dir = Path(args.output_dir) / args.run_id
    if args.setting == "complete" and args.ratio is not None:
        output_dir = output_dir / f"ratio_{args.ratio}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_adapter_path(adapter_path):
    if not adapter_path:
        return ""

    path = Path(adapter_path).expanduser()
    if path.is_dir():
        direct_config = path / "adapter_config.json"
        if direct_config.exists():
            return str(path.resolve())

        if (path / "config.json").exists():
            return ""

        epoch_4_config = path / "epoch_4" / "adapter_config.json"
        if epoch_4_config.exists():
            return str(epoch_4_config.parent.resolve())

        raise FileNotFoundError(
            f"Adapter path does not contain adapter_config.json: {path}. "
            "Pass a checkpoint directory such as .../epoch_4."
        )

    return adapter_path


def load_runtime(args):
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    except ImportError as error:  # pragma: no cover
        raise ImportError("transformers is required to run eval.py") from error

    if args.pretrain:
        model_source = args.model
    elif not args.cache_path:
        raise ValueError("--cache_path is required when --pretrain is not set")
    else:
        model_source = args.cache_path

    adapter_path = resolve_adapter_path(args.adapter_path)
    model_family = resolve_model_family(model_source)
    processor_source = model_source
    print(f"Loading model from {model_source}......")
    world_size = torch.cuda.device_count()

    # print("Initialize the vLLM")
    llm = LLM(
        model=model_source,
        tensor_parallel_size=world_size,
        gpu_memory_utilization=0.6,
        max_model_len=86256,
        # enable_lora=True,
    )

    # if model_family == "kimi":
    #     _ensure_kimi_transformers_compatibility()
    #     processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    #     tokenizer = AutoTokenizer.from_pretrained(processor_source, trust_remote_code=True)
    #     model = load_pretrained_with_dtype(
    #         AutoModelForCausalLM,
    #         model_source,
    #         device_map="auto",
    #         low_cpu_mem_usage=True,
    #         local_files_only=not args.pretrain,
    #         trust_remote_code=True,
    #         prefer_dtype_auto=True,
    #     )
    # else:
    #     processor = AutoProcessor.from_pretrained(processor_source)
    #     tokenizer = AutoTokenizer.from_pretrained(processor_source)
    #     if model_family == "llava":
    #         try:
    #             from transformers import LlavaForConditionalGeneration
    #         except ImportError as error:  # pragma: no cover
    #             raise ImportError("Current transformers install does not provide LlavaForConditionalGeneration") from error
    #         model = LlavaForConditionalGeneration.from_pretrained(
    #             model_source,
    #             device_map="auto",
    #             low_cpu_mem_usage=True,
    #             local_files_only=not args.pretrain,
    #         )
    #     elif model_family == "gemma3":
    #         try:
    #             from transformers import Gemma3ForConditionalGeneration
    #         except ImportError as error:  # pragma: no cover
    #             raise ImportError("Current transformers install does not provide Gemma3ForConditionalGeneration") from error
    #         model = load_pretrained_with_dtype(
    #             Gemma3ForConditionalGeneration,
    #             model_source,
    #             device_map="auto",
    #             low_cpu_mem_usage=True,
    #             local_files_only=not args.pretrain,
    #             trust_remote_code=True,
    #             prefer_dtype_auto=True,
    #         )
    #     elif model_family == "qwen3":
    #         try:
    #             from transformers import Qwen3VLForConditionalGeneration
    #         except ImportError as error:  # pragma: no cover
    #             raise ImportError("Current transformers install does not provide Qwen3VLForConditionalGeneration") from error
    #         model = load_pretrained_with_dtype(
    #             Qwen3VLForConditionalGeneration,
    #             model_source,
    #             device_map="auto",
    #             low_cpu_mem_usage=True,
    #             local_files_only=not args.pretrain,
    #             prefer_dtype_auto=True,
    #         )
    #     else:  # pragma: no cover
    #         raise ValueError(f"Unsupported model type: {args.model}")

    # if adapter_path:
    #     try:
    #         from peft import PeftModel
    #     except ImportError as error:  # pragma: no cover
    #         raise ImportError("peft is required when --adapter_path is provided") from error
    #     print(f"Loading LoRA Adapter from {adapter_path}......")
    #     model = PeftModel.from_pretrained(model, adapter_path)

    return llm, model_family


def load_image(image_path):
    if not image_path:
        return None
    if isinstance(image_path, Image.Image):
        return image_path.convert("RGB")
    resolved_path = Path(image_path)
    if not resolved_path.is_absolute():
        resolved_path = REPO_ROOT / resolved_path
    with Image.open(resolved_path) as image:
        return image.convert("RGB")


def load_image_from_row(row):
    image = row.get("image")
    if image is not None:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, dict):
            img_bytes = image.get("bytes")
            if img_bytes:
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_path = image.get("path")
            if img_path:
                return load_image(img_path)
        return load_image(image)
    image_bytes = row.get("image_bytes")
    if image_bytes is not None:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_path = row.get("image_path")
    if image_path:
        return load_image(image_path)
    return None


def _cleanup_cuda_cache():
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    is_available = getattr(cuda, "is_available", None)
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(is_available) and is_available() and callable(empty_cache):
        empty_cache()


def build_classification_prompt(row):
    # prompt = ""
    # for shot in few_shots:
    #     options_text = "\n".join(
    #         [
    #             f"A: {shot['option_a']}",
    #             f"B: {shot['option_b']}",
    #             f"C: {shot['option_c']}",
    #             f"D: {shot['option_d']}",
    #         ]
    #     )
    #     if shot["modality"] == "vqa":
    #         prompt += (
    #             "USER: <image>\n"
    #             f"{shot['question']}\n{options_text}\n"
    #             f"Correct Answer: {shot['answer_label']}\n"
    #         )
    #     else:
    #         prompt += (
    #             "USER:\n"
    #             f"{shot['question']}\n{options_text}\n"
    #             f"Correct Answer: {shot['answer_label']}\n"
    #         )

    options_text = "\n".join(
        [
            f"A: {row['option_a']}",
            f"B: {row['option_b']}",
            f"C: {row['option_c']}",
            f"D: {row['option_d']}",
        ]
    )
    prompt = (
        f"{row['question']}\n{options_text}\n"
        "Just give ONE letter representing the answer directly.\n"
    )
    return prompt


def build_cloze_prompt(row):
    # prompt = ""
    # for shot in few_shots:
    #     shot_question = str(shot["question"]).replace("___", "[Blank]")
    #     if shot["modality"] == "vqa":
    #         prompt += f"USER: <image>\n{shot_question}\nCorrect Answer: {shot['answer_text']}\n"
    #     else:
    #         prompt += f"USER:\n{shot_question}\nCorrect Answer: {shot['answer_text']}\n"

    question = str(row["question"]).replace("___", "[Blank]")
    suffix = "\nPlease ONLY provide the correct answer that should replace the [Blank]."
    # if row["modality"] == "vqa":
    #     return f"{question}{suffix}\n"
    return f"{question}{suffix}\n"


def build_generation_prompt(row):
    if row["modality"] == "vqa":
        return f"{row['question']}\nAnswer the question based on the image in one sentence accurately in ENGLISH.\n"
    return f"{row['question']}\nAnswer the question in one sentence in ENGLISH.\n"


def strip_legacy_chat_markers(prompt):
    text = str(prompt).strip()
    replacements = (
        ("USER: <image>\n", ""),
        ("USER:\n", ""),
        ("USER: ", ""),
        ("<image>\n", ""),
        ("<image>", ""),
        ("\nASSISTANT:", ""),
        ("ASSISTANT:", ""),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text.strip()


def generate_text(processor, tokenizer, model, model_family, prompt, image=None, max_new_tokens=50):
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if model_family == "qwen3":
        conversation = [{"role": "user", "content": []}]
        if image is not None:
            conversation[0]["content"].append({"type": "image"})
        conversation[0]["content"].append({"type": "text", "text": prompt})
        rendered_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=rendered_prompt, return_tensors="pt").to(model.device)
    elif model_family == "kimi":
        message_content = []
        if image is not None:
            message_content.append({"type": "image", "image": image})
        message_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": message_content}]
        rendered_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = processor(
            images=image,
            text=rendered_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        generation_kwargs["use_cache"] = False
    elif model_family == "gemma3":
        message_content = []
        if image is not None:
            message_content.append({"type": "image", "image": image})
        message_content.append({"type": "text", "text": strip_legacy_chat_markers(prompt)})
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {"role": "user", "content": message_content},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
    elif model_family == "llava":
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported model family: {model_family}")

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    out_wo_prompt = outputs[:, inputs.input_ids.shape[-1] :]
    if model_family == "kimi":
        return processor.batch_decode(
            out_wo_prompt,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
    return tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True).strip()


def append_prediction(predictions_path, record):
    with predictions_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def load_existing_predictions(predictions_path):
    if not predictions_path.exists():
        return []

    existing_predictions = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            existing_predictions.append(json.loads(line))
    return existing_predictions


def build_prediction_key(row):
    return (
        str(row.get("task_type")),
        str(row.get("modality")),
        str(row.get("sample_id")),
    )


def evaluate_rows(task_type, rows, llm, model_family, args, data_manager, predictions_path, existing_predictions=None):
    # shot_num = parse_shot_num(args.shot_num)
    predictions = list(existing_predictions or [])
    existing_keys = {build_prediction_key(row) for row in predictions}
    row_records = rows.to_dict(orient="records")
    pending_rows = [row for row in row_records if build_prediction_key(row) not in existing_keys]
    progress_bar = tqdm(
        pending_rows,
        total=len(row_records),
        initial=len(row_records) - len(pending_rows),
        desc=f"{args.setting}/{task_type}/{args.subset}",
        dynamic_ncols=True,
    )
    
    test_data = []
    for row in progress_bar:
        # few_shots = []
        # if task_type in {"classification", "cloze"}:
        #     few_shots = data_manager.sample_few_shot(
        #         task=task_type,
        #         modality=row["modality"],
        #         shot_num=shot_num,
        #         few_shot_data=args.few_shot_data or None,
        #         exclude_sample_ids={row["sample_id"]},
        #         setting=args.setting,
        #         ratio=args.ratio,
        #     )

        image = load_image_from_row(row)
        if task_type == "classification":
            prompt = build_classification_prompt(row)
        elif task_type == "cloze":
            prompt = build_cloze_prompt(row)
        elif task_type == "generation":
            prompt = build_generation_prompt(row)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported task type: {task_type}")

        if model_family == 'qwen3':
            chat_prompt_vqa = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
            chat_prompt_qa = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif model_family == 'gemma3':
            chat_prompt_vqa = f"<bos><start_of_turn>user\n<start_of_image>{prompt}<end_of_turn>\n<start_of_turn>model\n"
            chat_prompt_qa = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        if row['modality'] == 'qa':
            test_data.append(
                {
                    "prompt": chat_prompt_qa,
                }
            )
        elif row['modality'] == 'vqa':
            test_data.append(
                {
                    "prompt": chat_prompt_vqa,
                    "multi_modal_data": {"image": image}
                }
            )
    test_data = test_data 
    print(f"Loaded {len(test_data)} data for generation")
    print(f"Example data: \n{test_data[0]}\n{test_data[1]}\n{test_data[1]}\n{test_data[2]}\n{test_data[3]}\n{test_data[4]}")

    gen_params = [SamplingParams(temperature=0, max_tokens=args.max_new_tokens) for _ in range(len(test_data))]
    # print("Generating ......")
    outputs = llm.generate(test_data, gen_params, use_tqdm=True)
    gen_res = []

    for o in outputs:
        generated_text = o.outputs[0].text
        # print(generated_text)
        gen_res.append(generated_text)

    new_res = gen_res

    assert len(new_res) == len(test_data), f"Length of results and data should be equal. Got {len(new_res)} and {len(test_data)}"

    for i in range(len(test_data)):
        record = dict(pending_rows[i])
        record.pop("image", None)
        record['prompt'] = test_data[i]['prompt']
        record["prediction"] = new_res[i]
        if task_type == "classification":
            record["parsed_label"] = parse_choice_label(new_res[i])
        predictions.append(record)
        append_prediction(predictions_path, record)

        # prediction = generate_text(processor, tokenizer, model, model_family, prompt, image=image, max_new_tokens=args.max_new_tokens)
        # if model_family == "kimi":
        #     _cleanup_cuda_cache()

        # record = dict(row)
        # record["prompt"] = prompt
        # record["prediction"] = prediction
        # if task_type == "classification":
        #     record["parsed_label"] = parse_choice_label(prediction)
        # predictions.append(record)
        # append_prediction(predictions_path, record)

    return predictions


def save_outputs(output_dir, config, predictions, metrics):
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    with (output_dir / "predictions_indent.json").open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2)


def load_hf_data(args, normalized_task):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library is required when --hf is set. Install with: pip install datasets")

    if args.subset in ADV_SUBSETS:
        if args.setting == "complete":
            setting_suffix = f"complete{args.ratio}"
        else:
            setting_suffix = args.setting
        hf_config = f"{args.subset}_{setting_suffix}"
        print(f"Loading HF dataset: closerG/ppu-bench [{hf_config}]")
        ds = load_dataset("closerG/ppu-bench", hf_config)["test"]
        df = ds.to_pandas()
        df["modality"] = df["modality"].replace({"text": "qa", "text_image": "vqa"})
        return df.reset_index(drop=True)

    if args.testset is not None:
        stem = setting_file_stem(args.setting, args.ratio)
        hf_config = f"cross_image_{stem}"
    else:
        hf_config = f"ppu_eval_{normalized_task}"

    print(f"Loading HF dataset: closerG/ppu-bench [{hf_config}]")
    ds = load_dataset("closerG/ppu-bench", hf_config)["test"]
    df = ds.to_pandas()

    df["modality"] = df["modality"].replace({"text": "qa", "text_image": "vqa"})

    if args.testset is not None:
        image_col = f"image_{args.testset:03d}"
        if image_col not in df.columns:
            raise ValueError(f"Testset image column {image_col} not found. Available: {list(df.columns)}")
        df["image"] = df[image_col]

    data_manager = PPUEvalData(
        complete_split_path=args.complete_split_path,
        selective_split_path=getattr(args, "selective_split_path", None),
        persona_split_path=getattr(args, "persona_split_path", None),
    )
    data_manager.set_dataframe(df)
    rows = data_manager.get_rows(
        setting=args.setting,
        task=normalized_task,
        subset=args.subset,
        ratio=args.ratio,
        modality=args.modality,
    )
    if rows.empty:
        raise ValueError("No rows found for the requested eval slice")
    return rows

def main():
    args = parse_arguments()
    normalized_task = normalize_task_name(args.task)
    if args.setting == "complete" and args.ratio not in {5, 10, 15, 30}:
        raise ValueError("--ratio must be one of 5/10/15/30 when --setting complete")

    if args.hf:
        rows = load_hf_data(args, normalized_task)
        args.resolved_data_file = "hf://closerG/ppu-bench"
        args.resolved_few_shot_data = ""
        data_manager = None
    elif args.subset in ADV_SUBSETS:
        local_dir_map = {
            "random_prefix": "easy_prefix",
            "jailbreak_style_prompt": "hard_prefix",
            "paraphrase": "para",
        }
        mode = local_dir_map[args.subset]
        data_path = os.path.join(
            REPO_ROOT / "eval" / "cache" / "adv",
            mode,
            f"{args.setting}_{args.task}_forget_para.parquet",
        )
        print("=" * 60)
        print(f"DATA_FILE={data_path}")
        print("=" * 60)
        rows = pd.read_parquet(data_path)
        rows["sample_id"] = rows["sample_id"].astype(str)
        rows["subject_id"] = rows["subject_id"].astype(str)
        rows["task_type"] = rows["task_type"].astype(str)
        rows["modality"] = rows["modality"].astype(str)
        rows = rows.reset_index(drop=True)
        args.resolved_data_file = data_path
        args.resolved_few_shot_data = ""
        data_manager = None
    else:
        args.resolved_data_file = resolve_eval_data_file(args)
        args.resolved_few_shot_data = resolve_few_shot_data(args)
        print(f"DATA_FILE={args.resolved_data_file}")
        if args.testset is not None:
            print(f"TESTSET={args.testset}")
            print(f"FEW_SHOT_DATA={args.resolved_few_shot_data}")

        data_manager = PPUEvalData(canonical_path=args.resolved_data_file, complete_split_path=args.complete_split_path)
        rows = data_manager.get_rows(
            setting=args.setting,
            task=normalized_task,
            subset=args.subset,
            ratio=args.ratio,
            modality=args.modality,
        )
        if rows.empty:
            raise ValueError("No rows found for the requested eval slice")

    output_dir = get_output_dir(args, normalized_task)
    predictions_path = output_dir / "predictions.jsonl"
    if args.resume:
        existing_predictions = load_existing_predictions(predictions_path)
    else:
        existing_predictions = []
    if existing_predictions:
        print(f"Resuming from existing predictions: {len(existing_predictions)} rows already completed.")
    elif not predictions_path.exists():
        predictions_path.write_text("", encoding="utf-8")

    config = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "pretrain": args.pretrain,
        "cache_path": args.cache_path,
        "data_file": args.resolved_data_file,
        "base_data_file": args.data_file,
        "testset": args.testset,
        "testset_root": args.testset_root,
        "run_id": args.run_id,
        "setting": args.setting,
        "task": normalized_task,
        "subset": args.subset,
        "ratio": args.ratio,
        "modality": args.modality,
        "few_shot_data": args.resolved_few_shot_data or "",
        "shot_num": args.shot_num,
        "max_new_tokens": args.max_new_tokens,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    llm, model_family = load_runtime(args)

    predictions = evaluate_rows(
        normalized_task,
        rows,
        llm,
        model_family,
        args,
        data_manager,
        predictions_path,
        existing_predictions=existing_predictions,
    )

    metrics = {
        "task": normalized_task,
        "setting": args.setting,
        "subset": args.subset,
        "ratio": args.ratio,
        "num_rows": len(predictions),
        "metrics": {},
    }
    modalities = sorted({row["modality"] for row in predictions})
    for modality in modalities:
        modality_predictions = [row for row in predictions if row["modality"] == modality]
        metrics["metrics"][modality] = summarize_task_metrics(normalized_task, modality_predictions)

    save_outputs(output_dir, config, predictions, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

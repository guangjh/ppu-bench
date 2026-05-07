#!/usr/bin/env python3
"""
Evaluate one model on QA or VQA parquet datasets.

Supported model families:
- gemma3
- llava
- qwen3_vl
- kimi_vl

The script expands each record's `qas` field into individual evaluation samples,
runs generation, and reports Exact Match plus token-level F1 against the reference answer.

If you want to change how each sample is turned into a prompt, edit
`prepare_evaluation_sample` below.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from tqdm import tqdm

try:
    from generation.data_loader import load_evaluation_samples, load_pil_image
except ImportError:
    from data_loader import load_evaluation_samples, load_pil_image

try:
    from generation.metric import exact_match_score, token_f1_score
except ImportError:
    from metric import exact_match_score, token_f1_score

try:
    from generation.runner import infer_model_type, load_runner
except ImportError:
    from runner import infer_model_type, load_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one model on QA or VQA parquet datasets."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or local model path.",
    )
    parser.add_argument(
        "--data",
        default="data/vqa_500.parquet",
        help="Path to the parquet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root directory used to store run outputs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id. Defaults to a sanitized version of `--model`.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of newly generated tokens per answer.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of flattened QA samples to evaluate.",
    )
    parser.add_argument(
        "--subject-limit",
        type=int,
        default=None,
        help="Maximum number of subject records to load before flattening QAs.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Passed to `from_pretrained`. Use `auto`, `cuda:0`, `cpu`, or `none`.',
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype used when loading the model.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["none", "flash_attention_2", "sdpa", "eager"],
        default="none",
        help="Optional attention implementation passed to `from_pretrained`.",
    )
    parser.add_argument(
        "--gemma3-backend",
        choices=["transformers", "modelscope"],
        default="transformers",
        help="Backend used for Gemma3 only. `modelscope` uses the official image-text-to-text pipeline.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing prediction file if it exists.",
    )
    parser.add_argument(
        "--task",
        choices=["auto", "qa", "vqa"],
        default="auto",
        help="How to treat each sample. `auto` uses VQA when an image is available, otherwise QA.",
    )
    parser.add_argument(
        "--hf-config",
        default="",
        choices=["", "unlearning_target"],
        help="Load data from HuggingFace config instead of a local file (overrides --data).",
    )
    return parser.parse_args()


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return sanitized.strip("._-") or "run"


def infer_sample_task(sample: dict[str, Any], requested_task: str) -> str:
    if requested_task != "auto":
        return requested_task

    has_image = (
        bool(sample.get("image_path"))
        or isinstance(sample.get("image_blob"), (dict, Image.Image))
    )
    return "vqa" if has_image else "qa"


def prepare_evaluation_sample(sample: dict[str, Any], *, requested_task: str) -> dict[str, str]:
    """
    Edit this function if you want to change how each sample is turned
    into a prompt/reference pair for evaluation.
    """
    task = infer_sample_task(sample, requested_task)
    if task == "vqa":
        prompt_text = (
            "Answer the question based on the image in one sentence accurately in ENGLISH.\n"
            # "Answer the question based on the image in a word or a short phrase.\n"
            f"{sample['question']}"
        )
    else:
        prompt_text = (
            "Answer the question in one sentence accurately in ENGLISH.\n"
            f"{sample['question']}"
        )
    return {
        "task": task,
        "prompt_text": prompt_text,
        "reference_answer": sample["ground_truth"],
    }


def load_existing_results(
    prediction_path: Path,
) -> tuple[set[str], float, float, int]:
    completed_ids: set[str] = set()
    em_total = 0.0
    f1_total = 0.0
    count = 0

    if not prediction_path.exists():
        return completed_ids, em_total, f1_total, count

    with prediction_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("id") or row.get("sample_id")
            if sample_id:
                completed_ids.add(str(sample_id))

            em_total += float(row.get("exact_match", 0.0))
            f1_total += float(row.get("token_f1", 0.0))
            count += 1

    return completed_ids, em_total, f1_total, count


def load_prediction_rows(prediction_path: Path) -> list[dict[str, Any]]:
    rows = []
    if not prediction_path.exists():
        return rows

    with prediction_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_pretty_predictions(prediction_path: Path, indent: int = 4) -> None:
    pretty_path = prediction_path.with_name("predictions_indent.json")
    rows = load_prediction_rows(prediction_path)
    with pretty_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=indent)
        file.write("\n")


def evaluate_one_model(
    model_name_or_path: str,
    model_type: str,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_output_dir = Path(args.output_dir) / args.run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_output_dir / "predictions.jsonl"
    summary_path = run_output_dir / "summary.json"

    # if prediction_path.exists() and not args.resume:
    #     raise FileExistsError(
    #         f"{prediction_path} already exists. Use `--resume` or change `--run-id`."
    #     )

    completed_ids, em_total, f1_total, completed_count = load_existing_results(prediction_path)
    remaining_samples = [
        sample for sample in samples if sample["id"] not in completed_ids
    ]

    if not remaining_samples and completed_count:
        write_pretty_predictions(prediction_path)
        dataset_desc = args.hf_config or str(Path(args.data).resolve())
        summary = {
            "run_id": args.run_id,
            "model": model_name_or_path,
            "model_type": model_type,
            "dataset": dataset_desc,
            "prediction_path": str(prediction_path.resolve()),
            "evaluated_samples": completed_count,
            "exact_match": em_total / completed_count,
            "token_f1": f1_total / completed_count,
            "max_new_tokens": args.max_new_tokens,
            "gemma3_backend": args.gemma3_backend,
            "limit": args.limit,
            "subject_limit": args.subject_limit,
        }
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        return summary

    runner = load_runner(
        model_name_or_path,
        model_type=model_type,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        gemma3_backend=args.gemma3_backend,
    )

    image_cache: dict[str, Image.Image] = {}
    sample_fields_logged = False
    progress = tqdm(
        remaining_samples,
        desc=f"Evaluating {args.run_id}",
    )

    prediction_file_mode = "a" if completed_count else "w"
    with prediction_path.open(prediction_file_mode, encoding="utf-8") as prediction_file:
        for sample_index, sample in enumerate(progress):
            current_subject_key = sample.get("subject_id") or sample.get("record_id") or sample.get("subject")
            next_sample = remaining_samples[sample_index + 1] if sample_index + 1 < len(remaining_samples) else None
            next_subject_key = (
                next_sample.get("subject_id") or next_sample.get("record_id") or next_sample.get("subject")
                if next_sample is not None
                else None
            )

            prepared = prepare_evaluation_sample(sample, requested_task=args.task)
            image, image_ref = load_pil_image(
                sample,
                image_cache,
                require_image=prepared["task"] == "vqa",
            )
            prompt_text = prepared["prompt_text"]
            reference_answer = prepared["reference_answer"]

            error_message = None
            prediction = ""
            prediction = runner.generate(image=image, image_ref=image_ref, prompt_text=prompt_text)
            # print(prediction)
            # exit()
            # except Exception as exc:
            #     error_message = f"{type(exc).__name__}: {exc}"
            #     print("Generation_error:", error_message)

            em = exact_match_score(prediction, reference_answer)
            f1 = token_f1_score(prediction, reference_answer)

            em_total += em
            f1_total += f1
            completed_count += 1
            result_row = {
                "subject_id": sample.get("subject_id"),
                "id": sample["id"],
                "task": prepared["task"],
                "subject": sample["subject"],
                "question": sample["question"],
                "ground_truth": reference_answer,
                "qa_source": sample["qa_source"],
                "generated_answer": prediction,
                "token_f1": f1,
                "Exact_match": em,
            }
            if error_message:
                result_row["error"] = error_message

            print("Question:", sample["question"])
            print("Ground_truth:", reference_answer)
            print("Generated_answer:", prediction)
            print('Exact_match:', em)
            print("Token_f1:", f1)

            prediction_file.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            prediction_file.flush()

            progress.set_postfix(
                em=f"{(em_total / completed_count):.4f}",
                f1=f"{(f1_total / completed_count):.4f}",
            )
            if current_subject_key != next_subject_key:
                write_pretty_predictions(prediction_path)
	    
    write_pretty_predictions(prediction_path)

    dataset_desc = args.hf_config or str(Path(args.data).resolve())
    summary = {
        "run_id": args.run_id,
        "model": model_name_or_path,
        "model_type": model_type,
        "dataset": dataset_desc,
        "prediction_path": str(prediction_path.resolve()),
        "evaluated_samples": completed_count,
        "exact_match": em_total / completed_count if completed_count else 0.0,
        "token_f1": f1_total / completed_count if completed_count else 0.0,
        "max_new_tokens": args.max_new_tokens,
        "gemma3_backend": args.gemma3_backend,
        "limit": args.limit,
        "subject_limit": args.subject_limit,
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    for image in image_cache.values():
        image.close()

    del runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main() -> None:
    args = parse_args()
    model_type = infer_model_type(args.model)
    if model_type is None:
        raise ValueError(
            f"Unsupported model path or name for automatic runner selection: {args.model}"
        )

    if args.run_id is None:
        args.run_id = sanitize_path_component(Path(args.model).name)

    data_path = Path(args.data).resolve()
    samples = load_evaluation_samples(
        data_path,
        hf_config=args.hf_config,
        subject_limit=args.subject_limit,
        qa_limit=args.limit,
    )

    if not samples:
        raise RuntimeError("No evaluation samples were loaded.")

    summary = evaluate_one_model(args.model, model_type, samples, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import ConcatDataset, DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "baselines"))

from baselines.MMunlearner import load_model_and_processor
from baselines.ppu_data import (
    DEFAULT_COMPLETE_SPLIT_FILE,
    DEFAULT_PERSONA_SPLIT_FILE,
    DEFAULT_SELECTIVE_SPLIT_FILE,
    LLAVA_multimodal_Dataset,
    get_complete_ds,
    get_persona_ds,
    get_selective_ds,
    load_hf_train_records,
    load_train_records,
)
from data_process.CLEAR_process import train_collate_clear, train_collate_clear_ansonly
from data_process.SFRon import Mask_Our, Mask_grad


def parse_csv_arg(value):
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def default_mask_modules(model_id):
    model_id = model_id.lower()
    if "qwen" in model_id:
        return ["visual", "merger"], ["language_model"]
    elif "kimi" in model_id or "moonshot" in model_id or "kimi-vl-a3b" in model_id:
        return ["vision_tower", "multi_modal_projector"], ["language_model"]
    elif "gemma" in model_id:
        return ["vision_tower", "multi_modal_projector"], ["language_model"]
    elif "llava" in model_id:
        return ["vision_tower", "vision_model", "multi_modal_projector"], ["language_model"]
    return ["vision_tower", "vision_model", "multi_modal_projector", "visual"], ["language_model"]


def get_train_datasets(args):
    if args.task == "complete":
        forget_dataset, retain_dataset = get_complete_ds(
            train_ds_path=args.train_dataset_dir,
            hf_config=args.hf_config,
            ratio=args.forget_split_ratio,
            vqa=args.vqa,
        )
        split_reference = args.data_split_dir or str(DEFAULT_COMPLETE_SPLIT_FILE)
    elif args.task == "selective":
        forget_dataset, retain_dataset = get_selective_ds(
            train_ds_path=args.train_dataset_dir,
            hf_config=args.hf_config,
            vqa=args.vqa,
        )
        split_reference = args.data_split_dir or str(DEFAULT_SELECTIVE_SPLIT_FILE)
    elif args.task == "persona":
        forget_dataset, retain_dataset = get_persona_ds(
            train_ds_path=args.train_dataset_dir,
            hf_config=args.hf_config,
            vqa=args.vqa,
        )
        split_reference = args.data_split_dir or str(DEFAULT_PERSONA_SPLIT_FILE)
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    return forget_dataset, retain_dataset, split_reference


def build_dataloaders(args, processor):
    forget_dataset, retain_dataset, split_reference = get_train_datasets(args)

    if args.hf_config:
        full_text_dataset = LLAVA_multimodal_Dataset(load_hf_train_records(args.hf_config), vqa=False)
    else:
        full_text_dataset = LLAVA_multimodal_Dataset(load_train_records(args.train_dataset_dir), vqa=False)

    # full_preserve_dataset = ConcatDataset([full_text_dataset, retain_dataset])
    full_preserve_dataset = retain_dataset

    if args.ans_only:
        train_collate_function = train_collate_clear_ansonly
        print("Answer only mode enabled.")
    else:
        train_collate_function = train_collate_clear
        print("Answer only mode disabled.")

    print("Train Dataset: ", args.train_dataset_dir)
    print("HF Config: ", args.hf_config)
    print("Split Reference: ", split_reference)
    print("Task: ", args.task)
    if args.task == "complete":
        print("Forget Split Ratio: ", args.forget_split_ratio)
    print("VQA Mode: ", args.vqa)
    print("Forget Dataset Size: ", len(forget_dataset))
    print("Retain Dataset Size: ", len(retain_dataset))
    print("Full Preserve Dataset Size: ", len(full_preserve_dataset))

    if len(forget_dataset) == 0 or len(retain_dataset) == 0:
        raise ValueError("Forget and retain datasets must both be non-empty.")

    forget_dataloader = DataLoader(
        forget_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_function(x, processor, None, True),
    )
    full_preserve_dataloader = DataLoader(
        full_preserve_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_function(x, processor, None, True),
    )

    return forget_dataloader, full_preserve_dataloader


def get_cache_dir(root_dir, name):
    if not root_dir:
        return ""

    cache_dir = os.path.join(root_dir, name)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def save_mask(mask_tuple, path):
    weight_mask, forget_grad, preserve_grad = mask_tuple
    result = {
        "weight": {name: mask for name, mask in weight_mask.items() if torch.is_tensor(mask)},
        "forget_grad": {name: grad for name, grad in forget_grad.items() if torch.is_tensor(grad)},
        "preserve_grad": {name: grad for name, grad in preserve_grad.items() if torch.is_tensor(grad)},
    }
    torch.save(result, path)
    print(f"Mask saved to: {path} ({len(result['weight'])} tensor masks)")


def main(args):
    model, processor = load_model_and_processor(args)
    tokenizer = processor.tokenizer
    print("Tokenizer Length: ", len(tokenizer))

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    forget_dataloader, full_preserve_dataloader = build_dataloaders(args, processor)

    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise ValueError("Mask generation currently expects one process. Run this script with plain python.")

    model, forget_dataloader, full_preserve_dataloader = accelerator.prepare(
        model,
        forget_dataloader,
        full_preserve_dataloader,
    )

    # mask_cls = Mask_grad if args.mask_method == "grad" else Mask_Our
    # mask_generator = mask_cls(model, args.mask_lr)
    mg=Mask_Our(model,1e-5)

    default_vision_modules, default_language_modules = default_mask_modules(args.model_id)
    vision_modules = parse_csv_arg(args.vision_modules) or default_vision_modules
    language_modules = parse_csv_arg(args.language_modules) or default_language_modules
    full_modules = vision_modules + language_modules

    print("Start masking ...")
    print("Full mask modules: ", full_modules)

    os.makedirs(args.mask_save_dir, exist_ok=True)
    both_path = os.path.join(args.mask_save_dir, f"mllmu_both_mask_{args.mask_label}.pt")

    fisher_cache_root = args.fisher_cache_dir or os.path.join(args.mask_save_dir, "fisher_cache")

    # both_mask = mask_generator.prepare_weight_saliency_mask(
    #     modules=full_modules,
    #     forget_loader=forget_dataloader,
    #     preserve_loader=full_preserve_dataloader,
    #     threshold=args.mask_threshold,
    #     save_path=get_cache_dir(fisher_cache_root, "both"),
    # )
    # save_mask(both_mask, both_path)

    weight_mask,forget_grad,preserve_grad=mg.prepare_weight_saliency_mask(modules=vision_modules, forget_loader=forget_dataloader, preserve_loader=full_preserve_dataloader, threshold=1,save_path="")
    # res={"weight":weight_mask,"forget_grad":forget_grad,"preserve_grad":preserve_grad}
    # torch.save(res,f"{folder}/mllmu_vision_mask_{label}.pt")
    weight_mask,forget_grad,preserve_grad=mg.prepare_weight_saliency_mask(modules=language_modules, forget_loader=forget_dataloader, preserve_loader=full_preserve_dataloader, threshold=1,save_path="")
    # res={"weight":weight_mask,"forget_grad":forget_grad,"preserve_grad":preserve_grad}
    # torch.save(res,f"{folder}/mllmu_language_mask_{label}.pt")

    weight_mask,forget_grad,preserve_grad=mg.get_weight_saliency_mask()
    res={"weight":weight_mask,"forget_grad":forget_grad,"preserve_grad":preserve_grad}
    torch.save(res, both_path)

    with open(os.path.join(args.mask_save_dir, "mask_config.json"), "wt", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate saliency masks for GA-Diff-Mask")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Pretrained model directory")
    parser.add_argument(
        "--data_split_dir",
        type=str,
        default=None,
        help=(
            "Split file path or split root directory. "
            f"Defaults to task-specific paths: complete={DEFAULT_COMPLETE_SPLIT_FILE}, "
            f"selective={DEFAULT_SELECTIVE_SPLIT_FILE}, "
            f"persona={DEFAULT_PERSONA_SPLIT_FILE}"
        ),
    )
    parser.add_argument(
        "--forget_split_ratio",
        type=int,
        default=5,
        help="Forget split ratio for complete unlearning: 5, 10 or 15",
    )
    parser.add_argument(
        "--train_dataset_dir",
        type=str,
        default="",
        help="Path to the model-specific train_pair.json or train_pair.parquet",
    )
    parser.add_argument(
        "--hf_config",
        type=str,
        default="",
        choices=["train_pair_qwen3", "train_pair_gemma3"],
        help="HuggingFace config name (overrides --train_dataset_dir)",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for mask generation")
    parser.add_argument("--ans_only", action="store_true", help="Answer only for mask generation")
    parser.add_argument(
        "--vqa",
        action="store_true",
        help="Use image + question pairs if true, otherwise train in QA-only mode",
    )
    parser.add_argument("--task", type=str, default="complete", choices=["complete", "selective", "persona"], help="Unlearning task type")
    parser.add_argument("--mask_save_dir", type=str, required=True, help="Directory to save generated masks")
    parser.add_argument("--mask_label", type=str, default="ours", help="Suffix used in saved mask filenames")
    parser.add_argument("--mask_threshold", type=float, default=1.0, help="Saliency threshold")
    parser.add_argument("--mask_lr", type=float, default=1e-5, help="Learning rate used by the mask optimizer")
    parser.add_argument("--mask_method", type=str, default="our", choices=["our", "grad"], help="Mask generator implementation")
    parser.add_argument("--vision_modules", type=str, default=None, help="Comma-separated vision module name filters")
    parser.add_argument("--language_modules", type=str, default=None, help="Comma-separated language module name filters")
    parser.add_argument(
        "--fisher_cache_dir",
        type=str,
        default=None,
        help="Optional cache root for forget/preserve fisher tensors. A single both/ subdir is used.",
    )

    main(parser.parse_args())

import sys
import os
from math import ceil
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel

sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

import torch
import json
import argparse
from torch.optim import AdamW
from transformers import get_scheduler
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from data_process.CLEAR_process import (
    CLEAR_Dataset,
    CAPTION_MODE,
    RECOGNITION_MODE,
    train_collate_clear,
    NONE_MODE,
    train_collate_clear_ansonly,
)
from accelerate import Accelerator

sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

from data_process.CLEAR_process import train_collate_clear, train_collate_clear_ansonly

try:
    from ppu_data import (
        DEFAULT_COMPLETE_SPLIT_FILE,
        DEFAULT_PERSONA_SPLIT_FILE,
        DEFAULT_SELECTIVE_SPLIT_FILE,
        get_complete_ds,
        get_persona_ds,
        get_selective_ds,
    )
except ImportError:
    from baselines.ppu_data import (
        DEFAULT_COMPLETE_SPLIT_FILE,
        DEFAULT_PERSONA_SPLIT_FILE,
        DEFAULT_SELECTIVE_SPLIT_FILE,
        get_complete_ds,
        get_persona_ds,
        get_selective_ds,
    )


NON_LORA_MODULE_NAME_PARTS = {
    "multi_modal_projector",  # LLaVA/Kimi projector
    "vision_model",  # LLaVA CLIP internals
    "vision_tower",  # LLaVA/Kimi visual encoder root
    "visual",  # Qwen3-VL visual encoder root
}

DEFAULT_MASK_NAME_FILTERS = ("proj", "fc", "linear", "mlp", "qkv", "lora")


def is_non_lora_linear_name(module_name):
    name_parts = set(module_name.split("."))
    return bool(name_parts & NON_LORA_MODULE_NAME_PARTS)


def get_lora_target_name(module_name):
    names = module_name.split(".")
    return names[0] if len(names) == 1 else ".".join(names[-2:])


def find_all_linear_names(model):
    print(model)
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if is_non_lora_linear_name(name):
            continue
        if isinstance(module, cls):
            lora_module_names.add(get_lora_target_name(name))

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return sorted(lora_module_names)


def resize_token_embeddings_if_needed(model, processor):
    tokenizer = processor.tokenizer
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size before applying LoRA.")
        model.resize_token_embeddings(len(tokenizer))
    return model



def configure_kimi_moe_for_training(model):
    updated_gate_count = 0
    updated_attention_count = 0

    for module in model.modules():
        if getattr(module, "topk_method", None) == "noaux_tc":
            module.topk_method = "greedy"
            updated_gate_count += 1

        if hasattr(module, "attn_implementation") and getattr(module, "attn_implementation", None) != "eager":
            module.attn_implementation = "eager"
            updated_attention_count += 1

    if getattr(model.config, "topk_method", None) == "noaux_tc":
        model.config.topk_method = "greedy"

    model_dtype = next(model.parameters()).dtype

    if hasattr(model, "vision_tower"):
        model.vision_tower.to(dtype=model_dtype)
    if hasattr(model, "multi_modal_projector"):
        model.multi_modal_projector.to(dtype=model_dtype)

    if updated_gate_count:
        print(f"Set {updated_gate_count} Kimi MoE gates to greedy top-k for training.")
    if updated_attention_count:
        print(f"Set {updated_attention_count} Kimi vision attention modules to eager attention.")

    return model

def load_model_and_processor(args):
    if "llava" in args.model_id.lower():
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(
            args.vanilla_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
        model = resize_token_embeddings_if_needed(model, processor)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    elif "gemma" in args.model_id.lower():
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            args.vanilla_dir,
            trust_remote_code=True,
            local_files_only=True,
        )

        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "right"

        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        model = resize_token_embeddings_if_needed(model, processor)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    elif "kimi" in args.model_id.lower():
        from transformers import AutoProcessor, AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.vanilla_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model = configure_kimi_moe_for_training(model)

        processor = AutoProcessor.from_pretrained(
            args.vanilla_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        processor.tokenizer.padding_side = "right"
        model = resize_token_embeddings_if_needed(model, processor)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = configure_kimi_moe_for_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    elif "qwen3" in args.model_id.lower():
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(
            args.vanilla_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        processor.tokenizer.padding_side = "right"
        model = resize_token_embeddings_if_needed(model, processor)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor


def parse_csv_arg(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [item.strip() for item in value.split(",") if item.strip()]


def calc_sparsity(tensor):
    num_zero_elements = tensor.numel() - torch.count_nonzero(tensor)
    total_elements = tensor.numel()
    return num_zero_elements / total_elements, total_elements, num_zero_elements


def load_gradient_mask(mask_path, name_filters):
    if not mask_path:
        return None

    grad_data = torch.load(mask_path, map_location="cpu")
    if isinstance(grad_data, dict) and "weight" in grad_data:
        grad_mask = grad_data["weight"]
    else:
        grad_mask = grad_data

    if not isinstance(grad_mask, dict):
        raise ValueError(f"Gradient mask must be a dict or contain a 'weight' dict: {mask_path}")

    filtered_mask = {}
    total_cnt = 0
    zero_cnt = 0
    name_filters = tuple(name_filters)

    for name, mask in grad_mask.items():
        if not torch.is_tensor(mask):
            continue
        if name_filters and not any(name_filter in name for name_filter in name_filters):
            continue

        filtered_mask[name] = mask.cpu()
        sparsity, total_elements, num_zero_elements = calc_sparsity(mask)
        total_cnt += total_elements
        zero_cnt += num_zero_elements

    if not filtered_mask:
        raise ValueError(f"No tensor masks matched filters {name_filters} in {mask_path}")

    print("Saliency mask loaded!")
    if total_cnt > 0:
        print(f"Total sparsity among weight: {zero_cnt / total_cnt * 100:.4f}")
    print("Masked parameter groups:")
    print("\n".join(sorted(filtered_mask.keys())))
    return filtered_mask


def get_mask_for_param(grad_mask, param_name):
    if param_name in grad_mask:
        return grad_mask[param_name]

    if param_name.startswith("module."):
        stripped_name = param_name[len("module.") :]
        if stripped_name in grad_mask:
            return grad_mask[stripped_name]

    module_name = f"module.{param_name}"
    if module_name in grad_mask:
        return grad_mask[module_name]

    return None


def apply_gradient_mask(model, grad_mask):
    if grad_mask is None:
        return 0, 0

    matched = 0
    mismatched = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        mask = get_mask_for_param(grad_mask, name)
        if mask is None:
            continue

        if tuple(mask.shape) != tuple(param.grad.shape):
            mismatched += 1
            continue

        param.grad.mul_(mask.to(device=param.grad.device, dtype=param.grad.dtype))
        matched += 1

    return matched, mismatched


def get_train_datasets(args):
    if args.task == "complete":
        multimodal_forget_dataset, multimodal_retain_dataset = get_complete_ds(
            train_ds_path=args.train_dataset_dir,
            hf_config=args.hf_config,
            ratio=args.forget_split_ratio,
            vqa=args.vqa,
        )
        split_reference = str(DEFAULT_COMPLETE_SPLIT_FILE)
    elif args.task == "selective":
        multimodal_forget_dataset, multimodal_retain_dataset = get_selective_ds(
            train_ds_path=args.train_dataset_dir,
            hf_config=args.hf_config,
            vqa=args.vqa,
        )
        split_reference = str(DEFAULT_SELECTIVE_SPLIT_FILE)
    elif args.task == "persona":
        multimodal_forget_dataset, multimodal_retain_dataset = get_persona_ds(
            train_ds_path=args.train_dataset_dir,
            hf_config=args.hf_config,
            vqa=args.vqa,
        )
        split_reference = str(DEFAULT_PERSONA_SPLIT_FILE)
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    return multimodal_forget_dataset, multimodal_retain_dataset, split_reference


def main(args):
    model, processor = load_model_and_processor(args)
    tokenizer = processor.tokenizer
    print("Tokenizer Length: ", len(tokenizer))

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    multimodal_forget_dataset, multimodal_retain_dataset, split_reference = get_train_datasets(args)

    print("Train Dataset: ", args.train_dataset_dir)
    print("Split Reference: ", split_reference)
    print("Task: ", args.task)
    if args.task == "complete":
        print("Forget Split Ratio: ", args.forget_split_ratio)
    print("VQA Mode: ", args.vqa)
    print("Forget Dataset Size: ", len(multimodal_forget_dataset))
    print("Retain Dataset Size: ", len(multimodal_retain_dataset))

    if len(multimodal_forget_dataset) == 0 or len(multimodal_retain_dataset) == 0:
        raise ValueError("Forget and retain datasets must both be non-empty.")

    if args.ans_only:
        train_collate_function = train_collate_clear_ansonly
        print("Answer only mode enabled.")
    else:
        train_collate_function = train_collate_clear
        print("Answer only mode disabled.")

    train_dataloader_forget = DataLoader(
        multimodal_forget_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_function(x, processor, None, True),
    )
    train_dataloader_retain = DataLoader(
        multimodal_retain_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_function(x, processor, None, True),
    )

    mask_name_filters = parse_csv_arg(args.mask_name_filters)
    grad_mask = load_gradient_mask(args.grad_mask_path, mask_name_filters)

    gradient_accumulation_steps = args.gradient_accumulation_steps if args.gradient_accumulation else 1
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    if args.gradient_accumulation:
        print(f"Gradient accumulation enabled. step={gradient_accumulation_steps}")
    else:
        print("Gradient accumulation disabled.")

    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr)
    num_update_steps_per_epoch = ceil(len(train_dataloader_forget) / gradient_accumulation_steps) + ceil(
        len(train_dataloader_retain) / gradient_accumulation_steps
    )
    max_train_steps = num_update_steps_per_epoch * args.num_epochs
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler
    )

    len_forget = len(train_dataloader_forget)
    len_retain = len(train_dataloader_retain)
    n_iters = len_retain
    forget_freq = max(1, len_retain // len_forget)
    accelerator.print(f"Each epoch has {n_iters} retain iterations, with forget_freq={forget_freq}.")

    for epoch in range(args.num_epochs):
        model.train()
        total_loss_forget = 0.0
        total_loss_retain = 0.0
        forget_update_count = 0
        retain_update_count = 0
        mask_matched_count = 0
        mask_mismatched_count = 0

        forget_iterator = iter(train_dataloader_forget)
        progress_bar = tqdm(
            enumerate(train_dataloader_retain),
            total=n_iters,
            desc=f"Epoch {epoch + 1} [interleaved]",
            disable=not accelerator.is_local_main_process,
        )

        for retain_step, retain_batch in progress_bar:
            latest_forget_loss = None

            if retain_step % forget_freq == 0:
                try:
                    forget_batch = next(forget_iterator)
                except StopIteration:
                    forget_batch = None

                if forget_batch is not None:
                    with accelerator.accumulate(model):
                        forget_outputs = model(**forget_batch)
                        loss_forget = -args.forget_alpha * forget_outputs.loss
                        accelerator.backward(loss_forget)

                        matched, mismatched = apply_gradient_mask(model, grad_mask)
                        mask_matched_count += matched
                        mask_mismatched_count += mismatched

                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    latest_forget_loss = loss_forget.detach().float().item()
                    total_loss_forget += latest_forget_loss
                    forget_update_count += 1

            with accelerator.accumulate(model):
                retain_outputs = model(**retain_batch)
                loss_retain = retain_outputs.loss
                accelerator.backward(loss_retain)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss_retain += loss_retain.detach().float().item()
            retain_update_count += 1

            postfix = {"retain_loss": f"{total_loss_retain / retain_update_count:.4f}"}
            if latest_forget_loss is not None:
                postfix["forget_loss"] = f"{-latest_forget_loss:.4f}"
            progress_bar.set_postfix(postfix)

        if grad_mask is not None:
            accelerator.print(
                f"Epoch {epoch + 1} gradient mask applications: matched={mask_matched_count}, "
                f"shape_mismatch={mask_mismatched_count}"
            )
            if mask_matched_count == 0:
                raise ValueError(
                    "Gradient mask was provided but did not match any trainable gradients. "
                    "Regenerate the mask with the same model/LoRA configuration used for training."
                )

        avg_forget_loss = total_loss_forget / max(1, forget_update_count)
        avg_retain_loss = total_loss_retain / max(1, retain_update_count)
        accelerator.print(
            f"Epoch {epoch + 1} Forget Loss: {avg_forget_loss:.4f} | Retain Loss: {avg_retain_loss:.4f}"
        )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            epoch_dir = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)
            unwrapped_model.save_pretrained(epoch_dir, save_function=accelerator.save)
            processor.save_pretrained(epoch_dir)
            accelerator.print(f"Checkpoint for epoch {epoch + 1} saved to: {epoch_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune different models with GA-Diff and a saliency mask")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Pretrained model directory")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument(
        "--forget_split_ratio",
        type=int,
        default=5,
        choices=[5, 10, 15, 30],
        help="Forget split ratio for complete unlearning: 5, 10, 15 or 30",
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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", action="store_true", help="Enable gradient accumulation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--ans_only", action="store_true", help="Answer only for training")
    parser.add_argument(
        "--vqa",
        action="store_true",
        help="Use image + question pairs if true, otherwise train in QA-only mode",
    )
    parser.add_argument("--task", type=str, default="complete", choices=["complete", "selective", "persona"], help="Unlearning task type")
    parser.add_argument("--grad_mask_path", type=str, default=None, help="Mask for gradient update")
    parser.add_argument("--forget_alpha", type=float, default=1.0, help="Forget loss weight")
    parser.add_argument(
        "--mask_name_filters",
        type=str,
        default=",".join(DEFAULT_MASK_NAME_FILTERS),
        help="Comma-separated substrings used to keep trainable mask parameter names",
    )

    args = parser.parse_args()

    main(args)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(f"{args.save_dir}/trainer_config.json", "wt", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

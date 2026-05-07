import os
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import torch.nn.functional as F
import torch
from welford_torch import Welford
import psutil
import gc
from pathlib import Path
import torch.nn.utils.prune as prune
import h5py
from tqdm import tqdm


def identify_layer_keywords(model, pruning_mask_keys):
    # Initialize a list to store found keywords
    found_keywords = []

    # Check vision model layers
    for layer_idx, layer in enumerate(model.vision_tower.vision_model.encoder.layers):
        # Check if the layers like 'vision_fc1_{layer_idx}' and 'vision_fc2_{layer_idx}' exist
        if hasattr(layer.mlp, 'fc1'):
            vision_fc1_key = f"vision_fc1_{layer_idx}"
            if vision_fc1_key in pruning_mask_keys:
                found_keywords.append(vision_fc1_key)

        if hasattr(layer.mlp, 'fc2'):
            vision_fc2_key = f"vision_fc2_{layer_idx}"
            if vision_fc2_key in pruning_mask_keys:
                found_keywords.append(vision_fc2_key)

    # Check language model layers
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        # Check for layers like 'lang_gate_proj_{layer_idx}', 'lang_up_proj_{layer_idx}', 'lang_down_proj_{layer_idx}'
        if hasattr(layer.mlp, 'gate_proj'):
            lang_gate_proj_key = f"lang_gate_proj_{layer_idx}"
            if lang_gate_proj_key in pruning_mask_keys:
                found_keywords.append(lang_gate_proj_key)

        if hasattr(layer.mlp, 'up_proj'):
            lang_up_proj_key = f"lang_up_proj_{layer_idx}"
            if lang_up_proj_key in pruning_mask_keys:
                found_keywords.append(lang_up_proj_key)

        if hasattr(layer.mlp, 'down_proj'):
            lang_down_proj_key = f"lang_down_proj_{layer_idx}"
            if lang_down_proj_key in pruning_mask_keys:
                found_keywords.append(lang_down_proj_key)

    return found_keywords


def count_parameters(model):
    """
    Counts the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_most_available_gpu():
    """
    Get the GPU with the most available memory.
    Returns:
        torch.device: The device with the most available memory.
    """
    available_memory = []
    for device_id in range(torch.cuda.device_count()):
        free_memory, _ = torch.cuda.mem_get_info(device_id)
        available_memory.append((device_id, free_memory))
        print(f"Device {device_id}: {free_memory / 1e9:.2f} GB available")

    # Select the GPU with the most available memory
    best_device_id = max(available_memory, key=lambda x: x[1])[0]
    print(f"Selected device: cuda:{best_device_id}")
    return torch.device(f"cuda:{best_device_id}")

def _selected_layer_indices(num_layers, all_layers=False, tail_count=3):
    if all_layers or num_layers <= tail_count:
        return set(range(num_layers))
    return set(range(num_layers - tail_count, num_layers))


def _get_attr_path(obj, path):
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _first_attr_path(obj, paths):
    for path in paths:
        value = _get_attr_path(obj, path)
        if value is not None:
            return value
    return None


def _vision_mlp_pair(mlp):
    fc1 = getattr(mlp, "fc1", None) or getattr(mlp, "linear_fc1", None)
    fc2 = getattr(mlp, "fc2", None) or getattr(mlp, "linear_fc2", None)
    return fc1, fc2


def _gemma3_vision_layers(model):
    return _first_attr_path(
        model,
        (
            "model.vision_tower.vision_model.encoder.layers",
            "vision_tower.vision_model.encoder.layers",
            "model.vision_model.encoder.layers",
            "vision_model.encoder.layers",
        ),
    )


def _gemma3_text_layers(model):
    return _first_attr_path(
        model,
        (
            "model.language_model.layers",
            "model.language_model.model.layers",
            "language_model.layers",
            "language_model.model.layers",
        ),
    )


def _neuron_scores(activations):
    activations = activations.float().clamp(min=-1e3, max=1e3)
    activations = activations.reshape(-1, activations.shape[-1])
    return {
        "I_abs": activations.abs().mean(dim=0),
        "I_freq": (activations.abs() > 1e-1).float().mean(dim=0),
        "I_var": activations.std(dim=0),
        "I_rms": torch.sqrt((activations**2).mean(dim=0)),
    }


def register_feedforward_hooks(model, collector, device, model_type, all_layers=False, score_only=False):
    """Register feed-forward hooks for supported VLM backbones."""
    model_type_lower = model_type.lower()

    if model_type == "Llava":
        vision_layers = model.vision_tower.vision_model.encoder.layers
        vision_indices = _selected_layer_indices(len(vision_layers), all_layers=all_layers)
        for layer_idx, layer in enumerate(vision_layers):
            if layer_idx not in vision_indices:
                continue
            collector.register_hook(layer.mlp.fc1, f"vision_fc1_{layer_idx}", device, modality="multimodal", score_only=score_only)
            collector.register_hook(layer.mlp.fc2, f"vision_fc2_{layer_idx}", device, modality="multimodal", score_only=score_only)

        text_layers = model.language_model.model.layers
        text_indices = _selected_layer_indices(len(text_layers), all_layers=all_layers)
        for layer_idx, layer in enumerate(text_layers):
            if layer_idx not in text_indices:
                continue
            collector.register_hook(layer.mlp.gate_proj, f"lang_gate_proj_{layer_idx}", device, modality="unimodal", score_only=score_only)
            collector.register_hook(layer.mlp.up_proj, f"lang_up_proj_{layer_idx}", device, modality="unimodal", score_only=score_only)
            collector.register_hook(layer.mlp.down_proj, f"lang_down_proj_{layer_idx}", device, modality="unimodal", score_only=score_only)

    elif model_type == "Idefics2":
        vision_layers = model.model.vision_model.encoder.layers
        vision_indices = _selected_layer_indices(len(vision_layers), all_layers=all_layers)
        for layer_idx, layer in enumerate(vision_layers):
            if layer_idx not in vision_indices:
                continue
            collector.register_hook(layer.mlp.fc1, f"vision_fc1_{layer_idx}", device, modality="multimodal", score_only=score_only)
            if hasattr(layer.mlp, "fc2"):
                collector.register_hook(layer.mlp.fc2, f"vision_fc2_{layer_idx}", device, modality="multimodal", score_only=score_only)

        text_layers = model.model.text_model.layers
        text_indices = _selected_layer_indices(len(text_layers), all_layers=all_layers)
        for layer_idx, layer in enumerate(text_layers):
            if layer_idx not in text_indices:
                continue
            collector.register_hook(layer.mlp.gate_proj, f"text_gate_proj_{layer_idx}", device, modality="unimodal", score_only=score_only)
            collector.register_hook(layer.mlp.up_proj, f"text_up_proj_{layer_idx}", device, modality="unimodal", score_only=score_only)
            collector.register_hook(layer.mlp.down_proj, f"text_down_proj_{layer_idx}", device, modality="unimodal", score_only=score_only)

    elif model_type_lower in {"gemma", "gemma3", "gemma-3"}:
        vision_layers = _gemma3_vision_layers(model)
        if vision_layers is None:
            raise AttributeError("Could not locate Gemma3 vision layers for activation hooks.")
        vision_indices = _selected_layer_indices(len(vision_layers), all_layers=all_layers)
        for layer_idx, layer in enumerate(vision_layers):
            if layer_idx not in vision_indices:
                continue
            fc1, fc2 = _vision_mlp_pair(layer.mlp)
            if fc1 is not None:
                collector.register_hook(fc1, f"gemma3_vision_fc1_{layer_idx}", device, modality="multimodal", score_only=score_only)
            if fc2 is not None:
                collector.register_hook(fc2, f"gemma3_vision_fc2_{layer_idx}", device, modality="multimodal", score_only=score_only)

        text_layers = _gemma3_text_layers(model)
        if text_layers is None:
            raise AttributeError("Could not locate Gemma3 text layers for activation hooks.")
        text_indices = _selected_layer_indices(len(text_layers), all_layers=all_layers)
        for layer_idx, layer in enumerate(text_layers):
            if layer_idx not in text_indices:
                continue
            collector.register_hook(layer.mlp.gate_proj, f"gemma3_text_gate_proj_{layer_idx}", device, modality="unimodal", score_only=score_only)
            collector.register_hook(layer.mlp.up_proj, f"gemma3_text_up_proj_{layer_idx}", device, modality="unimodal", score_only=score_only)
            collector.register_hook(layer.mlp.down_proj, f"gemma3_text_down_proj_{layer_idx}", device, modality="unimodal", score_only=score_only)

    elif model_type_lower in {"qwen3", "qwen3vl", "qwen3-vl"}:
        vision_blocks = model.model.visual.blocks
        vision_indices = _selected_layer_indices(len(vision_blocks), all_layers=all_layers)
        for layer_idx, layer in enumerate(vision_blocks):
            if layer_idx not in vision_indices:
                continue
            collector.register_hook(layer.mlp.linear_fc1, f"qwen3_vision_linear_fc1_{layer_idx}", device, modality="multimodal", score_only=score_only)
            collector.register_hook(layer.mlp.linear_fc2, f"qwen3_vision_linear_fc2_{layer_idx}", device, modality="multimodal", score_only=score_only)

        text_layers = model.model.language_model.layers
        text_indices = _selected_layer_indices(len(text_layers), all_layers=all_layers)
        for layer_idx, layer in enumerate(text_layers):
            if layer_idx not in text_indices:
                continue
            collector.register_hook(layer.mlp.gate_proj, f"qwen3_text_gate_proj_{layer_idx}", device, modality="unimodal", score_only=score_only)
            collector.register_hook(layer.mlp.up_proj, f"qwen3_text_up_proj_{layer_idx}", device, modality="unimodal", score_only=score_only)
            collector.register_hook(layer.mlp.down_proj, f"qwen3_text_down_proj_{layer_idx}", device, modality="unimodal", score_only=score_only)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def collect_feedforward_activations(
    model,
    collector,
    dataloader,
    modality,
    model_type,
    device,
    num_batches=None,
    chunk_size=10,
    best_device="cuda:1",
    progress_desc=None,
):
    """
    Collect activations and compute importance scores batch-by-batch with chunked aggregation.
    Ensures consistent device allocation during aggregation.
    """
    collector.clear_activations()
    model.eval()
    num_batches = len(dataloader) if num_batches is None else num_batches

    # Initialize aggregated metrics
    aggregated_scores = defaultdict(lambda: defaultdict(list))

    # Choose a consistent aggregation device
    # aggregation_device = get_most_available_gpu()
    # aggregation_device = torch.device("cuda:1")
    aggregation_device = torch.device(best_device)
    desc = progress_desc or f"{modality} activation batches"
    with tqdm(total=num_batches, desc=desc, unit="batch") as progress_bar:
        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            tqdm.write(f"Processing chunk {chunk_start + 1} to {chunk_end}...")

            # Initialize chunk scores
            chunk_scores = defaultdict(lambda: defaultdict(list))

            try:
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx < chunk_start or batch_idx >= chunk_end:
                        continue

                    progress_bar.set_postfix(chunk=f"{chunk_start + 1}-{chunk_end}", batch=batch_idx + 1)

                    # Prepare inputs
                    if isinstance(batch, Mapping) or hasattr(batch, "keys"):
                        inputs = {key: batch[key] for key in batch.keys()}
                    elif modality == "multimodal":
                        input_ids, attention_mask, pixel_values, labels = batch
                        inputs = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "pixel_values": pixel_values,
                            "labels": labels,
                        }
                    elif modality == "unimodal":
                        input_ids, attention_mask, labels = batch
                        inputs = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "labels": labels,
                        }
                    else:
                        raise ValueError(f"Unsupported modality: {modality}")

                    inputs = {key: value for key, value in inputs.items() if value is not None}

                    model_type_lower = model_type.lower()
                    model_dtype = next(device.unwrap_model(model).parameters()).dtype
                    inputs = {
                        key: value.to(dtype=model_dtype) if torch.is_tensor(value) and value.is_floating_point() else value
                        for key, value in inputs.items()
                    }

                    # Forward pass
                    with torch.no_grad():
                        if model_type == "Llava":
                            if modality == "multimodal":
                                device.unwrap_model(model)(**inputs)
                            elif modality == "unimodal":
                                device.unwrap_model(model).language_model(**inputs)

                        elif model_type == "Idefics2":
                            if modality == "multimodal":
                                device.unwrap_model(model)(**inputs)
                            elif modality == "unimodal":
                                device.unwrap_model(model).model.text_model(**inputs)
                        elif model_type_lower in {"gemma", "gemma3", "gemma-3", "qwen3", "qwen3vl", "qwen3-vl"}:
                            device.unwrap_model(model)(**inputs)

                    # Compute metrics for each layer
                    for layer_name in collector.list_collected_layers(modality=modality):
                        try:
                            metric_scores = collector.get_scores(layer_name, modality=modality)
                            batch_activations = None
                        except KeyError:
                            batch_activations = collector.get_activations(layer_name, modality=modality)
                            # print(f"Layer: {layer_name}, Batch {batch_idx + 1}, Activation Shape: {batch_activations.shape}")

                            # Compute per-neuron metrics for the batch. Linear hooks can
                            # emit either [tokens, hidden] or [batch, seq, hidden], so
                            # reduce every dimension except the output-neuron dimension.
                            metric_scores = _neuron_scores(batch_activations)
                        for metric_name, metric_score in metric_scores.items():
                            chunk_scores[metric_name][layer_name].append(metric_score)

                        del metric_scores, batch_activations
                        torch.cuda.empty_cache()


                    collector.clear_activations()
                    torch.cuda.empty_cache()
                    progress_bar.update(1)

                # Aggregate chunk scores into the final scores
                tqdm.write(f"Aggregating scores for chunk {chunk_start + 1} to {chunk_end}...")
                for metric, layer_dict in chunk_scores.items():
                    for layer_name, scores in layer_dict.items():
                        # Concatenate scores for this chunk
                        scores = [score.to(aggregation_device) for score in scores]
                        # chunk_mean = torch.cat(scores).mean(dim=0).to(aggregation_device)
                        chunk_mean = torch.stack(scores).mean(dim=0).to(aggregation_device)
                        aggregated_scores[metric][layer_name].append(chunk_mean)

            except RuntimeError as e:
                tqdm.write(f"Chunk aggregation failed due to OOM error for chunk {chunk_start + 1} to {chunk_end}: {e}")
                last_successful_batches = chunk_start
                tqdm.write(f"Using aggregated scores from previous {last_successful_batches} batches.")
                break  # Skip this chunk and retain previously aggregated results

            del chunk_scores
            torch.cuda.empty_cache()

    # Finalize results by computing mean across all aggregated chunks
    final_scores = defaultdict(dict)
    try:
        for metric, layer_dict in aggregated_scores.items():
            for layer_name, chunk_means in layer_dict.items():
                # Concatenate chunk means and compute the final mean
                chunk_means = [score.to(aggregation_device) for score in chunk_means]
                final_scores[metric][layer_name] = torch.stack(chunk_means).mean(dim=0).to(aggregation_device)
    except RuntimeError as e:
        print(f"Final aggregation failed: {e}")
        return aggregated_scores  # Return the last successfully aggregated chunk

    return final_scores



def compute_combined_scores_incremental(forget_scores, retain_scores, weights=None, epsilon=1e-5):
    """
    Combine importance scores from forget and retain sets using weighted metrics.
    Args:
        forget_scores: Importance scores from the forget set.
        retain_scores: Importance scores from the retain set.
        weights: Dictionary of weights for each metric. Defaults to equal weights.
        epsilon: Small value to avoid division by zero.
    Returns:
        combined_scores: Combined importance scores for each layer.
    """
    if weights is None:
        weights = {"I_abs": 2.0, "I_freq":  0.0, "I_var":  2.0, "I_rms":  2.0}
        print("weights: ", weights)

    combined_scores = {}
    for metric in forget_scores:  # Iterate over all metrics (I_abs, I_freq, etc.)
        for layer_name in forget_scores[metric]:
            if layer_name not in retain_scores[metric]:
                raise KeyError(f"Layer {layer_name} not found in retain scores for metric {metric}.")

            # Extract per-neuron importance scores
            forget_importance = forget_scores[metric][layer_name]
            retain_importance = retain_scores[metric][layer_name]

            # Ensure both tensors are on the same device
            best_device = forget_importance.device  # Use the device of forget_importance
            forget_importance = forget_importance.to(best_device)
            retain_importance = retain_importance.to(best_device)

            # Combine scores for the given metric and layer
            combined_metric_score = weights[metric] * (
                (forget_importance / (retain_importance + epsilon)) - 1
            )

            # Accumulate the combined scores for each layer
            if layer_name not in combined_scores:
                combined_scores[layer_name] = combined_metric_score
            else:
                # Ensure the existing score is on the same device as the new score
                combined_scores[layer_name] = combined_scores[layer_name].to(combined_metric_score.device)
                combined_scores[layer_name] += combined_metric_score

    return combined_scores


def compute_top_k_pruning_mask(combined_scores_dict, top_k_percent):
    """
    Compute pruning masks for multiple layers based on the top-k percent of neurons.
    Args:
        combined_scores_dict: Dictionary of combined importance scores for multiple layers.
                              {layer_name: tensor_of_scores}
        top_k_percent: Float indicating the percentage of neurons to prune (e.g., 2 for top 2%).
    Returns:
        pruning_masks: Dictionary of binary masks (1 for pruned neurons, 0 for retained neurons) for each layer.
                       {layer_name: tensor_of_mask}
    """
    # Flatten all scores across layers to compute a global threshold
    all_scores = torch.cat([scores.flatten() for scores in combined_scores_dict.values()])

    # Determine the number of neurons to prune
    num_neurons = all_scores.numel()
    k = int((top_k_percent / 100) * num_neurons)

    # Find the global threshold score for the top-k neurons
    top_k_threshold, _ = torch.topk(all_scores, k, largest=True)
    # print(top_k_percent)
    # print(num_neurons)
    # print(all_scores)
    # print(k)
    # print(top_k_threshold)
    threshold = top_k_threshold[-1]

    # Create a pruning mask for each layer based on the global threshold
    pruning_masks = {}
    for layer_name, scores in combined_scores_dict.items():
        pruning_masks[layer_name] = (scores >= threshold).float()

    return pruning_masks


def apply_structural_pruning(model, pruning_masks, model_type):
    """Zero selected output neurons while preserving the original architecture."""
    applied_masks = {}
    model_type_lower = model_type.lower()

    def apply_if_present(layer, key, linear):
        if linear is None or key not in pruning_masks:
            return
        mask = pruning_masks[key].to(linear.weight.device)
        print(f"Applying mask to layer: {key}")
        apply_mask_to_layer(linear, mask)
        applied_masks[key] = mask

    if model_type == "Llava":
        for layer_idx, layer in enumerate(model.vision_tower.vision_model.encoder.layers):
            apply_if_present(layer, f"vision_fc1_{layer_idx}", getattr(layer.mlp, "fc1", None))
            apply_if_present(layer, f"vision_fc2_{layer_idx}", getattr(layer.mlp, "fc2", None))

        for layer_idx, layer in enumerate(model.language_model.model.layers):
            apply_if_present(layer, f"lang_gate_proj_{layer_idx}", getattr(layer.mlp, "gate_proj", None))
            apply_if_present(layer, f"lang_up_proj_{layer_idx}", getattr(layer.mlp, "up_proj", None))
            apply_if_present(layer, f"lang_down_proj_{layer_idx}", getattr(layer.mlp, "down_proj", None))

    elif model_type == "Idefics2":
        for layer_idx, layer in enumerate(model.model.vision_model.encoder.layers):
            apply_if_present(layer, f"vision_fc1_{layer_idx}", getattr(layer.mlp, "fc1", None))
            apply_if_present(layer, f"vision_fc2_{layer_idx}", getattr(layer.mlp, "fc2", None))

        for layer_idx, layer in enumerate(model.model.text_model.layers):
            apply_if_present(layer, f"text_gate_proj_{layer_idx}", getattr(layer.mlp, "gate_proj", None))
            apply_if_present(layer, f"text_up_proj_{layer_idx}", getattr(layer.mlp, "up_proj", None))
            apply_if_present(layer, f"text_down_proj_{layer_idx}", getattr(layer.mlp, "down_proj", None))

    elif model_type_lower in {"gemma", "gemma3", "gemma-3"}:
        vision_layers = _gemma3_vision_layers(model)
        if vision_layers is None:
            raise AttributeError("Could not locate Gemma3 vision layers for pruning.")
        for layer_idx, layer in enumerate(vision_layers):
            fc1, fc2 = _vision_mlp_pair(layer.mlp)
            apply_if_present(layer, f"gemma3_vision_fc1_{layer_idx}", fc1)
            apply_if_present(layer, f"gemma3_vision_fc2_{layer_idx}", fc2)

        text_layers = _gemma3_text_layers(model)
        if text_layers is None:
            raise AttributeError("Could not locate Gemma3 text layers for pruning.")
        for layer_idx, layer in enumerate(text_layers):
            apply_if_present(layer, f"gemma3_text_gate_proj_{layer_idx}", getattr(layer.mlp, "gate_proj", None))
            apply_if_present(layer, f"gemma3_text_up_proj_{layer_idx}", getattr(layer.mlp, "up_proj", None))
            apply_if_present(layer, f"gemma3_text_down_proj_{layer_idx}", getattr(layer.mlp, "down_proj", None))

    elif model_type_lower in {"qwen3", "qwen3vl", "qwen3-vl"}:
        for layer_idx, layer in enumerate(model.model.visual.blocks):
            apply_if_present(layer, f"qwen3_vision_linear_fc1_{layer_idx}", getattr(layer.mlp, "linear_fc1", None))
            apply_if_present(layer, f"qwen3_vision_linear_fc2_{layer_idx}", getattr(layer.mlp, "linear_fc2", None))

        for layer_idx, layer in enumerate(model.model.language_model.layers):
            apply_if_present(layer, f"qwen3_text_gate_proj_{layer_idx}", getattr(layer.mlp, "gate_proj", None))
            apply_if_present(layer, f"qwen3_text_up_proj_{layer_idx}", getattr(layer.mlp, "up_proj", None))
            apply_if_present(layer, f"qwen3_text_down_proj_{layer_idx}", getattr(layer.mlp, "down_proj", None))

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return applied_masks


# def apply_mask_to_layer(layer, mask):
#     """
#     Helper function to apply pruning mask to a given layer's weights without changing architecture.
#
#     Args:
#         layer: The layer to mask
#         mask: The pruning mask tensor
#     """
#     if len(mask.shape) == 1:
#         # For 1D masks, expand to match weight matrix
#         expanded_mask = mask.view(-1, 1).expand_as(layer.weight.data)
#         layer.weight.data *= expanded_mask
#
#         # Apply to bias if it exists
#         if hasattr(layer, 'bias') and layer.bias is not None:
#             layer.bias.data *= mask
#     elif len(mask.shape) == 2:
#         # For 2D masks, apply directly
#         assert mask.shape == layer.weight.shape, f"Mask shape {mask.shape} doesn't match weight shape {layer.weight.shape}"
#         layer.weight.data *= mask
#
#         # Apply to bias if it exists
#         if hasattr(layer, 'bias') and layer.bias is not None:
#             # For 2D masks, we typically want to mask based on output neurons
#             bias_mask = mask.any(dim=1)  # If any input connection is kept, keep the bias
#             layer.bias.data *= bias_mask
#     else:
#         raise ValueError(f"Unsupported mask shape: {mask.shape}")

def apply_mask_to_layer(layer, mask):
    """
    Apply neuron-level pruning mask to a given layer's weights.
    Mask is 1D where each element represents whether to prune (1) or keep (0) a neuron.
    """
    if len(mask.shape) == 1:
        # Verify mask matches number of output neurons
        assert mask.shape[0] == layer.weight.shape[0], \
            f"Mask length {mask.shape[0]} doesn't match number of output neurons {layer.weight.shape[0]}"

        # Expand mask to cover all input connections for each neuron
        expanded_mask = mask.view(-1, 1).expand_as(layer.weight.data)
        # Zero out weights where mask is 1 (pruned neurons)
        layer.weight.data *= (1 - expanded_mask)  # Note the (1 - expanded_mask)

        # Also mask the bias if it exists
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.data *= (1 - mask)  # Note the (1 - mask)

def apply_pruning(model, pruning_masks):
    """
    Apply pruning to the model based on provided pruning masks.

    Args:
        model: The model to prune.
        pruning_masks: Dictionary of pruning masks for each layer.
                       {layer_name: tensor_of_mask}
    """
    for name, param in model.named_parameters():
        if name in pruning_masks:  # Match parameter name directly to pruning mask
            mask = pruning_masks[name].to(param.device)
            if "weight" in name:
                # Ensure the mask has the same shape as the parameter
                if mask.shape != param.data.shape:
                    raise ValueError(f"Shape mismatch for mask and parameter {name}: {mask.shape} vs {param.data.shape}")
                param.data *= mask
            elif "bias" in name:
                # For biases, ensure mask shape matches
                if mask.shape != param.data.shape:
                    raise ValueError(f"Shape mismatch for bias mask and parameter {name}: {mask.shape} vs {param.data.shape}")
                param.data *= mask
    print("Pruning completed!")


def log_memory(prefix=""):
    """
    Log GPU and CPU memory usage.
    Args:
        prefix: String to identify the log context.
    """
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{prefix} | GPU Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB | "
          f"GPU Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def log_cpu_memory(prefix="CPU Memory"):
    # Get total, used, and free memory in GB
    memory = psutil.virtual_memory()
    total_memory = memory.total / 1e9
    used_memory = memory.used / 1e9
    free_memory = memory.available / 1e9

    # Print CPU memory usage
    print(f"{prefix} | Total: {total_memory:.2f} GB | Used: {used_memory:.2f} GB | Free: {free_memory:.2f} GB")

    # Get CPU utilization percentage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    print(f"{prefix} | CPU Utilization: {cpu_percent:.2f}%")


def collect_feedforward_activations_single_batch(model, collector, dataloader, modality, model_type, device=None):
    """
    Collect activations for a single batch to debug OOM issues.
    Args:
        model: Multimodal model.
        collector: ActivationCollector instance.
        dataloader: PyTorch DataLoader for the specific modality.
        modality: "multimodal" or "unimodal".
        model_type: "Llava" or "Idefics2".
        device: Device to use for inference (default: None, uses model's device).
    Returns:
        collector.activations: Collected activations for the specified modality.
    """
    collector.clear_activations()  # Clear previous activations
    model.eval()  # Set the model to evaluation mode

    # Use the model's device if no specific device is provided
    device = device or next(model.parameters()).device

    # Process only the first batch
    batch = next(iter(dataloader))

    # Ensure inputs are on the correct device
    if modality == "multimodal":
        if model_type == "Llava":
            input_ids, attention_mask, pixel_values, labels = batch
            inputs = {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
                "pixel_values": pixel_values.to(device) if pixel_values is not None else None,
                "labels": labels.to(device) if labels is not None else None,
            }
    elif modality == "unimodal":
        if model_type == "Llava":
            input_ids, attention_mask, labels = batch
            inputs = {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
                "labels": labels.to(device) if labels is not None else None,
            }

    # Forward pass
    with torch.no_grad():
        if model_type == "Llava":
            if modality == "multimodal":
                model(**inputs)
            elif modality == "unimodal":
                model.language_model(**inputs)

    return collector.activations


def collect_feedforward_activations_multiple_batches(
    model, collector, dataloader, modality, model_type, device=None, num_batches=5
):
    """
    Collect activations for multiple batches with improved logic and memory management.
    Args:
        model: Multimodal model.
        collector: ActivationCollector instance.
        dataloader: PyTorch DataLoader for the specific modality.
        modality: "multimodal" or "unimodal".
        model_type: "Llava" or "Idefics2".
        device: Device to use for inference (default: None, uses model's device).
        num_batches: Number of batches to process (default: 5).
    Returns:
        collector.activations: Dictionary of collected activations for the specified modality.
    """
    collector.clear_activations()  # Clear any previous activations
    model.eval()  # Set the model to evaluation mode

    # Use the model's device if not explicitly provided
    # device = device or next(model.parameters()).device
    device = device or next(model.parameters()).device
    num_batches = len(dataloader) if num_batches is None else num_batches

    # Loop over the specified number of batches
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break  # Stop after processing the specified number of batches

        print(f"Processing batch {batch_idx + 1}/{num_batches}...")

        # Move batch data to the appropriate device
        if modality == "multimodal":
            if model_type == "Llava":
                input_ids, attention_mask, pixel_values, labels = batch
                inputs = {
                    "input_ids": input_ids.to(device),
                    "attention_mask": attention_mask.to(device),
                    "pixel_values": pixel_values.to(device) if pixel_values is not None else None,
                    "labels": labels.to(device) if labels is not None else None,
                }
        elif modality == "unimodal":
            if model_type == "Llava":
                input_ids, attention_mask, labels = batch
                inputs = {
                    "input_ids": input_ids.to(device),
                    "attention_mask": attention_mask.to(device),
                    "labels": labels.to(device) if labels is not None else None,
                }
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # Perform a forward pass to collect activations
        with torch.no_grad():
            if model_type == "Llava":
                if modality == "multimodal":
                    model(**inputs)
                elif modality == "unimodal":
                    model.language_model(**inputs)

        torch.cuda.empty_cache()  # Clear GPU cache

        # gc.collect()  # Trigger Python's garbage collector
        #
        # # Log memory usage after processing the batch
        # print(f"Batch {batch_idx + 1}/{num_batches} processed.")
        # log_memory(f"After batch {batch_idx + 1}")
        # log_cpu_memory("After batch processing")

    return collector.activations


def count_pruned_parameters(model, model_type, masks=None):
    """Count zeroed parameters in the feed-forward layers this script can prune."""
    stats_dict = {}
    total_params = 0
    total_pruned = 0
    model_type_lower = model_type.lower()

    def count_zeros_and_total(weight_tensor, mask=None):
        if mask is not None:
            if len(mask.shape) == 1:
                expanded_mask = mask.view(-1, 1).expand_as(weight_tensor)
                zeros = (expanded_mask != 0).sum().item()
            else:
                zeros = (mask != 0).sum().item()
            total = weight_tensor.numel()
        else:
            zeros = (weight_tensor == 0).sum().item()
            total = weight_tensor.numel()
        return zeros, total

    def add_layer(key, linear):
        nonlocal total_params, total_pruned
        if linear is None:
            return
        mask = masks.get(key) if masks else None
        zeros, total = count_zeros_and_total(linear.weight, mask)
        stats_dict[key] = {
            'total_params': total,
            'pruned_params': zeros,
            'pruned_percentage': (zeros / total) * 100 if total > 0 else 0
        }
        total_params += total
        total_pruned += zeros

    if model_type == "Llava":
        for layer_idx, layer in enumerate(model.vision_tower.vision_model.encoder.layers):
            add_layer(f"vision_fc1_{layer_idx}", getattr(layer.mlp, "fc1", None))
            add_layer(f"vision_fc2_{layer_idx}", getattr(layer.mlp, "fc2", None))
        for layer_idx, layer in enumerate(model.language_model.model.layers):
            add_layer(f"lang_gate_proj_{layer_idx}", getattr(layer.mlp, "gate_proj", None))
            add_layer(f"lang_up_proj_{layer_idx}", getattr(layer.mlp, "up_proj", None))
            add_layer(f"lang_down_proj_{layer_idx}", getattr(layer.mlp, "down_proj", None))

    elif model_type == "Idefics2":
        for layer_idx, layer in enumerate(model.model.vision_model.encoder.layers):
            add_layer(f"vision_fc1_{layer_idx}", getattr(layer.mlp, "fc1", None))
            add_layer(f"vision_fc2_{layer_idx}", getattr(layer.mlp, "fc2", None))
        for layer_idx, layer in enumerate(model.model.text_model.layers):
            add_layer(f"text_gate_proj_{layer_idx}", getattr(layer.mlp, "gate_proj", None))
            add_layer(f"text_up_proj_{layer_idx}", getattr(layer.mlp, "up_proj", None))
            add_layer(f"text_down_proj_{layer_idx}", getattr(layer.mlp, "down_proj", None))

    elif model_type_lower in {"gemma", "gemma3", "gemma-3"}:
        vision_layers = _gemma3_vision_layers(model)
        if vision_layers is None:
            raise AttributeError("Could not locate Gemma3 vision layers for pruning stats.")
        for layer_idx, layer in enumerate(vision_layers):
            fc1, fc2 = _vision_mlp_pair(layer.mlp)
            add_layer(f"gemma3_vision_fc1_{layer_idx}", fc1)
            add_layer(f"gemma3_vision_fc2_{layer_idx}", fc2)
        text_layers = _gemma3_text_layers(model)
        if text_layers is None:
            raise AttributeError("Could not locate Gemma3 text layers for pruning stats.")
        for layer_idx, layer in enumerate(text_layers):
            add_layer(f"gemma3_text_gate_proj_{layer_idx}", getattr(layer.mlp, "gate_proj", None))
            add_layer(f"gemma3_text_up_proj_{layer_idx}", getattr(layer.mlp, "up_proj", None))
            add_layer(f"gemma3_text_down_proj_{layer_idx}", getattr(layer.mlp, "down_proj", None))

    elif model_type_lower in {"qwen3", "qwen3vl", "qwen3-vl"}:
        for layer_idx, layer in enumerate(model.model.visual.blocks):
            add_layer(f"qwen3_vision_linear_fc1_{layer_idx}", getattr(layer.mlp, "linear_fc1", None))
            add_layer(f"qwen3_vision_linear_fc2_{layer_idx}", getattr(layer.mlp, "linear_fc2", None))
        for layer_idx, layer in enumerate(model.model.language_model.layers):
            add_layer(f"qwen3_text_gate_proj_{layer_idx}", getattr(layer.mlp, "gate_proj", None))
            add_layer(f"qwen3_text_up_proj_{layer_idx}", getattr(layer.mlp, "up_proj", None))
            add_layer(f"qwen3_text_down_proj_{layer_idx}", getattr(layer.mlp, "down_proj", None))

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    stats_dict['overall'] = {
        'total_params': total_params,
        'pruned_params': total_pruned,
        'pruned_percentage': (total_pruned / total_params) * 100 if total_params > 0 else 0
    }
    return stats_dict


def print_pruning_stats(stats_dict):
    """
    Pretty print the pruning statistics.

    Args:
        stats_dict: Dictionary of pruning statistics from count_pruned_parameters
    """
    print("\nPruning Statistics:")
    print("-" * 80)
    print(f"{'Layer':<30} {'Total Params':<15} {'Pruned Params':<15} {'Pruned %':<10}")
    print("-" * 80)

    for layer_name, stats in stats_dict.items():
        if layer_name != 'overall':
            print(
                f"{layer_name:<30} {stats['total_params']:<15} {stats['pruned_params']:<15} {stats['pruned_percentage']:.2f}%")

    print("-" * 80)
    overall = stats_dict['overall']
    print(
        f"{'Overall':<30} {overall['total_params']:<15} {overall['pruned_params']:<15} {overall['pruned_percentage']:.2f}%")


# def collect_feedforward_activations(
#     model, collector, dataloader, modality, model_type, device=None, num_batches=5
# ):
#     """
#     Collect activations and compute importance scores batch-by-batch.
#     """
#     collector.clear_activations()
#     model.eval()
#     device = device or next(model.parameters()).device
#     num_batches = len(dataloader) if num_batches is None else num_batches
#
#     # Store metrics for each batch
#     batch_metrics = defaultdict(lambda: defaultdict(list))
#
#     for batch_idx, batch in enumerate(dataloader):
#         if batch_idx >= num_batches:
#             break
#
#         print(f"Processing batch {batch_idx + 1}/{num_batches}...")
#
#         # Prepare inputs
#         if modality == "multimodal":
#             input_ids, attention_mask, pixel_values, labels = batch
#             inputs = {
#                 "input_ids": input_ids.to(device),
#                 "attention_mask": attention_mask.to(device),
#                 "pixel_values": pixel_values.to(device),
#                 "labels": labels.to(device) if labels is not None else None,
#             }
#         elif modality == "unimodal":
#             input_ids, attention_mask, labels = batch
#             inputs = {
#                 "input_ids": input_ids.to(device),
#                 "attention_mask": attention_mask.to(device),
#                 "labels": labels.to(device) if labels is not None else None,
#             }
#
#         # Forward pass
#         with torch.no_grad():
#             if model_type == "Llava":
#                 if modality == "multimodal":
#                     model(**inputs)
#                 elif modality == "unimodal":
#                     model.language_model(**inputs)
#
#         # Compute metrics for each layer
#         for layer_name in collector.list_collected_layers(modality=modality):
#             batch_activations = collector.get_activations(layer_name, modality=modality, to_cpu=True)
#             print(f"Layer: {layer_name}, Batch {batch_idx + 1}, Activation Shape: {batch_activations.shape}")
#
#             # Compute metrics for the batch
#             batch_metrics["I_abs"][layer_name].append(batch_activations.abs().mean(dim=0))
#             batch_metrics["I_freq"][layer_name].append((batch_activations.abs() > 1e-3).float().mean(dim=0))
#             batch_metrics["I_var"][layer_name].append(batch_activations.var(dim=0))
#             batch_metrics["I_rms"][layer_name].append(torch.sqrt((batch_activations**2).mean(dim=0)))
#
#         collector.clear_activations()
#         torch.cuda.empty_cache()
#
#     # Aggregate metrics across all batches
#     final_scores = {}
#     for metric, layer_dict in batch_metrics.items():
#         final_scores[metric] = {}
#         for layer_name, scores in layer_dict.items():
#             # print(f"Aggregating scores for {modality} in layer {layer_name}")
#             # print(scores)
#             final_scores[metric][layer_name] = torch.cat(scores).mean(dim=0)
#
#     for metric, layers in final_scores.items():
#         print(f"Checking {metric}")
#         for layer_name, values in layers.items():
#             print(f"Layer {layer_name}: Min {values.min()}, Max {values.max()}, Mean {values.mean()}")
#
#     return final_scores
# def compute_absolute_importance(collector, modality):
#     """
#     Compute absolute activation magnitude (I_abs) for a single batch.
#     Args:
#         collector: ActivationCollector instance with stored activations.
#         modality: "unimodal" or "multimodal".
#     Returns:
#         importance_scores: Dictionary of importance scores for each layer.
#     """
#     importance_scores = {}
#
#     # Iterate through all layers with collected activations
#     for layer_name in collector.list_collected_layers(modality=modality):
#         print(f"Computing importance for {modality} in layer {layer_name}")
#
#         # Compute mean absolute activation for the batch
#         batch_activations = torch.cat(collector.activations[modality][layer_name], dim=0).cpu()
#         importance_scores[layer_name] = batch_activations.abs().mean(dim=0)
#
#     return importance_scores

# def compute_frequency_importance(collector, modality):
#     """
#     Compute frequency-based activation (I_freq) for one batch of activations.
#     Args:
#         collector: ActivationCollector instance with stored activations.
#         modality: "unimodal" or "multimodal".
#     Returns:
#         importance_scores: Dictionary of frequency importance scores for each layer.
#     """
#     importance_scores = {}
#
#     for layer_name in collector.list_collected_layers(modality=modality):
#         print(f"Computing frequency importance for {modality} in layer {layer_name}")
#         # Retrieve activations for the batch
#         threshold = 1e-6  # Small value to filter near-zero activations
#         batch_activations = torch.cat(collector.activations[modality][layer_name], dim=0).cpu()
#         importance_scores[layer_name] = (batch_activations.abs() > threshold).float().mean(dim=0)
#         # Compute frequency of non-zero activations
#         # batch_activations = collector.activations[modality][layer_name][0].cpu()
#         # importance_scores[layer_name] = (batch_activations != 0).float().mean(dim=0)
#
#     return importance_scores
#
#
# def compute_variance_importance(collector, modality):
#     """
#     Compute variance of activations (I_var) for one batch of activations.
#     Args:
#         collector: ActivationCollector instance with stored activations.
#         modality: "unimodal" or "multimodal".
#     Returns:
#         importance_scores: Dictionary of variance importance scores for each layer.
#     """
#     importance_scores = {}
#
#     for layer_name in collector.list_collected_layers(modality=modality):
#         print(f"Computing variance importance for {modality} in layer {layer_name}")
#         # Retrieve activations for the batch
#         batch_activations = torch.cat(collector.activations[modality][layer_name], dim=0).cpu()
#         # batch_activations = collector.activations[modality][layer_name][0].cpu()
#         importance_scores[layer_name] = batch_activations.var(dim=0)
#
#     return importance_scores
#
#
# def compute_rms_importance(collector, modality):
#     """
#     Compute root mean square activation (I_rms) for one batch of activations.
#     Args:
#         collector: ActivationCollector instance with stored activations.
#         modality: "unimodal" or "multimodal".
#     Returns:
#         importance_scores: Dictionary of RMS importance scores for each layer.
#     """
#     importance_scores = {}
#
#     for layer_name in collector.list_collected_layers(modality=modality):
#         print(f"Computing RMS importance for {modality} in layer {layer_name}")
#         # Retrieve activations for the batch
#         # batch_activations = collector.activations[modality][layer_name][0].cpu()
#         batch_activations = torch.cat(collector.activations[modality][layer_name], dim=0).cpu()
#         importance_scores[layer_name] = torch.sqrt((batch_activations**2).mean(dim=0))
#
#     return importance_scores

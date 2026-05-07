import torch
from collections import defaultdict
import torch.nn.functional as F
import torch
import numpy as np
from collections import defaultdict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


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

    # Select the GPU with the most available memory
    best_device_id = max(available_memory, key=lambda x: x[1])[0]
    return torch.device(f"cuda:{best_device_id}")

class ActivationCollector:
    """
    Collect activations from feed-forward layers in multimodal models,
    with support for modality-specific storage.
    """
    def __init__(self):
        # Store activations separately for text-only and multimodal inputs
        self.activations = {
            "unimodal": defaultdict(list),
            "multimodal": defaultdict(list)
        }
        self.scores = {
            "unimodal": defaultdict(lambda: defaultdict(list)),
            "multimodal": defaultdict(lambda: defaultdict(list))
        }

    def register_hook(self, module, layer_name, device = "cuda:1", modality="multimodal", score_only=False):
        """
        Register a forward hook for a specific module to collect activations.
        Handles mismatched sequence lengths by padding and offloads activations to the GPU with the most available memory.
        """

        def hook(module, inputs, outputs):
            if score_only:
                activations = outputs.detach().float().clamp(min=-1e3, max=1e3)
                activations = activations.reshape(-1, activations.shape[-1])
                abs_activations = activations.abs()
                layer_scores = {
                    "I_abs": abs_activations.mean(dim=0),
                    "I_freq": (abs_activations > 1e-1).float().mean(dim=0),
                    "I_var": activations.std(dim=0),
                    "I_rms": torch.sqrt((activations**2).mean(dim=0)),
                }
                for metric_name, metric_score in layer_scores.items():
                    self.scores[modality][layer_name][metric_name].append(metric_score.to(device))
                del abs_activations, activations, layer_scores
                return

            # Detach activations and move them to the selected storage device
            # before padding, so CPU storage does not allocate extra CUDA memory.
            outputs = outputs.detach().to(device)

            # Determine max sequence length for padding
            if layer_name in self.activations[modality]:
                max_len = max(tensor.size(1) for tensor in self.activations[modality][layer_name])
            else:
                max_len = outputs.size(1)

            # Pad the current outputs to match max_len
            padded_outputs = F.pad(outputs, (0, 0, 0, max_len - outputs.size(1)))

            # Store the activations
            self.activations[modality][layer_name].append(padded_outputs)

        # Attach the hook to the specified module
        module.register_forward_hook(hook)

    def clear_activations(self, layer_name=None, modality=None):
        """
        Efficiently clear stored activations for a specific layer, modality, or all.

        Args:
            layer_name (str): Name of the specific layer to clear activations for. Defaults to None (clear all).
            modality (str): The modality to clear activations for ("unimodal" or "multimodal"). Defaults to None (clear all).
        """
        if modality is None:
            # Clear all modalities
            for mod in self.activations:
                for layer in self.activations[mod]:
                    # Ensure tensors are detached and references are cleared
                    self.activations[mod][layer] = []
            self.activations = {"unimodal": defaultdict(list), "multimodal": defaultdict(list)}
            self.scores = {
                "unimodal": defaultdict(lambda: defaultdict(list)),
                "multimodal": defaultdict(lambda: defaultdict(list))
            }
        elif layer_name is None:
            # Clear all layers for a specific modality
            for layer in self.activations[modality]:
                self.activations[modality][layer] = []
            self.activations[modality] = defaultdict(list)
            self.scores[modality] = defaultdict(lambda: defaultdict(list))
        else:
            # Clear a specific layer for a specific modality
            if layer_name in self.activations[modality]:
                self.activations[modality][layer_name] = []
            if layer_name in self.scores[modality]:
                self.scores[modality][layer_name] = defaultdict(list)

        # Explicitly release GPU memory
        torch.cuda.empty_cache()

    def get_activations(self, layer_name, modality="multimodal"):
        """
        Retrieve activations for a specific layer and modality, ensuring all activations are on the same GPU.
        Args:
            layer_name (str): Name of the layer.
            modality (str): "unimodal" or "multimodal".
        Returns:
            torch.Tensor: Concatenated activations for the specified layer and modality on the GPU.
        """
        if layer_name in self.activations[modality]:
            # Ensure all tensors are on the same device as the first tensor
            target_device = self.activations[modality][layer_name][0].device
            self.activations[modality][layer_name] = [
                tensor.to(target_device) for tensor in self.activations[modality][layer_name]
            ]

            # Concatenate activations
            activations = torch.cat(self.activations[modality][layer_name], dim=0)

            # Keep the activations on GPU
            return activations
        else:
            raise KeyError(f"No activations collected for layer: {layer_name} (modality: {modality})")

    def get_scores(self, layer_name, modality="multimodal"):
        """
        Retrieve per-neuron scores collected directly in hooks.
        """
        if layer_name not in self.scores[modality]:
            raise KeyError(f"No scores collected for layer: {layer_name} (modality: {modality})")

        layer_scores = {}
        for metric_name, scores in self.scores[modality][layer_name].items():
            target_device = scores[0].device
            layer_scores[metric_name] = torch.stack(
                [score.to(target_device) for score in scores]
            ).mean(dim=0)
        return layer_scores

    def list_collected_layers(self, modality="multimodal", only_nonempty=False):
        """
        List all layers for which activations have been collected.
        Args:
            modality (str): "unimodal" or "multimodal".
            only_nonempty (bool): If True, only list layers with collected activations. Defaults to False.
        Returns:
            List[str]: List of layer names with collected activations.
        """
        if only_nonempty:
            activation_layers = {
                layer_name for layer_name, activations in self.activations[modality].items()
                if len(activations) > 0
            }
            score_layers = {
                layer_name for layer_name, metrics in self.scores[modality].items()
                if any(len(scores) > 0 for scores in metrics.values())
            }
            return sorted(activation_layers | score_layers)
        return sorted(set(self.activations[modality].keys()) | set(self.scores[modality].keys()))

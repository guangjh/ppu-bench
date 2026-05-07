#!/usr/bin/env python3
"""Download all PPU-Bench configs from HuggingFace to local cache.

Usage:
  python3 download_ppubench.py                # download all 22 configs
  python3 download_ppubench.py --repo closerG/ppu-bench-test  # mini version
  python3 download_ppubench.py cloze classification  # specific configs only
"""

import argparse
import sys
from datasets import get_dataset_config_names, load_dataset

DEFAULT_REPO = "closerG/ppu-bench"
MINI_REPO = "closerG/ppu-bench-test"

ALL_CONFIGS = [
    "train_pair_qwen3",
    "train_pair_gemma3",
    "train_pair_qwen3_qa",
    "train_pair_gemma3_qa",
    "unlearning_target",
    "ppu_eval_cloze",
    "ppu_eval_classification",
    "ppu_eval_generation",
    "cross_image_complete_r5",
    "cross_image_complete_r15",
    "cross_image_complete_r30",
    "cross_image_selective",
    "cross_image_persona",
    "random_prefix_complete30",
    "random_prefix_selective",
    "random_prefix_persona",
    "jailbreak_style_prompt_complete30",
    "jailbreak_style_prompt_selective",
    "jailbreak_style_prompt_persona",
    "paraphrase_complete30",
    "paraphrase_selective",
    "paraphrase_persona",
]


def main():
    parser = argparse.ArgumentParser(description="Download PPU-Bench to local cache")
    parser.add_argument("configs", nargs="*", help="Configs to download (default: all)")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"HF repo (default: {DEFAULT_REPO})")
    parser.add_argument("--mini", action="store_true",
                        help=f"Use mini review version ({MINI_REPO})")
    args = parser.parse_args()

    repo = MINI_REPO if args.mini else args.repo

    targets = args.configs if args.configs else ALL_CONFIGS

    # Validate against actual configs on HF
    try:
        existing = set(get_dataset_config_names(repo))
    except Exception as e:
        print(f"Failed to list configs from {repo}: {e}")
        sys.exit(1)

    unknown = set(targets) - existing
    if unknown:
        print(f"Warning: configs not found on {repo}: {unknown}")

    targets = [t for t in targets if t in existing]
    if not targets:
        print("No configs to download.")
        sys.exit(0)

    print(f"Downloading {len(targets)} configs from {repo} ...\n")

    failed = []
    for i, name in enumerate(targets, 1):
        try:
            print(f"[{i:2d}/{len(targets)}] Loading {name} ...", end=" ", flush=True)
            ds = load_dataset(repo, name, trust_remote_code=True)
            split_name = list(ds.keys())[0]
            n_rows = len(ds[split_name])
            print(f"{n_rows} rows (split={split_name})")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(name)

    print()
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
    print("All configs cached successfully!")


if __name__ == "__main__":
    main()

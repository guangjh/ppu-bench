# Evaluation

We provide a unified evaluation pipeline for PPU-bench. Eval data is loaded from [HuggingFace](https://huggingface.co/datasets/closerG/ppu-bench) by default.

## Quick Start

```bash
# PPU-bench
MODEL_PATH=/path/to/your/model bash eval/eval.sh

# Testset (cross-image generalization)
MODEL_PATH=/path/to/your/model bash eval/eval_testset.sh

# Adversarial evaluation
MODEL_PATH=/path/to/your/model bash eval/eval_adv.sh
```

## Configuration

All scripts are configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | *required* | Path or HuggingFace model ID |
| `OUTPUT_DIR` | `eval/results` | Where to save predictions and metrics |
| `CUDA_DEVICE` | `0` | GPU device ID |
| `USE_HF` | `1` | `1` = load from HuggingFace, `0` = load from local parquet |

## Eval Scripts

### 1. Main Experiment (`eval.sh`)

Evaluates on the standard forget/retain benchmark across three settings:

- **complete** — Complete unlearning (requires `ratio`: 5, 15, or 30)
- **selective** — Selective unlearning
- **persona** — Personalized unlearning

Tasks: `generation`, `class` (classification), `cloze`

Edit the EVAL MATRIX section at the bottom of the script to uncomment the evaluations you need:

```bash
# ---- persona ----
run_eval "persona" "generation" "forget" "" "0" &
run_eval "persona" "generation" "retain" "" "1" &
run_eval "persona" "class" "forget" "" "2" &
run_eval "persona" "class" "retain" "" "3" &
wait
```

### 2. Testset (`eval_testset.sh`)

Evaluates cross-image generalization on alternate images. By default runs all three testsets (2, 3, 4) sequentially. Only the forget subset — the testset measures whether unlearning truly transfers to unseen images.

```bash
# Run all testsets (2, 3, 4)
MODEL_PATH=/path/to/model bash eval/eval_testset.sh

# Run a single testset only
TESTSETS="3" MODEL_PATH=/path/to/model bash eval/eval_testset.sh
```

### 3. Adversarial (`eval_adv.sh`)

Evaluates robustness against adversarial attacks. All use the `generation` task only.

| Subset | Description |
|--------|-------------|
| `random_prefix` | Random text prefix prepended to the question |
| `jailbreak_style_prompt` | Jailbreak-style prompts |
| `paraphrase` | Paraphrased questions |

```bash
# Persona setting, all three attack types
MODEL_PATH=/path/to/model bash eval/eval_adv.sh
```

Edit the EVAL MATRIX section in the script to choose specific setting/subset combinations.

## Output

Each run produces:

```
eval/results/{setting}/{task}_{subset}_r{ratio}/
├── config.json          # Run configuration
├── metrics.json         # Aggregated metrics (accuracy, rougeL, etc.)
├── predictions.jsonl    # Per-sample predictions
└── predictions_indent.json  # Formatted predictions
```

## Multi-GPU Parallelism

Assign different GPU devices to different runs by adding `&` and varying `CUDA_DEVICE`:

```bash
CUDA_DEVICE=0 run_eval "persona" "generation" "forget" "" &
CUDA_DEVICE=1 run_eval "persona" "class" "forget" "" &
wait
```


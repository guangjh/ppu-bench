# PPU-Bench: Real-World Multimodal Benchmark for Personalized Partial Unlearning

PPU-Bench is a real-world multimodal benchmark designed to evaluate personalized partial unlearning in vision-language models. It supports multiple unlearning settings and provides training/evaluation data for different VLM backbones.

---

## Requirements

We recommend using Python 3.10 with the following dependencies. 

```bash
python==3.10
torch==2.8.0
transformers==4.57.0
vllm==0.11.0
flash-attn==2.8.3
```

You can install the required packages by running the following commands:

For running baselines:
```bash
conda create --name ppubench_training python=3.10
conda activate ppubench_training
pip install -r requirements_train.txt
```

For running evaluation:
```bash
conda create --name ppubench_eval python=3.10
conda activate ppubench_eval
pip install -r requirements_eval.txt
```

---

## Dataset Download

The dataset is available on Hugging Face: [Huggingface Dataset](https://huggingface.co/datasets/closerG/ppu-bench)

You can use the following command to download the data:
```shell
python3 data/download_ppubench.py
```

---

## Training Data

We provide training data for different model backbones.

For **Qwen3-VL-8B**, use:
```shell
from datasets import load_dataset
ds = load_dataset("closerG/ppu-bench", "train_pair_qwen3")["train"]
```

For **Gemma3-12B**, use:
```shell
from datasets import load_dataset
ds = load_dataset("closerG/ppu-bench", "train_pair_gemma3")["train"]
```

You can also generate your own training data following the guidance below:
```shell
python "data/generation/eval_vqa_models.py" \
    --model [your model path] \
    --hf-config "unlearning_target"  \
    --output-dir [your output path] \
    --run-id [your run id] \
    --task [choose vqa for multimodal dataset and qa for unimodal dataset]
```

---

## Split IDs

The split files for different unlearning settings are organized as follows:

```text
Complete unlearning:      data/split/complete_split.json
Selective unlearning:     data/split/selective_split.json
Personalized unlearning:  data/split/persona_split.json
```

---

## Training Baselines

You can train baseline unlearning methods using the provided training data.

Detailed instructions are available in the [baselines](./baselines/README.md) directory.

Currently, we support the following unlearning algorithms:

- GA
- GA Difference
- KL Minimization
- NPO
- MMUnlearner
- MANU

More baselines may be added as the field develops.

---

## Evaluation

### Merge LoRA Model into Full Model

We support three unlearning settings:

```text
complete:complete unlearning
selective:selective unlearning
persona:personalized unlearning
```

Before evaluation, you can merge a LoRA checkpoint into the base model using:

```shell
python eval/merge_lora.py \
  --base_model_path [your_base_model_path] \
  --lora_path [your_lora_model_path] \
  --output_path [your_output_path] \
  --model_type [choose: gemma3 or qwen3vl] \
  --dtype bfloat16 
```

---

Eval data is loaded from [HuggingFace](https://huggingface.co/datasets/closerG/ppu-bench) by default.

Run the main experiment:

```bash
MODEL_PATH=/path/to/your/model bash eval/eval.sh
```

Edit the EVAL MATRIX section at the bottom of each script to uncomment the evaluations you need. Use `&` to parallelize across GPUs.

### Complete Unlearning

For complete unlearning, specify the forget ratio, such as `5`, `15`, or `30`.

```bash
# Complete unlearning

run_eval "complete" "generation" "forget" "forget_ratio" "device" &
run_eval "complete" "generation" "retain" "forget_ratio" "device" &

run_eval "complete" "cloze" "forget" "forget_ratio" "device" &
run_eval "complete" "cloze" "retain" "forget_ratio" "device" &

run_eval "complete" "class" "forget" "forget_ratio" "device" &
run_eval "complete" "class" "retain" "forget_ratio" "device" &
```

Example:

```bash
run_eval "complete" "generation" "forget" "30" "0" &
```

---

### Selective Unlearning

For selective unlearning, the forget/retain split is predefined by the dataset. Therefore, the forget ratio can be left empty.

```bash
# Selective unlearning

run_eval "selective" "generation" "forget" "" "device" &
run_eval "selective" "generation" "retain" "" "device" &

run_eval "selective" "cloze" "forget" "" "device" &
run_eval "selective" "cloze" "retain" "" "device" &

run_eval "selective" "class" "forget" "" "device" &
run_eval "selective" "class" "retain" "" "device" &
```

Example:

```bash
run_eval "selective" "generation" "forget" "" "0" &
```

---

### Personalized Unlearning

For personalized unlearning, the forget/retain split is also predefined by the dataset. The forget ratio can be left empty.

```bash
# Personalized unlearning

run_eval "persona" "generation" "forget" "" "device" &
run_eval "persona" "generation" "retain" "" "device" &

run_eval "persona" "cloze" "forget" "" "device" &
run_eval "persona" "cloze" "retain" "" "device" &

run_eval "persona" "class" "forget" "" "device" &
run_eval "persona" "class" "retain" "" "device" &
```

Example:

```bash
run_eval "persona" "class" "retain" "" "0" &
```

---

## For Attack test

### Cross-image Generalization Testing

For cross-image generalization testing, run `eval/eval_testset.sh`. By default evaluates all three testsets (2, 3, 4):

```bash
MODEL_PATH=/path/to/model bash eval/eval_testset.sh
```

You should also edit the EVAL MATRIX section at the bottom of each script to uncomment the evaluations you need. Use `&` to parallelize across GPUs.

---

### Adversarial Attack Evaluation

Three attack types are supported:

| Subset | Description |
|--------|-------------|
| `random_prefix` | Random text prefix injection |
| `jailbreak_style_prompt` | Jailbreak-style prompts |
| `paraphrase` | Paraphrased questions |

Use `eval/eval_adv.sh`:

```bash
MODEL_PATH=/path/to/model bash eval/eval_adv.sh
```

#### Complete Unlearning (Adv)

```bash
run_eval "complete" "random_prefix" "forget_ratio" "device" &
run_eval "complete" "jailbreak_style_prompt" "forget_ratio" "device" &
run_eval "complete" "paraphrase" "forget_ratio" "device" &
```

Example:

```bash
run_eval "complete" "random_prefix" "30" "0" &
```

#### Selective Unlearning (Adv)

```bash
run_eval "selective" "random_prefix" "" "device" &
run_eval "selective" "jailbreak_style_prompt" "" "device" &
run_eval "selective" "paraphrase" "" "device" &
```

Example:

```bash
run_eval "selective" "random_prefix" "" "0" &
```

#### Personalized Unlearning (Adv)

```bash
run_eval "persona" "random_prefix" "" "device" &
run_eval "persona" "jailbreak_style_prompt" "" "device" &
run_eval "persona" "paraphrase" "" "device" &
```

Example:

```bash
run_eval "persona" "random_prefix" "" "0" &
```

---

## BAO

To address intra-subject control of factual boundaries in Personalized Unlearning, we propose Boundary-Aware Optimization (BAO).

### Appling BAO on GA_diff
```bash
python BAGD.py \
  --model_id [your model path] \
  --vanilla_dir [your model path] \
  # --train_dataset_dir "YOUR_TRAIN_PAIR_PARQUET_PATH" \  # use it if you want to use your own data
  --hf_config [train_pair_qwen3 or train_pair_gemma3 or your own path] \ 
  --save_dir [your save directory] \
  --vqa \
  --task persona \
  --ans_only \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 2 \
  --boundary_aware \
  --boundary_lambda 1.0 \
  --boundary_margin 1.0
```

### Applying BAO to MMUnlearner

```bash
python BAMM.py \
  --model_id [your model path] \
  --vanilla_dir [your model path] \
  # --train_dataset_dir "YOUR_TRAIN_PAIR_PARQUET_PATH" \  # use it if you want to use your own data
  --hf_config [train_pair_qwen3 or train_pair_gemma3 or your own path] \ 
  --save_dir [your save directory] \
  --vqa \
  --task persona \
  --ans_only \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 4 \
  --grad_mask_path [choose your mask directory] \
  --boundary_aware \
  --boundary_lambda 1.0 \
  --boundary_margin 1.0
```

### Evaluation

Please refer to the [Evaluation](#Evaluation) section for detailed evaluation instructions.
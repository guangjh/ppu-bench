# Baselines Training

Here we provide instructions on how to train your own baselines. Firstly, you need to git pull the data from HF to your local folder.

## GA

```shell
python GA.py \
  --model_id [your model path] \
  --vanilla_dir [your model path] \
  # --train_dataset_dir "YOUR_TRAIN_PAIR_PARQUET_PATH" \  # use it if you want to use your own data
  --hf_config [train_pair_qwen3 or train_pair_gemma3 or your own path] \ 
  --save_dir [your model save path] \ 
  --vqa \
  --ans_only \
  --task [choose from: persona, selective, complete] \
  --forget_split_ratio [choose 5 or 15 or 30 if --task is complete] \
  --batch_size 2 \
  --lr 1e-6 \
  --num_epochs 2 
```

## GA Difference

```shell
python GA_Diff.py \
  --model_id [your model path] \
  --vanilla_dir [your model path] \
  # --train_dataset_dir "YOUR_TRAIN_PAIR_PARQUET_PATH" \  # use it if you want to use your own data
  --hf_config [train_pair_qwen3 or train_pair_gemma3 or your own path] \ 
  --save_dir [your model save path] \ 
  --vqa \
  --ans_only \
  --task [choose from: persona, selective, complete] \
  --forget_split_ratio [choose 5 or 15 or 30 if --task is complete] \
  --batch_size 2 \
  --lr 1e-6 \
  --num_epochs 2 
```

## KL Minimization

```shell
python KL_Min.py \
  --model_id [your model path] \
  --vanilla_dir [your model path] \
  # --train_dataset_dir "YOUR_TRAIN_PAIR_PARQUET_PATH" \  # use it if you want to use your own data
  --hf_config [train_pair_qwen3 or train_pair_gemma3 or your own path] \ 
  --save_dir [your model save path] \ 
  --vqa \
  --ans_only \
  --task [choose from: persona, selective, complete] \
  --forget_split_ratio [choose 5 or 15 or 30 if --task is complete] \
  --batch_size 2 \
  --lr 1e-6 \
  --num_epochs 2 
```

## NPO

NPO demands a reference model where we call "oracle model". In our case, we regard origin model as the reference model.

```shell
python NPO.py \
  --model_id [your model path] \
  --vanilla_dir [your model path] \
  --oracle_model_id [your model path] \
  # --train_dataset_dir "YOUR_TRAIN_PAIR_PARQUET_PATH" \  # use it if you want to use your own data
  --hf_config [train_pair_qwen3 or train_pair_gemma3 or your own path] \ 
  --save_dir [your model save path] \ 
  --vqa \
  --ans_only \
  --task [choose from: persona, selective, complete] \
  --forget_split_ratio [choose 5 or 15 or 30 if --task is complete] \
  --batch_size 2 \
  --lr 1e-6 \
  --num_epochs 2 
```

## MMUnlearner

1. MMUnlearner needs to generate the gradient mask first, run:

```shell
python MMunlearner_Mask.py \
  --model_id [your model path] \
  --vanilla_dir [your model path] \
  # --train_dataset_dir "YOUR_TRAIN_PAIR_PARQUET_PATH" \  # use it if you want to use your own data
  --hf_config [train_pair_qwen3 or train_pair_gemma3 or your own path] \ 
  --mask_save_dir [your mask saving directory] \
  --task [choose from: persona, selective, complete] \
  --vqa \
  --batch_size 2
```

2. Then you can run the training process:

```shell
python MMunlearner.py \
  --model_id [your model path] \
  --vanilla_dir [your model path] \
  # --train_dataset_dir "YOUR_TRAIN_PAIR_PARQUET_PATH" \  # use it if you want to use your own data
  --hf_config [train_pair_qwen3 or train_pair_gemma3 or your own path] \ 
  --save_dir [your model save path] \ 
  --grad_mask_path [your mask saving directory] \
  --task [choose from: persona, selective, complete] \
  --vqa \
  --ans_only \
  --forget_split_ratio [choose 5 or 15 or 30 if --task is complete] \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 4
```

## MANU

```shell
python MANU/prune_incremental.py \
  --model_save_name [your model save name] \
  --model_id [your model path] \
  --vanilla_dir [your model path] \
  --multimodal_hf_config [train_pair_qwen3 or train_pair_gemma3 or your own path] \
  --unimodal_hf_config [train_pair_qwen3_qa or train_pair_gemma3_qa or your own path] \
  # --multimodal_train_dataset [your data path] \ ## # use it if you want to use your own data
  # --unimodal_train_dataset [your data path] \ ## # use it if you want to use your own data
  --task [choose from: persona, selective, complete] \
  --batch_size 4 \
  --max_length 384 \
  --forget_ratio [choose 5 or 15 or 30 if --task is complete] \
  --num_iterations 1 \
  --prune_percent 50 \
  --model_type "auto" \
  --activation_device "auto" \
  --save_path [your model save directory]"
```

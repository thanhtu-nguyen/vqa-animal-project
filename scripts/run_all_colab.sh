#!/usr/bin/env bash
set -e

ROOT=/content/drive/MyDrive/vqa_dataset

pip install -r requirements.txt

# Chuẩn bị dữ liệu
python -m src.prepare_data \
  --train_csv $ROOT/train.csv \
  --test_csv  $ROOT/test.csv \
  --image_dir $ROOT \
  --out_dir   $ROOT/outputs/processed

# Hướng A — LSTM decoder
python -m src.train_a --config configs/default.yaml --decoder lstm
python -m src.evaluate_a \
  --config configs/default.yaml \
  --ckpt   $ROOT/outputs/A_lstm/best.pt

# Hướng A — Transformer decoder
python -m src.train_a --config configs/default.yaml --decoder transformer
python -m src.evaluate_a \
  --config configs/default.yaml \
  --ckpt   $ROOT/outputs/A_transformer/best.pt

# Hướng B1 — PaliGemma zero-shot
python -m src.evaluate_paligemma \
  --config configs/default.yaml \
  --name   B1_paligemma_zeroshot

# Hướng B2 — PaliGemma SFT (LoRA)
python -m src.train_paligemma_sft --config configs/default.yaml
python -m src.evaluate_paligemma \
  --config  configs/default.yaml \
  --adapter $ROOT/outputs/B2_paligemma_lora_sft \
  --name    B2_paligemma_lora_sft

# Hướng B2 — PaliGemma DPO (RL)
python -m src.train_paligemma_dpo \
  --config      configs/default.yaml \
  --sft_adapter $ROOT/outputs/B2_paligemma_lora_sft
python -m src.evaluate_paligemma \
  --config  configs/default.yaml \
  --adapter $ROOT/outputs/B2_paligemma_lora_dpo \
  --name    B2_paligemma_lora_dpo
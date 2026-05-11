#!/usr/bin/env bash
# run_all_colab.sh — Chạy toàn bộ pipeline VQA (Hướng A + B) trên Colab/Linux
# Cách dùng: bash run_all_colab.sh
set -e

ROOT=/content/drive/MyDrive/vqa_dataset

pip install -r requirements.txt

# ────────────────────────────────────────────────────────────────
# Bước 0: Chuẩn bị dữ liệu
# ────────────────────────────────────────────────────────────────
python -m src.prepare_data \
  --train_csv $ROOT/train.csv \
  --test_csv  $ROOT/test.csv \
  --image_dir $ROOT \
  --out_dir   $ROOT/outputs/processed

# ────────────────────────────────────────────────────────────────
# Hướng A1 — EfficientNet-B2 + PhoBERT + LSTM decoder
# ────────────────────────────────────────────────────────────────
python -m src.train_a --config configs/default.yaml --decoder lstm
python -m src.evaluate_a \
  --config configs/default.yaml \
  --ckpt   $ROOT/outputs/A_lstm/best.pt

# ────────────────────────────────────────────────────────────────
# Hướng A2 — EfficientNet-B2 + PhoBERT + Transformer decoder
# ────────────────────────────────────────────────────────────────
python -m src.train_a --config configs/default.yaml --decoder transformer
python -m src.evaluate_a \
  --config configs/default.yaml \
  --ckpt   $ROOT/outputs/A_transformer/best.pt

# ────────────────────────────────────────────────────────────────
# Hướng B1 — Qwen2.5-VL zero-shot (không cần train)
# ────────────────────────────────────────────────────────────────
python -m src.evaluate_qwen \
  --config configs/default.yaml \
  --name   B1_qwen_zeroshot

# ────────────────────────────────────────────────────────────────
# Hướng B2 — Qwen2.5-VL SFT với QLoRA (4-bit)
# ────────────────────────────────────────────────────────────────
python -m src.train_qwen_sft --config configs/default.yaml
python -m src.evaluate_qwen \
  --config  configs/default.yaml \
  --adapter $ROOT/outputs/B2_qwen_lora_sft \
  --name    B2_qwen_lora_sft

# ────────────────────────────────────────────────────────────────
# Hướng B2 DPO — Qwen2.5-VL DPO sau SFT (cần preference.csv)
# ────────────────────────────────────────────────────────────────
if [ -f "$ROOT/preference.csv" ]; then
  python -m src.train_qwen_dpo \
    --config      configs/default.yaml \
    --sft_adapter $ROOT/outputs/B2_qwen_lora_sft
  python -m src.evaluate_qwen \
    --config  configs/default.yaml \
    --adapter $ROOT/outputs/B2_qwen_lora_dpo \
    --name    B2_qwen_lora_dpo
else
  echo "[SKIP] preference.csv không tồn tại — bỏ qua bước DPO"
fi
"""
train_qwen_sft.py
SFT Qwen2.5-VL với QLoRA (4-bit) — Supervised Fine-Tuning stage.

Cách dùng:
    python -m src.train_qwen_sft --config configs/default.yaml

So với PaliGemma:
  - Dùng AutoProcessor.apply_chat_template thay vì truyền text + images riêng biệt
  - Processor tự sinh ra attention_mask, pixel_values, image_grid_thw
  - Label mask: chỉ học phần answer, prompt và pad đều bị mask = -100
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from collections import deque

from .utils import load_config, seed_everything, ensure_dir, save_json
from .qwen_common import (
    QwenSFTDataset,
    make_messages,
    load_qwen_lora,
)


# ─────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────

class QwenSFTCollator:
    """
    Collate một batch:
      1. Tạo messages với chat-template (prompt + answer)
      2. Tokenize qua processor — sinh pixel_values tự động
      3. Tạo labels: mask prompt và pad, chỉ học answer
    """

    def __init__(self, processor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        texts_full = []    # prompt + answer (để tokenize labels)
        texts_prompt = []  # chỉ prompt (để tính offset mask)
        images = []

        for ex in batch:
            image = Image.open(ex["image_path"]).convert("RGB")
            images.append(image)

            # Tạo messages chứa ảnh + câu hỏi
            msgs = make_messages(ex["question"])

            # apply_chat_template với add_generation_prompt=False sẽ trả về text đầy đủ
            # Qwen tokenizer cần truyền images riêng khi gọi processor
            text_prompt = self.processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,  # Thêm <|im_start|>assistant\n
            )
            texts_prompt.append(text_prompt)
            texts_full.append(text_prompt + ex["answer"] + "<|im_end|>\n")

        # ── Tokenize full (prompt + answer) ──────────────────────────────
        inputs_full = self.processor(
            text=texts_full,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # ── Tokenize chỉ prompt (để lấy độ dài mask) ────────────────────
        # Dùng padding=False để lấy độ dài thực của từng prompt
        inputs_prompt = self.processor(
            text=texts_prompt,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = inputs_full["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id

        # Mask pad tokens
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Mask phần prompt — chỉ học phần answer
        for i in range(len(batch)):
            if pad_id is not None:
                prompt_len = (inputs_prompt["input_ids"][i] != pad_id).sum().item()
            else:
                prompt_len = inputs_prompt["input_ids"][i].shape[0]
            labels[i, :prompt_len] = -100

        inputs_full["labels"] = labels
        return inputs_full


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train_csv", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    out_dir = ensure_dir(os.path.join(cfg["output_dir"], "B2_qwen_lora_sft"))
    train_csv = args.train_csv or os.path.join(
        cfg["output_dir"], "processed", "train_prepared.csv"
    )

    # ── Tải model + LoRA ──────────────────────────────────────────────────
    model, processor = load_qwen_lora(
        cfg["qwen_model_name"],
        load_in_4bit=True,
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
    )

    # ── Dataset + DataLoader ─────────────────────────────────────────────
    ds = QwenSFTDataset(train_csv, cfg["image_dir"])
    loader = DataLoader(
        ds,
        batch_size=cfg["qwen_batch_size"],
        shuffle=True,
        collate_fn=QwenSFTCollator(processor, cfg["qwen_max_length"]),
        num_workers=0,  # 0 để tránh lỗi multiprocessing trên Windows
        pin_memory=False,
    )

    # ── Optimizer ────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["qwen_lr"]),
        weight_decay=0.01,
    )

    # Cosine LR scheduler
    total_steps = len(loader) * int(cfg["qwen_epochs"]) // int(cfg["qwen_grad_accum"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))

    grad_accum = int(cfg["qwen_grad_accum"])

    # ── Training loop ────────────────────────────────────────────────────
    model.train()
    step = 0
    losses = []
    running_loss = deque(maxlen=20)  # trung bình 20 bước gần nhất

    for ep in range(1, int(cfg["qwen_epochs"]) + 1):
        epoch_loss = 0.0
        pbar = tqdm(
            loader,
            desc=f"Epoch {ep}/{cfg['qwen_epochs']}",
            unit="batch",
            dynamic_ncols=True,       # tự co giãn theo terminal
        )

        for batch_idx, batch in enumerate(pbar):
            batch = {
                k: v.to(model.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()

            raw_loss = float(loss.item() * grad_accum)
            running_loss.append(raw_loss)
            epoch_loss += raw_loss

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)

            losses.append(raw_loss)
            step += 1

            # Cập nhật thanh tiến trình
            pbar.set_postfix(
                loss=f"{raw_loss:.4f}",
                avg20=f"{sum(running_loss)/len(running_loss):.4f}",  # avg 20 bước
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                step=step,
            )

        avg_epoch_loss = epoch_loss / len(loader)
        print(f"\n✓ Epoch {ep} hoàn tất — avg loss: {avg_epoch_loss:.4f}")

        ep_dir = os.path.join(out_dir, f"epoch_{ep}")
        model.save_pretrained(ep_dir)
        processor.save_pretrained(ep_dir)
        print(f"✓ Saved checkpoint: {ep_dir}\n")

    # Lưu adapter cuối cùng
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    save_json({"losses": losses}, os.path.join(out_dir, "train_history.json"))
    print(f"\n[✓] Saved LoRA adapter: {out_dir}")
    print(f"    Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
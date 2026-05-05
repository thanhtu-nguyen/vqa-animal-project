"""
train_paligemma_sft.py
Thay thế train_qwen_sft.py — SFT PaliGemma với LoRA.

PaliGemma processor nhận (text, images) riêng biệt — không dùng chat-template như Qwen.
Label: toàn bộ sequence; pad token được mask thành -100.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from .utils import load_config, seed_everything, ensure_dir, save_json
from .paligemma_common import (
    PaliGemmaSFTDataset,
    make_prompt,
    load_paligemma_lora,
)


class PaliGemmaSFTCollator:
    """Collate batch: tokenise text + process images cùng lúc."""

    def __init__(self, processor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        texts = []
        images = []
        for ex in batch:
            texts.append(make_prompt(ex["question"]) + ex["answer"])
            images.append(Image.open(ex["image_path"]).convert("RGB"))

        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        inputs["labels"] = labels
        return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train_csv", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    out_dir = ensure_dir(os.path.join(cfg["output_dir"], "B2_paligemma_lora_sft"))
    train_csv = args.train_csv or os.path.join(
        cfg["output_dir"], "processed", "train_prepared.csv"
    )

    model, processor = load_paligemma_lora(
        cfg["paligemma_model_name"],
        load_in_4bit=True,
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
    )

    ds = PaliGemmaSFTDataset(train_csv, cfg["image_dir"])
    loader = DataLoader(
        ds,
        batch_size=cfg["qwen_batch_size"],   # tái sử dụng key batch_size từ config
        shuffle=True,
        collate_fn=PaliGemmaSFTCollator(processor, cfg["qwen_max_length"]),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["qwen_lr"]))
    grad_accum = int(cfg["qwen_grad_accum"])

    model.train()
    step = 0
    losses = []

    for ep in range(1, int(cfg["qwen_epochs"]) + 1):
        pbar = tqdm(loader, desc=f"paligemma-sft epoch {ep}")
        for batch in pbar:
            batch = {k: v.to(model.device) for k, v in batch.items() if torch.is_tensor(v)}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            losses.append(float(loss.item() * grad_accum))
            step += 1
            pbar.set_postfix(loss=losses[-1])

        # Lưu checkpoint mỗi epoch
        ep_dir = os.path.join(out_dir, f"epoch_{ep}")
        model.save_pretrained(ep_dir)
        processor.save_pretrained(ep_dir)

    # Lưu adapter cuối cùng
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    save_json({"losses": losses}, os.path.join(out_dir, "train_history.json"))
    print("Saved LoRA adapter:", out_dir)


if __name__ == "__main__":
    main()
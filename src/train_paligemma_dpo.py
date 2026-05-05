"""
train_paligemma_dpo.py
Thay thế train_qwen_dpo.py — DPO stage cho B2 sau SFT với PaliGemma.

Preference CSV: image, question, chosen, rejected.
TRL DPOTrainer hỗ trợ multimodal qua trường "images" trong dataset.
"""

import os
import argparse
import torch
from datasets import Dataset as HFDataset
from trl import DPOTrainer, DPOConfig

from .utils import load_config, seed_everything, ensure_dir
from .paligemma_common import (
    PaliGemmaPreferenceDataset,
    make_prompt,
    load_paligemma_lora,
)


def to_hf_dataset(pref_ds: PaliGemmaPreferenceDataset) -> HFDataset:
    """Chuyển PaliGemmaPreferenceDataset sang HuggingFace Dataset cho DPOTrainer."""
    rows = []
    for ex in pref_ds:
        rows.append(
            {
                "images": [ex["image"]],          # TRL multimodal DPO dùng key "images"
                "prompt": ex["prompt"],            # đã qua make_prompt()
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
            }
        )
    return HFDataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sft_adapter", default=None, help="Đường dẫn LoRA SFT adapter.")
    ap.add_argument("--preference_csv", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    out_dir = ensure_dir(os.path.join(cfg["output_dir"], "B2_paligemma_lora_dpo"))
    pref_csv = args.preference_csv or cfg["preference_csv"]

    model, processor = load_paligemma_lora(
        cfg["paligemma_model_name"],
        load_in_4bit=True,
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
    )

    # Nếu có SFT adapter, nạp vào trước khi DPO
    if args.sft_adapter and os.path.isdir(args.sft_adapter):
        model.load_adapter(args.sft_adapter, adapter_name="default")
        print(f"[INFO] Loaded SFT adapter from: {args.sft_adapter}")

    pref_ds = PaliGemmaPreferenceDataset(pref_csv, cfg["image_dir"])
    hf_ds = to_hf_dataset(pref_ds)

    training_args = DPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(cfg["qwen_grad_accum"]),
        num_train_epochs=1,
        learning_rate=5e-6,
        bf16=True,
        logging_steps=5,
        save_steps=50,
        remove_unused_columns=False,
        beta=0.1,
        max_length=int(cfg["qwen_max_length"]),
        max_prompt_length=256,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # implicit reference (peft)
        args=training_args,
        train_dataset=hf_ds,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    print("Saved DPO adapter:", out_dir)


if __name__ == "__main__":
    main()
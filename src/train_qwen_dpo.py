"""
train_qwen_dpo.py
DPO (Direct Preference Optimization) cho Qwen2.5-VL sau SFT.

Preference CSV cần có các cột: image, question, chosen, rejected
TRL DPOTrainer hỗ trợ multimodal qua key "images" trong HF Dataset.

Cách dùng:
    python -m src.train_qwen_dpo --config configs/default.yaml --sft_adapter outputs/B2_qwen_lora_sft
"""

import os
import argparse
import torch
from datasets import Dataset as HFDataset
from trl import DPOTrainer, DPOConfig

from .utils import load_config, seed_everything, ensure_dir
from .qwen_common import (
    QwenPreferenceDataset,
    load_qwen_lora,
)

from transformers import TrainerCallback

class VerboseProgressCallback(TrainerCallback):
    """In tiến trình chi tiết hơn mỗi logging_step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step   = state.global_step
        total  = state.max_steps
        pct    = 100 * step / total if total else 0

        parts = [f"Step {step}/{total} ({pct:.1f}%)"]
        for key in ["loss", "rewards/chosen", "rewards/rejected", "rewards/margins", "logits/chosen"]:
            if key in logs:
                parts.append(f"{key.split('/')[-1]}={logs[key]:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        print("  ".join(parts))
# ─────────────────────────────────────────────
# Convert dataset
# ─────────────────────────────────────────────

def to_hf_dataset(pref_ds: QwenPreferenceDataset) -> HFDataset:
    """Chuyển QwenPreferenceDataset sang HuggingFace Dataset cho DPOTrainer."""
    rows = []
    for ex in pref_ds:
        rows.append(
            {
                "images": [ex["image"]],   # TRL multimodal DPO dùng key "images"
                "prompt": ex["prompt"],
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
            }
        )
    return HFDataset.from_list(rows)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sft_adapter", default=None, help="Đường dẫn LoRA SFT adapter.")
    ap.add_argument("--preference_csv", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    out_dir = ensure_dir(os.path.join(cfg["output_dir"], "B2_qwen_lora_dpo"))
    pref_csv = args.preference_csv or cfg["preference_csv"]

    # ── Tải model ─────────────────────────────────────────────────────────
    model, processor = load_qwen_lora(
        cfg["qwen_model_name"],
        load_in_4bit=True,
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
    )

    # Nạp SFT adapter (nếu có) trước khi DPO
    if args.sft_adapter and os.path.isdir(args.sft_adapter):
        model.load_adapter(args.sft_adapter, adapter_name="default")
        print(f"[✓] Loaded SFT adapter from: {args.sft_adapter}")

    # ── Dataset ───────────────────────────────────────────────────────────
    pref_ds = QwenPreferenceDataset(pref_csv, cfg["image_dir"])
    hf_ds = to_hf_dataset(pref_ds)
    print(f"[INFO] Preference dataset: {len(hf_ds)} examples")

    # ── DPO Config ────────────────────────────────────────────────────────
    training_args = DPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(cfg["qwen_grad_accum"]),
        num_train_epochs=1,
        learning_rate=5e-6,
        bf16=True,
        logging_steps=2,
        logging_first_step=True,
        save_steps=50,
        remove_unused_columns=False,
        beta=0.1,                           # KL penalty coefficient
        max_length=int(cfg["qwen_max_length"]),
        max_prompt_length=int(cfg["qwen_max_length"]) // 2,
        loss_type="sigmoid",                # DPO loss gốc
        report_to="none",
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        ref_model=None,       # implicit reference khi dùng PEFT
        args=training_args,
        train_dataset=hf_ds,
        processing_class=processor,
        callbacks=[VerboseProgressCallback()], 
    )
    trainer.train()
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    print(f"\n[✓] Saved DPO adapter: {out_dir}")


if __name__ == "__main__":
    main()
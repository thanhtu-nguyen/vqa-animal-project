"""
paligemma_common.py
Thay thế qwen_common.py — dùng google/paligemma2-3b-pt-224 (hoặc bất kỳ variant PaliGemma nào)
thay vì Qwen2.5-VL.

PaliGemma dùng PaliGemmaProcessor (vision + text cùng một processor).
Prompt format: "<image>\n{question}"  (không có system-role như Qwen).
"""

import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from .path_utils import resolve_image_path

# PaliGemma không có system-prompt role; ta nhúng chỉ dẫn vào prefix của user prompt.
PROMPT_PREFIX = "Trả lời ngắn gọn bằng tiếng Việt, tối đa 10 từ: "


def make_prompt(question: str) -> str:
    """Tạo text prompt cho PaliGemma. Ảnh được truyền riêng qua processor."""
    return PROMPT_PREFIX + str(question)


class PaliGemmaSFTDataset(Dataset):
    """Dataset cho SFT — trả về image_path, question, answer."""

    def __init__(self, csv_path: str, image_dir: str):
        self.df = (
            pd.read_csv(csv_path)
            .dropna(subset=["image", "question", "answer"])
            .reset_index(drop=True)
        )
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        image_path = resolve_image_path(
            self.image_dir,
            str(r.image),
            r.get("split", None),
            r.get("image_folder", None),
        )
        return {
            "image_path": image_path,
            "question": str(r.question),
            "answer": str(r.answer),
        }


class PaliGemmaPreferenceDataset(Dataset):
    """Dataset cho DPO — trả về image (PIL), prompt, chosen, rejected."""

    def __init__(self, csv_path: str, image_dir: str):
        self.df = (
            pd.read_csv(csv_path)
            .dropna(subset=["image", "question", "chosen", "rejected"])
            .reset_index(drop=True)
        )
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        image_path = resolve_image_path(
            self.image_dir,
            str(r.image),
            r.get("split", None),
            r.get("image_folder", None),
        )
        return {
            "image": Image.open(image_path).convert("RGB"),
            "prompt": make_prompt(str(r.question)),
            "chosen": str(r.chosen),
            "rejected": str(r.rejected),
            "image_path": image_path,
        }


def load_paligemma_lora(
    model_name: str,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Load PaliGemma với LoRA (tuỳ chọn 4-bit quantization)."""
    bnb = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        if load_in_4bit
        else None
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = PaliGemmaProcessor.from_pretrained(model_name)

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # PaliGemma target modules (Gemma-2 language backbone + SigLIP vision)
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, processor


def load_paligemma_for_infer(base_model: str, adapter_dir: str = None):
    """Load PaliGemma (+ LoRA adapter tuỳ chọn) ở chế độ inference."""
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    processor = PaliGemmaProcessor.from_pretrained(base_model)
    model.eval()
    return model, processor
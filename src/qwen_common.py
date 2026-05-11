"""
qwen_common.py
Thay thế paligemma_common.py — các hàm dùng chung cho Qwen2.5-VL.

Qwen2.5-VL dùng chat-template với <|vision_start|>...<|vision_end|> tokens
và xử lý ảnh qua processor.apply_chat_template.
"""

import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


# ─────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────

def make_messages(question: str) -> list[dict]:
    """
    Trả về list messages theo định dạng Qwen2.5-VL chat-template.
    <image> placeholder sẽ được processor tự thay bằng vision tokens.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "<placeholder>"},  # sẽ được thay bằng PIL image
                {
                    "type": "text",
                    "text": (
                        "Trả lời ngắn gọn bằng tiếng Việt, tối đa 10 từ: "
                        + question
                    ),
                },
            ],
        }
    ]


def make_prompt_text(question: str) -> str:
    """Chỉ trả về phần text prompt (không có vision) để tính độ dài mask."""
    return f"Trả lời ngắn gọn bằng tiếng Việt, tối đa 10 từ: {question}"


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────

def _resolve_image_path(image_dir: str, image_name: str, split=None, image_folder=None) -> str:
    """
    Thứ tự tìm kiếm:
    1. image_path tuyệt đối (nếu image_name là path đầy đủ và tồn tại)
    2. image_dir / split / image_name
    3. image_dir / image_folder / image_name
    4. image_dir / image_name
    """
    if os.path.isabs(image_name) and os.path.exists(image_name):
        return image_name

    candidates = []
    if split:
        candidates.append(os.path.join(image_dir, split, image_name))
    if image_folder:
        candidates.append(os.path.join(image_dir, image_folder, image_name))
    candidates.append(os.path.join(image_dir, image_name))

    for c in candidates:
        if os.path.exists(c):
            return c

    # fallback — trả về candidate đầu tiên dù không tồn tại
    return candidates[0]


class QwenSFTDataset(Dataset):
    """Dataset cho SFT: trả về image_path, question, answer."""

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
        image_path = _resolve_image_path(
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


class QwenPreferenceDataset(Dataset):
    """Dataset cho DPO: trả về image (PIL), prompt, chosen, rejected."""

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
        image_path = _resolve_image_path(
            self.image_dir,
            str(r.image),
            r.get("split", None),
            r.get("image_folder", None),
        )
        return {
            "image": Image.open(image_path).convert("RGB"),
            "prompt": make_prompt_text(str(r.question)),
            "chosen": str(r.chosen),
            "rejected": str(r.rejected),
            "image_path": image_path,
        }


# ─────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────

# Các module Qwen2.5-VL thường dùng cho LoRA
QWEN_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_qwen_lora(
    model_name: str,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """
    Tải Qwen2.5-VL với QLoRA (4-bit) + LoRA adapter sẵn sàng để train.

    Returns:
        model  — PEFT model với LoRA injected
        processor — AutoProcessor (xử lý cả text + ảnh)
    """
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,   # Double quantization tiết kiệm thêm ~0.4 bit/param
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=QWEN_LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, processor


def load_qwen_for_infer(base_model: str, adapter_dir: str = None):
    """
    Tải Qwen2.5-VL để inference (không cần QLoRA).
    Nếu có adapter_dir, nạp LoRA adapter lên trên base model.
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if adapter_dir and os.path.isdir(adapter_dir):
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()   # Merge LoRA vào weights để inference nhanh hơn

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model.eval()
    return model, processor
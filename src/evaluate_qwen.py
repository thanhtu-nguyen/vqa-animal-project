"""
evaluate_qwen.py
Đánh giá B1 (zero-shot) và B2 (fine-tuned) với Qwen2.5-VL.

Cách dùng:
    # B1 zero-shot:
    python -m src.evaluate_qwen --config configs/default.yaml --name B1_qwen_zeroshot

    # B2 fine-tuned:
    python -m src.evaluate_qwen --config configs/default.yaml \\
        --adapter outputs/B2_qwen_lora_sft --name B2_qwen_lora_sft
"""

import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

from .utils import load_config, save_json
from .metrics import compute_text_metrics
from .qwen_common import make_messages, load_qwen_for_infer


# ─────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────

def _resolve_image_path(image_dir, image_name, split=None, image_folder=None):
    if os.path.isabs(str(image_name)) and os.path.exists(str(image_name)):
        return str(image_name)
    candidates = []
    if split:
        candidates.append(os.path.join(image_dir, split, image_name))
    if image_folder:
        candidates.append(os.path.join(image_dir, image_folder, image_name))
    candidates.append(os.path.join(image_dir, image_name))
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


@torch.no_grad()
def qwen_answer(model, processor, image_path: str, question: str, max_new_tokens: int = 64) -> str:
    """
    Chạy inference một câu hỏi + ảnh với Qwen2.5-VL.

    Returns:
        answer (str) — câu trả lời tiếng Việt
    """
    image = Image.open(image_path).convert("RGB")
    msgs = make_messages(question)

    # apply_chat_template tạo text prompt chuẩn Qwen chat format
    text_prompt = processor.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        repetition_penalty=1.1,
    )

    # Decode chỉ phần mới sinh ra (bỏ phần input)
    new_tokens = output_ids[0][input_len:]
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()

    return answer if answer else "không có câu trả lời"


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--csv", default=None)
    ap.add_argument(
        "--adapter",
        default=None,
        help="Đường dẫn LoRA adapter (B2). Để trống cho zero-shot (B1).",
    )
    ap.add_argument("--name", default="B1_qwen_zeroshot")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    csv_path = args.csv or os.path.join(cfg["output_dir"], "processed", "test_prepared.csv")

    print(f"[INFO] Model  : {cfg['qwen_model_name']}")
    print(f"[INFO] Adapter: {args.adapter or 'None (zero-shot)'}")
    print(f"[INFO] CSV    : {csv_path}")

    # ── Tải model ─────────────────────────────────────────────────────────
    model, processor = load_qwen_for_infer(cfg["qwen_model_name"], args.adapter)

    # ── Load test data ────────────────────────────────────────────────────
    df = (
        pd.read_csv(csv_path)
        .dropna(subset=["image", "question", "answer"])
        .reset_index(drop=True)
    )
    print(f"[INFO] Test samples: {len(df)}")

    # ── Inference ─────────────────────────────────────────────────────────
    preds = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"qwen-eval [{args.name}]"):
        img_path = _resolve_image_path(
            cfg["image_dir"],
            str(r.image),
            r.get("split", None),
            r.get("image_folder", None),
        )

        try:
            answer = qwen_answer(
                model,
                processor,
                img_path,
                str(r.question),
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            print(f"\n[!] Lỗi tại ảnh {img_path}: {e}")
            answer = "lỗi"

        preds.append(answer)

    # ── Metrics ───────────────────────────────────────────────────────────
    refs = df["answer"].astype(str).tolist()
    metrics = compute_text_metrics(preds, refs)

    # ── Lưu kết quả ──────────────────────────────────────────────────────
    out_dir = os.path.join(cfg["output_dir"], args.name)
    os.makedirs(out_dir, exist_ok=True)

    save_json(metrics, os.path.join(out_dir, "test_metrics.json"))
    df["prediction"] = preds
    df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    print("\n─── KẾT QUẢ ĐÁNH GIÁ ──────────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print(f"\n[✓] Saved to: {out_dir}")


if __name__ == "__main__":
    main()
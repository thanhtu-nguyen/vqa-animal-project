"""
evaluate_paligemma.py
Thay thế evaluate_qwen.py — đánh giá B1 (zero-shot) và B2 (fine-tuned) với PaliGemma.
"""

import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

from .utils import load_config, save_json
from .metrics import compute_text_metrics
from .paligemma_common import make_prompt, load_paligemma_for_infer
from .path_utils import resolve_image_path


@torch.no_grad()
def paligemma_answer(
    model,
    processor,
    image_path: str,
    question: str,
    max_new_tokens: int = 32,
) -> str:
    """Chạy inference 1 cặp (ảnh, câu hỏi) với PaliGemma."""
    image = Image.open(image_path).convert("RGB")
    prompt = make_prompt(question)

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Chỉ giải mã phần sinh ra (bỏ phần prompt)
    generated = outputs[0][input_len:]
    answer = processor.decode(generated, skip_special_tokens=True)
    return answer.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--adapter", default=None, help="Đường dẫn LoRA adapter (B2). Bỏ trống cho zero-shot (B1).")
    ap.add_argument("--name", default="B1_paligemma_zeroshot")
    args = ap.parse_args()

    cfg = load_config(args.config)
    csv_path = args.csv or os.path.join(cfg["output_dir"], "processed", "test_prepared.csv")

    model, processor = load_paligemma_for_infer(cfg["paligemma_model_name"], args.adapter)

    df = (
        pd.read_csv(csv_path)
        .dropna(subset=["image", "question", "answer"])
        .reset_index(drop=True)
    )

    preds = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="paligemma-eval"):
        img_path = resolve_image_path(
            cfg["image_dir"],
            str(r.image),
            r.get("split", None),
            r.get("image_folder", None),
        )
        preds.append(paligemma_answer(model, processor, img_path, str(r.question)))

    metrics = compute_text_metrics(preds, df["answer"].astype(str).tolist())
    out_dir = os.path.join(cfg["output_dir"], args.name)
    os.makedirs(out_dir, exist_ok=True)
    save_json(metrics, os.path.join(out_dir, "test_metrics.json"))
    df["prediction"] = preds
    df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
    print(metrics)


if __name__ == "__main__":
    main()
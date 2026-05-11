"""
demo_gradio.py
Giao diện Gradio cho cả Hướng A (LSTM/Transformer) và Hướng B (Qwen2.5-VL).

Cách dùng:
    # Demo Hướng A:
    python -m src.demo_gradio --mode a --ckpt outputs/A_lstm/best.pt

    # Demo Hướng B (zero-shot):
    python -m src.demo_gradio --mode qwen

    # Demo Hướng B (fine-tuned):
    python -m src.demo_gradio --mode qwen --adapter outputs/B2_qwen_lora_sft
"""

import argparse
import gradio as gr

from .infer_a import predict as predict_a
from .evaluate_qwen import qwen_answer
from .qwen_common import load_qwen_for_infer
from .utils import load_config


def launch_a(ckpt: str, config: str):
    def fn(image, question):
        return predict_a(image, question, ckpt, config)

    gr.Interface(
        fn=fn,
        inputs=[gr.Image(type="filepath"), gr.Textbox(label="Câu hỏi")],
        outputs="text",
        title="VQA A1/A2 Demo — EfficientNet-B2 + PhoBERT",
        description="Hướng A: mô hình nhẹ (LSTM/Transformer decoder), trả lời tiếng Việt.",
    ).launch(share=True)


def launch_qwen(adapter: str, config: str):
    cfg = load_config(config)
    model, processor = load_qwen_for_infer(cfg["qwen_model_name"], adapter)

    def fn(image, question):
        if image is None:
            return "Vui lòng upload ảnh."
        if not question or not question.strip():
            return "Vui lòng nhập câu hỏi."
        return qwen_answer(model, processor, image, question)

    gr.Interface(
        fn=fn,
        inputs=[
            gr.Image(type="filepath", label="Ảnh đầu vào"),
            gr.Textbox(label="Câu hỏi (tiếng Việt)"),
        ],
        outputs=gr.Textbox(label="Câu trả lời"),
        title="VQA Qwen2.5-VL Demo",
        description=(
            f"Model: {cfg['qwen_model_name']}\n"
            f"Adapter: {adapter or 'None (zero-shot)'}"
        ),
        examples=[
            ["examples/butterfly.jpg", "Con bướm trong ảnh có màu gì?"],
            ["examples/horse.jpg", "Con ngựa đang làm gì?"],
        ] if False else None,  # Bật nếu có ảnh mẫu
    ).launch(share=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["a", "qwen"], required=True)
    ap.add_argument("--ckpt", default=None, help="Checkpoint cho Hướng A")
    ap.add_argument("--adapter", default=None, help="LoRA adapter cho Qwen B2")
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()

    if a.mode == "a":
        launch_a(a.ckpt, a.config)
    else:
        launch_qwen(a.adapter, a.config)
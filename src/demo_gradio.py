"""
demo_gradio.py
Giao diện Gradio cho cả Hướng A (LSTM/Transformer) và Hướng B (PaliGemma).
"""

import argparse
import gradio as gr

from .infer_a import predict as predict_a
from .evaluate_paligemma import paligemma_answer
from .paligemma_common import load_paligemma_for_infer
from .utils import load_config


def launch_a(ckpt: str, config: str):
    def fn(image, question):
        return predict_a(image, question, ckpt, config)

    gr.Interface(
        fn=fn,
        inputs=[gr.Image(type="filepath"), gr.Textbox(label="Câu hỏi")],
        outputs="text",
        title="VQA A1/A2 Demo (EfficientNet-B2 + PhoBERT)",
    ).launch(share=True)


def launch_paligemma(adapter: str, config: str):
    cfg = load_config(config)
    model, processor = load_paligemma_for_infer(cfg["paligemma_model_name"], adapter)

    def fn(image, question):
        return paligemma_answer(model, processor, image, question)

    gr.Interface(
        fn=fn,
        inputs=[gr.Image(type="filepath"), gr.Textbox(label="Câu hỏi")],
        outputs="text",
        title="VQA PaliGemma Demo",
    ).launch(share=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["a", "paligemma"], required=True)
    ap.add_argument("--ckpt", default=None, help="Checkpoint cho Hướng A")
    ap.add_argument("--adapter", default=None, help="LoRA adapter cho PaliGemma B2")
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()

    if a.mode == "a":
        launch_a(a.ckpt, a.config)
    else:
        launch_paligemma(a.adapter, a.config)

import os
import sys
import argparse
import platform
import traceback

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from .utils import (
    load_config,
    seed_everything,
    ensure_dir,
    save_json,
    get_device,
    count_trainable_params,
)
from .vocab import AnswerVocab
from .dataset_a import VQADatasetA
from .model_a import VQAModelA
from .metrics import compute_text_metrics
import transformers
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def safe_join(*parts):
    return os.path.abspath(os.path.join(*map(str, parts)))


def print_path_check(name, path, must_exist=True):
    path = os.path.abspath(str(path))
    ok = os.path.exists(path)
    print(f"[PATH] {name}: {path}")
    print(f"       exists: {ok}")
    if must_exist and not ok:
        raise FileNotFoundError(f"Không tìm thấy {name}: {path}")
    return path


def make_scaler(device):
    """Compatible with both newer and older PyTorch versions."""
    enabled = device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(device):
    """Compatible autocast context for CUDA/CPU and old/new PyTorch."""
    enabled = device.type == "cuda"
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, grad_clip=1.0):
    model.train()
    total = 0.0

    for batch in tqdm(loader, desc="train"):
        image = batch["image"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        ans = batch["answer_ids"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device):
            logits = model(image, ids, mask, ans)
            loss = criterion(logits.reshape(-1, logits.size(-1)), ans[:, 1:].reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item())

    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate_a(model, loader, vocab, criterion, device, max_len):
    model.eval()
    total = 0.0
    preds, refs = [], []

    for batch in tqdm(loader, desc="eval"):
        image = batch["image"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        ans = batch["answer_ids"].to(device, non_blocking=True)

        with autocast_ctx(device):
            logits = model(image, ids, mask, ans)
            loss = criterion(logits.reshape(-1, logits.size(-1)), ans[:, 1:].reshape(-1))

        total += float(loss.item())
        gen = model.generate(image, ids, mask, vocab.bos_id, vocab.eos_id, max_len=max_len)
        preds += [vocab.decode(x.detach().cpu().tolist()) for x in gen]
        refs += list(batch["raw_answer"])

    metrics = compute_text_metrics(preds, refs)
    metrics["loss"] = total / max(len(loader), 1)
    return metrics, preds, refs


def make_optimizer(model, lr=2e-4, encoder_lr=5e-5, weight_decay=0.01):
    encoder_keywords = ["text_encoder", "phobert", "image_encoder", "efficientnet", "backbone"]
    no_decay_keywords = ["bias", "LayerNorm.weight", "norm.weight"]

    groups = {
        "encoder_decay": [],
        "encoder_no_decay": [],
        "main_decay": [],
        "main_no_decay": [],
    }
    seen = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        is_encoder = any(k in name for k in encoder_keywords)
        is_no_decay = any(k in name for k in no_decay_keywords)

        if is_encoder:
            groups["encoder_no_decay" if is_no_decay else "encoder_decay"].append(param)
        else:
            groups["main_no_decay" if is_no_decay else "main_decay"].append(param)

    optimizer_grouped_parameters = [
        {"params": groups["encoder_decay"], "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": groups["encoder_no_decay"], "lr": encoder_lr, "weight_decay": 0.0},
        {"params": groups["main_decay"], "lr": lr, "weight_decay": weight_decay},
        {"params": groups["main_no_decay"], "lr": lr, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW([g for g in optimizer_grouped_parameters if len(g["params"]) > 0])


def resolve_num_workers(cfg):
    # Windows/Jupyter thường lỗi spawn/pickle nếu num_workers > 0.
    # Dùng 0 là an toàn nhất cho máy local.
    requested = int(cfg.get("num_workers", 0))
    if platform.system().lower().startswith("win"):
        if requested != 0:
            print(f"[WARN] Windows detected: override num_workers {requested} -> 0")
        return 0
    return requested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--decoder", choices=["lstm", "transformer"], required=True)
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--val_csv", default=None)
    args = ap.parse_args()

    print("[INFO] Python:", sys.executable)
    print("[INFO] Platform:", platform.platform())
    print("[INFO] PyTorch:", torch.__version__)
    print("[INFO] CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[INFO] CUDA device:", torch.cuda.get_device_name(0))

    config_path = print_path_check("config", args.config, must_exist=True)
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))

    # Chuẩn hóa các path quan trọng.
    output_dir = safe_join(cfg["output_dir"])
    image_dir = safe_join(cfg["image_dir"])
    out_dir = ensure_dir(os.path.join(output_dir, f"A_{args.decoder}"))

    train_csv = args.train_csv or os.path.join(output_dir, "processed", "train_prepared.csv")
    val_csv = args.val_csv or os.path.join(output_dir, "processed", "val_prepared.csv")

    train_csv = print_path_check("train_csv", train_csv, must_exist=True)
    val_csv = print_path_check("val_csv", val_csv, must_exist=True)
    image_dir = print_path_check("image_dir", image_dir, must_exist=True)

    # Optional sanity checks.
    train_folder = os.path.join(image_dir, "train")
    test_folder = os.path.join(image_dir, "test")
    print(f"[PATH] train folder: {train_folder} | exists: {os.path.exists(train_folder)}")
    print(f"[PATH] test folder : {test_folder} | exists: {os.path.exists(test_folder)}")

    train_df = pd.read_csv(train_csv)
    print("[INFO] train_df shape:", train_df.shape)
    print("[INFO] train_df columns:", list(train_df.columns))
    if "answer" not in train_df.columns:
        raise KeyError("File train_prepared.csv phải có cột 'answer'.")

    vocab = AnswerVocab.build(train_df["answer"].astype(str).tolist(), min_freq=1)
    vocab.save(os.path.join(out_dir, "answer_vocab.json"))

    train_ds = VQADatasetA(
        train_csv,
        image_dir,
        vocab,
        cfg["phobert_name"],
        cfg["max_question_len"],
        cfg["max_answer_len"],
        cfg["image_size"],
        train=True,
    )
    val_ds = VQADatasetA(
        val_csv,
        image_dir,
        vocab,
        cfg["phobert_name"],
        cfg["max_question_len"],
        cfg["max_answer_len"],
        cfg["image_size"],
        train=False,
    )

    num_workers = resolve_num_workers(cfg)
    pin_memory = torch.cuda.is_available()
    batch_size = int(cfg.get("batch_size", 4))

    print(f"[INFO] batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    device = get_device()
    print("[INFO] device:", device)

    model = VQAModelA(
        len(vocab),
        decoder_type=args.decoder,
        phobert_name=cfg["phobert_name"],
        pad_id=vocab.pad_id,
        max_len=cfg["max_answer_len"],
    ).to(device)

    print("[INFO] trainable params:", count_trainable_params(model))

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    scaler = make_scaler(device)

    optimizer = make_optimizer(
        model,
        lr=cfg.get("lr_a", 2e-4),
        encoder_lr=cfg.get("lr_phobert", 5e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    best = -1.0
    bad = 0
    history = []
    epochs = int(cfg.get("epochs", 5))
    patience = int(cfg.get("patience", 3))

    for ep in range(1, epochs + 1):
        print(f"\n========== Epoch {ep}/{epochs} ==========")
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
        metrics, preds, refs = evaluate_a(model, val_loader, vocab, criterion, device, cfg["max_answer_len"])
        metrics["epoch"] = ep
        metrics["train_loss"] = tr_loss
        history.append(metrics)
        print("[METRICS]", metrics)

        score = metrics.get("soft_vqa_f1", 0)
        save_json(history, os.path.join(out_dir, "history.json"))

        if score > best:
            best = score
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "decoder": args.decoder,    
                    "vocab_path": "answer_vocab.json",
                },
                os.path.join(out_dir, "best.pt"),
            )
            pd.DataFrame({"pred": preds, "ref": refs}).to_csv(
                os.path.join(out_dir, "val_predictions.csv"), index=False
            )
            print(f"[SAVE] New best checkpoint saved. score={best}")
        else:
            bad += 1
            if bad >= patience:
                print("[INFO] Early stopping")
                break

    print("[DONE] Saved to", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] train_a.py crashed. Full traceback:")
        traceback.print_exc()
        raise

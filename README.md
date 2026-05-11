<div align="center">

# 🔍 VQA Tiếng Việt — Visual Question Answering

**Hệ thống trả lời câu hỏi dựa trên hình ảnh (VQA) hoàn chỉnh bằng tiếng Việt**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21F)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

<br/>

> Dự án triển khai **hai hướng tiếp cận** song song để giải quyết bài toán VQA tiếng Việt:  
> **Hướng A** — Mô hình nhẹ tự xây dựng (EfficientNet-B2 + PhoBERT + Decoder)  
> **Hướng B** — Fine-tune Large Multimodal Model (Qwen2.5-VL với QLoRA)

</div>

---

## 📋 Mục lục

- [Tổng quan kiến trúc](#-tổng-quan-kiến-trúc)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Chuẩn bị dữ liệu](#-chuẩn-bị-dữ-liệu)
- [Hướng A — Mô hình tự xây dựng](#-hướng-a--mô-hình-tự-xây-dựng)
- [Hướng B — Qwen2.5-VL](#-hướng-b--qwen25-vl)
- [Đánh giá & Metrics](#-đánh-giá--metrics)
- [Demo Gradio](#-demo-gradio)
- [Cấu hình](#-cấu-hình)
- [Kết quả mong đợi](#-kết-quả-mong-đợi)

---

## 🏗 Tổng quan kiến trúc

### Hướng A — Mô hình tự xây dựng (Lightweight)

```
Ảnh đầu vào ──► EfficientNet-B2 ──► Image Feature (512-d)
                                              │
                                              ▼
                                         Fusion Layer ──► Context (512-d)
                                              ▲                  │
                                              │                  ▼
Câu hỏi ──► PhoBERT (vinai) ──► Text Feature     ┌─────────────────────────┐
                    (512-d)                        │  A1: LSTM Decoder       │
                                                   │  A2: Transformer Decoder│
                                                   └─────────────────────────┘
                                                              │
                                                              ▼
                                                        Câu trả lời
```

| Module | Chi tiết |
|--------|----------|
| **Image Encoder** | EfficientNet-B2 (pretrained ImageNet), AdaptiveAvgPool → Linear(1408→512) |
| **Text Encoder** | PhoBERT `vinai/phobert-base`, lấy `[CLS]` token → Linear(768→512) |
| **Fusion** | Concat `[image, text]` → Linear(1024→512) + ReLU + Dropout + LayerNorm |
| **A1 Decoder** | LSTM 1 layer, teacher forcing, context-initialized h₀/c₀ |
| **A2 Decoder** | Transformer Decoder (3 layers, 8 heads), positional embedding |

### Hướng B — Qwen2.5-VL (Large Multimodal)

```
Ảnh + Câu hỏi ──► Qwen2.5-VL-3B-Instruct (frozen) ──► LoRA Adapter ──► Câu trả lời
                        (QLoRA 4-bit, NF4)                 (r=16, α=32)
```

| Variant | Chi tiết |
|---------|----------|
| **B1 — Zero-shot** | Qwen2.5-VL-3B không fine-tune, prompt tiếng Việt |
| **B2 — SFT** | QLoRA 4-bit, Supervised Fine-Tuning trên tập train |

---

## 📁 Cấu trúc dự án

```
📦 vqa-vietnamese/
│
├── 📂 configs/
│   └── default.yaml              # Toàn bộ hyperparameters & đường dẫn
│
├── 📂 src/
│   ├── __init__.py
│   │
│   ├── 🗃️  DỮ LIỆU
│   ├── prepare_data.py           # Tiền xử lý, split train/val/test
│   ├── dataset_a.py              # PyTorch Dataset cho Hướng A
│   ├── vocab.py                  # AnswerVocab (tokenizer tiếng Việt)
│   ├── path_utils.py             # Resolve đường dẫn ảnh linh hoạt
│   │
│   ├── 🧠  MÔ HÌNH
│   ├── model_a.py                # EfficientNet-B2 + PhoBERT + LSTM/Transformer
│   ├── qwen_common.py            # Load Qwen2.5-VL, Dataset, LoRA config
│   │
│   ├── 🏋️  HUẤN LUYỆN
│   ├── train_a.py                # Training loop Hướng A (mixed precision, early stopping)
│   ├── train_qwen_sft.py         # SFT Qwen với QLoRA (Hướng B2)
│   │
│   ├── 📊  ĐÁNH GIÁ
│   ├── evaluate_a.py             # Đánh giá Hướng A trên tập test
│   ├── evaluate_qwen.py          # Đánh giá Hướng B (zero-shot & fine-tuned)
│   ├── metrics.py                # EM, Token-F1, BLEU, ROUGE-L, BERTScore
│   │
│   ├── 🚀  INFERENCE
│   ├── infer_a.py                # Single-image inference Hướng A
│   └── demo_gradio.py            # Web demo (Gradio)
│
├── 📂 outputs/                   # Checkpoint, log, kết quả (tự sinh)
│   ├── A_lstm/
│   │   ├── best.pt
│   │   ├── answer_vocab.json
│   │   ├── history.json
│   │   └── test_metrics.json
│   ├── A_transformer/
│   ├── B1_qwen_zeroshot/
│   └── B2_qwen_lora_sft/
│
├── 📂 train/                     # Ảnh train (train/*.jpg)
├── 📂 test/                      # Ảnh test (test/*.jpg)
├── 📄 train.csv                  # Nhãn train: image, question, answer
├── 📄 test.csv                   # Nhãn test:  image, question, answer
├── 📄 configs/default.yaml
├── 📄 requirements.txt
└── 📄 run_all_colab.sh           # Chạy toàn bộ pipeline 1 lệnh
```

### Format dữ liệu CSV

`train.csv` và `test.csv` cần có **ít nhất** 3 cột:

| Cột | Kiểu | Ví dụ |
|-----|------|-------|
| `image` | str | `img_001.jpg` |
| `question` | str | `Con bướm trong ảnh có màu gì?` |
| `answer` | str | `màu vàng` |

---

## 💻 Yêu cầu hệ thống

| Thành phần | Tối thiểu | Khuyến nghị |
|-----------|-----------|------------|
| GPU VRAM | 6 GB (Hướng A) | 16 GB (Hướng B) |
| RAM | 8 GB | 16 GB |
| Python | 3.10+ | 3.11 |
| CUDA | 11.8+ | 12.1+ |
| Dung lượng | 10 GB | 30 GB |

> **Google Colab:** T4 (16 GB) chạy được Hướng A + B1 + B2 (QLoRA 4-bit).  
> **Hướng A** có thể chạy trên CPU (chậm hơn đáng kể).

---

## ⚙️ Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/thanhtu-nguyen/vqa-animal-project.git
cd vqa-animal-project

# 2. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Cài dependencies
pip install -r requirements.txt
```

**Các thư viện chính:**

```
torch >= 2.0
torchvision
transformers >= 4.37
peft >= 0.8
bitsandbytes >= 0.41        # QLoRA 4-bit (cần CUDA)
qwen-vl-utils               # Qwen2.5-VL processor
trl >= 0.8                  # SFT/DPO trainer
evaluate                    # BLEU, ROUGE
bert_score                  # BERTScore
gradio >= 4.0               # Web demo
scikit-learn
pandas
PyYAML
Pillow
```

---

## 📦 Chuẩn bị dữ liệu

### Bước 1 — Đặt ảnh đúng cấu trúc

```
vqa_dataset/
├── train/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── test/
│   ├── img_test_001.jpg
│   └── ...
├── train.csv
└── test.csv
```

### Bước 2 — Chạy prepare_data

```bash
python -m src.prepare_data \
  --train_csv /path/to/train.csv \
  --test_csv  /path/to/test.csv \
  --image_dir /path/to/vqa_dataset \
  --out_dir   /path/to/vqa_dataset/outputs/processed
```

Script sẽ tự động:
- Kiểm tra ảnh tồn tại, báo cáo ảnh thiếu
- Tạo split `val` (10%) từ `train`
- Lưu ra `train_prepared.csv`, `val_prepared.csv`, `test_prepared.csv`

**Output mẫu:**
```
[INFO] Train hợp lệ: 4500 dòng
[INFO] Test hợp lệ:  500 dòng
[DONE] Đã lưu vào: outputs/processed/
```

---

## 🤖 Hướng A — Mô hình tự xây dựng

### Huấn luyện

```bash
# A1: LSTM Decoder
python -m src.train_a \
  --config  configs/default.yaml \
  --decoder lstm

# A2: Transformer Decoder
python -m src.train_a \
  --config  configs/default.yaml \
  --decoder transformer
```

**Quá trình training sẽ:**
- In progress bar từng epoch với train loss
- Đánh giá trên val set sau mỗi epoch (EM, soft-F1, BLEU, ROUGE-L)
- Lưu `best.pt` khi soft-F1 cải thiện
- Early stopping nếu không cải thiện sau `patience` epochs

**Log mẫu:**
```
========== Epoch 1/30 ==========
train: 100%|████████| 563/563 [02:14<00:00]
eval:  100%|████████|  63/63  [00:18<00:00]
[METRICS] {'exact_match': 0.312, 'soft_vqa_f1': 0.487, 'bleu': 0.231, 'loss': 1.823, 'epoch': 1}
[SAVE] New best checkpoint saved. score=0.487
```

### Đánh giá

```bash
python -m src.evaluate_a \
  --config configs/default.yaml \
  --ckpt   outputs/A_lstm/best.pt
```

### Inference đơn lẻ

```bash
python -m src.infer_a \
  --image    /path/to/image.jpg \
  --question "Con chó trong ảnh có màu gì?" \
  --ckpt     outputs/A_lstm/best.pt \
  --config   configs/default.yaml
```

---

## 🦙 Hướng B — Qwen2.5-VL

### B1 — Zero-shot (không cần train)

```bash
python -m src.evaluate_qwen \
  --config configs/default.yaml \
  --name   B1_qwen_zeroshot
```

> Model được tải tự động từ HuggingFace (~6 GB). Cần GPU với ≥ 8 GB VRAM (bf16) hoặc 6 GB (4-bit).

### B2 — Supervised Fine-Tuning (QLoRA)

```bash
python -m src.train_qwen_sft \
  --config configs/default.yaml
```

**Cấu hình mặc định:**

| Tham số | Giá trị |
|---------|---------|
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Batch size | 1 × grad_accum 8 = effective 8 |
| Epochs | 3 |
| Learning rate | 2e-4 (cosine decay) |

**Đánh giá sau SFT:**

```bash
python -m src.evaluate_qwen \
  --config  configs/default.yaml \
  --adapter outputs/B2_qwen_lora_sft \
  --name    B2_qwen_lora_sft
```

---

## 📊 Đánh giá & Metrics

Hệ thống tính toán **5 metrics** tự động sau mỗi lần đánh giá:

| Metric | Mô tả | Tốt khi |
|--------|-------|---------|
| **Exact Match (EM)** | Khớp hoàn toàn sau normalize | > 0.4 |
| **Soft VQA F1** | Token-level F1 (precision × recall), dùng làm monitor chính | > 0.6 |
| **BLEU** | N-gram precision (SacreBLEU) | > 0.3 |
| **ROUGE-L** | Longest Common Subsequence | > 0.5 |
| **BERTScore F1** | Semantic similarity (PhoBERT embeddings, `lang=vi`) | > 0.7 |

Kết quả lưu tại `outputs/{model_name}/test_metrics.json`.

**Xem kết quả:**

```bash
cat outputs/A_lstm/test_metrics.json
cat outputs/B2_qwen_lora_sft/test_metrics.json
```

---

## 🎨 Demo Gradio

```bash
# Demo Hướng A (LSTM)
python -m src.demo_gradio \
  --mode a \
  --ckpt outputs/A_lstm/best.pt

# Demo Hướng B (zero-shot)
python -m src.demo_gradio \
  --mode qwen

# Demo Hướng B (fine-tuned)
python -m src.demo_gradio \
  --mode  qwen \
  --adapter outputs/B2_qwen_lora_sft
```

Mở trình duyệt tại `http://localhost:7860` (hoặc link public nếu dùng Colab `--share`).

---

## ⚙️ Cấu hình

Tất cả hyperparameters được quản lý trong `configs/default.yaml`:

```yaml
# ── Đường dẫn ───────────────────────────────────────────────────
project_root: /content/drive/MyDrive/vqa_dataset
image_dir:    /content/drive/MyDrive/vqa_dataset
train_csv:    /content/drive/MyDrive/vqa_dataset/train.csv
test_csv:     /content/drive/MyDrive/vqa_dataset/test.csv
output_dir:   /content/drive/MyDrive/vqa_dataset/outputs

# ── Hướng A ─────────────────────────────────────────────────────
seed: 42
image_size: 288           # Input resolution EfficientNet-B2
batch_size: 8
epochs: 30
patience: 3               # Early stopping
lr_a: 2.0e-4              # Learning rate decoder
lr_phobert: 2.0e-5        # Learning rate PhoBERT (nhỏ hơn để fine-tune nhẹ)
max_question_len: 64      # Max tokens câu hỏi
max_answer_len: 10        # Max tokens câu trả lời

# ── Hướng B — Qwen2.5-VL ────────────────────────────────────────
qwen_model_name: Qwen/Qwen2.5-VL-3B-Instruct
qwen_epochs: 3
qwen_lr: 2.0e-4
qwen_batch_size: 1
qwen_grad_accum: 8        # effective batch = 1 × 8 = 8

# ── LoRA ────────────────────────────────────────────────────────
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
```

**Đổi model Qwen lớn hơn** (nếu có đủ VRAM):

```yaml
# ~12 GB VRAM
qwen_model_name: Qwen/Qwen2.5-VL-7B-Instruct

# Multi-GPU, ~80+ GB VRAM tổng
qwen_model_name: Qwen/Qwen2.5-VL-72B-Instruct
```

---

## 🚀 Chạy toàn bộ pipeline (1 lệnh)

```bash
# Cấu hình đường dẫn trong configs/default.yaml trước
bash run_all_colab.sh
```

Pipeline sẽ tự chạy theo thứ tự:

```
Bước 0: prepare_data
   ↓
Hướng A1: train (LSTM)  →  evaluate_a
   ↓
Hướng A2: train (Transformer)  →  evaluate_a
   ↓
Hướng B1: evaluate_qwen (zero-shot)
   ↓
Hướng B2: train_qwen_sft  →  evaluate_qwen
```

---

## 📈 Kết quả thực nghiệm

Kết quả đo trên tập **test** của dự án:

| Hướng | Model | Exact Match | Soft F1 | BLEU | ROUGE-L | BERTScore |
|-------|-------|:-----------:|:-------:|:----:|:-------:|:---------:|
| **B2** | Qwen2.5-VL-3B + QLoRA SFT | **0.0575** | **0.6955** | **0.4086** | **0.7079** | **0.9027** |
| B1 | Qwen2.5-VL-3B (zero-shot) | 0.0375 | 0.6483 | 0.3298 | 0.6333 | 0.8776 |
| A2 | EfficientNet + PhoBERT + Transformer | 0.0025 | 0.4749 | 0.1286 | 0.4977 | 0.8285 |
| A1 | EfficientNet + PhoBERT + LSTM | 0.0000 | 0.3887 | 0.0871 | 0.4396 | 0.8109 |

### 🔍 Nhận xét

- **B2 (QLoRA SFT) đạt kết quả tốt nhất** trên tất cả metrics, đặc biệt BERTScore đạt **0.9027** — cho thấy câu trả lời sinh ra có độ tương đồng ngữ nghĩa rất cao so với ground truth.
- **Exact Match thấp** ở tất cả các hướng do tiếng Việt có nhiều cách diễn đạt tương đương (ví dụ: *"màu đỏ"* vs *"đỏ"*) — Soft F1 và BERTScore phản ánh chất lượng thực tế chính xác hơn.
- **B1 zero-shot** đã đạt Soft F1 = 0.648 mà không cần fine-tune, chứng minh sức mạnh của Qwen2.5-VL trên tiếng Việt.
- **Hướng A** (mô hình nhẹ ~50M params) đạt BERTScore > 0.81, cho thấy học được ngữ nghĩa nhất định dù kiến trúc đơn giản hơn nhiều so với LMM.

> 💡 **Gợi ý chọn hướng:**  
> — Cần **độ chính xác cao nhất** → dùng **B2** (QLoRA SFT)  
> — Không có dữ liệu train / cần triển khai nhanh → dùng **B1** (zero-shot)  
> — Cần mô hình **nhẹ, deployable, không phụ thuộc GPU lớn** → dùng **A2** (Transformer decoder)

---

## 🐛 Xử lý lỗi thường gặp

**`FileNotFoundError: Không tìm thấy ảnh`**
```bash
# Kiểm tra cấu trúc thư mục
ls vqa_dataset/train/ | head -5
# Đảm bảo tên ảnh trong CSV khớp với file thực tế
```

**`CUDA out of memory` (Hướng B)**
```yaml
# Giảm trong default.yaml
qwen_batch_size: 1
qwen_grad_accum: 16       # Tăng grad_accum để bù
qwen_max_length: 256      # Giảm max_length
```

**`num_workers` lỗi trên Windows**
```yaml
# Script tự detect và override về 0 trên Windows
num_workers: 0            # Hoặc đặt tường minh
```

**BitsAndBytes không cài được (no CUDA)**
```bash
# Hướng A không cần bitsandbytes
# Chỉ cần cho Hướng B (QLoRA)
pip install bitsandbytes --prefer-binary
```

---

## 📄 License

MIT License — xem [LICENSE](LICENSE).

---

<div align="center">

Made for Vietnamese NLP

**[⬆ Về đầu trang](#-vqa-tiếng-việt--visual-question-answering)**

</div>

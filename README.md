# **Hệ thống Trả lời Câu hỏi Thị giác (VQA) Tiếng Việt về Động vật**

Dự án này triển khai các mô hình Deep Learning để giải quyết bài toán Visual Question Answering (VQA) bằng tiếng Việt. Hệ thống tập trung vào việc nhận diện và trả lời các câu hỏi liên quan đến 10 loài động vật: bướm, mèo, bò, chó, voi, gà, ngựa, cừu, nhện và sóc.

## **🚀 Các Hướng Tiếp Cận**

Dự án triển khai hai chiến lược mô hình hóa chính để so sánh hiệu năng:

### **Hướng A: Kiến trúc Tùy chỉnh (Modular VQA)**

Kết hợp sức mạnh của thị giác máy tính và xử lý ngôn ngữ tự nhiên thông qua cơ chế fusion:

* **Image Encoder:** EfficientNet-B2 (Pre-trained trên ImageNet).  
* **Text Encoder:** PhoBERT (Pre-trained cho tiếng Việt).  
* **Decoder:** Hỗ trợ hai biến thể **LSTM** và **Transformer** để sinh câu trả lời.

### **Hướng B: Mô hình Thị giác-Ngôn ngữ Lớn (VLM)**

Sử dụng mô hình nền tảng hiện đại từ Google:

* **PaliGemma:** Chạy ở chế độ **Zero-shot** (không cần huấn luyện) và **Fine-tuning (SFT)** bằng kỹ thuật LoRA để tối ưu hóa trên tập dữ liệu đặc thù.

## **📂 Cấu trúc Thư mục**

`├── configs/`  
`│   └── default.yaml         # Cấu hình tham số huấn luyện và đường dẫn`  
`├── src/`  
`│   ├── model_a.py           # Định nghĩa kiến trúc EfficientNet + PhoBERT`  
`│   ├── train_a.py           # Script huấn luyện Hướng A`  
`│   ├── train_paligemma_sft.py # Tinh chỉnh PaliGemma với LoRA`  
`│   ├── dataset_a.py         # Xử lý dữ liệu cho Hướng A`  
`│   ├── metrics.py           # Tính toán EM, F1, BLEU, ROUGE-L`  
`│   ├── demo_gradio.py       # Giao diện web demo tương tác`  
`│   └── ...`  
`├── train.csv                # Dữ liệu huấn luyện`  
`├── test.csv                 # Dữ liệu kiểm tra`  
`└── requirements.txt         # Các thư viện cần thiết`

## **🛠 Hướng dẫn Cài đặt**

1. Cài đặt các thư viện phụ thuộc:

`pip install -r requirements.txt`

2\. Cập nhật đường dẫn trong configs/default.yaml phù hợp với môi trường của bạn (Local hoặc Google Colab).

## **📈 Quy trình Huấn luyện & Đánh giá**

Sử dụng script run\_all\_colab.sh hoặc chạy thủ công các lệnh sau:  
**1\. Chuẩn bị dữ liệu:**

`python -m src.prepare_data --train_csv train.csv --test_csv test.csv --image_dir ./data`

**2\. Huấn luyện Hướng A (Ví dụ với Transformer):**

`python -m src.train_a --config configs/default.yaml --decoder transformer`

**3\. Tinh chỉnh PaliGemma (SFT):**

`python -m src.train_paligemma_sft --config configs/default.yaml`

## **🖥 Demo Giao diện (Gradio)**

Bạn có thể chạy giao diện web để thử nghiệm trực tiếp với ảnh cá nhân:

`# Thử nghiệm mô hình Hướng A`  
`python -m src.demo_gradio --mode a --ckpt ./outputs/A_transformer/best.pt`

`# Thử nghiệm mô hình PaliGemma`  
`python -m src.demo_gradio --mode paligemma --adapter ./outputs/B2_paligemma_lora_sft`

## **📊 Kết quả & Đánh giá**

Hệ thống sử dụng các thang đo tiêu chuẩn để đánh giá chất lượng câu trả lời:

| Metric | Mô tả |
| :---- | :---- |
| **Exact Match (EM)** | Tỷ lệ khớp hoàn toàn 100% với đáp án chuẩn. |
| **Soft VQA Accuracy (F1)** | Đánh giá độ tương đồng về mặt ngữ nghĩa (Token-level F1). |
| **BLEU & ROUGE-L** | Đo lường chất lượng văn bản sinh ra so với câu tham chiếu. |

*Dự án được thực hiện như một phần của môn học Deep Learning tại Đại học Tôn Đức Thắng.*

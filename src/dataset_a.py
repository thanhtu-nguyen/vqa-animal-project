import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from .vocab import AnswerVocab
from .path_utils import resolve_image_path


def get_image_transform(image_size=288, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

class VQADatasetA(Dataset):
    def __init__(self, csv_path, image_dir, vocab: AnswerVocab, phobert_name='vinai/phobert-base', max_q_len=64, max_a_len=16, image_size=288, train=True):
        self.df = pd.read_csv(csv_path).dropna(subset=['image','question','answer']).reset_index(drop=True)
        self.image_dir = image_dir
        self.vocab = vocab
        self.tokenizer = AutoTokenizer.from_pretrained(phobert_name, use_fast=False)
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.tf = get_image_transform(image_size, train=train)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img_path = resolve_image_path(
            self.image_dir,
            str(r.image),
            split=r.get('split', None),
            image_folder=r.get('image_folder', None),
        )
        image = Image.open(img_path).convert('RGB')
        image = self.tf(image)
        q = self.tokenizer(str(r.question), padding='max_length', truncation=True, max_length=self.max_q_len, return_tensors='pt')
        ans_ids = torch.tensor(self.vocab.encode(str(r.answer), self.max_a_len), dtype=torch.long)
        return {
            'image': image,
            'input_ids': q['input_ids'].squeeze(0),
            'attention_mask': q['attention_mask'].squeeze(0),
            'answer_ids': ans_ids,
            'raw_answer': str(r.answer),
            'question': str(r.question),
            'image_name': str(r.image),
        }

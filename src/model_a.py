import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from transformers import AutoModel

class ImageEncoderEffB2(nn.Module):
    def __init__(self, out_dim=512, train_backbone=False):
        super().__init__()
        m = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1408, out_dim)
        for p in self.features.parameters():
            p.requires_grad = train_backbone
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)

class TextEncoderPhoBERT(nn.Module):
    def __init__(self, name='vinai/phobert-base', out_dim=512, freeze=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(name)
        self.proj = nn.Linear(self.bert.config.hidden_size, out_dim)
        if freeze:
            for p in self.bert.parameters(): p.requires_grad = False
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # PhoBERT dùng token đầu như CLS pooled representation.
        cls = out.last_hidden_state[:,0]
        return self.proj(cls)

class LSTMAnswerDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512, context_dim=512, pad_id=0, max_len=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.h0 = nn.Linear(context_dim, hidden_dim)
        self.c0 = nn.Linear(context_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len
    def forward(self, context, tgt_ids):
        # teacher forcing input: bỏ token cuối
        x = self.emb(tgt_ids[:, :-1])
        h0 = torch.tanh(self.h0(context)).unsqueeze(0)
        c0 = torch.tanh(self.c0(context)).unsqueeze(0)
        o, _ = self.lstm(x, (h0, c0))
        return self.out(o)
    @torch.no_grad()
    def generate(self, context, bos_id, eos_id, max_len=16):
        B = context.size(0)
        h = torch.tanh(self.h0(context)).unsqueeze(0)
        c = torch.tanh(self.c0(context)).unsqueeze(0)
        cur = torch.full((B,1), bos_id, dtype=torch.long, device=context.device)
        outs = []
        for _ in range(max_len-1):
            emb = self.emb(cur[:, -1:])
            o, (h,c) = self.lstm(emb, (h,c))
            nxt = self.out(o[:, -1]).argmax(-1)
            outs.append(nxt)
            cur = torch.cat([cur, nxt[:,None]], dim=1)
        return torch.stack(outs, dim=1)

class TransformerAnswerDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, nhead=8, num_layers=3, context_dim=512, pad_id=0, max_len=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, emb_dim)
        self.mem = nn.Linear(context_dim, emb_dim)
        layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(emb_dim, vocab_size)
        self.max_len = max_len
        self.pad_id = pad_id
    def _mask(self, T, device):
        return torch.triu(torch.ones(T,T,device=device), diagonal=1).bool()
    def forward(self, context, tgt_ids):
        inp = tgt_ids[:, :-1]
        B,T = inp.shape
        pos = torch.arange(T, device=inp.device).unsqueeze(0)
        x = self.emb(inp) + self.pos(pos)
        memory = self.mem(context).unsqueeze(1)
        y = self.dec(x, memory, tgt_mask=self._mask(T, inp.device))
        return self.out(y)
    @torch.no_grad()
    def generate(self, context, bos_id, eos_id, max_len=16):
        B = context.size(0)
        ys = torch.full((B,1), bos_id, dtype=torch.long, device=context.device)
        memory = self.mem(context).unsqueeze(1)
        for _ in range(max_len-1):
            T = ys.size(1)
            pos = torch.arange(T, device=context.device).unsqueeze(0)
            x = self.emb(ys) + self.pos(pos)
            out = self.dec(x, memory, tgt_mask=self._mask(T, context.device))
            nxt = self.out(out[:,-1]).argmax(-1)
            ys = torch.cat([ys, nxt[:,None]], dim=1)
        return ys[:,1:]

class VQAModelA(nn.Module):
    def __init__(self, vocab_size, decoder_type='lstm', phobert_name='vinai/phobert-base', pad_id=0, max_len=16):
        super().__init__()
        self.image_encoder = ImageEncoderEffB2(out_dim=512, train_backbone=False)
        self.text_encoder = TextEncoderPhoBERT(phobert_name, out_dim=512, freeze=False)
        self.fusion = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.2), nn.LayerNorm(512))
        if decoder_type == 'lstm':
            self.decoder = LSTMAnswerDecoder(vocab_size, context_dim=512, pad_id=pad_id, max_len=max_len)
        elif decoder_type == 'transformer':
            self.decoder = TransformerAnswerDecoder(vocab_size, context_dim=512, pad_id=pad_id, max_len=max_len)
        else:
            raise ValueError('decoder_type phải là lstm hoặc transformer')
    def encode_context(self, image, input_ids, attention_mask):
        im = self.image_encoder(image)
        txt = self.text_encoder(input_ids, attention_mask)
        return self.fusion(torch.cat([im, txt], dim=-1))
    # Sửa trong lớp VQAModelA (file model_a.py)
    # Trong file model_a.py, lớp VQAModelA
    def forward(self, image, input_ids, attention_mask, answer_ids):
        ctx = self.encode_context(image, input_ids, attention_mask)
        logits = self.decoder(ctx, answer_ids) # Decoder đã trả về (B, T-1, Vocab)
        
        # SỬA TẠI ĐÂY: Trả về trực tiếp logits, không cắt thêm nữa
        return logits.contiguous()
    @torch.no_grad()
    def generate(self, image, input_ids, attention_mask, bos_id, eos_id, max_len=16):
        ctx = self.encode_context(image, input_ids, attention_mask)
        return self.decoder.generate(ctx, bos_id, eos_id, max_len=max_len)

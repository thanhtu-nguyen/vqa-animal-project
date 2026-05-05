import re, json
from collections import Counter

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

def tokenize_vi(text: str):
    text = str(text).strip().lower()
    # Giữ tiếng Việt, số, dấu câu đơn giản để answer decoder sinh câu ngắn.
    return re.findall(r"[\wÀ-ỹ]+|[^\s\w]", text, flags=re.UNICODE)

class AnswerVocab:
    def __init__(self, stoi=None):
        self.stoi = stoi or {tok:i for i,tok in enumerate(SPECIALS)}
        self.itos = {i:t for t,i in self.stoi.items()}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    @classmethod
    def build(cls, answers, min_freq=1, max_size=None):
        counter = Counter()
        for a in answers:
            counter.update(tokenize_vi(a))
        stoi = {tok:i for i,tok in enumerate(SPECIALS)}
        for tok, freq in counter.most_common():
            if freq < min_freq: continue
            if tok not in stoi:
                stoi[tok] = len(stoi)
            if max_size and len(stoi) >= max_size: break
        return cls(stoi)

    def encode(self, text, max_len=16):
        ids = [self.bos_id] + [self.stoi.get(t, self.unk_id) for t in tokenize_vi(text)] + [self.eos_id]
        ids = ids[:max_len]
        ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        toks = []
        for i in ids:
            tok = self.itos.get(int(i), "<unk>")
            if tok in ["<bos>", "<pad>"]: continue
            if tok == "<eos>": break
            toks.append(tok)
        s = " ".join(toks)
        s = s.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
        return s.strip()

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            return cls(json.load(f))

    def __len__(self): return len(self.stoi)

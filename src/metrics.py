import re
import numpy as np
from collections import Counter


def normalize_text(s):
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.,;:!\?\"'“”‘’]", "", s)
    return s


def exact_match(preds, refs):
    return float(np.mean([normalize_text(p)==normalize_text(r) for p,r in zip(preds, refs)]))


def token_f1(pred, ref):
    p = normalize_text(pred).split(); r = normalize_text(ref).split()
    if len(p)==0 and len(r)==0: return 1.0
    if len(p)==0 or len(r)==0: return 0.0
    common = Counter(p) & Counter(r)
    num = sum(common.values())
    if num == 0: return 0.0
    precision = num / len(p); recall = num / len(r)
    return 2*precision*recall/(precision+recall)


def soft_vqa_accuracy(preds, refs):
    # Với mỗi câu chỉ có 1 đáp án chuẩn, dùng token-F1 làm soft accuracy gần đúng.
    return float(np.mean([token_f1(p,r) for p,r in zip(preds, refs)]))


def compute_text_metrics(preds, refs):
    out = {"exact_match": exact_match(preds, refs), "soft_vqa_f1": soft_vqa_accuracy(preds, refs)}
    try:
        import evaluate
        bleu = evaluate.load("sacrebleu")
        rouge = evaluate.load("rouge")
        out["bleu"] = bleu.compute(predictions=preds, references=[[r] for r in refs])["score"] / 100.0
        out["rougeL"] = rouge.compute(predictions=preds, references=refs)["rougeL"]
    except Exception as e:
        out["bleu"] = None; out["rougeL"] = None; out["metric_warning"] = str(e)
    try:
        from bert_score import score
        _, _, F1 = score(preds, refs, lang="vi", verbose=False, rescale_with_baseline=False)
        out["bertscore_f1"] = float(F1.mean().item())
    except Exception as e:
        out["bertscore_f1"] = None
        out["bertscore_warning"] = str(e)
    return out

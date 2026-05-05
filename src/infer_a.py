import os, argparse, torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from .utils import load_config, get_device
from .vocab import AnswerVocab
from .dataset_a import get_image_transform
from .model_a import VQAModelA

@torch.no_grad()
def predict(image_path, question, ckpt_path, config='configs/default.yaml'):
    cfg=load_config(config); out_dir=os.path.dirname(ckpt_path)
    vocab=AnswerVocab.load(os.path.join(out_dir,'answer_vocab.json'))
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    device=get_device()
    model=VQAModelA(len(vocab),ckpt['decoder'],cfg['phobert_name'],vocab.pad_id,cfg['max_answer_len']).to(device)
    model.load_state_dict(ckpt['model']); model.eval()
    tf=get_image_transform(cfg['image_size'],train=False)
    image=tf(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    tok=AutoTokenizer.from_pretrained(cfg['phobert_name'],use_fast=False)
    q=tok(question,padding='max_length',truncation=True,max_length=cfg['max_question_len'],return_tensors='pt')
    ids=q['input_ids'].to(device); mask=q['attention_mask'].to(device)
    gen=model.generate(image,ids,mask,vocab.bos_id,vocab.eos_id,cfg['max_answer_len'])
    return vocab.decode(gen[0].cpu().tolist())

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--image',required=True); ap.add_argument('--question',required=True); ap.add_argument('--ckpt',required=True); ap.add_argument('--config',default='configs/default.yaml')
    a=ap.parse_args(); print(predict(a.image,a.question,a.ckpt,a.config))

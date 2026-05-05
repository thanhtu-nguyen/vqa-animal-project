import os, argparse, torch, pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from .utils import load_config, get_device, save_json
from .vocab import AnswerVocab
from .dataset_a import VQADatasetA
from .model_a import VQAModelA
from .train_a import evaluate_a


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--csv', default=None)
    args=ap.parse_args(); cfg=load_config(args.config)
    out_dir=os.path.dirname(args.ckpt); csv_path=args.csv or os.path.join(cfg['output_dir'],'processed','test_prepared.csv')
    vocab=AnswerVocab.load(os.path.join(out_dir,'answer_vocab.json'))
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    device=get_device()
    model=VQAModelA(len(vocab),ckpt['decoder'],cfg['phobert_name'],vocab.pad_id,cfg['max_answer_len']).to(device)
    model.load_state_dict(ckpt['model'])
    ds=VQADatasetA(csv_path,cfg['image_dir'],vocab,cfg['phobert_name'],cfg['max_question_len'],cfg['max_answer_len'],cfg['image_size'],train=False)
    loader=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['num_workers'])
    metrics,preds,refs=evaluate_a(model,loader,vocab,nn.CrossEntropyLoss(ignore_index=vocab.pad_id),device,cfg['max_answer_len'])
    print(metrics)
    save_json(metrics, os.path.join(out_dir,'test_metrics.json'))
    base=pd.read_csv(csv_path)
    base['prediction']=preds
    base.to_csv(os.path.join(out_dir,'test_predictions.csv'),index=False)
if __name__=='__main__': main()

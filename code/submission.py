"""Compact NumPy inference; no training framework or references at evaluation."""
import json
import re
from pathlib import Path
import numpy as np
from features import build_features, aggregate, fusion_features, tree_predict, arithmetic_evidence


def score_bank(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    root=Path(checkpoint) if checkpoint else Path(__file__).parent/'checkpoints'/'final'
    config=json.loads((root/'ensemble.json').read_text())
    cache={}; total=np.zeros((len(bank),4),np.float32);family_totals={};family_weights={}
    for member in config['members']:
        mode=member['mode']
        kind=member.get('kind','nn')
        if kind=='lexical':
            import joblib
            vectorizer,model=joblib.load(root/member['file'])
            texts=[re.sub(r'\d+(?:[.,/]\d+)*',' NUM ',c['text'].lower()) for r in bank for c in r['candidates']]
            s=model.decision_function(vectorizer.transform(texts)).reshape(-1,4)
            total+=member.get('weight',1.)*s/member.get('temperature',1.)
            continue
        if mode not in cache:
            cache[mode]=build_features(bank,arrays,mode)
        x,equal=cache[mode]
        if kind=='tree':
            from catboost import CatBoostClassifier
            model=CatBoostClassifier();model.load_model(str(root/member['file']))
            s=model.predict(x.reshape(-1,x.shape[-1]),prediction_type='RawFormulaVal',thread_count=4).reshape(-1,4)
            total+=member.get('weight',1.)*s/member.get('temperature',1.)
            continue
        with np.load(root/member['file'],allow_pickle=False) as w:
            h=((x-w['mean'])/w['scale']).clip(-10,10)
            if kind=='kernel':
                z=(h-w['pca_mean'])@w['projection'];flat=z.reshape(-1,z.shape[-1])
                support=w['support'];out=[]
                norms=(support**2).sum(1)
                for start in range(0,len(flat),256):
                    zz=flat[start:start+256]
                    d=np.maximum((zz**2).sum(1)[:,None]+norms[None,:]-2*zz@support.T,0)
                    out.append(np.exp(-w['gamma']*d)@w['dual']+w['bias'])
                h=np.concatenate(out).reshape(len(bank),4,1)
            elif kind=='set_nn':
                h=np.maximum(h@w['w0']+w['b0'],0)
                valid=equal.sum(-1,keepdims=True)>0
                avg=(h*valid).sum(1,keepdims=True)/np.maximum(valid.sum(1,keepdims=True),1)
                maximum=np.where(valid,h,-1e9).max(1,keepdims=True)
                same=equal@h/np.maximum(equal.sum(-1,keepdims=True),1)
                context=np.concatenate([h,np.broadcast_to(avg,h.shape),h-avg,h*avg,same,
                                        np.broadcast_to(maximum,h.shape)],axis=-1)
                h=np.maximum(context@w['w1']+w['b1'],0)@w['w2']+w['b2']
            else:
                for i in range(3):
                    h=h@w[f'w{i}']+w[f'b{i}']
                    if i<2:
                        h=np.maximum(h,0)
        s=h[...,0]/member.get('temperature',1.)
        weight=member.get('weight',1.)
        total+=weight*s
        family=Path(member['file']).stem.rsplit('_',1)[0]
        family_totals[family]=family_totals.get(family,0)+weight*s
        family_weights[family]=family_weights.get(family,0)+weight
    if 'fusion' in config:
        fs=[family_totals[f]/family_weights[f] for f in config['fusion']['families']]
        with np.load(root/config['fusion']['file'],allow_pickle=False) as w:
            total=tree_predict(fusion_features(fs,equal),w)
    else:
        total=aggregate(total,equal,config.get('answer_aggregation',0.))
    if 'arithmetic_penalty' in config:
        c=config['arithmetic_penalty']
        if c.get('version')=='v2':
            from arithmetic_v2 import evidence
            ev=evidence(bank);p=ev[...,c['column']]+c.get('symbol_weight',0.)*ev[...,3]
        else:
            p=arithmetic_evidence(bank)[...,c['column']]
            if c['column'] in (3,4):p=np.log1p(10*p)
        total-=c['weight']*aggregate(p,equal,c['pooling'])
    total=np.where(equal.sum(-1)>0,total,-1e9)
    return total


def predict(bank, arrays, checkpoint):
    """Return one native Python candidate index for every provided question."""
    return [int(i) for i in score_bank(bank, arrays, checkpoint).argmax(1)]

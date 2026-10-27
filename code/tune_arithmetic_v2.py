"""Choose arithmetic-v2 penalty using cached training-only OOF scores."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json
from pathlib import Path
import numpy as np
from common import load_inputs
from features import aggregate,text_features,arithmetic_evidence
from arithmetic_v2 import evidence
from train import labels,metrics
from train_cv import hard_mask


def main():
    root=Path('experiments/cross_validation');out=Path('experiments/optimization85');out.mkdir(exist_ok=True)
    bank,_=load_inputs('train');use=hard_mask(bank);y=labels('train',bank)[use];_,eq=text_features(bank);eq=eq[use]
    choice=json.loads((root/'selected.json').read_text())
    raw=np.mean([np.load(root/(f['family']['name']+'-oof.npy'))/f['temperature'] for f in choice['families']],0)
    base=aggregate(raw,eq,choice['pooling']);ev=evidence(bank)[use];old=arithmetic_evidence(bank)[use]
    rows=[]
    # The previous rule remains a candidate, evaluated under the same OOF proxy.
    for version in ('old','v2'):
        for column in (1,2):
            for symbol_weight in ((0.,.5,1.) if version=='v2' else (0.,)):
                p=(ev if version=='v2' else old)[...,column]
                if version=='v2':p=p+symbol_weight*ev[...,3]
                for pooling in (0.,.5,1.):
                    for weight in (0.,.1,.25,.5,1.):
                        scores=np.where(eq.sum(-1)>0,base-weight*aggregate(p,eq,pooling),-1e9)
                        rows.append(dict(version=version,column=column,symbol_weight=symbol_weight,pooling=pooling,weight=weight,
                                         **metrics(scores,y,np.ones(len(y),bool))))
    best=max(rows,key=lambda r:r['correct'])
    record=dict(selected=best,search=rows,correct_mean_evidence=ev[y.astype(bool)].mean(0).tolist(),wrong_mean_evidence=ev[~y.astype(bool)].mean(0).tolist())
    (out/'arithmetic_v2.json').write_text(json.dumps(record,indent=2)+'\n')
    print('TRAIN OOF SELECTED',best)
    # Validation is a single check after training-OOF rule selection.
    from submission import score_bank
    vb,va=load_inputs('validation');vy=labels('validation',vb);_,veq=text_features(vb)
    scores=score_bank(vb,va,Path('code/checkpoints/final'))+.5*arithmetic_evidence(vb)[...,1]
    if best['version']=='v2':
        v=evidence(vb);p=v[...,best['column']]+best['symbol_weight']*v[...,3]
    else:p=arithmetic_evidence(vb)[...,best['column']]
    scores-=best['weight']*aggregate(p,veq,best['pooling'])
    record['validation']=metrics(scores,vy,hard_mask(vb))
    (out/'arithmetic_v2.json').write_text(json.dumps(record,indent=2)+'\n')
    print('VALIDATION',record['validation'])


if __name__=='__main__':main()

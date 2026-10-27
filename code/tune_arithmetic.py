"""Select a small arithmetic consistency penalty using train OOF scores only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json
from pathlib import Path
import numpy as np
from common import load_inputs
from features import arithmetic_evidence,text_features,aggregate
from train import labels,metrics
from train_cv import hard_mask

def main():
    b,_=load_inputs('train');use=hard_mask(b);y=labels('train',b)[use];_,eq=text_features(b);eq=eq[use]
    choice=json.loads(Path('experiments/cross_validation/selected.json').read_text())
    family_scores=[np.load(Path('experiments/cross_validation')/(f['family']['name']+'-oof.npy'))/f['temperature'] for f in choice['families']]
    scores=aggregate(np.mean(family_scores,0),eq,choice['pooling'])
    evidence=arithmetic_evidence(b)[use];rows=[];best=None
    for column in (1,2,3,4):
        penalty=evidence[...,column]
        if column in (3,4):penalty=np.log1p(10*penalty)
        for pooling in (0.,.5,1.):
            p=aggregate(penalty,eq,pooling)
            for weight in (0.,.1,.25,.5,1.,2.):
                s=np.where(eq.sum(-1)>0,scores-weight*p,-1e9)
                r={'column':column,'pooling':pooling,'weight':weight,**metrics(s,y,np.ones(len(y),bool))}
                rows.append(r)
                if best is None or r['accuracy']>best['accuracy']:best=r
    Path('experiments/cross_validation/arithmetic_penalty.json').write_text(json.dumps({'selected':best,'search':rows,
        'correct_mean_evidence':evidence[y.astype(bool)].mean(0).tolist(),
        'wrong_mean_evidence':evidence[~y.astype(bool)].mean(0).tolist()},indent=2)+'\n')
    print(best)

if __name__=='__main__':main()

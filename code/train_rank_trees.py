"""GPU query-ranking trees evaluated by grouped train-only OOF predictions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRanker,Pool
from common import load_inputs
from features import build_features
from train import labels,metrics
from train_cv import hard_mask

def main():
    tick=time.perf_counter();b,a=load_inputs('train');use=hard_mask(b);y=labels('train',b)[use]
    x,eq=build_features(b,a,'full');x=x[use];eq=eq[use]
    oof=np.zeros((4,len(y),4),np.float32);steps=[150,300,500,700]
    models=[]
    for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=2026).split(y)):
        model=CatBoostRanker(iterations=700,depth=4,learning_rate=.05,l2_leaf_reg=10,loss_function='YetiRank',
            task_type='GPU',border_count=32,random_seed=42+fold,thread_count=4,verbose=200,allow_writing_files=False)
        train=Pool(x[tr].reshape(-1,x.shape[-1]),y[tr].ravel(),group_id=np.repeat(np.arange(len(tr)),4))
        model.fit(train)
        for j,step in enumerate(steps):oof[j,va]=model.predict(x[va].reshape(-1,x.shape[-1]),ntree_end=step).reshape(-1,4)
        print('FOLD COMPLETE',fold+1,flush=True)
    out=Path('experiments/cross_validation');np.save(out/'rank_trees_oof.npy',oof)
    rows=[]
    from features import aggregate
    for i,step in enumerate(steps):
        for w in (0.,.5,1.):
            s=np.where(eq.sum(-1)>0,aggregate(oof[i],eq,w),-1e9)
            r={'trees':step,'pooling':w,**metrics(s,y,np.ones(len(y),bool))};rows.append(r);print(r,flush=True)
    (out/'rank_trees.json').write_text(json.dumps({'seconds':time.perf_counter()-tick,'results':rows},indent=2)+'\n')

if __name__=='__main__':main()

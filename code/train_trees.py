"""Candidate-wise boosted-tree ablation; train/validation only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import argparse,json,time
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
from common import load_inputs,parse_answer
from features import build_features
from train import labels,metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',default='style');ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--depth',type=int,default=5);ap.add_argument('--device',choices=['CPU','GPU'],default='CPU');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
    b,a=load_inputs('train');v,va=load_inputs('validation');x,_=build_features(b,a,args.mode);vx,eq=build_features(v,va,args.mode)
    y=labels('train',b);vy=labels('validation',v)
    hard=np.array([len({parse_answer(c['text']) for c in r['candidates']}-{None})>=2 for r in v])
    use=(y.min(1)<1)
    model=CatBoostClassifier(iterations=1500,depth=args.depth,learning_rate=.035,l2_leaf_reg=5,thread_count=4,
                             random_seed=args.seed,verbose=200,allow_writing_files=False,task_type=args.device)
    model.fit(x[use].reshape(-1,x.shape[-1]),y[use].ravel())
    best=-1;record=[]
    for n in range(100,1501,100):
        s=model.predict(vx.reshape(-1,vx.shape[-1]),prediction_type='RawFormulaVal',ntree_end=n).reshape(-1,4)
        m=metrics(np.where(eq.sum(-1)>0,s,-1e9),vy,hard);record.append({'trees':n,**m});print(n,m,flush=True)
        if m['hard_accuracy']>best:
            best=m['hard_accuracy'];np.save(args.out/'validation_logits.npy',s)
            frozen=model.copy();frozen.shrink(n);frozen.save_model(str(args.out/'trees.cbm'))
            c={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()};c.update(trees=n,validation=m,kind='tree')
            (args.out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (args.out/'history.json').write_text(json.dumps({'seconds':time.perf_counter()-tick,'epochs':record},indent=2)+'\n')

if __name__=='__main__':main()

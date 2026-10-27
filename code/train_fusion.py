"""Train a compact meta verifier on grouped out-of-fold training scores only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from common import load_inputs
from features import text_features,fusion_features,tree_predict
from train import labels,metrics
from train_cv import hard_mask

def main():
    t=time.perf_counter();b,_=load_inputs('train');use=hard_mask(b);y=labels('train',b)[use];_,eq=text_features(b);eq=eq[use]
    choice=json.loads(Path('experiments/cross_validation/selected.json').read_text())
    family_scores=[np.load(Path('experiments/cross_validation')/(f['family']['name']+'-oof.npy'))/f['temperature'] for f in choice['families']]
    x=fusion_features(family_scores,eq);valid=eq.sum(-1)>0
    records=[];best=None
    for depth in (2,3):
        for iterations in (150,400,800):
            scores=np.zeros_like(y)
            for train,val in KFold(5,shuffle=True,random_state=2027).split(y):
                model=CatBoostClassifier(iterations=iterations,depth=depth,learning_rate=.035,l2_leaf_reg=20,
                    random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
                model.fit(x[train][valid[train]],y[train][valid[train]])
                scores[val]=model.predict(x[val].reshape(-1,x.shape[-1]),prediction_type='RawFormulaVal').reshape(-1,4)
            result=metrics(np.where(valid,scores,-1e9),y,np.ones(len(y),bool))
            record={'depth':depth,'iterations':iterations,**result};records.append(record);print(record,flush=True)
            if best is None or record['accuracy']>best['accuracy']:best=record
    model=CatBoostClassifier(iterations=best['iterations'],depth=best['depth'],learning_rate=.035,l2_leaf_reg=20,
        random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
    model.fit(x[valid],y[valid]);out=Path('code/checkpoints/cv_augmented')
    export=Path('runs/fusion.json');model.save_model(str(export),format='json');data=json.loads(export.read_text())
    trees=data['oblivious_trees'];features=np.zeros((len(trees),3),np.int32);borders=np.full((len(trees),3),np.inf,np.float32)
    leaves=np.zeros((len(trees),8),np.float32)
    mapping={i:f['flat_feature_index'] for i,f in enumerate(data['features_info']['float_features'])}
    for i,tree in enumerate(trees):
        for j,split in enumerate(tree['splits'] or []):
            features[i,j]=mapping[split['float_feature_index']];borders[i,j]=split['border']
        leaves[i,:len(tree['leaf_values'])]=tree['leaf_values']
    np.savez(out/'fusion.npz',features=features,borders=borders,leaves=leaves,
             scale_tree=np.float32(data['scale_and_bias'][0]),bias_tree=np.float32(data['scale_and_bias'][1][0]))
    with np.load(out/'fusion.npz') as w:
        ss=tree_predict(x,w)
    exact=model.predict(x.reshape(-1,x.shape[-1]),prediction_type='RawFormulaVal').reshape(-1,4)
    error=float(abs(ss-exact).max());assert error<1e-4,error
    config=json.loads((out/'ensemble.json').read_text())
    config['fusion']={'file':'fusion.npz','families':[f['family']['name'] for f in choice['families']],
                      'selection':best,'training':'Grouped OOF train scores; no validation or test references.'}
    (out/'ensemble.json').write_text(json.dumps(config,indent=2)+'\n')
    Path('experiments/cross_validation/fusion_training.json').write_text(json.dumps({'candidates':records,'selected':best,
        'seconds':time.perf_counter()-t,'export_error':error,
        'caveat':'Meta cross-validation reuses base OOF scores whose epoch choices were made on these train groups. It is a development estimate, not nested unbiased validation.'},indent=2)+'\n')

if __name__=='__main__':main()

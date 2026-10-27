"""Validation-only baseline, ensemble search, and freezing. Never reads test gold."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='4'
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from common import load_inputs,parse_answer
from features import aggregate,text_features,arithmetic_evidence
from train import labels,metrics

def asset_name(config):
    return {'tree':'trees.cbm','kernel':'kernel.npz','lexical':'lexical.joblib'}.get(config.get('kind'),'weights.npz')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--freeze',action='store_true'); args=ap.parse_args()
    bank,_=load_inputs('validation'); y=labels('validation',bank)
    _,eq=text_features(bank)
    arithmetic=json.loads(Path('experiments/cross_validation/arithmetic_penalty.json').read_text())['selected']
    penalty=arithmetic_evidence(bank)[...,arithmetic['column']]
    if arithmetic['column'] in (3,4):penalty=np.log1p(10*penalty)
    penalty=arithmetic['weight']*aggregate(penalty,eq,arithmetic['pooling'])
    hard=np.array([len({parse_answer(c['text']) for c in r['candidates']}-{None})>=2 for r in bank])
    paths=sorted(p for p in Path('runs').glob('*/validation_logits.npy') if (p.parent/'history.json').exists() and not p.parent.name.startswith('reproduce_'))
    scores=[]; configs=[]; results=[]
    results.append({'name':'first',**metrics(np.tile([1,0,0,0],(len(y),1)),y,hard)})
    results.append({'name':'majority',**metrics(eq.sum(-1),y,hard)})
    for p in paths:
        s=np.load(p); c=json.loads((p.parent/'config.json').read_text())
        # Remove arbitrary score scale before equal-weight ensembling.
        temp=float(np.std(s-s.mean(1,keepdims=True)))
        scores.append(s/temp); configs.append((p,c,temp))
        results.append({'name':p.parent.name,'seconds':json.loads((p.parent/'history.json').read_text())['seconds'],**metrics(np.where(eq.sum(-1)>0,s,-1e9),y,hard)})
    stack=np.array(scores)
    best=None
    # Fixed model families and a short interpretable aggregation grid.
    families={
        'full_rank':[i for i,(p,c,t) in enumerate(configs) if c.get('mode')=='full' and c.get('loss')=='rank'],
        'full_mixed':[i for i,(p,c,t) in enumerate(configs) if c.get('loss')=='mixed'],
        'full_all':[i for i,(p,c,t) in enumerate(configs) if c.get('mode')=='full'],
        'all_embeddings':[i for i,(p,c,t) in enumerate(configs) if c.get('mode')!='style' and c.get('kind')!='tree'],
        'enhanced':[i for i,(p,c,t) in enumerate(configs) if c.get('mode')=='enhanced'],
        'bce':[i for i,(p,c,t) in enumerate(configs) if c.get('loss')=='bce'],
        'small':[i for i,(p,c,t) in enumerate(configs) if c.get('width')==64],
        'mixed_bce':[i for i,(p,c,t) in enumerate(configs) if c.get('mode')=='full' and c.get('loss') in ('mixed','bce')],
        'rank_mixed128':[i for i,(p,c,t) in enumerate(configs) if c.get('mode')=='full' and c.get('width')==128 and c.get('loss') in ('mixed','rank')],
        'hard':[i for i,(p,c,t) in enumerate(configs) if p.parent.name.startswith('hard')],
        'context':[i for i,(p,c,t) in enumerate(configs) if p.parent.name.startswith('context')],
        'hard_context':[i for i,(p,c,t) in enumerate(configs) if p.parent.name.startswith(('hard','context'))],
    }
    families.update({p.parent.name:[i] for i,(p,c,t) in enumerate(configs) if c.get('kind')!='tree'})
    for family,inds in families.items():
        inds=[i for i in inds if configs[i][1].get('kind') not in ('tree','kernel','lexical')]
        if not inds: continue
        if sum((configs[i][0].parent/asset_name(configs[i][1])).stat().st_size for i in inds)>16*1024**2-32000:continue
        s=stack[inds].mean(0)
        for weight in (0.,.25,.5,.75,1.):
            ss=aggregate(s,eq,weight)-penalty; ss=np.where(eq.sum(-1)>0,ss,-1e9); m=metrics(ss,y,hard)
            r={'name':family,'answer_aggregation':weight,'members':[configs[i][0].parent.name for i in inds],**m}
            results.append(r)
            # First best wins; test is deliberately absent from this selection process.
            key=(m['hard_accuracy'],m['accuracy'],-len(inds))
            if best is None or key>best[0]: best=(key,r,inds,ss)
    # Forward selection is validation tuning, with at most six equal-weight models.
    candidates=list(range(len(configs)))
    chosen=[];current=-1
    for step in range(6):
        proposal=None
        for i in candidates:
            if i in chosen:continue
            inds=chosen+[i]
            if sum((configs[j][0].parent/asset_name(configs[j][1])).stat().st_size for j in inds)>16*1024**2-32000:continue
            for w in (0.,.25,.5,.75,1.):
                ss=aggregate(stack[inds].mean(0),eq,w)-penalty;ss=np.where(eq.sum(-1)>0,ss,-1e9)
                m=metrics(ss,y,hard)
                r={'name':f'forward_step{step+1}','answer_aggregation':w,'members':[configs[j][0].parent.name for j in inds],**m}
                results.append(r);key=(m['hard_accuracy'],m['accuracy'],-len(inds))
                if proposal is None or key>proposal[0]:proposal=(key,r,inds,ss)
        if proposal is None or proposal[0][0]<=current:break
        chosen=proposal[2];current=proposal[0][0]
        if best is None or proposal[0]>best[0]:best=proposal
    Path('experiments').mkdir(exist_ok=True)
    Path('experiments/validation_results.json').write_text(json.dumps(results,indent=2)+'\n')
    for r in results: print(json.dumps(r))
    print('SELECTED',json.dumps(best[1]))
    if args.freeze:
        out=Path('code/checkpoints/final');out.mkdir(parents=True,exist_ok=True)
        members=[]
        for i in best[2]:
            p,c,temp=configs[i]; asset=asset_name(c);name=p.parent.name+Path(asset).suffix
            shutil.copyfile(p.parent/asset,out/name)
            members.append({'file':name,'kind':c.get('kind','nn'),'mode':c['mode'],'temperature':temp,'weight':1/len(best[2])})
        config={'members':members,'answer_aggregation':best[1]['answer_aggregation'],'selection':best[1],'arithmetic_penalty':arithmetic}
        (out/'ensemble.json').write_text(json.dumps(config,indent=2)+'\n')
        needed={m['file'] for m in members}|{'ensemble.json'}
        for p in out.iterdir():
            if p.is_file() and p.name not in needed:p.unlink()
        print('FROZEN',sum(p.stat().st_size for p in out.iterdir()),'bytes')


if __name__=='__main__':main()

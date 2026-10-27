"""Train only on train references; select using validation (especially hard groups)."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from common import DATA, load_inputs, read_jsonl, parse_answer
from features import build_features, aggregate


def labels(split, bank):
    if split not in ('train','validation'):
        raise ValueError('Training/development may never open test references')
    refs = read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert [r['id'] for r in refs] == [r['id'] for r in bank]
    return np.array([[parse_answer(c['text']) == parse_answer(r['answer'])
                      and parse_answer(c['text']) is not None for c in b['candidates']]
                     for b,r in zip(bank,refs)],np.float32)


def metrics(scores,y,hard):
    chosen = y[np.arange(len(y)),scores.argmax(1)]
    return {'accuracy':100*int(chosen.sum())/len(y),'hard_accuracy':100*int(chosen[hard].sum())/int(hard.sum()),
            'hard_n':int(hard.sum()),'correct':int(chosen.sum())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',default='full',choices=['full','relative','embedding','style','enhanced','answer','interaction'])
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--width',type=int,default=128)
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--decay',type=float,default=.01)
    ap.add_argument('--dropout',type=float,default=.2)
    ap.add_argument('--loss',default='rank',choices=['rank','bce','mixed','list','mixedlist'])
    ap.add_argument('--save-epoch',type=int,help='Reproduce a previously frozen epoch instead of selecting again')
    ap.add_argument('--hard-only',action='store_true',help='Train on groups with at least two valid distinct answers')
    ap.add_argument('--valid-only',action='store_true',help='Exclude unparseable candidates from the training loss')
    ap.add_argument('--ema',type=float,default=0.,help='Exponential moving average of parameters after each update')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    import torch
    from torch import nn
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    tick=time.perf_counter()
    train,ta=load_inputs('train'); val,va=load_inputs('validation')
    x,eq=build_features(train,ta,args.mode); vx,veq=build_features(val,va,args.mode)
    y=labels('train',train); vy=labels('validation',val)
    hard=(veq.sum((1,2))<16) & ((veq.sum(-1)>0).sum(-1)>=2)
    # Match the test input condition exactly: at least two distinct valid parsed answers.
    hard=np.array([len({parse_answer(c['text']) for c in r['candidates']} - {None})>=2 for r in val])
    mean=x.reshape(-1,x.shape[-1]).mean(0)
    scale=x.reshape(-1,x.shape[-1]).std(0).clip(.15)
    x=((x-mean)/scale).clip(-10,10); vx=((vx-mean)/scale).clip(-10,10)
    net=nn.Sequential(nn.Linear(x.shape[-1],args.width),nn.ReLU(),nn.Dropout(args.dropout),
                      nn.Linear(args.width,32),nn.ReLU(),nn.Dropout(args.dropout),nn.Linear(32,1))
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=args.decay)
    averaged=copy.deepcopy(net) if args.ema else net
    tx=torch.from_numpy(x); ty=torch.from_numpy(y); tv=torch.from_numpy(vx)
    active=np.where(y.min(1)<1)[0]
    if args.hard_only:
        active=np.array([i for i,r in enumerate(train) if len({parse_answer(c['text']) for c in r['candidates']}-{None})>=2])
    valid=torch.from_numpy((eq.sum(-1)>0).astype(np.float32))
    records=[]; best=-1
    for epoch in range(1,args.epochs+1):
        net.train()
        order=np.random.permutation(active)
        losses=[]
        for inds in np.array_split(order,max(1,len(order)//128)):
            s=net(tx[inds]).squeeze(-1); target=ty[inds]
            ranked=s.masked_fill(valid[inds]==0,-1e9) if args.valid_only else s
            rank=(torch.logsumexp(ranked,1)-torch.logsumexp(ranked.masked_fill(target==0,-1e9),1)).mean()
            losses_bce=nn.functional.binary_cross_entropy_with_logits(s,target,reduction='none')
            bce=(losses_bce*valid[inds]).sum()/valid[inds].sum() if args.valid_only else losses_bce.mean()
            listloss=-(target/target.sum(1,keepdim=True)*torch.log_softmax(ranked,1)).sum(1).mean()
            loss={'rank':rank,'bce':bce,'mixed':rank+.5*bce,'list':listloss,'mixedlist':listloss+bce}[args.loss]
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
            if args.ema:
                with torch.no_grad():
                    for moving,current in zip(averaged.parameters(),net.parameters()):
                        moving.mul_(args.ema).add_(current,alpha=1-args.ema)
        averaged.eval()
        with torch.no_grad():
            s=averaged(tv).squeeze(-1).numpy()
        m=metrics(np.where(veq.sum(-1)>0,s,-1e9),vy,hard)
        records.append({'epoch':epoch,'loss':float(np.mean(losses)),**m})
        if (epoch==args.save_epoch) if args.save_epoch else (m['hard_accuracy']>best):
            best=m['hard_accuracy']
            weights={'mean':mean,'scale':scale}
            for i,j in enumerate((0,3,6)):
                weights[f'w{i}']=averaged[j].weight.detach().numpy().T
                weights[f'b{i}']=averaged[j].bias.detach().numpy()
            np.savez(args.out/'weights.npz',**weights)
            np.save(args.out/'validation_logits.npy',s)
            config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
            config.update(best_epoch=epoch,validation=m)
            (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        if epoch%10==0 or epoch==1:
            print(json.dumps({'epoch':epoch,'best_hard':best,**m}),flush=True)
    record={'config':json.loads((args.out/'config.json').read_text()),'seconds':time.perf_counter()-tick,'epochs':records}
    (args.out/'history.json').write_text(json.dumps(record,indent=2)+'\n')
    print('FINISHED',args.out,record['seconds'],flush=True)


if __name__=='__main__':
    main()

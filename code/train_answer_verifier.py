"""Training-only CV for answer-balanced objectives and SAM regularization."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from common import load_inputs
from features import build_features,aggregate
from train import labels,metrics
from train_cv import hard_mask

FAMILIES=[dict(name=name,mode=mode,width=width,objective=objective,rho=rho)
          for name,mode,width,objective,rho in (
              ('answer_full','full',128,'balanced',0.),
              ('answer_context','interaction',64,'balanced',0.),
              ('answer_list_full','full',128,'answer_list',0.),
              ('answer_list_context','interaction',64,'answer_list',0.),
              ('sam_full','full',128,'balanced',.05),
              ('sam_context','interaction',64,'balanced',.05))]


def fit(x,y,eq,vx,c,seed,epochs,callback=None):
    import torch
    from torch import nn
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    mean=x.reshape(-1,x.shape[-1]).mean(0);scale=x.reshape(-1,x.shape[-1]).std(0).clip(.15)
    def tensor(a):return torch.as_tensor(a,device='cuda')
    tx=tensor(((x-mean)/scale).clip(-10,10));ty=tensor(y);te=tensor(eq)
    tv=tensor(((vx-mean)/scale).clip(-10,10))
    net=nn.Sequential(nn.Linear(x.shape[-1],c['width']),nn.ReLU(),nn.Dropout(.4),
                      nn.Linear(c['width'],32),nn.ReLU(),nn.Dropout(.4),nn.Linear(32,1)).cuda()
    averaged=copy.deepcopy(net);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    for epoch in range(epochs):
        net.train()
        for batch in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
            ix=tensor(batch);target=ty[ix];equal=te[ix];count=equal.sum(-1)
            weights=(count>0)/count.clamp(min=1)
            def objective():
                s=net(tx[ix]).squeeze(-1)
                bce=nn.functional.binary_cross_entropy_with_logits(s,target,reduction='none')
                loss=((bce*weights).sum(1)/weights.sum(1)).mean()
                if c['objective']=='answer_list':
                    pooled=(equal@s.unsqueeze(-1)).squeeze(-1)/count.clamp(min=1)
                    adjusted=(pooled-count.clamp(min=1).log()).masked_fill(count==0,-1e9)
                    loss=loss+.5*(torch.logsumexp(adjusted,1)-torch.logsumexp(adjusted.masked_fill(target==0,-1e9),1)).mean()
                return loss
            opt.zero_grad();loss=objective();loss.backward()
            if c['rho']:
                # Two-pass sharpness-aware minimization. Dropout is stochastic
                # in both passes; EMA is updated only after the optimizer step.
                norm=torch.stack([p.grad.norm(2) for p in net.parameters()]).norm(2)
                shifts=[]
                with torch.no_grad():
                    for p in net.parameters():
                        shift=p.grad*c['rho']/(norm+1e-12);p.add_(shift);shifts.append(shift)
                opt.zero_grad();objective().backward()
                with torch.no_grad():
                    for p,shift in zip(net.parameters(),shifts):p.sub_(shift)
            opt.step()
            with torch.no_grad():
                for a,b in zip(averaged.parameters(),net.parameters()):a.mul_(.99).add_(b,alpha=.01)
        if callback is not None or epoch==epochs-1:
            averaged.eval()
            with torch.no_grad():s=averaged(tv).squeeze(-1).cpu().numpy()
            if callback is not None:callback(epoch,s)
    w=dict(mean=mean,scale=scale)
    for i,j in enumerate((0,3,6)):
        w[f'w{i}']=averaged[j].weight.detach().cpu().numpy().T.copy();w[f'b{i}']=averaged[j].bias.detach().cpu().numpy().copy()
    return w,s


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--out',type=Path,default=Path('runs/optimization85/answer'));a=ap.parse_args()
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for training')
    a.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    b,arrays=load_inputs('train');use=hard_mask(b);y=labels('train',b)[use];cache={}
    for mode in ('full','interaction'):
        x,eq=build_features(b,arrays,mode);cache[mode]=(x[use],eq[use])
    eq=cache['full'][1];results=[]
    for c in FAMILIES:
        t=time.perf_counter();x=cache[c['mode']][0];path=a.out/(c['name']+'-curves.npy')
        if path.exists():oof=np.load(path)
        else:
            oof=np.zeros((a.epochs,len(y),4),np.float32)
            for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=2026).split(y)):
                def collect(epoch,s):oof[epoch,va]=s
                fit(x[tr],y[tr],eq[tr],x[va],c,42+fold,a.epochs,collect)
                print(c['name'],'fold',fold+1,'complete',flush=True)
            np.save(path,oof)
        for pooling in (0.,.5,1.):
            counts=[int(y[np.arange(len(y)),np.where(eq.sum(-1)>0,aggregate(s,eq,pooling),-1e9).argmax(1)].sum()) for s in oof]
            smooth=np.convolve(counts,np.ones(9)/9,'valid');epoch=int(smooth.argmax())+5
            s=oof[epoch-1];temp=float((s-s.mean(1,keepdims=True)).std())
            r=dict(family=c,epoch=epoch,pooling=pooling,temperature=temp,n=len(y),oof_accuracy=100*counts[epoch-1]/len(y),
                   smoothed_oof_accuracy=100*float(smooth.max())/len(y),seconds=time.perf_counter()-t,correct_by_epoch=counts)
            results.append(r);print('OOF',c['name'],pooling,epoch,r['smoothed_oof_accuracy'],flush=True)
    (a.out/'search.json').write_text(json.dumps(results,indent=2)+'\n')
    best=max(results,key=lambda r:r['smoothed_oof_accuracy']);(a.out/'selected.json').write_text(json.dumps(best,indent=2)+'\n')
    print('SELECTED',json.dumps({k:v for k,v in best.items() if k!='correct_by_epoch'}),flush=True)
    c=best['family'];vb,va=load_inputs('validation');vx,veq=build_features(vb,va,c['mode']);vy=labels('validation',vb)
    checkpoint=a.out/'checkpoint';checkpoint.mkdir(exist_ok=True);members=[];scores=[]
    for seed in (17,42,73):
        w,s=fit(cache[c['mode']][0],y,eq,vx,c,seed,best['epoch'])
        name=f'{c["name"]}_{seed}.npz';np.savez(checkpoint/name,**w);scores.append(s/best['temperature'])
        members.append(dict(kind='nn',mode=c['mode'],file=name,temperature=best['temperature'],weight=1/3));print('REFIT',seed,flush=True)
    combined=np.mean(scores,axis=0);np.save(a.out/'validation_logits.npy',combined)
    m=metrics(np.where(veq.sum(-1)>0,aggregate(combined,veq,best['pooling']),-1e9),vy,hard_mask(vb))
    (checkpoint/'ensemble.json').write_text(json.dumps(dict(members=members,answer_aggregation=best['pooling'],selection=best,validation=m),indent=2)+'\n')
    (a.out/'resources.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(0),torch=torch.__version__,folds=5,split_seed=2026,refit_seeds=[17,42,73]),indent=2)+'\n')
    print('VALIDATION',m,flush=True)


if __name__=='__main__':main()

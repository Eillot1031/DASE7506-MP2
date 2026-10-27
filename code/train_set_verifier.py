"""GPU DeepSets verifier: select using training-group CV, then refit once.

Only train/validation inputs and labels are opened. Candidate order is never a
feature. Exported models use NumPy and fit within the course asset budget.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='4'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, copy, json, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from common import load_inputs
from features import build_features, aggregate
from train import labels, metrics
from train_cv import hard_mask

FAMILIES=[
    dict(name='set_bce64',mode='full',width=64,loss='bce',dropout=.4,ema=.99,decay=.01),
    dict(name='set_bce128',mode='full',width=128,loss='bce',dropout=.4,ema=.99,decay=.01),
    dict(name='set_pair64',mode='full',width=64,loss='pair',dropout=.4,ema=0.,decay=.1),
    dict(name='set_list64',mode='full',width=64,loss='mixedlist',dropout=.4,ema=0.,decay=.1),
    dict(name='set_context64',mode='interaction',width=64,loss='bce',dropout=.4,ema=.99,decay=.01),
    dict(name='set_norm128',mode='norm',width=128,loss='bce',dropout=.25,ema=.9,decay=.01),
]


def fit(x,y,eq,vx,veq,c,seed,epochs,callback=None):
    import torch
    from torch import nn
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    mean=x.reshape(-1,x.shape[-1]).mean(0)
    scale=x.reshape(-1,x.shape[-1]).std(0).clip(.15)
    def tensor(a):return torch.as_tensor(a,device='cuda')
    tx=tensor(((x-mean)/scale).clip(-10,10));ty=tensor(y);te=tensor(eq)
    tv=tensor(((vx-mean)/scale).clip(-10,10));tve=tensor(veq)
    class SetNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Linear(x.shape[-1],c['width'])
            self.hidden=nn.Linear(6*c['width'],32)
            self.output=nn.Linear(32,1)
        def forward(self,z,e):
            h=nn.functional.dropout(torch.relu(self.encoder(z)),c['dropout'],self.training)
            valid=(e.sum(-1,keepdim=True)>0)
            avg=(h*valid).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp(min=1)
            maximum=h.masked_fill(~valid,-1e9).max(1,keepdim=True).values
            same=e@h/e.sum(-1,keepdim=True).clamp(min=1)
            context=torch.cat([h,avg.expand_as(h),h-avg,h*avg,same,maximum.expand_as(h)],-1)
            h=nn.functional.dropout(torch.relu(self.hidden(context)),c['dropout'],self.training)
            return self.output(h).squeeze(-1)
    net=SetNet().cuda();averaged=copy.deepcopy(net) if c['ema'] else net
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=c['decay'])
    for epoch in range(epochs):
        net.train()
        for batch in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
            ix=tensor(batch);s=net(tx[ix],te[ix]);target=ty[ix]
            mask=(te[ix].sum(-1)>0).float()
            bce=(nn.functional.binary_cross_entropy_with_logits(s,target,reduction='none')*mask).sum()/mask.sum()
            if c['loss']=='pair':
                pair=target[:,:,None]*(1-target[:,None,:])*mask[:,None,:]
                loss=(nn.functional.softplus(s[:,None,:]-s[:,:,None])*pair).sum()/pair.sum().clamp(min=1)+.25*bce
            elif c['loss']=='mixedlist':
                loss=bce-(target/target.sum(1,keepdim=True)*torch.log_softmax(s.masked_fill(mask==0,-1e9),1)).sum(1).mean()
            else:loss=bce
            opt.zero_grad();loss.backward();opt.step()
            if c['ema']:
                with torch.no_grad():
                    for a,b in zip(averaged.parameters(),net.parameters()):a.mul_(c['ema']).add_(b,alpha=1-c['ema'])
        if callback is not None or epoch==epochs-1:
            averaged.eval()
            with torch.no_grad():s=averaged(tv,tve).cpu().numpy()
            if callback is not None:callback(epoch,s)
    weights=dict(mean=mean,scale=scale)
    for i,layer in enumerate((averaged.encoder,averaged.hidden,averaged.output)):
        weights[f'w{i}']=layer.weight.detach().cpu().numpy().T.copy()
        weights[f'b{i}']=layer.bias.detach().cpu().numpy().copy()
    return weights,s


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--stage',choices=['search','refit','both'],default='both')
    ap.add_argument('--out',type=Path,default=Path('runs/optimization85/set'));a=ap.parse_args()
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this development script')
    a.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    bank,arrays=load_inputs('train');use=hard_mask(bank);y=labels('train',bank)[use]
    cache={}
    for mode in dict.fromkeys(c['mode'] for c in FAMILIES):
        x,eq=build_features(bank,arrays,mode);cache[mode]=(x[use],eq[use])
    eq=cache['full'][1]
    if a.stage in ('search','both'):
        results=[]
        for c in FAMILIES:
            t=time.perf_counter();x=cache[c['mode']][0];path=a.out/(c['name']+'-curves.npy')
            if path.exists():oof=np.load(path)
            else:
                oof=np.zeros((a.epochs,len(y),4),np.float32)
                for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=2026).split(y)):
                    def collect(epoch,s):oof[epoch,va]=s
                    fit(x[tr],y[tr],eq[tr],x[va],eq[va],c,42+fold,a.epochs,collect)
                    print(c['name'],'fold',fold+1,'complete',flush=True)
                np.save(path,oof)
            for pooling in (0.,.5,1.):
                counts=[int(y[np.arange(len(y)),np.where(eq.sum(-1)>0,aggregate(s,eq,pooling),-1e9).argmax(1)].sum()) for s in oof]
                smooth=np.convolve(counts,np.ones(9)/9,'valid');epoch=int(smooth.argmax())+5
                s=oof[epoch-1];temp=float((s-s.mean(1,keepdims=True)).std())
                r=dict(family=c,epoch=epoch,pooling=pooling,temperature=temp,n=len(y),
                       oof_accuracy=100*counts[epoch-1]/len(y),smoothed_oof_accuracy=100*float(smooth.max())/len(y),
                       seconds=time.perf_counter()-t,correct_by_epoch=counts)
                results.append(r);print('OOF',c['name'],pooling,epoch,r['smoothed_oof_accuracy'],flush=True)
        (a.out/'search.json').write_text(json.dumps(results,indent=2)+'\n')
        best=max(results,key=lambda r:r['smoothed_oof_accuracy'])
        (a.out/'selected.json').write_text(json.dumps(best,indent=2)+'\n')
        print('SELECTED',json.dumps({k:v for k,v in best.items() if k!='correct_by_epoch'}),flush=True)
    if a.stage in ('refit','both'):
        selected=json.loads((a.out/'selected.json').read_text());c=selected['family']
        vb,va=load_inputs('validation');vx,veq=build_features(vb,va,c['mode']);vy=labels('validation',vb)
        checkpoint=a.out/'checkpoint';checkpoint.mkdir(exist_ok=True);members=[];scores=[]
        for seed in (17,42,73):
            w,s=fit(cache[c['mode']][0],y,eq,vx,veq,c,seed,selected['epoch'])
            name=f'{c["name"]}_{seed}.npz';np.savez(checkpoint/name,**w);scores.append(s/selected['temperature'])
            members.append(dict(kind='set_nn',mode=c['mode'],file=name,temperature=selected['temperature'],weight=1/3))
            print('REFIT',seed,flush=True)
        combined=np.mean(scores,axis=0);np.save(a.out/'validation_logits.npy',combined)
        m=metrics(np.where(veq.sum(-1)>0,aggregate(combined,veq,selected['pooling']),-1e9),vy,hard_mask(vb))
        (checkpoint/'ensemble.json').write_text(json.dumps(dict(members=members,answer_aggregation=selected['pooling'],selection=selected,validation=m),indent=2)+'\n')
        print('VALIDATION',m,flush=True)
    (a.out/'resources.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(0),torch=torch.__version__,folds=5,split_seed=2026,refit_seeds=[17,42,73]),indent=2)+'\n')


if __name__=='__main__':main()

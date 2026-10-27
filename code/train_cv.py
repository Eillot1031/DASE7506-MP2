"""Group cross-validation chooses architecture, epoch and answer pooling on TRAIN only.

Validation is evaluated after refitting, never used by this script for selection.
CPU and one CUDA GPU are supported; evaluation assets contain NumPy weights only.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,json,hashlib,time
from itertools import combinations
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from common import load_inputs,parse_answer
from features import build_features,aggregate
from train import labels,metrics

FAMILIES=[
    {'name':'cv_ema_full','mode':'full','width':128,'loss':'bce','dropout':.4,'decay':.01,'ema':.99},
    {'name':'cv_context_bce','mode':'interaction','width':64,'loss':'bce','dropout':.4,'decay':.01,'ema':0.},
    {'name':'cv_context_list','mode':'interaction','width':64,'loss':'mixedlist','dropout':.4,'decay':.1,'ema':0.},
    {'name':'cv_raw','mode':'raw','width':128,'loss':'bce','dropout':.4,'decay':.01,'ema':.99},
    {'name':'cv_raw_pair','mode':'raw_interaction','width':64,'loss':'pair','dropout':.4,'decay':.1,'ema':0.},
    {'name':'cv_norm','mode':'norm','width':128,'loss':'bce','dropout':.4,'decay':.01,'ema':.99},
    {'name':'cv_rank','mode':'full','width':128,'loss':'rank','dropout':.4,'decay':.1,'ema':0.},
    {'name':'cv_pair','mode':'full','width':128,'loss':'pair','dropout':.4,'decay':.1,'ema':0.},
    {'name':'cv_arithmetic','mode':'enhanced','width':128,'loss':'pair','dropout':.4,'decay':.1,'ema':0.},
    {'name':'cv_all_full','mode':'full','width':128,'loss':'bce','dropout':.4,'decay':.01,'ema':.99,'augment':.25},
    {'name':'cv_all_context','mode':'interaction','width':64,'loss':'bce','dropout':.4,'decay':.01,'ema':0.,'augment':.25},
    {'name':'cv_all_raw','mode':'raw','width':128,'loss':'bce','dropout':.4,'decay':.01,'ema':.99,'augment':.25},
    {'name':'cv_all_input','mode':'norm','width':128,'loss':'bce','dropout':.4,'decay':.01,'ema':.99,'augment':.25,'input_dropout':.2},
    {'name':'cv_plain128','mode':'full','width':128,'loss':'bce','dropout':.2,'decay':.01,'ema':0.},
    {'name':'cv_plain64','mode':'full','width':64,'loss':'bce','dropout':.2,'decay':.01,'ema':0.},
    {'name':'cv_plain256','mode':'full','width':256,'loss':'bce','dropout':.2,'decay':.01,'ema':0.},
    {'name':'cv_plain512','mode':'full','width':512,'loss':'bce','dropout':.5,'decay':.01,'ema':0.},
    {'name':'cv_norm_plain','mode':'norm','width':128,'loss':'bce','dropout':.2,'decay':.01,'ema':0.},
    {'name':'cv_norm_ema90','mode':'norm','width':128,'loss':'bce','dropout':.2,'decay':.01,'ema':.9},
    {'name':'cv_raw_plain','mode':'raw','width':128,'loss':'bce','dropout':.2,'decay':.01,'ema':0.},
    {'name':'cv_mixed','mode':'full','width':128,'loss':'mixed','dropout':.2,'decay':.01,'ema':0.},
    {'name':'cv_relations','mode':'relations','width':64,'loss':'bce','dropout':.4,'decay':.01,'ema':.99},
    {'name':'cv_relations_pair','mode':'relations','width':64,'loss':'pair','dropout':.4,'decay':.1,'ema':0.},
    {'name':'cv_relations_plain','mode':'relations','width':64,'loss':'bce','dropout':.2,'decay':.01,'ema':0.},
]

def hard_mask(bank):
    return np.array([len({parse_answer(c['text']) for c in r['candidates']}-{None})>=2 for r in bank])

def fit_model(x,y,valid,vx,c,seed,epochs,device,callback=None,group_weight=None):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    mean=x.reshape(-1,x.shape[-1]).mean(0);scale=x.reshape(-1,x.shape[-1]).std(0).clip(.15)
    tx=torch.as_tensor(((x-mean)/scale).clip(-10,10),device=device)
    ty=torch.as_tensor(y,device=device);mask=torch.as_tensor(valid,device=device)
    group_weight=np.ones(len(x),np.float32) if group_weight is None else group_weight
    sample_weight=torch.as_tensor(group_weight,device=device)
    tv=torch.as_tensor(((vx-mean)/scale).clip(-10,10),device=device)
    net=nn.Sequential(nn.Linear(x.shape[-1],c['width']),nn.ReLU(),nn.Dropout(c['dropout']),
                      nn.Linear(c['width'],32),nn.ReLU(),nn.Dropout(c['dropout']),nn.Linear(32,1)).to(device)
    averaged=copy.deepcopy(net) if c['ema'] else net
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=c['decay'])
    for epoch in range(epochs):
        net.train()
        for batch in np.array_split(rng.permutation(len(x)),max(1,len(x)//128)):
            ix=torch.as_tensor(batch,device=device)
            inp=nn.functional.dropout(tx[ix],p=c.get('input_dropout',0.),training=True)
            s=net(inp).squeeze(-1);target=ty[ix]
            loss_weights=mask[ix]*sample_weight[ix,None]
            bce=(nn.functional.binary_cross_entropy_with_logits(s,target,reduction='none')*loss_weights).sum()/loss_weights.sum()
            if c['loss']=='mixedlist':
                ls=-(target/target.sum(1,keepdim=True)*torch.log_softmax(s.masked_fill(mask[ix]==0,-1e9),1)).sum(1).mean()
                loss=bce+ls
            elif c['loss']=='rank':
                ranked=s.masked_fill(mask[ix]==0,-1e9)
                loss=(torch.logsumexp(ranked,1)-torch.logsumexp(ranked.masked_fill(target==0,-1e9),1)).mean()
            elif c['loss']=='mixed':
                ranked=s.masked_fill(mask[ix]==0,-1e9)
                loss=(torch.logsumexp(ranked,1)-torch.logsumexp(ranked.masked_fill(target==0,-1e9),1)).mean()+.5*bce
            elif c['loss']=='pair':
                pair=target[:,:,None]*(1-target[:,None,:])*mask[ix][:,None,:]
                loss=(nn.functional.softplus(s[:,None,:]-s[:,:,None])*pair).sum()/pair.sum().clamp(min=1)+.25*bce
            else:loss=bce
            opt.zero_grad();loss.backward();opt.step()
            if c['ema']:
                with torch.no_grad():
                    for a,b in zip(averaged.parameters(),net.parameters()):a.mul_(c['ema']).add_(b,alpha=1-c['ema'])
        if callback is not None or epoch==epochs-1:
            averaged.eval()
            with torch.no_grad():s=averaged(tv).squeeze(-1).cpu().numpy()
            if callback is not None:callback(epoch,s)
    weights={'mean':mean,'scale':scale}
    for i,j in enumerate((0,3,6)):
        weights[f'w{i}']=averaged[j].weight.detach().cpu().numpy().T.copy()
        weights[f'b{i}']=averaged[j].bias.detach().cpu().numpy().copy()
    return weights,s

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['search','refit','both'],default='both')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--out',type=Path,default=Path('code/checkpoints/cv'));args=ap.parse_args()
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
    start=time.perf_counter();bank,arrays=load_inputs('train');y=labels('train',bank);use=hard_mask(bank)
    cache={};easy={};easy_y=y[~use]
    for mode in dict.fromkeys(c['mode'] for c in FAMILIES):
        x,eq=build_features(bank,arrays,mode);cache[mode]=(x[use],eq[use]);easy[mode]=(x[~use],eq[~use])
    y=y[use];eq=cache['full'][1];valid=(eq.sum(-1)>0).astype(np.float32)
    outlog=Path('experiments/cross_validation');outlog.mkdir(parents=True,exist_ok=True)
    if args.stage in ('search','both'):
        summary=[];best=None;family_choices=[];oofs=[]
        folds=list(KFold(n_splits=5,shuffle=True,random_state=2026).split(y))
        for c in FAMILIES:
            t=time.perf_counter();x=cache[c['mode']][0];oof=np.zeros((args.epochs,len(y),4),np.float32)
            cached=outlog/(c['name']+'-curves.npy')
            if cached.exists():oof=np.load(cached)
            else:
                for fold,(tr,va) in enumerate(folds):
                    def callback(epoch,s):oof[epoch,va]=s
                    xx,yy,mm=x[tr],y[tr],valid[tr];gw=None
                    if c.get('augment'):
                        ex,ee=easy[c['mode']];xx=np.concatenate([xx,ex]);yy=np.concatenate([yy,easy_y]);mm=np.concatenate([mm,(ee.sum(-1)>0).astype(np.float32)])
                        gw=np.r_[np.ones(len(tr),np.float32),np.full(len(ex),c['augment'],np.float32)]
                    fit_model(xx,yy,mm,x[va],c,42+fold,args.epochs,args.device,callback,gw)
                    print(c['name'],'fold',fold+1,'complete',flush=True)
                np.save(cached,oof)
            family_best=None
            for w in (0.,.5,1.):
                counts=[]
                for epoch in range(args.epochs):
                    s=aggregate(oof[epoch],eq,w);s=np.where(valid>0,s,-1e9)
                    counts.append(int(y[np.arange(len(y)),s.argmax(1)].sum()))
                smooth=np.convolve(counts,np.ones(5)/5,mode='valid')
                # Central epoch of the best five-epoch window, avoiding isolated spikes.
                epoch=int(smooth.argmax())+3
                r={'family':c,'pooling':w,'epoch':epoch,'smoothed_oof_accuracy':100*float(smooth.max())/len(y),
                   'oof_accuracy':100*counts[epoch-1]/len(y),'n':len(y),'correct_by_epoch':counts,'seconds':time.perf_counter()-t}
                summary.append(r)
                if best is None or r['smoothed_oof_accuracy']>best['smoothed_oof_accuracy']:best=r
                if family_best is None or r['smoothed_oof_accuracy']>family_best['smoothed_oof_accuracy']:family_best=r
                print(c['name'],'pool',w,'epoch',epoch,'OOF',r['smoothed_oof_accuracy'],flush=True)
            family_choices.append(family_best)
            s=oof[family_best['epoch']-1]
            temperature=float((s-s.mean(1,keepdims=True)).std())
            family_best['temperature']=temperature
            oofs.append(s/temperature)
            np.save(outlog/(c['name']+'-oof.npy'),s)
        fusion=[];fusion_best=None
        for size in (1,2,3):
            for ids in combinations(range(len(FAMILIES)),size):
                estimate=0
                for i in ids:
                    c=FAMILIES[i];d=cache[c['mode']][0].shape[-1];width=c['width']
                    estimate+=3*(4*(d*width+width+width*32+32+32+1+2*d)+3000)
                if estimate>16*1024**2-50000:continue
                s=np.mean([oofs[i] for i in ids],axis=0)
                for w in (0.,.5,1.):
                    scores=np.where(valid>0,aggregate(s,eq,w),-1e9)
                    ncorrect=int(y[np.arange(len(y)),scores.argmax(1)].sum())
                    r={'families':[family_choices[i] for i in ids],'pooling':w,'oof_accuracy':100*ncorrect/len(y),
                       'correct':ncorrect,'n':len(y),'rule':'At most three families; equal family weights; three fixed refit seeds each.'}
                    fusion.append({'families':[FAMILIES[i]['name'] for i in ids],
                                   'epochs':[family_choices[i]['epoch'] for i in ids],
                                   'pooling':w,'oof_accuracy':r['oof_accuracy'],'correct':ncorrect,'n':len(y)})
                    if fusion_best is None or ncorrect>fusion_best['correct']:fusion_best=r
        best=fusion_best
        (outlog/'fusion.json').write_text(json.dumps(fusion,indent=2)+'\n')
        (outlog/'search.json').write_text(json.dumps(summary,indent=2)+'\n')
        (outlog/'selected.json').write_text(json.dumps(best,indent=2)+'\n')
    if args.stage in ('refit','both'):
        choice=json.loads((outlog/'selected.json').read_text())
        vb,va=load_inputs('validation')
        args.out.mkdir(parents=True,exist_ok=True);members=[];sum_scores=np.zeros((len(vb),4),np.float32)
        for family in choice['families']:
            c=family['family'];x=cache[c['mode']][0];vx,veq=build_features(vb,va,c['mode'])
            for seed in (17,42,73):
                xx,yy,mm=x,y,valid;gw=None
                if c.get('augment'):
                    ex,ee=easy[c['mode']];xx=np.concatenate([x,ex]);yy=np.concatenate([y,easy_y]);mm=np.concatenate([valid,(ee.sum(-1)>0).astype(np.float32)])
                    gw=np.r_[np.ones(len(x),np.float32),np.full(len(ex),c['augment'],np.float32)]
                t=time.perf_counter();weights,s=fit_model(xx,yy,mm,vx,c,seed,family['epoch'],args.device,group_weight=gw)
                name=f'{c["name"]}_{seed}.npz';np.savez(args.out/name,**weights)
                np.save(outlog/f'{c["name"]}-validation-{seed}.npy',s)
                weight=1/(len(choice['families'])*3);sum_scores+=s/family['temperature']*weight
                members.append({'file':name,'kind':'nn','mode':c['mode'],'temperature':family['temperature'],'weight':weight})
                print('Refit',c['name'],seed,'seconds',time.perf_counter()-t,flush=True)
        final=aggregate(sum_scores,veq,choice['pooling']);final=np.where(veq.sum(-1)>0,final,-1e9)
        m=metrics(final,labels('validation',vb),hard_mask(vb))
        config={'members':members,'answer_aggregation':choice['pooling'],'selection':choice,'validation':m,
                'selection_rule':'Architecture, common epochs and pooling selected only using train-group OOF scores; fixed three-seed mean per family; no validation labels used for selection.'}
        (args.out/'ensemble.json').write_text(json.dumps(config,indent=2)+'\n')
        print('REFIT VALIDATION',m,flush=True)
    (outlog/'resources.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'device':args.device,
        'device_name':torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU','torch':torch.__version__,
        'seed_split':2026,'folds':5,'seeds_refit':[17,42,73]},indent=2)+'\n')

if __name__=='__main__':main()

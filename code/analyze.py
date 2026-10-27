"""Post-freeze validation analysis and report figures. No test reference access."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import argparse,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_inputs,parse_answer,read_jsonl,DATA
from features import build_features,aggregate,text_features
from submission import score_bank
from train import labels,metrics
from train_cv import hard_mask

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,default=Path('code/checkpoints/final'))
    ap.add_argument('--output-dir',type=Path,default=Path('experiments/analysis'))
    ap.add_argument('--figures',type=Path,default=Path('report/figures'));args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    figs=args.figures;figs.mkdir(parents=True,exist_ok=True)
    root=args.checkpoint;cfg=json.loads((root/'ensemble.json').read_text())
    b,a=load_inputs('validation');y=labels('validation',b);hard=hard_mask(b);_,eq=text_features(b)
    scores=score_bank(b,a,root);choices=scores.argmax(1);correct=y[np.arange(len(b)),choices].astype(bool)
    majority=eq.sum(-1).argmax(1);vote_correct=y[np.arange(len(b)),majority].astype(bool)
    table=[{'name':'First candidate',**metrics(np.tile([1,0,0,0],(len(b),1)),y,hard)},
           {'name':'Majority vote',**metrics(eq.sum(-1),y,hard)}]
    old=json.loads(Path('experiments/exploratory/runs.json').read_text())
    for name,label in [('answer42','Question and answer'),('style42','Handcrafted descriptors'),
                       ('embedding42','Embedding only'),('relative42','Relative embeddings'),('full42','Combined single network')]:
        r=next(r for r in old if r['name']==name)
        with np.load('experiments/validation_logits.npz') as values:s=values[name]
        s=np.where(eq.sum(-1)>0,s,-1e9)
        table.append({'name':label,**metrics(s,y,hard)})
    if any(m.get('kind')=='set_nn' for m in cfg['members']):
        table.append({'name':'Previous frozen ensemble',**metrics(score_bank(b,a,Path('code/checkpoints/final')),y,hard)})
    table.append({'name':'Final frozen ensemble',**metrics(scores,y,hard)})
    # Pooling ablation uses the already-frozen weights, never re-fits them.
    raw=np.zeros_like(scores);cache={}
    for m in cfg['members']:
        if m['mode'] not in cache:cache[m['mode']]=build_features(b,a,m['mode'])[0]
        x=cache[m['mode']]
        with np.load(root/m['file']) as w:
            h=((x-w['mean'])/w['scale']).clip(-10,10)
            if m.get('kind')=='set_nn':
                h=np.maximum(h@w['w0']+w['b0'],0);valid=eq.sum(-1,keepdims=True)>0
                avg=(h*valid).sum(1,keepdims=True)/np.maximum(valid.sum(1,keepdims=True),1)
                maximum=np.where(valid,h,-1e9).max(1,keepdims=True);same=eq@h/np.maximum(eq.sum(-1,keepdims=True),1)
                h=np.concatenate([h,np.broadcast_to(avg,h.shape),h-avg,h*avg,same,np.broadcast_to(maximum,h.shape)],-1)
                h=np.maximum(h@w['w1']+w['b1'],0)@w['w2']+w['b2']
            else:
                for i in range(3):
                    h=h@w[f'w{i}']+w[f'b{i}']
                    if i<2:h=np.maximum(h,0)
        raw+=h[...,0]*m['weight']/m['temperature']
    pooling=[{'weight':w,**metrics(np.where(eq.sum(-1)>0,aggregate(raw,eq,w),-1e9),y,hard)} for w in (0.,.5,1.)]
    bins=[];coverage=[];gaps=[];conf=[];indices=np.flatnonzero(hard)
    for row,s in zip(b,scores):
        values={}
        for c,ss in zip(row['candidates'],s):
            ans=parse_answer(c['text'])
            if ans is not None:values[ans]=max(float(ss),values.get(ans,-1e9))
        vv=sorted(values.values(),reverse=True)
        gaps.append(vv[0]-vv[1] if len(vv)>1 else float('inf'))
        p=np.exp(np.array(vv)-vv[0]);conf.append(float(1/p.sum()))
    conf=np.array(conf);gaps=np.array(gaps)
    order=indices[np.argsort(-gaps[indices],kind='stable')]
    for fraction in (.25,.5,.75,1.):
        ids=order[:math.ceil(len(order)*fraction)]
        coverage.append({'coverage':len(ids)/len(order),'accuracy':100*correct[ids].mean(),'n':len(ids)})
    for ids in np.array_split(indices[np.argsort(conf[indices])],5):
        bins.append({'n':len(ids),'confidence':float(conf[ids].mean()),'accuracy':float(correct[ids].mean())})
    ece=sum(r['n']*abs(r['confidence']-r['accuracy']) for r in bins)/len(indices)
    refs=read_jsonl(DATA/'gsm8k_validation.jsonl')
    cases=[]
    wrong=indices[~correct[indices]];wrong=wrong[np.argsort(-gaps[wrong])]
    for i in wrong[:3]:
        cases.append({'question':b[i]['question'],'gold_answer':parse_answer(refs[i]['answer']),
                      'candidate_answers':[parse_answer(c['text']) for c in b[i]['candidates']],
                      'selected':int(choices[i]),'margin':float(gaps[i]),
                      'selected_reasoning':b[i]['candidates'][choices[i]]['text'],
                      'correct_reasoning':next(c['text'] for j,c in enumerate(b[i]['candidates']) if y[i,j])})
    breakdown=[]
    for count in (2,3,4):
        mask=np.array([len({parse_answer(c['text']) for c in r['candidates']}-{None})==count for r in b])
        breakdown.append({'distinct_answers':count,'n':int(mask.sum()),'final_accuracy':100*float(correct[mask].mean()),
                          'majority_accuracy':100*float(vote_correct[mask].mean())})
    arithmetic_ablation={'without':metrics(np.where(eq.sum(-1)>0,aggregate(raw,eq,cfg['answer_aggregation']),-1e9),y,hard),
                         'with':metrics(scores,y,hard)}
    record={'comparisons':table,'pooling_ablation_without_arithmetic':pooling,'arithmetic_ablation':arithmetic_ablation,'coverage':coverage,'uncalibrated_reliability_bins':bins,
            'uncalibrated_ece':ece,'breakdown':breakdown,'error_cases':cases,
            'vs_majority_hard':{'fixed':int((correct&~vote_correct&hard).sum()),'regressed':int((~correct&vote_correct&hard).sum())}}
    (out/'validation_analysis.json').write_text(json.dumps(record,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,ax=plt.subplots(figsize=(7.3,3.4));names=[r['name'] for r in table]
    ax.barh(names,[r['hard_accuracy'] for r in table],color=['#b3bdc9']*(len(table)-1)+['#244a70'])
    ax.invert_yaxis();ax.set_xlim(45,100);ax.set_xlabel('Accuracy on difficult validation questions (%)')
    for i,r in enumerate(table):ax.text(r['hard_accuracy']+.6,i,f'{r["hard_accuracy"]:.2f}',va='center',fontsize=9)
    fig.tight_layout();fig.savefig(figs/'validation_comparison.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.3,2.8))
    axes[0].plot([r['coverage']*100 for r in coverage],[r['accuracy'] for r in coverage],'o-',color='#244a70')
    axes[0].set(xlabel='Coverage (%)',ylabel='Accuracy (%)',ylim=(60,102),title='Selection by answer score margin')
    axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].plot([r['confidence'] for r in bins],[r['accuracy'] for r in bins],'o-',color='#244a70')
    axes[1].set(xlabel='Uncalibrated confidence',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1),title='Five bins on difficult validation')
    fig.tight_layout();fig.savefig(figs/'reliability.png');plt.close(fig)
    print(json.dumps({k:v for k,v in record.items() if k!='error_cases'},indent=2))

if __name__=='__main__':main()

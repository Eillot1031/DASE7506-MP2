"""Train-only PCA and nonlinear kernel verifier, selected on validation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json,time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from common import load_inputs,parse_answer
from features import build_features
from train import labels,metrics

def main():
    tick=time.perf_counter();b,a=load_inputs('train');v,va=load_inputs('validation')
    x,eq=build_features(b,a,'full');vx,veq=build_features(v,va,'full')
    y=labels('train',b);vy=labels('validation',v)
    hard=lambda rows:np.array([len({parse_answer(c['text']) for c in r['candidates']}-{None})>=2 for r in rows])
    use=hard(b);vh=hard(v)
    mean=x.reshape(-1,x.shape[-1]).mean(0);scale=x.reshape(-1,x.shape[-1]).std(0).clip(.15)
    x=((x-mean)/scale).clip(-10,10);vx=((vx-mean)/scale).clip(-10,10)
    pca=PCA(n_components=128,svd_solver='randomized',random_state=42)
    z=pca.fit_transform(x.reshape(-1,x.shape[-1]));vz=pca.transform(vx.reshape(-1,vx.shape[-1]))
    z=z.reshape(len(b),4,-1);vz=vz.reshape(len(v),4,-1)
    valid=eq[use].sum(-1)>0;tz=z[use][valid];ty=y[use][valid]
    records=[]
    for c in (.1,1.,10.,100.):
        for gamma in (.0001,.0003,.001,.003):
            out=Path(f'runs/kernel_c{c}_g{gamma}');out.mkdir(exist_ok=True)
            t=time.perf_counter();model=SVC(C=c,gamma=gamma,cache_size=512)
            model.fit(tz,ty);s=model.decision_function(vz.reshape(-1,128)).reshape(-1,4)
            m=metrics(np.where(veq.sum(-1)>0,s,-1e9),vy,vh)
            config={'kind':'kernel','mode':'full','C':c,'gamma':gamma,'validation':m,'seed':42}
            (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
            np.savez(out/'kernel.npz',mean=mean,scale=scale,pca_mean=pca.mean_,projection=pca.components_.T,
                     support=model.support_vectors_.astype(np.float32),dual=model.dual_coef_.ravel().astype(np.float32),
                     bias=model.intercept_.astype(np.float32),gamma=np.float32(gamma))
            np.save(out/'validation_logits.npy',s)
            (out/'history.json').write_text(json.dumps({'seconds':time.perf_counter()-t},indent=2)+'\n')
            print(config,'seconds',time.perf_counter()-t,flush=True);records.append(config)
    Path('experiments/kernel_search.json').write_text(json.dumps({'total_seconds':time.perf_counter()-tick,'records':records},indent=2)+'\n')

if __name__=='__main__':main()

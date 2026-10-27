"""Sparse lexical ablation with train-only vocabulary and regularized weights."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json,re,time
from pathlib import Path
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from common import load_inputs,parse_answer
from features import text_features
from train import labels,metrics

def texts(bank):
    return [re.sub(r'\d+(?:[.,/]\d+)*',' NUM ',c['text'].lower()) for r in bank for c in r['candidates']]

def main():
    tick=time.perf_counter();b,_=load_inputs('train');v,_=load_inputs('validation')
    y=labels('train',b);vy=labels('validation',v);_,eq=text_features(b);_,veq=text_features(v)
    hard=lambda rows:np.array([len({parse_answer(c['text']) for c in r['candidates']}-{None})>=2 for r in rows])
    use=hard(b);vh=hard(v)
    vectorizer=FeatureUnion([('word',TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=25000,sublinear_tf=True)),
                             ('char',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=5,max_features=25000,sublinear_tf=True))])
    x=vectorizer.fit_transform(texts(b));vx=vectorizer.transform(texts(v))
    valid=(np.repeat(use,4))&(eq.sum(-1).ravel()>0)
    records=[]
    for c in (.1,1.,10.,100.):
        out=Path(f'runs/lexical_c{c}');out.mkdir(exist_ok=True)
        t=time.perf_counter();m=LogisticRegression(C=c,solver='liblinear',max_iter=500,random_state=42)
        m.fit(x[valid],y.ravel()[valid]);s=m.decision_function(vx).reshape(-1,4)
        result=metrics(np.where(veq.sum(-1)>0,s,-1e9),vy,vh)
        joblib.dump((vectorizer,m),out/'lexical.joblib',compress=0)
        np.save(out/'validation_logits.npy',s)
        config={'kind':'lexical','mode':'lexical','C':c,'validation':result,'seed':42}
        (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (out/'history.json').write_text(json.dumps({'seconds':time.perf_counter()-t},indent=2)+'\n')
        records.append(config);print(config,flush=True)
    Path('experiments/lexical_search.json').write_text(json.dumps({'seconds':time.perf_counter()-tick,'records':records},indent=2)+'\n')

if __name__=='__main__':main()

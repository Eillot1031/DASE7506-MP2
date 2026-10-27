"""Validation-only audit of bank independence, equivariance and exported weights."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import copy,json
from pathlib import Path
import numpy as np
from common import load_inputs,parse_answer
from submission import predict,score_bank
from features import aggregate,fusion_features,tree_predict,arithmetic_evidence

def main():
    root=Path('code/checkpoints/final');bank,arrays=load_inputs('validation')
    actual=score_bank(bank,arrays,root)
    config=json.loads((root/'ensemble.json').read_text())
    expected=np.zeros_like(actual);families={};family_weights={}
    for m in config['members']:
        p=Path('runs')/Path(m['file']).stem/'validation_logits.npy'
        packaged=Path('experiments/validation_logits.npz')
        if packaged.exists():
            with np.load(packaged) as values:s=values[Path(m['file']).stem]
            p=None
        elif not p.exists():
            family,seed=Path(m['file']).stem.rsplit('_',1)
            p=Path('experiments/cross_validation')/f'{family}-validation-{seed}.npy'
            if not p.exists():p=Path('experiments/cross_validation')/f'validation_logits_{seed}.npy'
        if p is not None:s=np.load(p)
        weighted=s/m['temperature']*m['weight'];expected+=weighted
        family=Path(m['file']).stem.rsplit('_',1)[0]
        families[family]=families.get(family,0)+weighted
        family_weights[family]=family_weights.get(family,0)+m['weight']
    from features import text_features
    _,eq=text_features(bank)
    if 'fusion' in config:
        fs=[families[f]/family_weights[f] for f in config['fusion']['families']]
        with np.load(root/config['fusion']['file']) as w:expected=tree_predict(fusion_features(fs,eq),w)
    else:expected=aggregate(expected,eq,config['answer_aggregation'])
    if 'arithmetic_penalty' in config:
        c=config['arithmetic_penalty'];p=arithmetic_evidence(bank)[...,c['column']]
        if c['column'] in (3,4):p=np.log1p(10*p)
        expected-=c['weight']*aggregate(p,eq,c['pooling'])
    expected=np.where(eq.sum(-1)>0,expected,-1e9)
    err=float(abs(expected[actual>-1e8]-actual[actual>-1e8]).max())
    assert err<1e-4,err
    assert np.array_equal(actual.argmax(1),expected.argmax(1))
    order=np.random.default_rng(17).permutation(len(bank))[:31]
    small=copy.deepcopy([bank[i] for i in order]); ar={'embeddings':arrays['embeddings'][order].copy()}
    for i,r in enumerate(small):r['id']=f'unseen-audit-{i}'
    assert predict(small,ar,root)==actual[order].argmax(1).tolist()
    perm=np.array([2,0,3,1]);swapped=copy.deepcopy(small)
    for r in swapped:
        r['candidates']=[r['candidates'][i] for i in perm]
        for i,c in enumerate(r['candidates']):c['index']=i
    selected=predict(swapped,{'embeddings':ar['embeddings'][:,perm]},root)
    original=predict(small,ar,root)
    assert all(parse_answer(small[i]['candidates'][a]['text'])==parse_answer(swapped[i]['candidates'][b]['text'])
               for i,(a,b) in enumerate(zip(original,selected)))
    record={'export_max_abs_error':err,'prediction_parity':True,'arbitrary_ids_and_batch_size':True,'candidate_permutation':True}
    Path('experiments/method_verification.json').write_text(json.dumps(record,indent=2)+'\n');print(record)

if __name__=='__main__':main()

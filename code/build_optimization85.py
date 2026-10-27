"""Package a validation-selected extension without modifying the 84.154% baseline."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import copy,datetime,hashlib,json,shutil
from pathlib import Path
import numpy as np
from common import load_inputs,parse_answer
from features import arithmetic_evidence,aggregate,text_features
from arithmetic_v2 import evidence
from submission import score_bank,predict
from train import labels,metrics
from train_cv import hard_mask


def main():
    out=Path('experiments/optimization85');target=Path('code/checkpoints/candidate85')
    if (out/'freeze.json').exists():raise RuntimeError('Candidate already frozen; do not replace it after scoring')
    selected=max(json.loads((out/'blends.json').read_text()),key=lambda r:r['correct'])
    penalty=json.loads((out/'arithmetic_v2.json').read_text())['selected']
    assert selected['family']=='set' and selected['pooling']==.5
    base=Path('code/checkpoints/final');addition=Path('runs/optimization85/set/checkpoint')
    config=json.loads((base/'ensemble.json').read_text());new=json.loads((addition/'ensemble.json').read_text())
    target.mkdir(parents=True,exist_ok=True);members=[]
    for root,source,factor in ((base,config,1-selected['weight']),(addition,new,selected['weight'])):
        for member in source['members']:
            m=copy.deepcopy(member);m['weight']*=factor;members.append(m)
            shutil.copyfile(root/m['file'],target/m['file'])
    config=dict(members=members,answer_aggregation=.5,arithmetic_penalty=penalty,selection=selected,
                selection_rule='Train-only five-fold CV chose the new family and epoch. Validation selected one of 17 fixed blends; train OOF selected the arithmetic rule. Earlier aggregate public test feedback had already been seen.')
    (target/'ensemble.json').write_text(json.dumps(config,indent=2)+'\n')
    b,a=load_inputs('validation');_,eq=text_features(b)
    baseline=score_bank(b,a,base)+.5*arithmetic_evidence(b)[...,1]
    extension=np.load('runs/optimization85/set/validation_logits.npy')
    ev=evidence(b);p=penalty['weight']*aggregate(ev[...,penalty['column']]+penalty['symbol_weight']*ev[...,3],eq,penalty['pooling'])
    expected=(1-selected['weight'])*baseline+selected['weight']*aggregate(extension,eq,.5)-p
    expected=np.where(eq.sum(-1)>0,expected,-1e9);actual=score_bank(b,a,target)
    error=float(abs(expected[actual>-1e8]-actual[actual>-1e8]).max());assert error<1e-4,error
    assert np.array_equal(expected.argmax(1),actual.argmax(1))
    assert metrics(actual,labels('validation',b),hard_mask(b))['correct']==selected['correct']
    ix=np.random.default_rng(17).permutation(len(b))[:47]
    small=copy.deepcopy([b[i] for i in ix]);ar={'embeddings':a['embeddings'][ix].copy()}
    for i,r in enumerate(small):r['id']=f'new-independent-{i}'
    assert predict(small,ar,target)==actual[ix].argmax(1).tolist()
    for perm in (np.array([2,0,3,1]),np.array([3,2,1,0])):
        swapped=copy.deepcopy(small)
        for r in swapped:
            r['candidates']=[r['candidates'][i] for i in perm]
            for j,c in enumerate(r['candidates']):c['index']=j
        choices=predict(swapped,{'embeddings':ar['embeddings'][:,perm]},target)
        assert all(parse_answer(small[i]['candidates'][int(actual[ix[i]].argmax())]['text'])==parse_answer(swapped[i]['candidates'][j]['text']) for i,j in enumerate(choices))
    np.savez(out/'validation_scores.npz',scores=actual)
    verification=dict(export_max_abs_error=error,prediction_parity=True,arbitrary_ids_and_batch_size=True,candidate_permutations=2)
    (out/'method_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    for family in ('set','answer'):
        for name in ('search.json','selected.json','resources.json'):
            shutil.copyfile(Path('runs/optimization85')/family/name,out/(family+'_'+name))
    sources=[Path('code')/n for n in ('submission.py','features.py','arithmetic_v2.py','train_set_verifier.py','train_answer_verifier.py','tune_arithmetic_v2.py','build_optimization85.py','common.py','predict.py','score_predictions.py')]
    files=sources+sorted(target.glob('*'))
    manifest=dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_test_evaluations=6,
                  validation=selected,verification=verification,
                  disclosure='This is a further adaptive development round after the six previously reported aggregate test scores. No individual test labels or errors were used. Candidate settings were frozen before the next complete official score.',
                  files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'freeze.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(checkpoint=str(target),bytes=sum(p.stat().st_size for p in target.glob('*')),validation=selected,verification=verification),indent=2))


if __name__=='__main__':main()

"""Refit the three fixed DeepSets seeds on GPU and rebuild the released bundle.

No architecture, epoch, mixture or penalty selection occurs here. Use the
original reproduction script first to independently rebuild the baseline too.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from common import load_inputs
from features import build_features
from train import labels
from train_cv import hard_mask
from train_set_verifier import fit


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('runs/reproduced85'))
    ap.add_argument('--base',type=Path,default=Path('code/checkpoints/final'));a=ap.parse_args()
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('This training reproduction requires CUDA; inference remains CPU-only')
    start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=True)
    canonical=Path('code/checkpoints/candidate85');cfg=json.loads((canonical/'ensemble.json').read_text())
    selected=json.loads(Path('experiments/optimization85/set_selected.json').read_text())
    b,arrays=load_inputs('train');use=hard_mask(b);y=labels('train',b)[use]
    x,eq=build_features(b,arrays,selected['family']['mode']);x=x[use];eq=eq[use]
    records=[]
    for m in cfg['members']:
        target=a.output/m['file']
        if m.get('kind')=='set_nn':
            seed=int(Path(m['file']).stem.rsplit('_',1)[1])
            w,_=fit(x,y,eq,x[:1],eq[:1],selected['family'],seed,selected['epoch'])
            np.savez(target,**w)
        else:shutil.copyfile(a.base/m['file'],target)
        with np.load(canonical/m['file']) as original,np.load(target) as actual:
            error=max(float(abs(original[k]-actual[k]).max()) for k in original.files)
        records.append(dict(member=m['file'],max_abs_weight_error=error,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    shutil.copyfile(canonical/'ensemble.json',a.output/'ensemble.json')
    record=dict(seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(0),torch=torch.__version__,
                baseline_source=str(a.base),output=str(a.output),members=records)
    (a.output.parent/(a.output.name+'-training.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()

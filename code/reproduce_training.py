"""Rebuild the frozen selected checkpoints, without repeating model selection."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('runs/reproduced_bundle'));args=ap.parse_args()
    configs=json.loads(Path('experiments/final_training_configs.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True);tick=time.perf_counter();records=[]
    for name,c in configs.items():
        run=Path('runs/reproduction')/name
        command=[sys.executable,'code/train.py','--mode',c['mode'],'--seed',str(c['seed']),'--width',str(c['width']),
                 '--epochs',str(c['best_epoch']),'--save-epoch',str(c['best_epoch']),'--lr',str(c['lr']),
                 '--decay',str(c['decay']),'--dropout',str(c['dropout']),'--loss',c['loss'],
                 '--ema',str(c.get('ema',0.)),'--out',str(run)]
        if c.get('hard_only'):command.append('--hard-only')
        if c.get('valid_only'):command.append('--valid-only')
        subprocess.run(command,check=True)
        target=args.output/(name+'.npz');shutil.copyfile(run/'weights.npz',target)
        original=Path('code/checkpoints/final')/target.name
        import numpy as np
        with np.load(original) as a,np.load(target) as b:
            error=max(float(abs(a[k]-b[k]).max()) for k in a.files)
        records.append({'member':name,'max_abs_weight_error':error,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    shutil.copyfile('code/checkpoints/final/ensemble.json',args.output/'ensemble.json')
    record={'seconds':time.perf_counter()-tick,'python':sys.version,'members':records,'output':str(args.output)}
    (args.output/'training_reproduction.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()

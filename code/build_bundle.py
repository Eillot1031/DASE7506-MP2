"""Package frozen refit weights and train-OOF-selected arithmetic settings."""
import argparse,json,shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    config=json.loads((args.source/'ensemble.json').read_text())
    config['arithmetic_penalty']=json.loads(Path('experiments/cross_validation/arithmetic_penalty.json').read_text())['selected']
    args.output.mkdir(parents=True,exist_ok=True)
    required={'ensemble.json'}|{m['file'] for m in config['members']}
    for m in config['members']:shutil.copyfile(args.source/m['file'],args.output/m['file'])
    (args.output/'ensemble.json').write_text(json.dumps(config,indent=2)+'\n')
    for p in args.output.iterdir():
        if p.is_file() and p.name not in required:p.unlink()
    size=sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
    if size>16*1024**2:raise ValueError('Inference asset limit exceeded')
    print('Packaged',size,'bytes')

if __name__=='__main__':main()

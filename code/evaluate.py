"""Measure fresh prediction plus official scoring subprocesses under four threads."""
import argparse,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--checkpoint',type=Path,default=Path('code/checkpoints/final'))
    p.add_argument('--output-dir',type=Path,default=Path('experiments'));args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        env[k]='4'
    predictions=args.output_dir/f'{args.split}-predictions.json';score=args.output_dir/f'{args.split}-score.json'
    t=time.perf_counter()
    subprocess.run([sys.executable,'code/predict.py','--method','code/submission.py','--checkpoint',str(args.checkpoint),
                    '--split',args.split,'--output',str(predictions)],check=True,env=env)
    middle=time.perf_counter()
    subprocess.run([sys.executable,'code/score_predictions.py','--split',args.split,'--predictions',str(predictions),
                    '--output',str(score)],check=True,env=env)
    end=time.perf_counter()
    files={str(p.relative_to(args.checkpoint)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in sorted(args.checkpoint.rglob('*')) if p.is_file()}
    record={'split':args.split,'python':sys.version,'platform':platform.platform(),'threads':4,
            'prediction_process_seconds':middle-t,'scoring_process_seconds':end-middle,'total_seconds':end-t,
            'asset_file_bytes':sum(f['bytes'] for f in files.values()),'asset_files':files,
            'official_score':json.loads(score.read_text())}
    (args.output_dir/f'{args.split}-resources.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()

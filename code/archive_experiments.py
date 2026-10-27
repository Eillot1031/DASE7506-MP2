"""Retain exploratory configurations and learning curves without training checkpoints."""
import json,shutil
from pathlib import Path

def main():
    records=[];out=Path('experiments/exploratory');out.mkdir(exist_ok=True)
    for p in sorted(Path('runs').glob('*/config.json')):
        if p.parent.name.startswith('reproduce_') or not (p.parent/'history.json').exists():continue
        config=json.loads(p.read_text());history=json.loads((p.parent/'history.json').read_text())
        records.append({'name':p.parent.name,'configuration':config,'resources_and_curve':history})
    (out/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
    print(len(records),'completed exploratory runs archived')

if __name__=='__main__':main()

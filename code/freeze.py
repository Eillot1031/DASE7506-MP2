"""Record code and asset hashes before a frozen model is officially scored."""
import datetime,hashlib,json
from pathlib import Path

def main():
    sources=['submission.py','features.py','common.py','predict.py','score_predictions.py','train.py','develop.py']
    files=[Path('code')/s for s in sources]+sorted(Path('code/checkpoints/final').glob('*'))
    record={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'selection_split':'validation for base ensemble; train-group OOF for arithmetic penalty','prior_test_evaluations':5,
            'prior_result':'experiments/initial/test-score.json',
            'disclosure':'Development continued after the initial aggregate test score was seen. No test error cases or references were used to fit weights, design features, or select configurations. The final score is not an untouched one-shot holdout estimate.',
            'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    Path('experiments/freeze.json').write_text(json.dumps(record,indent=2)+'\n')
    print(record['frozen_at_utc'])

if __name__=='__main__':main()

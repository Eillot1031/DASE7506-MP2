# Selecting Reliable LLM Reasoning

DASE7506 Advanced Machine Learning Methods · Mini Project 2 · Eillot1031

Select one of four supplied mathematical solutions using compact candidate verifiers, comparison between answers, and a numerical consistency check. The frozen model scores **84.58244111349036% (395/467)** on the **complete public test bank** with the unchanged official scorer.

| Frozen result | Measurement |
|---|---:|
| Complete test accuracy | **84.582% (395/467)** |
| Complete validation accuracy | 94.862% (757/798) |
| Difficult validation accuracy | 88.451% (314/355) |
| Fresh CPU prediction plus independent scoring | 0.916 s |
| All required inference assets | 4,899,529 bytes / 4.673 MiB |
| Local CPU | AMD Ryzen 7 9700X, four active library threads |
| GPU used in exploratory cross-validation | NVIDIA RTX 5060 Ti 16 GB |

The reference Xeon determines official runtime eligibility; the timing above is a local measurement. The checkpoint is `code/checkpoints/candidate85/`. The project report is available in Markdown at [report/REPORT.md](report/REPORT.md).

**Experimental limitation:** development used aggregate public-test feedback, so this is an adaptive public-bank result rather than an untouched holdout estimate. Weights were fitted only using training supervision, and no individual test reference answers or error cases were inspected for model development. The [report](report/REPORT.md#experimental-protocol-and-test-history) discloses the evaluation history; instructor review and an independent audit bank remain necessary for assessing eligibility and generalization.

## Installation and Usage

Run these commands from the repository root with Python 3.12 to install the NumPy-only inference dependencies and verify the supplied data.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-inference.txt
python code/check_data.py
```

## Evaluate

Evaluate the supplied checkpoint on the complete test bank with four library threads and save predictions, the official score and resource measurements.

```bash
python code/evaluate.py --split test --checkpoint code/checkpoints/candidate85 \
  --output-dir runs/reproduction-evaluation
```

## Method

Three candidate-wise networks use the supplied 1024-dimensional representations, differences from the other candidates, answer agreement, and generic text descriptors. Two networks also receive an element-wise interaction with the group representation. Two use exponential parameter averaging. These networks supply 90% of the neural score.

A three-seed verifier inspired by [Deep Sets](https://arxiv.org/abs/1703.06114) supplies the remaining 10%. Its shared encoder produces 64 features per candidate; the scoring head compares these with the valid-candidate mean, maximum and mean for the same answer. Training uses only difficult training questions. Five-fold training CV selects the family and epoch, and validation selects a small blend from a fixed grid. The complete inference bundle contains six compact networks and uses NumPy only. [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412) was also explored but was not retained.

A candidate retains half its own neural score and receives half the mean score for the same parsed answer. The numeric checker supports percentages, powers and fraction notation, and avoids checking numeric fragments of symbolic expressions. The final penalty is 0.5 times the same-answer mean count of inconsistent numeric equalities, with relative-error threshold 0.001. Local symbolic evidence was explored but received zero final weight. Invalid answers are masked using the unchanged official parser. Arithmetic consistency is evidence, not proof of semantic correctness.

See [the bundle configuration](code/checkpoints/candidate85/ensemble.json), [train-OOF settings](experiments/optimization85/set_selected.json), [the pre-scoring freeze](experiments/optimization85/freeze.json), and [the report](report/REPORT.md) for the full experimental analysis.

## Reproduce

Refit the three DeepSets seeds with CUDA PyTorch 2.9.0 and package them with the supplied candidate-wise verifier weights using the frozen configuration.

```bash
python -m pip install -r code/requirements-training.txt
python -m pip install torch==2.9.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
python code/reproduce_optimization85.py --output runs/reproduced85
```

## Evidence and repository map

| Location | Contents |
|---|---|
| [report/REPORT.md](report/REPORT.md) | Markdown report with method, experiments, findings and declarations |
| `code/submission.py`, `code/features.py`, `code/arithmetic_v2.py` | Reference-free inference and arithmetic evidence |
| `code/checkpoints/candidate85/` | Latest complete inference bundle, stored without compression |
| `code/reproduce_optimization85.py` | Frozen configuration for reproducing the current bundle |
| `code/verify_method.py` | Export parity, new IDs, changed batch size and candidate permutation checks |
| `experiments/optimization85/test-score.json` | Latest official score, 395/467 |
| `experiments/optimization85/test-resources.json` | Latest fresh-process measurement and asset hashes |
| `experiments/optimization85/analysis/` | Latest validation ablations, uncertainty and errors |
| `experiments/cross_validation/` | Train OOF search, failed ideas and arithmetic-penalty selection |
| `report/figures85/` | Figures used in the Markdown report |
| `guide/GUIDE.md`, `code/README.md` | Original course guide and running instructions |

The official data, numeric parser and scoring code are unchanged. Evaluation predictions in `experiments/` are evidence only; inference never reads them. No IDs, cached choices, external datasets or additional pretrained weights enter the method.

## Conclusions

The compact verifier achieves **84.582%** on the complete public test bank using six small networks and numeric consistency evidence. Consistent arithmetic and high score margins provide useful signals, but semantic errors remain and the scores are not calibrated probabilities; independent audit evaluation is needed to assess generalization.

## Declarations

### AI assistance disclosure

OpenAI Codex provided substantive assistance with interpreting the requirements, implementing feature extraction and training code, running experiments, preparing reproducibility records, drafting the report and README, and publishing the repository. The student author is responsible for reviewing and understanding the implementation and report. Scores come from the unchanged official scorer; no LLM is called during evaluation, and no external dataset, additional pretrained weights, question-ID feature or answer lookup is used. The [report](report/REPORT.md#experimental-protocol-and-test-history) discloses the adaptive test-feedback limitation.

### Data attribution

This repository is a fork of [the course package](https://github.com/xudongwu-0/DASE7506-MP2). GSM8K's retained MIT license and attribution are in [LICENSE](code/data/LICENSE.txt) and [NOTICE](code/data/NOTICE.txt). Candidate traces and fixed representations were supplied by the course and produced with Qwen3-0.6B. The original data, parser and scorer remain unchanged.

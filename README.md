# Selecting Reliable LLM Reasoning

DASE7506 Advanced Machine Learning Methods · Mini Project 2 · Eillot1031

Select one of four supplied mathematical solutions using compact candidate verifiers, comparison between answers, and a numerical consistency check. The final frozen model scores **84.1541755888651% (393/467)** on the **complete public test bank** with the unchanged official scorer.

| Frozen result | Measurement |
|---|---:|
| Complete test accuracy | **84.154%** |
| Complete validation accuracy | 94.737% (756/798) |
| Difficult validation accuracy | 88.169% (313/355) |
| Fresh CPU prediction plus independent scoring | 0.829 s |
| All required inference assets | 2,952,607 bytes / 2.816 MiB |
| Local CPU | AMD Ryzen 7 9700X, four active library threads |
| GPU used in exploratory cross-validation | NVIDIA RTX 5060 Ti 16 GB |

The reference Xeon determines official runtime eligibility; the timing above is a local measurement. The report is being prepared as an editable Word review copy and will be added after author review.

**Experimental limitation:** six test evaluations were made while working toward the requested numerical target. The aggregate scores influenced continuation of development. Consequently this is an adaptive public-bank result, not an untouched final-only holdout estimate. No individual test reference answers or test error cases were inspected for model development, and weights were fitted only using training supervision. See the full [development protocol and score history](experiments/PROTOCOL.md) and [AI disclosure](AI_DISCLOSURE.md). Instructor review and an independent audit bank are appropriate before treating this workflow as eligible under the strict final-testing rule.

## Reproduce the frozen score without training

Run all commands from the repository root with Python 3.12. No model download, GPU, API key, PyTorch, scikit-learn or CatBoost is needed for final inference.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-inference.txt
python code/check_data.py
python code/evaluate.py --split test
```

`evaluate.py` launches a fresh official prediction process and then the independent official scoring process, setting the relevant thread limits to four. It writes predictions, official score JSON, external elapsed times, asset sizes and SHA-256 hashes into `experiments/`. To preserve the released records, use `--output-dir runs/reproduction-evaluation`.

Equivalent official commands are:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python code/predict.py --method code/submission.py \
  --checkpoint code/checkpoints/final --split test \
  --output runs/test-predictions.json
python code/score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

Use the same frozen method on a new audit bank:

```bash
python code/predict.py --method code/submission.py \
  --checkpoint code/checkpoints/final --split audit \
  --data-dir /path/to/audit_inputs --output runs/audit-predictions.json
```

## Method

Three small networks share a candidate-wise interface and use the supplied 1024-dimensional representations, differences from the other candidates, answer agreement, and generic text descriptors. Two networks also receive an element-wise interaction with the group representation. Two use exponential parameter averaging. Invalid numeric answers are masked using the unchanged supplied parser.

The normalized network scores are averaged with equal weights. A candidate retains half its own evidence and receives half the mean evidence for candidates with the same parsed answer. A fixed penalty of 0.5 per detected inconsistent numeric equality provides additional evidence. Symbolic expressions are skipped and small rounding differences are tolerated; arithmetic correctness is not treated as a proof of semantic correctness.

Final asset and training configurations are in [the model bundle](code/checkpoints/final/ensemble.json) and [the exact selected training configurations](experiments/final_training_configs.json). The inference source and assets were hashed before the final scoring run in [the freeze record](experiments/freeze.json).

## Reproduce the selected training runs

Install CPU training dependencies in a separate Python 3.12 environment:

```bash
python -m pip install numpy==2.3.5
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
python code/reproduce_training.py --output runs/reproduced_bundle
python code/evaluate.py --checkpoint runs/reproduced_bundle \
  --split test --output-dir runs/retrained-evaluation
```

The three CPU training runs were reproduced locally with **zero difference in every learned weight**, taking approximately 25 seconds in total. Their scores require no re-selection on test data. This is stronger than relying only on a seed: the release includes the actual weights and hashes. Other hardware/library builds may introduce floating-point differences during retraining; the supplied NumPy checkpoint is the canonical inference artifact.

To reproduce the broader exploratory search, use `code/train.py`, `code/train_trees.py`, `code/train_kernel.py`, `code/train_lexical.py`, and `code/develop.py`. The complete configurations and curves for 67 completed exploratory runs are in [the experiment archive](experiments/exploratory/runs.json). The training dependency lock is retained separately.

GPU cross-validation and arithmetic-setting selection can be rerun as follows. This search is optional for reproducing the frozen score:

```bash
python -m pip install -r code/requirements-training.txt
python -m pip install torch==2.9.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
python code/train_cv.py --device cuda --epochs 80 --out runs/cv-refit
python code/tune_arithmetic.py
```

The five folds keep all four candidates of a question together. The OOF proxy selects the arithmetic penalty; it is not an unbiased OOF estimate for the final validation-selected three-network ensemble. CPU execution is available with `--device cpu`.

## Evidence and repository map

| Location | Contents |
|---|---|
| `code/submission.py`, `code/features.py` | Reference-free inference and arithmetic evidence |
| `code/checkpoints/final/` | Every required inference asset, stored without compression |
| `code/reproduce_training.py` | Exact selected training commands |
| `code/verify_method.py` | Export parity, new IDs, changed batch size and candidate permutation checks |
| `experiments/test-score.json` | Official final score |
| `experiments/test-resources.json` | Fresh-process resource measurement and asset hashes |
| `experiments/analysis/` | Validation ablations, uncertainty and error analysis |
| `experiments/initial/` through `experiments/fifth/` | Earlier frozen scoring records |
| `experiments/cross_validation/` | Train OOF search, failed ideas and arithmetic-penalty selection |
| `report/figures/` | Figures used in the Word review copy |
| `guide/GUIDE.md`, `code/README.md` | Original course guide and running instructions |

The official data, numeric parser and scoring code are unchanged. Evaluation predictions in `experiments/` are evidence only; inference never reads them. No IDs, cached choices, external datasets or additional pretrained weights enter the method.

## Attribution and submission status

This repository is a fork of [the course package](https://github.com/xudongwu-0/DASE7506-MP2). GSM8K's retained MIT license and attribution are in [LICENSE](code/data/LICENSE.txt) and [NOTICE](code/data/NOTICE.txt). Candidate traces and fixed representations were supplied by the course and produced with Qwen3-0.6B.

Substantive OpenAI Codex assistance is disclosed in [AI_DISCLOSURE.md](AI_DISCLOSURE.md). The student author must review and understand the implementation and report. No course leaderboard issue has been created by this work; author review, the student ID, and the final submission action remain separate from publishing this repository.

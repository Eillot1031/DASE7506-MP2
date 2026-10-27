# Selecting Reliable LLM Reasoning

DASE7506 Advanced Machine Learning Methods · Mini Project 2 · Eillot1031

Select one of four supplied mathematical solutions using compact candidate verifiers, comparison between answers, and a numerical consistency check. The latest frozen model scores **84.58244111349036% (395/467)** on the **complete public test bank** with the unchanged official scorer. This improves the previous 84.154% model by two questions; two further correct questions would give 85.011%.

| Frozen result | Measurement |
|---|---:|
| Complete test accuracy | **84.582% (395/467)** |
| Complete validation accuracy | 94.862% (757/798) |
| Difficult validation accuracy | 88.451% (314/355) |
| Fresh CPU prediction plus independent scoring | 0.916 s |
| All required inference assets | 4,899,529 bytes / 4.673 MiB |
| Local CPU | AMD Ryzen 7 9700X, four active library threads |
| GPU used in exploratory cross-validation | NVIDIA RTX 5060 Ti 16 GB |

The reference Xeon determines official runtime eligibility; the timing above is a local measurement. The editable Word report is available locally for author review and has not been published. The latest checkpoint is `code/checkpoints/candidate85/`; the earlier 84.154% checkpoint remains unchanged in `code/checkpoints/final/` for historical reproduction. Always specify the desired checkpoint explicitly.

**Experimental limitation:** seven distinct frozen methods were tested while working toward the requested numerical targets. Aggregate scores influenced continuation of development. The latest round selected its candidate using train OOF and validation, froze it, then ran one complete test evaluation. This remains an adaptive public-bank result, not an untouched final-only holdout estimate. No individual test reference answers or test error cases were inspected for model development, and weights were fitted only using training supervision. See the full [development protocol and score history](experiments/PROTOCOL.md) and [AI disclosure](AI_DISCLOSURE.md). Instructor review and an independent audit bank are appropriate before treating this workflow as eligible under the strict final-testing rule.

## Reproduce the frozen score without training

Run all commands from the repository root with Python 3.12. No model download, GPU, API key, PyTorch, scikit-learn or CatBoost is needed for final inference.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-inference.txt
python code/check_data.py
python code/evaluate.py --split test --checkpoint code/checkpoints/candidate85 \
  --output-dir runs/reproduction-evaluation
```

`evaluate.py` launches a fresh official prediction process and then the independent official scoring process, setting the relevant thread limits to four. It writes predictions, official score JSON, external elapsed times, asset sizes and SHA-256 hashes to the chosen output directory. The historical default checkpoint remains the earlier model, so the explicit checkpoint argument above is required for 84.582%.

Equivalent official commands are:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python code/predict.py --method code/submission.py \
  --checkpoint code/checkpoints/candidate85 --split test \
  --output runs/test-predictions.json
python code/score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

Use the same frozen method on a new audit bank:

```bash
python code/predict.py --method code/submission.py \
  --checkpoint code/checkpoints/candidate85 --split audit \
  --data-dir /path/to/audit_inputs --output runs/audit-predictions.json
```

## Method

The previous three-network ensemble uses the supplied 1024-dimensional representations, differences from the other candidates, answer agreement, and generic text descriptors. Two networks also receive an element-wise interaction with the group representation. Two use exponential parameter averaging. This ensemble supplies 90% of the latest neural score.

A three-seed verifier inspired by [Deep Sets](https://arxiv.org/abs/1703.06114) supplies the remaining 10%. Its shared encoder produces 64 features per candidate; the scoring head compares these with the valid-candidate mean, maximum and mean for the same answer. Training uses only difficult training questions. Five-fold training CV selects the family and epoch, and validation selects a small blend from a fixed grid. The complete inference bundle contains six compact networks and uses NumPy only. [Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412) was also explored but was not retained.

A candidate retains half its own neural score and receives half the mean score for the same parsed answer. The revised numeric checker supports percentages, powers and additional fraction notation, and avoids checking numeric fragments of symbolic expressions. The final penalty is 0.5 times the same-answer mean count of inconsistent numeric equalities, with relative-error threshold 0.001. Local symbolic evidence was explored but received zero final weight. Invalid answers are masked using the unchanged official parser. Arithmetic consistency is evidence, not proof of semantic correctness.

See [the latest bundle configuration](code/checkpoints/candidate85/ensemble.json), [train-OOF settings](experiments/optimization85/set_selected.json), [the pre-scoring freeze](experiments/optimization85/freeze.json), and [the full latest-round log](experiments/optimization85/README.md). The original three-network configurations remain in [the baseline training record](experiments/final_training_configs.json).

## Reproduce the selected training runs

For the latest bundle, use CUDA PyTorch 2.9.0 and the frozen refit script. It trains the three new seeds at epoch 19 without any model selection, copying the original three released networks:

```bash
python -m pip install -r code/requirements-training.txt
python -m pip install torch==2.9.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
python code/reproduce_optimization85.py --output runs/reproduced85
python code/evaluate.py --checkpoint runs/reproduced85 \
  --split test --output-dir runs/retrained85-evaluation
```

The new networks reproduced every learned array exactly on the local GPU in 5.061 seconds. To rebuild the original networks as well, use the CPU procedure below and pass `--base runs/reproduced_bundle` to `reproduce_optimization85.py`. NumPy-only inference does not require a GPU.

Install CPU training dependencies in a separate Python 3.12 environment:

```bash
python -m pip install numpy==2.3.5
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
python code/reproduce_training.py --output runs/reproduced_bundle
python code/evaluate.py --checkpoint runs/reproduced_bundle \
  --split test --output-dir runs/retrained-evaluation
```

The three original CPU training runs were reproduced locally with **zero difference in every learned weight**, taking approximately 25 seconds in total. This command reproduces the earlier 84.154% model. No test-based re-selection occurs in either frozen training reproduction. Other hardware/library builds may introduce floating-point differences; the released NumPy checkpoints are canonical.

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

The latest optional GPU searches are `code/train_set_verifier.py` and `code/train_answer_verifier.py`, each comparing six configurations with five folds and 100 epochs. `code/tune_arithmetic_v2.py` selects the revised penalty using the saved train-OOF proxy. They do not need to be rerun for inference or frozen training reproduction.

## Evidence and repository map

| Location | Contents |
|---|---|
| `code/submission.py`, `code/features.py`, `code/arithmetic_v2.py` | Reference-free inference and arithmetic evidence |
| `code/checkpoints/candidate85/` | Latest complete inference bundle, stored without compression |
| `code/checkpoints/final/` | Preserved earlier 84.154% bundle |
| `code/reproduce_training.py` | Exact selected training commands |
| `code/verify_method.py` | Export parity, new IDs, changed batch size and candidate permutation checks |
| `experiments/optimization85/test-score.json` | Latest official score, 395/467 |
| `experiments/optimization85/test-resources.json` | Latest fresh-process measurement and asset hashes |
| `experiments/optimization85/analysis/` | Latest validation ablations, uncertainty and errors |
| `experiments/test-score.json`, `experiments/analysis/` | Preserved earlier score and analyses |
| `experiments/initial/` through `experiments/fifth/` | Earlier frozen scoring records |
| `experiments/cross_validation/` | Train OOF search, failed ideas and arithmetic-penalty selection |
| `report/figures85/` | Latest figures used in the Word review copy |
| `guide/GUIDE.md`, `code/README.md` | Original course guide and running instructions |

The official data, numeric parser and scoring code are unchanged. Evaluation predictions in `experiments/` are evidence only; inference never reads them. No IDs, cached choices, external datasets or additional pretrained weights enter the method.

## Attribution and submission status

This repository is a fork of [the course package](https://github.com/xudongwu-0/DASE7506-MP2). GSM8K's retained MIT license and attribution are in [LICENSE](code/data/LICENSE.txt) and [NOTICE](code/data/NOTICE.txt). Candidate traces and fixed representations were supplied by the course and produced with Qwen3-0.6B.

Substantive OpenAI Codex assistance is disclosed in [AI_DISCLOSURE.md](AI_DISCLOSURE.md). The student author must review and understand the implementation and report. No course leaderboard issue has been created by this work; author review, the student ID, and the final submission action remain separate from publishing this repository.

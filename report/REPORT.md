# Selecting Reliable LLM Reasoning

DASE7506 Advanced Machine Learning Methods  
Mini Project 2    Eillot1031    27 October 2026

## Abstract

A compact ensemble selects a reference-correct numerical answer for 395 of 467 questions in the complete public test bank, giving 84.582% accuracy. Three candidate verifiers are combined with three seeds of a DeepSets comparison network and a numerical equality check. The full CPU prediction and scoring processes take 0.916 seconds locally, with 4.673 MiB of inference assets. This improves the previous frozen result by two questions; the subsequent 85% aspiration remains two correct questions away.

Agreement alone is insufficient: difficult-validation accuracy improves from 69.014% for majority voting to 88.451%. The latest extension improves validation by only one question, so evidence for its incremental benefit is limited. High answer-score margins identify an easier subset, but scores are not calibrated probabilities. Seven distinct frozen methods were tested during development; the final result is an adaptive benchmark result rather than an untouched holdout estimate.

## Task and research question

Each question supplies four generated solution texts and a 1024-dimensional representation for every candidate. The output is one candidate index. The unchanged official parser and scorer compare the selected final numeric answer with the reference; they do not grade every intermediate reasoning step [1, 2]. The model performs offline selection and never calls an LLM during evaluation.

The research question is whether comparing candidate evidence and checking numerical consistency improves selection beyond consensus, while remaining small and fast enough for the course limits. A related question is whether the resulting scores meaningfully identify uncertainty.

| Split | Questions | Difficult groups |
| --- | --- | --- |
| Training | 4,755 | 2,114 |
| Validation | 798 | 355 |
| Test | 467 | 467 by protocol |

A difficult group has at least two distinct valid parsed answers. Each supplied group has at least one reference-correct candidate. Easier groups explain much of the difference between complete-validation and difficult-validation accuracy.

## Method and implementation

### Candidate evidence

The supplied embedding is normalized to length 32. Candidate features concatenate this representation, its difference from the within-question mean, and handcrafted descriptors. The descriptors include valid-answer agreement, logarithmic trace length and answer magnitude, numerical overlap with the question, formatting counts, and generic lexical indicators. Each descriptor is accompanied by its deviation from the group mean. These are general functions of the provided inputs; question IDs are never features.

Two networks additionally receive the element-wise product of the candidate embedding and the group-mean embedding. This supplies a simple interaction with the surrounding candidates. All features use training means and scales, with a standard-deviation floor of 0.15 and standardized values clipped to [-10, 10]. The final networks have two hidden layers, with widths shown below, ReLU activations and training dropout.

| Member | Hidden widths | Objective | Seed | Frozen epoch |
| --- | --- | --- | --- | --- |
| ema42 | 128 and 32 | BCE with EMA | 42 | 46 |
| ema_context73 | 64 and 32 | BCE with EMA | 73 | 8 |
| context17 | 64 and 32 | List loss plus BCE | 17 | 132 |
| DeepSets x3 | 64 and 32 | BCE with EMA | 17, 42, 73 | 19 |

Training uses AdamW with learning rate 0.001 and dropout 0.4. Weight decay is 0.01 for BCE and 0.1 for list loss; EMA is 0.99 where used. Gradients use the 2,114 difficult training groups and exclude unparseable candidates. The original three models standardize using all training inputs; DeepSets uses difficult training inputs. Following set-based shared encoding and symmetric aggregation [5], its 64-dimensional encoder is compared with the valid-candidate mean, maximum and same-answer mean. Concatenating these with the candidate, its mean difference and product produces 384 features for the 32-to-1 head, preserving candidate permutation equivariance.

### Answer comparison and arithmetic evidence

Scores are divided by frozen within-question scales. The original ensemble supplies 90% of the neural score; the mean of three DeepSets seeds supplies 10%. Same-answer evidence then supplies half the candidate score. The revised arithmetic penalty is also averaged within each numeric answer group.

```text
s_i = 0.9 mean_old(f_i/T) + 0.1 mean_set(f_i/T)
g_i = 0.5 s_i + 0.5 mean_same(s_j) - 0.5 mean_same(b_j)
```

Here b counts numeric equalities with relative error above 0.001, using max(1, |left|, |right|) as denominator. The checker handles fractions, percentages and bounded powers, and avoids treating numeric tails of symbolic terms as full equations. Local-variable evidence was explored but has zero final weight. Unparseable answers are masked. Locally consistent arithmetic does not establish semantic correctness.

## Experimental protocol and test history

Exploration compared answer-only features, handcrafted descriptors, supplied embeddings, relative embeddings, neural objectives, regularization, random seeds, boosted trees, sparse lexical models and kernel verifiers. Complete configurations and learning curves for 67 completed exploratory runs are retained. Validation selected base models and ensemble combinations, with difficult-validation accuracy as the primary criterion. Forward selection considered at most six members and a short answer-pooling grid.

GPU five-fold studies keep all four candidates together. The initial 24-family search selected common epochs using five-epoch-smoothed OOF curves; its refits did not exceed the original validation-selected ensemble. A further six DeepSets and six answer-balanced/SAM [6] configurations used nine-epoch smoothing over 100 epochs. Their best smoothed OOF scores were 82.103% and 82.093%. A fixed 17-setting validation blend grid selected the 10% DeepSets contribution. The train-OOF arithmetic proxy improved from 83.964% to 84.248% with the revised checker; this is not an OOF estimate of the final six-network ensemble.

| Evaluation | Method stage | Test correct | Test accuracy |
| --- | --- | --- | --- |
| 1 | Initial ensemble | 383 | 82.013% |
| 2 | Hard groups and averaging | 389 | 83.298% |
| 3 | Single CV family | 384 | 82.227% |
| 4 | Multiple CV families | 385 | 82.441% |
| 5 | OOF proxy and arithmetic | 387 | 82.869% |
| 6 | Original frozen ensemble | 393 | 84.154% |
| 7 | DeepSets blend and revised checker | 395 | 84.582% |

### Interpretation of the development history

Aggregate test scores monitored the requested 84% target and subsequent 85% aspiration; development continued after earlier scores. This is adaptive test feedback and does not satisfy a strictly one-shot, final-only protocol. No individual test references or error cases were inspected for fitting or feature design. The latest round froze its train/validation-selected candidate before one complete test evaluation and made no subsequent score-based changes. These safeguards do not make the test bank untouched.

All seven official results and freeze records are retained. The instructor determines eligibility under the strict course rule. An independent audit bank is needed for a clean assessment. Validation also reflects model and blend selection; its one-question latest gain is development evidence, not an independent significance result.

## Ablations and comparative results

| Method | Complete validation | Difficult validation |
| --- | --- | --- |
| First candidate | 78.195% | 54.930% |
| Majority vote | 86.216% | 69.014% |
| Question and answer | 88.471% | 74.085% |
| Handcrafted descriptors | 90.977% | 79.718% |
| Embedding only | 91.729% | 81.408% |
| Relative embeddings | 91.729% | 81.408% |
| Combined single network | 92.356% | 82.817% |
| Previous frozen ensemble | 94.737% | 88.169% |
| Final frozen ensemble | 94.862% | 88.451% |

![Validation comparison](figures85/validation_comparison.png)

Figure 1  Difficult-validation comparisons. Ablations are development runs; the final model also benefits from seed and ensemble selection.

Answer agreement provides a substantial baseline. Handcrafted descriptors and embeddings each add predictive information, while the combined model is stronger than either single information source in these runs. This supports using complementary evidence, but it does not establish that the model has learned a faithful mathematical proof checker: formatting, answer statistics and generator-specific patterns remain available shortcuts.

On the same six frozen networks, removing the arithmetic penalty gives 87.606% difficult-validation accuracy; adding it gives 88.451%, a gain of three validation questions. The entire extension improves the previous complete method by one validation question and two test questions. The answer-balanced/SAM family was rejected: its standalone difficult-validation accuracy was 85.634%, and its tested blends did not exceed the selected method. Adding complexity alone was insufficient.

## Uncertainty and selective accuracy

![Reliability](figures85/reliability.png)

Figure 2  Validation reliability diagnostics. Left: retaining questions with larger distinct-answer margins. Right: uncalibrated softmax confidence over distinct answers.

| Coverage on difficult validation | Retained questions | Accuracy |
| --- | --- | --- |
| 25.1% | 89 | 100.000% |
| 50.1% | 178 | 97.191% |
| 75.2% | 267 | 92.884% |
| 100.0% | 355 | 88.451% |

The margin is the difference between the best and second-best distinct-answer score. It ranks uncertainty more usefully than treating every selected answer as equally reliable: the highest-margin quarter contains 89 questions and is entirely reference-correct in this validation sample. This is an exploratory observation on the development split; it does not guarantee 100% precision on unseen questions.

The reliability plot uses five equal-count bins and an uncalibrated softmax over distinct-answer scores. Its expected calibration error is 13.53 percentage points. Score scales were normalized for fusion, not fitted as probabilities. These values therefore should not be interpreted as calibrated likelihoods of correctness. A new calibration split or nested calibration procedure would be required before deploying a confidence threshold.

Compared with majority voting, the final method fixes 82 difficult-validation questions while regressing on 13. The net gain is substantial, but individual high-margin errors remain. Correlated candidate traces can agree on the same omitted condition, and a confident learned score can reinforce that shared mistake.

## Failure analysis and limits of correctness labels

### An omitted quantity despite correct arithmetic

In a validation question, a class has 250 students, all 140 girls attend, and attending girls are twice as numerous as attending boys. The reference answer is 40 absent boys: 250 - 140 - 70. Three candidates instead return 180 by subtracting only 70 attending boys from the total. The selected wrong trace contains locally correct calculations. The numerical checker cannot detect the missing group of 140 girls; the failure is about representing the question conditions.

### A plausible reading can disagree with the reference

Another question says Yun starts with 20 paperclips and loses 12, while Marion has "1/4 more than" Yun now has, plus 7. The supplied reference is 9, following (20 - 12)/4 + 7. The model selects 17, following 8 + 8/4 + 7, which is a natural reading of "one quarter more than". The official scorer correctly marks it wrong under its reference, but the example cautions against equating every reference mismatch with demonstrably invalid reasoning. No reference or scorer was changed.

### Reversing a relation between entities

In a pens-and-pencils question with 108 utensils, the reference relation gives 16 pens, while the selected candidate reverses which entity equals five times the other plus 12 and returns 92. Both alternatives contain consistent algebra once the relation is chosen. Distinguishing them requires grounding the equation in the wording, beyond checking arithmetic or consensus.

### What the method can and cannot establish

The method is an empirical verifier of final answers within the supplied candidate distribution. It checks only a subset of explicit calculations and cannot prove completeness of a solution, validate arbitrary symbolic algebra, or guarantee the intended interpretation of ambiguous language. Conversely, a correct final answer may contain incorrect intermediate steps. The published metric should be interpreted according to this scope.

The strongest limitations are adaptive public-test feedback, a relatively small difficult-validation split, generator-specific features, and correlated mistakes across candidates. Increasing network capacity, adding many arithmetic-relation features, or learning a nonlinear score fusion did not reliably improve the train OOF results. Those failures argue for better evidence and cleaner evaluation, rather than further uncontrolled search.

## Resources and reproducibility

| Resource | Measured value or configuration |
| --- | --- |
| Frozen asset files | 4,899,529 bytes  4.673 MiB |
| Fresh prediction process | 0.859 seconds |
| Independent scoring process | 0.057 seconds |
| Total complete-bank CPU evaluation | 0.916 seconds |
| Evaluation hardware | Ryzen 7 9700X  four library threads |
| Training search GPU | RTX 5060 Ti 16 GB  PyTorch 2.9.0 CUDA 13.0 |
| Frozen training reproduction | Original CPU 24.987 s  new GPU seeds 5.061 s |

These totals include new-process startup, imports, supplied input loading, feature extraction, all ensemble members and independent scoring. They are local measurements; the Xeon Platinum 8457C reference machine remains authoritative for eligibility. The final assets are NumPy arrays stored without compression, including all learned means, scales, weights and score normalization constants.

The original 67 exploratory runs record 1,010.390 seconds of summed per-run work, not wall-clock duration. The initial GPU search contains 24 families times five folds times 80 epochs and reused caches. The latest two searches each contain six configurations times five folds times 100 epochs; measured times are retained in experiments/optimization85/. A single wall-clock total for all development was not captured. Frozen training reproduction does not repeat model selection.

```text
python -m pip install -r code/requirements-inference.txt
python code/evaluate.py --split test \
  --checkpoint code/checkpoints/candidate85 \
  --output-dir runs/reproduction85
```

Use Python 3.12 and NumPy 2.3.5. For training, code/reproduce_training.py rebuilds the original CPU models; code/reproduce_optimization85.py refits the new CUDA seeds at fixed epoch 19. Both reproduced all learned arrays exactly locally. The latter can consume the rebuilt baseline via --base. The supplied parser, data and scorer remain unchanged. Export parity, independent IDs/batches, two candidate permutations and arithmetic regression checks passed.

Pre-scoring manifest SHA256: 8532186c533b1086d05a55f28fabdc44acf17b9b5870e6f309d088651a7f4b5d

Project repository: https://github.com/Eillot1031/DASE7506-MP2

All asset sizes and full hashes are in experiments/optimization85/test-resources.json. Pre-scoring source and asset hashes are in experiments/optimization85/freeze.json. The earlier checkpoint and its six scoring records remain intact.

## AI disclosure attribution and further work

### AI assistance disclosure

OpenAI Codex provided substantive assistance in reading the project requirements, implementing feature extraction and training code, running experiments, diagnosing validation overfitting, preparing reproducibility records, drafting this report and the README, and publishing code. The reported numerical scores come from execution of the unchanged independent course scorer. No score was fabricated or manually inserted into the prediction output.

The student author is responsible for reviewing and understanding the implementation, checking the analysis, and accurately disclosing the assistance and the adaptive testing history. No LLM generation or API call occurs during evaluation. No external training dataset, additional pretrained weights, question-ID feature or answer lookup is used. Evaluation prediction files are retained as evidence and are not read by inference.

### Attribution and provenance

The data, fixed representations, interface and official evaluation tools are reused from the course repository [1, 2]. Questions and references derive from GSM8K under the retained MIT license. Candidate traces and representations were prepared by the course with Qwen3-0.6B, revision c1899de289a04d12100db370d81485cdf75e47ca [3, 4]. No Qwen weights are distributed here. The original NOTICE and LICENSE remain in code/data/.

### Further work

An independent audit bank is the next defensible evaluation. Better grounding of entities and quantities and a held-out calibration procedure remain useful directions. The latest public score is 84.582%, exceeding the original 84% target but missing the subsequent 85% aspiration by two correct questions. Additional public-score tuning was stopped after the single frozen evaluation in this round; it would compound the adaptive testing limitation.

### References

[1] DASE7506 Mini Project 2 guide. https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md

[2] DASE7506 code and running instructions. https://github.com/xudongwu-0/DASE7506-MP2/blob/main/code/README.md

[3] Course data manifest, including generation revision and split hashes. code/data/manifest.json in the frozen repository version.

[4] GSM8K attribution and MIT license retained by the course. code/data/NOTICE.txt and code/data/LICENSE.txt in the frozen repository version.

[5] Zaheer et al. Deep Sets. NeurIPS 2017. https://arxiv.org/abs/1703.06114

[6] Foret et al. Sharpness-Aware Minimization for Efficiently Improving Generalization. ICLR 2021. https://arxiv.org/abs/2010.01412

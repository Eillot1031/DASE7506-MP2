# Development protocol and score history

Training supervision was restricted to supplied training references. Validation was used for exploratory model and ensemble selection. GPU group cross-validation used training questions only; each question's four candidates remained in the same fold. The arithmetic penalty was selected using OOF training scores from a proxy ensemble. The final base ensemble was selected using validation, rather than adopting the lower-validation CV refits.

## Public test evaluations during development

| Order | Frozen method | Correct / 467 | Accuracy |
|---|---|---:|---:|
| 1 | Initial three-network ensemble | 383 | 82.01284796573876% |
| 2 | Hard-group training and parameter averaging | 389 | 83.29764453961457% |
| 3 | Single-family group-CV refit | 384 | 82.22698072805139% |
| 4 | Multi-family group-CV refit | 385 | 82.44111349036403% |
| 5 | OOF proxy ensemble with numeric consistency evidence | 387 | 82.86937901498929% |
| 6 | Validation-selected base ensemble with OOF-selected arithmetic penalty | **393** | **84.1541755888651%** |
| 7 | Frozen 90% previous ensemble, 10% three-seed DeepSets, revised OOF-selected arithmetic rule | **395** | **84.58244111349036%** |

Each evaluation followed a freeze of its inputs and configuration. The first five records are retained in `initial/`, `second/`, `third/`, `fourth/` and `fifth/`; evaluation six is preserved in `test-score.json`, and evaluation seven is in `optimization85/test-score.json`. Later verification of the same frozen weights, when recorded, is reproduction rather than further method selection.

The latest round compared six DeepSets architectures/objectives and six answer-balanced/SAM configurations using five-fold training-group CV. New common epochs used nine-epoch-smoothed curves. A fixed 17-setting validation blend check selected 10% DeepSets evidence, improving validation from 756/798 to 757/798. A training-OOF proxy selected the revised arithmetic parser, penalty strength and same-answer pooling. The single candidate was hashed in `optimization85/freeze.json` before its one complete test evaluation. No subsequent test-based tuning was performed in that round. The prior checkpoint remains intact.

The aggregate test score was used to monitor progress toward a user-specified numerical target; development continued after earlier shortfalls. This is adaptive test feedback, even though individual test reference values and error cases were not inspected for feature design or training. The result therefore cannot be represented as a strictly final-only or untouched holdout estimate. The instructor should assess whether this development history is acceptable under the course's strict test-use constraint. An independent audit bank is the appropriate way to measure generalization after this process.

## Evidence limits

- Complete validation includes 443 groups with only one distinct valid parsed answer. The difficult subset contains 355 groups, matching the test input condition of at least two distinct valid parsed answers.
- Validation results are development measurements, including seed, epoch and ensemble selection. Their uncertainty is larger than an unselected binomial interval suggests.
- The 83.964% OOF arithmetic result belongs to a CV-selected proxy ensemble on 2,114 training groups. It is not the OOF accuracy of the final three-network model.
- No test result was manually entered into predictions. The independent supplied scorer generated all reported accuracy fields. All data and scoring source remain unchanged.
- The final numeric answer is the target. A reference-labelled error can still have a linguistically plausible interpretation, and a correct numeric answer can follow flawed intermediate reasoning.
- Local runtime was measured on a Ryzen 7 9700X, not the official reference Xeon. Reference-machine eligibility remains subject to course reproduction.

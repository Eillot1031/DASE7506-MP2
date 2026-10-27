# Further optimization toward 85 percent

The frozen candidate scored **395/467 = 84.58244111349036%**, compared with the previous **393/467 = 84.1541755888651%**. Two more correct questions would yield 397/467 = 85.01070663811563%. The requested aspiration was not reached; both versions remain available.

## Selection before the new test evaluation

1. Keep the previous three networks and checkpoint unchanged.
2. Compare six DeepSets configurations on the 2,114 difficult training questions using five folds, split seed 2026, 100 epochs and nine-epoch smoothing. Best family: 64-dimensional encoder, full features, BCE, dropout 0.4, AdamW decay 0.01 and EMA 0.99, epoch 19. Smoothed training OOF accuracy was 82.103%. Refit seeds 17, 42, 73 without further epoch search. Standalone difficult-validation accuracy was 84.507%.
3. Compare six answer-balanced or SAM configurations under the same protocol. Best: full features, width 128, inverse-answer-frequency BCE, SAM radius 0.05 and EMA 0.99, epoch 30. Smoothed training OOF was 82.093%; standalone difficult validation was 85.634%. This family was not retained in the final blend.
4. Expand the numeric parser to handle percentages, bounded powers, spaced fractions and numeric parentheses. Prevent algebraic fragments such as `x + 1 = 9` from becoming the false numeric claim `+1 = 9`. Local single-letter-variable checks were also explored, but their final coefficient is zero. Regression checks cover correct and inconsistent syntax, symbolic fragments and unsafe/unbounded AST expressions.
5. On the existing train-OOF proxy, select the revised numeric error count, threshold 0.001, penalty weight 0.5, same-answer mean pooling. The proxy improves from 83.964% with the previous checker to 84.248%. This is not OOF accuracy of the final six-network model.
6. Check a fixed 17-setting validation grid: original only, or one of two new families at weight 0.1/0.25/0.5/0.75 and pooling 0.5/1. The winning configuration is 90% original neural score and 10% DeepSets score, pooling 0.5. Complete validation improves from 756/798 to 757/798; difficult validation from 313/355 to 314/355. Ties prefer the earlier, smaller-grid entry.
7. Verify NumPy export parity (maximum logit error 9.54e-7), independent IDs/batches and two candidate permutations. Record all source/asset hashes in `freeze.json`. Run one official complete test evaluation after this freeze. No settings were changed after its aggregate score was returned.

The DeepSets head sees each encoded candidate, the valid-candidate mean, their difference and product, the same-answer mean, and the valid-candidate maximum. All operations are symmetric over the candidate set, so candidate positions and question IDs supply no evidence. The exported head is a 384-to-32-to-1 MLP with ReLU; all inference uses NumPy.

## Resources and reproduction

Both searches used an RTX 5060 Ti 16 GB with PyTorch 2.9.0+cu130. `set_resources.json` and `answer_resources.json` retain measured elapsed times. Refitting the three selected new seeds took 5.061 seconds and reproduced every learned array exactly; the original three arrays were copied from their separately reproduced baseline. See `training_reproduction.json`.

The latest inference checkpoint is `code/checkpoints/candidate85/`, containing 4,899,529 bytes (4.673 MiB). Fresh CPU prediction plus independent scoring took 0.916 seconds on a Ryzen 7 9700X with four active library threads. Reference Xeon runtime eligibility remains subject to official reproduction.

```bash
python code/evaluate.py --checkpoint code/checkpoints/candidate85 \
  --split test --output-dir runs/reproduction85
python code/reproduce_optimization85.py --output runs/reproduced85
```

The second command requires the documented CUDA training environment. It only refits the frozen epoch and seeds. Reproducing the complete search is optional; search curves are generated under ignored `runs/optimization85/`, while their configurations, epoch counts and summaries are retained here.

## Evidence limits

Earlier aggregate test scores had already influenced continuation of the project. This seventh frozen-method score is therefore adaptive public-bank evidence, even though this round used only train/validation to select its candidate. No individual test labels or errors were inspected for development. The one-question validation gain is small and selected from a grid; it does not establish a statistically independent improvement. Independent audit evaluation is still needed.

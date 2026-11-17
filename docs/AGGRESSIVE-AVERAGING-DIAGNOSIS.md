# 100-BB aggressive play and inherited averaging diagnosis

Read-only audit of the final 969,520-iteration snapshot on the already consumed
300 development deals for the six-player 100-BB aggressive condition (seed
61461000, all six hero positions, 1,800 hands). No training, policy replacement,
new validation or fresh confirmation was performed. The replay returns exactly
+4.02778 bb/100, matching the campaign report's +4.03 for this condition.

## Findings

Across 2,455 learned decisions, 88.7% of average-policy mass came from the
900,520-iteration parent; 86.0% came from the original deployed incumbent.
Postflop, across 610 learned decisions, these fractions were 89.8% and 87.8%.
These are decision-weighted means on this replay, not global table inventories.
Inherited fractions compare nonnegative accumulated average rows at identical keys.

The average policy moved only 2.47 percentage points in total variation from its
parent on average (postflop 2.38 points). Average versus current positive-regret
policy disagreement was much larger: 22.16 points overall and 24.54 postflop.
This supports slow adaptation of the inherited average as a concrete mechanism
worth testing. It does not establish that the current regret policy is stronger,
or that discarding the inherited averages would improve returns.

On learned postflop decisions facing a bet, 102 actions continued when the equity
fallback preferred folding, versus six learned folds where fallback continued.
The equity fallback uses heuristic range priors; its recommendations are not an
oracle and these counts are neither independent samples nor proven errors.

Representative -100-BB outcomes include:

- AQ offsuit on 9h Jc 3c called the flop, called an Ac turn and raised on a 7s
  river. At the river coordinate the average assigned 78.7% probability to the
  raise and effectively zero to folding. Heuristic equity was 27.9% versus a
  33.3% immediate call-price threshold. This is a priority bluff-catching/value
  classification example, not an exact range-solved verdict.
- K8 suited raised preflop, reraised facing pressure, then continued for its
  remaining stack with a pair of eights on Js 6h 8d. The final learned row put
  88.5% on the raise. This motivates auditing weak preflop reraises and subsequent
  commitment; the hand's final loss alone cannot prove the earlier actions wrong.

Sixteen of 1,752 visited abstract keys appeared with more than one legal numeric
menu. This is expected from pot/price/SPR buckets and pot-relative action slots,
but confirms loss of exact sizing context. It is a candidate explanation for
strategy transfer failures, not proof of a corrupt action coordinate.

## Next controlled investigation

Before a larger training run, compare inherited averages with new-distribution-only
average deltas on the same consumed development deals, requiring identical keys,
legal menu lengths and explicit support checks. Preserve inherited regrets and
optimizer state for both arms; do not deploy an unvalidated current-regret policy.
Separately inspect the high-commitment preflop/postflop coordinates and whether
merged card/history/range situations require different actions. Only a promising,
repeatable development result should justify another training budget or fresh
confirmation seed.

## Reproduce and validation

```sh
.venv/bin/python -m scripts.diagnose_aggressive_history \
  --candidate runs/bot/aggressive-stages-001/stage-10.json \
  --parent runs/bot/live-curriculum-002/candidate-581/checkpoint.json \
  --incumbent runs/bot/history-sustained-001/checkpoint.json \
  --state runs/bot/aggressive-stages-001/training/state \
  --out runs/bot/aggressive-diagnosis-new.json
```

The report preserves checkpoint/source hashes, visited-row statistics, regret
comparisons, all learned decisions and worst-hand decision traces. The original
local artifact is `runs/bot/aggressive-diagnosis-001/report.json`. Large checkpoints
and detailed artifacts remain under ignored `runs/`; this report retains the key
findings in the repository. The new regression test verifies exact inherited mass,
visit deltas and average/regret disagreement using a small synthetic checkpoint.

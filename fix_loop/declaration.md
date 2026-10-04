# Fix declaration

_One page. Written BEFORE the fix is shipped._

## 1. Worst-performing gate (own benchmark)

- Gate: <name>
- Failing number: <measured value> vs. gate <threshold>
- Evidence run: `fix_loop/before/`

## 2. Root-cause hypothesis

- Hypothesis: <one sentence>
- Evidence: <plots, per-room errors, log excerpts, ablations that point at it>

## 3. Fix and prediction

- Fix to ship: <what changes, which files>
- Predicted number after fix: <value>
- Why this prediction: <reasoning>

## Result (fill after the fix)

- After run: `fix_loop/after/`
- Actual number: <value>
- Prediction vs. actual: <delta, post-mortem if wrong>
- Diff: `git diff <before-commit> <after-commit>`

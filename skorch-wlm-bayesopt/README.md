# codimango/skorch-wlm-bayesopt

## Description

Replaces the exhaustive `GridSearchCV(net, params)` hyperparameter search in
skorch's `examples/word_language_model/train.py` with **Bayesian optimization**
that samples a bounded number of configurations (an explicit iteration cap)
instead of evaluating the full Cartesian product. Two things make it non-trivial:
1. The search lives in module-level code that runs real PTB training on import,
   so the solution must refactor `train.py` to be **import-safe** (guard the
   training under `if __name__ == '__main__':`) and expose the search as an
   **importable, standalone builder function** that can be exercised in isolation.
2. That builder must accept a caller-provided estimator, search space, and
   iteration cap and return a configured search **without running any training**,
   so it can be unit-tested cheaply and deterministically.

- **Repo:** `skorch-dev/skorch` @ `e769a2b89dc6e946dfaf42654184aa01b7025964`
- **Language:** Python
- **Files touched by the reference solution:** `examples/word_language_model/train.py`

## Reference solution (oracle)

- `train.py`: imports change from `GridSearchCV` to `skopt.BayesSearchCV`
  (+ `skopt.space.Real`); a new `build_search(estimator, search_spaces,
  n_iter=…, random_state=…)` factory returns a capped `BayesSearchCV`; a new
  `--search-iter` flag (default `DEFAULT_SEARCH_ITERATIONS = 16`) is the
  iteration cap; the grid `[{'lr':[10,20,30]}]` becomes a continuous
  `{'lr': Real(1.0, 40.0, 'log-uniform')}` searched by `build_search(...)`; all
  the previously module-level code (argparse, corpus, `Net`, `.fit`) is wrapped
  in `main()` under `if __name__ == '__main__':`. `net.py` / `model.py` are
  untouched.

## Completion Rates

Measured by the platform during validation (commit `7b2701f`):

| Agent | Model | Attempts | Pass | Pass rate |
|-------|-------|----------|------|-----------|
| oracle | oracle | 3 | 3 | 1.000 |
| metacode | avocado_dvsc_tester | 5 | 4 | 0.800 |
| claude-code | claude-opus-4-6 | 5 | 2 | 0.400 |
| (aux) | gpt-5.5 | 5 | 0 | 0.000 |

Balance gate: **passed** — avocado not trivial (4/5, not a clean sweep) and
≥1 agent solved (opus 2/5).

## Model Analysis

The task is a real ML-engineering change (cap HPO cost by replacing exhaustive
grid with Bayesian optimization) plus an import-safety refactor that makes the
search unit-testable.

- **avocado 4/5, opus 2/5, gpt 0/5** — a genuine difficulty gradient; on the
  easy side (avocado nearly aces it) but not trivial, and gpt never solved it in
  this run. Not too easy (avocado 4/5, < 5/5), not unsolvable (opus 2/5).
- **Calibration history:** v1 (commit `9edbfe7`) failed balance **unsolvable**
  (avocado/opus/gpt all 0/5) — but the agents *had* solved the BO swap; they
  failed only an **under-specified factory contract** (some used a different
  parameter order → `TypeError`; others inlined the search with no importable
  builder → `AssertionError`). v2 fixed both: the test binds arguments to the
  agent's factory by name/position via `inspect.signature` (any reasonable
  signature works), and the instruction was tightened to require an importable
  standalone builder. The task then became solvable.
- **gpt-5.5 note:** some gpt trials across runs fail on agent-harness
  infrastructure (`codex` non-zero exit / rate-limit) rather than reasoning;
  discount infra exits when reading gpt's rate.

## Anti-Cheating Analysis

- **Tests are verifier-only.** The agent sees only `instruction.md` at solve
  time; `test_bayesopt.py` is applied via `test_patch` at verify time, so the
  interface cannot be read off the tests.
- **fail_to_pass executes the patched code** — imports the patched `train`
  module and calls the agent's search-builder; not a static-artifact check.
- **nop fails / oracle passes.** At base commit the four BO tests fail (import
  of `train` runs argparse/corpus and raises; no builder) and the RNN
  regression test passes; after the solution all five pass.
- **No oracle/ground-truth in agent-readable paths** — the Dockerfile only
  clones the repo and pip-installs pinned deps; no `COPY`/`ADD`.
- **Cheap, deterministic, behavioral tests** — the search is exercised on a
  tiny synthetic estimator (`DecisionTreeClassifier` + `make_classification`),
  never real PTB training; assertions are on cap respected, fewer-than-full-grid,
  and well-formed `best_params_`/`best_score_` — not implementation internals.

## Known caveats

- **Provenance: SUSPECT (review recommended)** — the AI-authorship classifier
  scored `instruction.md` highly (p3p ≈ 0.998); stylometric/trajectory signals
  are clean so it is not blocked. The instruction should be rephrased in the
  author's own voice before final submission.
- **Test leniency (review M2):** a capped *random* sampler (`RandomizedSearchCV`)
  would also pass — the tests verify capped, non-exhaustive, best-surfacing
  search but do not discriminate Bayesian optimization from random sampling.
  Acceptable for this library-BO task; a from-scratch escalation would add a
  BO-vs-random discriminator.

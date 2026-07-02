# codimango/ignite-fairness-subgroup-ratio

## Description

Adds a **`reduction`** parameter to PyTorch-Ignite's fairness
**`SubgroupDifference`** metric (and its subclasses **`SubgroupAccuracyDifference`**
and **`DemographicParityDifference`**). Alongside the existing `"difference"`
behaviour (`max − min` across subgroups, kept as the default), it introduces a
new `"ratio"` reduction (`min / max`, where `1.0` means perfectly balanced). This
is ignite's own subgroup-fairness framework (`versionadded 0.5.4`), not a port of
an external API — ignite currently has no such parameter (confirmed absent at HEAD).

- **Repo:** `pytorch/ignite` @ `d5208644ab5646ee250c3b5db08f5c5e9f128354`
- **Language:** Python
- **Files (reference):** `ignite/metrics/fairness/base.py` (`_SubgroupBase`,
  `SubgroupDifference`), `ignite/metrics/fairness/accuracy_difference.py`,
  `ignite/metrics/fairness/demographic_parity.py`

## Reference solution (oracle)

- `ignite/metrics/fairness/base.py`: `_SubgroupBase.__init__` gains a
  `reduction` kwarg (default `"difference"`) validated against
  `{"difference", "ratio"}` (else `ValueError`). `SubgroupDifference.compute()`
  branches on it:
  - **scalar base metric** (e.g. `Accuracy`): `"difference"` → `max − min`;
    `"ratio"` → `min / max`, with `0.0` returned when `max == 0`.
  - **tensor / per-class base metric** (e.g. `SelectionRate`): reduce **per index**
    first (per-class `min`/`max`), then take the worst case across indices — the
    **maximum** disparity for `"difference"`, the **minimum** ratio for `"ratio"`.
    Per-index zero denominators map to `0.0`.
- `accuracy_difference.py` / `demographic_parity.py`: thread `reduction` through
  to the base class.

## Completion Rates

Measured by the platform (balance is flaky — avocado swings on code-gen):

| Agent | Model | Pass rate (observed) |
|-------|-------|----------------------|
| oracle | oracle | 3/3 (1.000) |
| metacode | avocado_dvsc_tester | TBD |
| claude-code | claude-opus | TBD |
| (aux) | gpt-5.5 | TBD |

Local harness: **oracle reward = 1** (14/14), **nop reward = 0** (11 fail_to_pass
fail, 3 pass_to_pass pass).

## Model Analysis

Ignite-internal (non-sklearn-parity) feature in a less-mature module, chosen to
avoid both the parity "too-easy" ceiling and the named-algorithm novelty wall.
Difficulty comes from the **tensor/per-class path** (per-index `min`/`max` then a
worst-case reduction — the ratio's worst case is the *minimum*, the mirror of the
difference's *maximum*), the **zero-denominator convention**, and preserving
**backward-compatibility** (default `"difference"` unchanged across scalar and
tensor base metrics).

## Anti-Cheating Analysis

- **Tests are verifier-only** (`test_patch` adds
  `tests/ignite/metrics/fairness/test_subgroup_ratio.py`).
- **fail_to_pass execute the patched code** — construct the metrics, call
  `update`/`compute`, and assert the reduced value.
- **No hardcoded oracle** — expected values are derived from first principles
  (the ignite-internal spec: `max−min`, `min/max`, per-index worst case,
  `0.0` on zero denominator) and asserted with `pytest.approx`.
- **Backward-compat enforced** — `pass_to_pass` asserts the default
  (`reduction="difference"`) still returns `max−min` for both scalar and tensor
  base metrics, so a solution that changes the default is caught.
- **No git-history leak** — the Dockerfile strips `.git` and `.github` after the
  editable install and re-inits a fresh single-commit repo, so the base commit /
  any later upstream fix is not recoverable via `git log --all`.

## Notes

- `tests/run_script.sh` forces `--color=no -o addopts=` (ignite pins
  `addopts="--color=yes"`, whose ANSI codes break `parser.py`).
- The existing `test_accuracy_difference.py` / `test_demographic_parity.py`
  import `fairlearn` (not in the image), so `pass_to_pass` is kept self-contained
  in the new test file rather than referencing them.
- `fail_to_pass` (11 curated) is a config-only calibration lever — expand the
  scored subset to raise the bar, contract to lower it.

# codimango/ignite-fairness-auto-subgroups

## Description

Adds **automatic subgroup discovery** to PyTorch-Ignite's fairness
`SubgroupMetric` / `SubgroupDifference` framework (and the subclasses
`SubgroupAccuracyDifference`, `DemographicParityDifference`). The `groups`
argument becomes **optional** (`None` = auto-discover): subgroups are discovered
lazily from the `group_labels` seen during `update()`, creating a fresh
deep-copied base metric per newly-seen group. Pre-listing `groups` still works and
preserves the previous fixed-set behavior. This is an ignite-internal API-design
change (not a documented metric formula) — it exercises `__init__`, `update`,
`reset`, `compute`, and the `state_dict`/`load_state_dict` round-trip.

- **Repo:** `pytorch/ignite` @ `d5208644ab5646ee250c3b5db08f5c5e9f128354`
- **Language:** Python
- **Files (reference):** `ignite/metrics/fairness/base.py`
  (`_SubgroupBase`, `SubgroupMetric`, `SubgroupDifference`),
  `ignite/metrics/fairness/accuracy_difference.py`,
  `ignite/metrics/fairness/demographic_parity.py`

## Reference solution (oracle)

- `_SubgroupBase.__init__`: `groups` optional (`None` → `self._auto_discover=True`,
  empty metric set); a sequence seeds a fixed set (previous behavior).
- `update`: when auto-discovering, register any new `group_labels` value
  (`copy.deepcopy(base_metric)`) before accumulating.
- `reset`: rebuild the seeded set (empty when auto), so a new epoch rediscovers.
- `state_dict`: add a type-preserving `_groups` key alongside the existing
  per-group `_metrics`.
- `load_state_dict`: when `_groups` is present, re-create the full (possibly
  auto-discovered) set into a fresh instance and restore each metric's state;
  otherwise fall back to the legacy in-place match.
- subclasses thread the optional `groups` through.

## Completion Rates

Measured by the platform (balance is flaky — avocado swings on code-gen):

| Agent | Model | Pass rate (observed) |
|-------|-------|----------------------|
| oracle | oracle | 3/3 (1.000) |
| metacode | avocado_dvsc_tester | TBD |
| claude-code | claude-opus | TBD |
| (aux) | gpt-5.5 | TBD |

Local harness: **oracle reward = 1** (11/11), **nop reward = 0** (7 fail_to_pass
fail, 4 pass_to_pass pass). Harbor-sim (reset to `base_commit`, apply gold +
test_patch, run node ids): 11 passed.

## Model Analysis

Deliberately **non-formula** and **cross-file** to avoid the recall wall that a
named metric hits: the correct solution is a set of coordinated edits
(`__init__`/`update`/`reset`/`state_dict`/`load_state_dict`) with a subtle
round-trip requirement (restoring auto-discovered groups into a fresh instance),
not a single canonical expression an LLM regenerates verbatim.

## Anti-Cheating Analysis

- **Tests are verifier-only** (`test_patch` adds
  `tests/ignite/metrics/fairness/test_auto_subgroups.py`).
- **fail_to_pass execute the patched code** — construct with no `groups`, call
  `update`/`compute`, inspect discovered `_metrics`, exercise the state_dict
  round-trip into a fresh instance.
- **No hardcoded oracle** — expected accuracies/selection-rates are derived from
  first principles and asserted with `pytest.approx`.
- **Backward-compat enforced** — `pass_to_pass` uses the pre-listed `groups` API
  (fixed set, unknown group dropped, single-group raise, state_dict round-trip);
  these pass at base and after the patch, so a solution that regresses the
  pre-listed behavior is caught.
- **No git-history leak** — the Dockerfile detaches at `base_commit`, deletes all
  branches/tags and the remote, and gc's, so no descendant (any later upstream
  fix) is reachable via `git log --all`; `.github` is removed. The base commit
  itself stays a valid object so the eval/oracle harness can `git reset --hard`.

## Notes

- `tests/run_script.sh` forces `--color=no -o addopts=` (ignite pins
  `addopts="--color=yes"`, whose ANSI codes break `parser.py`).
- The existing `test_accuracy_difference.py` / `test_demographic_parity.py`
  import `fairlearn` (not in the image), so `pass_to_pass` is self-contained in
  the new test file rather than referencing them.
- `fail_to_pass` (7) is a config-only calibration lever.

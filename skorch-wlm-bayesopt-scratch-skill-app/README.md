# codimango/skorch-wlm-bayesopt-scratch-skill-app

## Description

Skill Application variant of `skorch-wlm-bayesopt-scratch`. The task replaces the exhaustive `GridSearchCV` hyperparameter search in skorch's `examples/word_language_model` example with a from-scratch bounded 1-D optimizer — a hand-written quadratic surrogate over observed `(config, score)` history plus a trust-region acquisition rule that proposes the next configuration — using only numpy/scipy (no BO library), capped by an explicit iteration budget.

The instruction is intentionally **super minimal + symptom-only** (3 lines): bounded 1-D search over learning rate with explicit bounds and iteration cap, `__main__` guard import-safe, optimizer class with bounds, preserve original skorch wiring. It **does NOT list flag names, defaults, constant names, or exact surrogate formulas** — those must be inferred from original `train.py` + skill. Tests strictly require `--search-iter` (default 16 via `DEFAULT_SEARCH_ITERATIONS`), `--lr-low/high` (1.0/40.0), `--seed`, `--save`, `--data`, `--bptt`, `my_train_split` with `Dataset(valid[:200])`, `data.Loader` with `bptt` and `device`, `cross_val_score` + `.numpy()` + `save_params`, and trust constants `N_SEED=3`, `INIT_RADIUS_FRAC=0.25`, `SHRINK=0.5`. Exact constants were **removed from skill** to make WITH guess them (abstract skill says "few seed points uniform, fraction of bound width, shrink on non-improvement" without numbers), causing Medium difficulty.

- **Repo:** `skorch-dev/skorch` @ `e769a2b89dc6e946dfaf42654184aa01b7025964`
- **Language:** Python
- **Files touched by reference solution:** `examples/word_language_model/train.py`
- **Skill:** `environment/skills/quadratic-trust-region-optimizer/SKILL.md` (authored)
- **Sub-track:** Skill Application (implicit prompt, no distractors required)

> Exact reference-solution details (formulas, constants, tie-breaks) intentionally live in `solution/` and `SKILL.md` only — not in `instruction.md` — to keep the task from being one-pass reconstructible.

## Reference solution (oracle)

`train.py` gains a small from-scratch optimizer class `QuadraticTrustRegionOptimizer` constructed with search `bounds`:

- `predict(history, x)`: least-squares degree-2 polyfit `np.polyfit(xs, scores, 2)` to all points in history, eval at `x`
- `propose(history, radius)`: `best_x = argmax score`, feasible = `[max(lo, best_x - radius), min(hi, best_x+radius)]` ∩ bounds, if quadratic coeff `a<0` → vertex `-b/(2a)` clipped to feasible, else feasible endpoint with higher predicted (tie→lower)
- `optimize(objective, n_iter, seed)`: `np.random.default_rng(seed).uniform(lo,hi)` for `N_SEED=3` seed points, `INIT_RADIUS_FRAC=0.25*(hi-lo)`, `SHRINK=0.5` on non-improvement, returns `(best_x, history)`
- `main()` under `if __name__ == '__main__'`: argparse with `--search-iter` (default `DEFAULT_SEARCH_ITERATIONS=16`), `--lr-low` (1.0), `--lr-high` (40.0), `--seed`, `--data`, `--bptt`, `--batch_size`, `--epochs`, `--data-limit`, `--save`; corpus loading, `Net` with `train_split`/`iterator_train`, objective `net.set_params(lr=float(x))` + `np.mean(cross_val_score(net, X))`, `optimizer.optimize`, `net.set_params(lr=best)`, `net.fit(X)`, `net.save_params(f_params=args.save)`.

`GridSearchCV` import and instantiation removed. `net.py`/`model.py` untouched. All training wrapped in `main()` (import-safe).

## Completion Rates

Local 5-trial ablation progression to Medium:

| Agent | Model | Pass rate WITH | Pass rate WITHOUT | Delta | Skill invoked? | Job |
|-------|-------|----------------|-------------------|-------|----------------|-----|
| oracle | oracle | 1/1 (1.00) | — | — | — | `01-55-29__bd78fb` |
| metacode | meta/avocado-5.14-code | 4/5 (0.80) Easy | 0/5 | 0.80 | 5/5 | `00-11-15__ec419f` (explicit skill + symptom-only) |
| metacode | meta/avocado-5.14-code | 5/5 (1.00) Easy (trivial) | 0/5 | 1.0 | 5/5 | `01-36-08__979693` (explicit wiring hints added → easier) |
| metacode | meta/avocado-5.14-code | 5/5 (1.00) Easy | 0/5 | 1.0 | 5/5 | `01-46-26__3d851c` (super minimal 3-line instruction, old detailed skill) |
| metacode | meta/avocado-5.14-code | 0/5 (0.00) Hard | 0/5 | 0.0 | 5/5 | `01-50-59__305528` (abstract skill no constants + strict constant test, extractor bug) |
| metacode | meta/avocado-5.14-code | **2/5 (0.40) Medium** | 0/5 (0.00) | **0.40** | 5/5 | `01-55-53__5bbe5f` (abstract skill + super minimal + `test_optimizer_constants` + fixed `_extract_best` handling `tuple(dict, history)`) |
| metacode | meta/avocado-5.14-code | 1/5 (0.20) Hard | 0/5 (0.00) | 0.20 | 5/5 | `09-33-54__6ca952` (same + skill name hint *"use quadratic-trust-region-optimizer skill"* → even harder 20%) |
| codex | gpt-5 | 0/5 (0.00) | 0/5 (0.00) | 0.0 | 0/5 suspect | `09-33-54__dd026e` (with hint, NonZeroAgentExitCodeError, no skill used) |
| codex | gpt-5 | 0/5 (0.00) | 0/5 (0.00) | 0.0 | 0/5 suspect | `09-28-03__f9a93e` (no hint, no skill used) |
| claude-code | claude-opus-4-8 | 0/5 (0.00) * | 0/5 (0.00) | 0.0 | 0/5 suspect | `01-53-10__8233fa` / `02-23-52__58588f` (UnknownApiError, API gateway) |

*Opus trials hit `UnknownApiError` (API gateway) after Docker dual-stack fix (`dualStackEnabled=1`). Previously hit `ECONNRESET` before fix. Skill staged at 4 COPY locations (`/app/skills`, `/app/.opencode/skills`, `/app/.codex/skills`, `/app/.claude/skills`) but agent failed before FS scan. Needs retry when API recovers. Avocado Medium 40% alone already passes Skill Application bar (1-of-3 + Δ=0.4 ≥0.4).

**Skill Application calibration:** Avocado WITH 2/5 (40% Medium) <5/5 passes triviality gate, >0 passes non-zero, WITHOUT 0/5. Δ=0.4 → 1-of-3 increase (0.4) with combined 0.4 ≥0.4 → **PASSES relaxed bar**. Difficulty **Medium** achieved (35-70%, target 45% ideal — 40% is 5% off, closer than 60%). Previous Easy versions were 80-100% (10% share).

## Model Analysis

### Skills Usage table

| Model | Pass Rate With | Pass Rate Without | Skill Invoked | Avg Tool Calls (passing WITH) | Notes |
|-------|----------------|-------------------|---------------|-------------------------------|-------|
| meta/avocado-5.14-code | 4/5 (0.80) Easy | 0/5 | Yes (5/5) | 8.0 | WITH `00-11-15__ec419f-skills-with`. Old symptom-only + CLI/main tests. Failure = missed `DEFAULT_SEARCH_ITERATIONS`. |
| meta/avocado-5.14-code | 5/5 (1.00) Easy trivial | 0/5 | Yes (5/5) | 6.6 | WITH `01-36-08__979693-skills-with`. Added explicit `my_train_split` + `data.Loader` wiring to instruction → easier, 5/5 fails triviality. |
| meta/avocado-5.14-code | 5/5 (1.00) Easy | 0/5 | Yes (5/5) | 8.2 | WITH `01-46-26__3d851c-skills-with`. Super minimal 3-line instruction, old detailed skill (with N_SEED=3,0.25,0.5). Still 5/5. |
| meta/avocado-5.14-code | 0/5 (0.00) Hard | 0/5 | Yes (5/5) | null | WITH `01-50-59__305528-skills-with`. Abstract skill (no constants) + `test_optimizer_constants` + bug in `_extract_best` (tuple dict) → 0/5 Hard. |
| meta/avocado-5.14-code | **2/5 (0.40) Medium** | 0/5 (0.00) 4 RuntimeErrors | Yes (5/5) | 7.5 | WITH `01-55-53__5bbe5f-skills-with`, WITHOUT `...-without`. **Medium achieved** after fixing `_extract_best` to handle `tuple(dict, history)`. 2 passing trials used skill via FS scan, 3 failed `test_optimize_loop_bounded_and_deterministic` / constants. Δ=0.4, relevant_used_rate 1.0. |
| claude-opus-4-8 | 0/5 (0.00) | 0/5 (0.00) | No (0/5) suspect | null | WITH `01-53-10__8233fa-skills-with` — `UnknownApiError` (API gateway), 5 errored, 0 used skill. Previously `ECONNRESET` before dual-stack fix. Needs retry. |

## Model Analysis

| Skill | Relationship | Skill Type | Skill Composition | Source | Distractor Level |
|-------|--------------|------------|-------------------|--------|------------------|
| quadratic-trust-region-optimizer | essential | domain_knowledge | atomic_skill | authored | n/a |
| random-search-baseline | distractor | n/a | n/a | authored | 2 |
| bayesian-optimization-basics | distractor | n/a | n/a | authored | 2 |
| grid-search-optimizer | distractor | n/a | n/a | authored | 1 |

Relevant skill defines bounded 1-D trust-region optimizer contract (predict least-squares quadratic, propose trust region logic, seeded determinism, constants). Distractors are random search, bayesian basics, grid search — substantive real techniques but not relevant to this task, not load-bearing (WITH never uses distractors, WITHOUT removal doesn't help, tests discovery).

## Summary classification table

| Skill | Relationship | Skill Type | Skill Composition | Source | Distractor Level |
|-------|--------------|------------|-------------------|--------|------------------|
| quadratic-trust-region-optimizer | essential | domain_knowledge | atomic_skill | authored | n/a |
| random-search-baseline | distractor | n/a | n/a | authored | 2 |
| bayesian-optimization-basics | distractor | n/a | n/a | authored | 2 |
| grid-search-optimizer | distractor | n/a | n/a | authored | 1 |

3 distractors present (recommended 3+ ideal) for Skill Application discovery test.

### Trajectory commentary

**Avocado WITH progression:**
- **4/5 Easy (00-11-15):** Read skill, implemented `QuadraticTrustRegionOptimizer(bounds)` with `predict` polyfit degree2, `propose` best_x feasible clipped vertex else endpoint, `optimize` N_SEED=3 RNG uniform 0.25 radius shrink 0.5. One failure missed `DEFAULT_SEARCH_ITERATIONS` / `--search-iter` default.
- **5/5 Easy trivial (01-36-08, 01-46-26):** Added explicit wiring hints (`my_train_split`, `data.Loader` bptt) to instruction → actually helped, 5/5 trivial (fails triviality gate).
- **0/5 Hard (01-50-59):** Made skill abstract (removed N_SEED=3,0.25,0.5,RNG details) + added `test_optimizer_constants` requiring exact values + bug in `_extract_best` (tuple dict) → all 5 failed `test_optimize_loop_bounded_and_deterministic` with `TypeError: float() arg must be real, not dict`.
- **2/5 Medium (01-55-53):** Fixed `_extract_best` to handle `tuple(dict, history)`, kept abstract skill + super minimal 3-line instruction + strict wiring + constant tests. 2 passing trials correctly guessed N_SEED=3,0.25,0.5, used skill via FS scan (implicit), implemented `train_split=my_train_split` with `Dataset(valid[:200])`, `data.Loader` bptt/device, `torch.manual_seed`, `.numpy()`, `cross_val_score`, `save_params`, `DEFAULT_SEARCH_ITERATIONS`. 3 failing trials failed constants or loop determinism. Avg 7.5 tool calls passing.
- Skill usage: 5/5 WITH used relevant skill (grep `quadratic`/`trust` in trajectory.json), 0/5 WITHOUT (skill removed). BEFORE 15.8 calls with implicit hints, AFTER 7.5 with abstract.

**Avocado WITHOUT:** All runs 0/5 (4 RuntimeErrors + 1 fail). Fails `test_predict_is_least_squares_quadratic` or `test_propose_*` or `test_skorch_wiring_preserved` or `test_optimizer_constants` — without skill guesses linear/GP or wrong constants. Confirms load-bearing.

**Opus WITH:** `01-53-10__8233fa` — 5 trials errored `UnknownApiError` (API gateway) after dual-stack fix, 0 used skill, suspect true. Previously `ECONNRESET` before fix. Needs retry when API recovers. Expected 1-4/5 Medium once API stable, given Avocado 2/5 Medium.

## Anti-Cheating Analysis

- **Tests are verifier-only** (`test_patch` applied at verify time); `fail_to_pass` import patched `train` and call optimizer methods.
- **Strong discrimination:** `test_predict_is_least_squares_quadratic` asserts true LS quadratic values at multiple points via computed oracles — GP/linear fails, only faithful quadratic passes. Instruction does NOT disclose values.
- **Return-shape robust:** `_extract_best` now handles scalar, `(best, history)` tuple, `tuple(dict, history)`, or dict (`best_x`, `best`, etc.) for `optimize()` return, and multiple method-name/arg-order variants for `predict`/`propose`.
- **Deterministic/RNG-robust:** `predict`/`propose` fixed histories, explicit radius, no randomness; `test_optimize_loop_bounded_and_deterministic` checks loop bounded and deterministic with seed.
- **No BO library** (skopt/optuna/hyperopt/bayes_opt/GPy/botorch absent); from-scratch enforced behaviorally.
- **CLI/main guard:** `test_cli_and_main_guard` checks `def main`, `__main__` guard, `--search-iter`, `--lr-low/high`, `--seed`, `--save`, `DEFAULT_SEARCH_ITERATIONS`, `cross_val_score`, `save_params`, optimizer class inside `train.py`.
- **Skorch wiring:** `test_skorch_wiring_preserved` checks `def my_train_split`, `Dataset(valid[:200])`, `train_split=my_train_split`, `iterator_train=data.Loader`, `iterator_train__bptt`, `iterator_valid`, `iterator_valid__bptt`, `data.Corpus`, `torch.manual_seed`, `data_limit`, `.numpy()` — preserves original training path, not in skill.
- **Constants:** `test_optimizer_constants` checks `N_SEED` concept, `0.25` and `0.5` appear, and if class attrs present they equal 3/0.25/0.5, plus behavioral n_iter=3 seeds only. Constants **removed from skill** (abstract says "few seed points, fraction of bound width, shrink on non-improvement" without numbers) → WITH must guess exact values, causing Medium 2/5.
- **Solution leakage:** skill describes contract abstractly (bounds, polyfit degree2, vertex clipped, endpoint fallback tie→lower, seeding, fraction, shrink, return shape) without exact numbers in latest version, without copy-pasted solution code.

## Calibration notes

- **Progression:** Explicit skill (with N_SEED=3,0.25,0.5) + symptom-only → 4/5 Easy (1 CLI miss) → adding explicit wiring → 5/5 trivial → super minimal 3-line instruction + old detailed skill → still 5/5 (Avocado reads original file) → abstract skill (no numbers) + strict constant test (bug in extractor) → 0/5 Hard → fixed extractor handling `tuple(dict, history)` → **2/5 Medium (40%)** target 45% ideal.
- **Difficulty achieved:** Medium 40% (2/5) is 5% off 45% ideal, better than 60% (15% off). Passes triviality (<5/5, >0) and Skill Application bar (1-of-3 increase 0.4 + combined 0.4 ≥0.4).
- **Dockerfile 4 COPY:** `COPY skills /app/skills`, `/app/.opencode/skills`, `/app/.codex/skills`, `/app/.claude/skills` — after `git clean -fd`, verified Avocado 5/5 used, Opus discovered but hit API errors.
- **Opus status:** `ECONNRESET` before dual-stack fix, `UnknownApiError` after fix (API gateway). Needs retry when API stable. Avocado Medium alone already passes bar, but 2-of-3 would be stronger.
- **Next steps:** Retry Opus/Codex when API recovers, update README, then `git add`, commit, push to codimango repo. Task already `difficulty=medium` in `task.toml`.

## Taxonomy

- category_usecase: compute_result
- category_subdomain: machine_learning
- reward_type: binary
- format: swe_bench_single_turn_skills
- workstream: swe_public_repo
- sub_track: skill_application
- prompt_style: implicit

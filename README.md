# swe-bench-pro-msl

A template repository for creating [SWE-Bench Pro](https://github.com/codimango/swe-bench-pro) tasks — repository-level coding challenges for evaluating AI coding agents.

SWE-Bench Pro tasks use the [Harbor](https://harborframework.com) task format as a base, with additional metadata files for SWE-Bench-specific test infrastructure.

## Tasks must be original

Do **not** base a task on an existing commit, PR, or closed issue — public fixes are likely in model training data and contaminate the benchmark. Author novel tasks against a recent `base_commit` from your own domain expertise.

Authoritative policy: `team-aai:swebench-hackathon-guide` skill + the AAI provenance & quality policy post.

## Getting Started

1. Click **"Use this template"** on GitHub to create your own repo under `codimango`
2. Clone your new repo and create a task folder for each task
3. Install [Harbor](https://harborframework.com/docs/getting-started): `uv tool install harbor` (or `pip install harbor`)

## Task Structure

Each task lives in its own directory:

```
my-task/
├── instruction.md              # The prompt given to the agent
├── task.toml                   # Task metadata and configuration (Harbor format)
├── environment/
│   └── Dockerfile              # Docker sandbox setup
├── solution/
│   └── solve.sh                # Ground-truth solution (oracle)
└── tests/
    ├── test.sh                 # Test entrypoint — runs tests, writes reward (Harbor format)
    ├── config.json             # MSL metadata (instance_id, base_commit, patch, test_patch, FAIL_TO_PASS, PASS_TO_PASS)
    ├── run_script.sh           # Test runner (framework-specific)
    └── parser.py               # Output parser (test output → JSON)
```

### Harbor Base Files

| File | Purpose |
|------|---------|
| `instruction.md` | The prompt the evaluated model receives |
| `task.toml` | Metadata: author, difficulty, timeouts, resource limits |
| `environment/Dockerfile` | Docker image setup |
| `solution/solve.sh` | Oracle solution (typically a git patch) |
| `tests/test.sh` | Test entrypoint — writes `1` or `0` to `/logs/verifier/reward.txt` |

### SWE-Bench Pro Extras

| File | Purpose |
|------|---------|
| `tests/config.json` | `instance_id`, `base_commit`, `patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `selected_test_files_to_run` |
| `tests/run_script.sh` | Framework-specific test runner (pytest, jest, etc.) |
| `tests/parser.py` | Parses test output into `{"tests": [{"name": ..., "status": ...}]}` |

## Validating Your Task

```bash
# Build the Docker image
docker build -t my-task -f my-task/environment/Dockerfile .

# Test the oracle solution (should produce reward=1)
docker run --rm -v $(pwd)/my-task/tests:/tests -v $(pwd)/my-task/solution:/solution my-task bash -c "bash /solution/solve.sh && bash /tests/test.sh && cat /logs/verifier/reward.txt"

# Test without solution (should produce reward=0)
docker run --rm -v $(pwd)/my-task/tests:/tests my-task bash -c "bash /tests/test.sh && cat /logs/verifier/reward.txt"

# Validate with Harbor
harbor run -d . -a oracle
```

Run the oracle 3+ times to check for flakiness.

## Included Tasks

| Task | Repo | Difficulty | Description |
|------|------|-----------|-------------|
| `hello-world` | `promplate/partial-json-parser` | Easy | Off-by-one bug in `rfind()` causing inconsistent JSON completion |

## Creating a New Task

> **First:** your task must be [original](#tasks-must-be-original) — not from any existing commit, PR, or issue.

1. Run `harbor tasks init "my-task"` or copy the `hello-world/` directory
2. Write `instruction.md` — the prompt the model will see
3. Set up `environment/Dockerfile` — clone the repo, install deps, reset to base commit
4. Write `solution/solve.sh` — the oracle patch
5. Fill in `tests/config.json` — `instance_id`, `base_commit`, `patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`
6. Write `tests/run_script.sh` — test runner for the repo's framework
7. Write `tests/parser.py` — parse test output to JSON
8. Update `tests/test.sh` if needed (the default usually works)
9. Validate: oracle passes 3/3, regression tests pass without solution

For detailed guidelines, see the [SWE-Bench Pro annotation guide](https://docs.google.com/document/d/1kOaMgZp3Phd2e0ZFq3kEFTT2TW6UDxB48h9Etifm-Ck/edit?tab=t.u5jldkf0fwl4).

## Consuming Tasks in MSL

Tasks authored with this template are ingested by Meta's MSL post-training pipeline through `agents/utils/sweagent/grading_spec/sweap.py`. See [`docs/msl-annotation-guideline.md`](docs/msl-annotation-guideline.md) for the full JSONL schema, an in-depth `parsing_script` contract, and a drop-in converter that turns each task directory into one JSONL record.

### Required files

Every task must supply these three files. MSL will reject (or silently mis-grade) tasks missing any of them.

| File | What it must contain |
|------|----------------------|
| `tests/config.json` | See the required-fields table below. |
| `tests/run_script.sh` | The sole entrypoint MSL uses to execute tests inside the SWEAP container. Must accept the selected test files as CLI arguments, invoke the test runner (pytest, jest, etc.), and leave raw runner output on stdout/stderr for the parser. Do **not** rely on `test.sh` — MSL does not use it. |
| `tests/parser.py` | Reads the run_script's stdout/stderr and writes `{"tests": [{"name": "<test_id>", "status": "PASSED\|FAILED\|SKIPPED\|ERROR"}, ...]}` to the path passed as `argv[3]`. The `name` strings must match `FAIL_TO_PASS` / `PASS_TO_PASS` entries **exactly**, and `status` must be one of the four uppercase enums. Every test listed in `FAIL_TO_PASS` / `PASS_TO_PASS` must appear in the output — missing entries cause silent grading failures. **See [`docs/test-output-contract.md`](docs/test-output-contract.md) for the full contract, common mistakes, and a self-check snippet.** The `hello-world/tests/parser.py` template includes a `validate_contract()` step that exits non-zero on shape violations — keep it. |

### Create a (solution) 'patch' and a 'test_patch'
* Make a git 'patch' for your solutions/ directory: git diff > ~/solution.patch -- path/to/solutions_directory/ (many other options of the git command)
* Make a git 'patch' for your tests/ directory: git diff > ~/tests.patch -- path/to/tests/ (many other options of the git command)
* Copy the *patch* (solution) and *test_patch* content into 'patch' and 'test_patch' fields respectively, in the config.json file (next section).

### Required `tests/config.json` fields

| Field | Type | Must be |
|-------|------|---------|
| `instance_id` | str | MSL form: `aai_{owner}__{repo}-{40-hex commit \| pr_id \| issue_id}`. Globally unique. |
| `repo` | str | `{owner}/{repo}` (e.g. `promplate/partial-json-parser`). Must be registered in MSL's `MAP_REPO_VERSION_TO_SPECS`. |
| `base_commit` | str | 40-hex SHA of the initial state the agent starts from (buggy or feature-incomplete). |
| `patch` | str | Unified `git diff` for the solution. |
| `test_patch` | str | Unified `git diff` that installs the new tests on top of `base_commit`. MSL applies this via `git apply` — not via `before_repo_set_cmd`. |
| `FAIL_TO_PASS` | list[str] | Uppercase. Test ids that must flip FAIL → PASS after the solution patch. Must match what `parser.py` emits. |
| `PASS_TO_PASS` | list[str] | Uppercase. Test ids that must remain PASS (regression guard). |
| `selected_test_files_to_run` | list[str] | Test file paths, relative to repo root, passed as CLI args to `run_script.sh`. |

Optional but recommended: `before_repo_set_cmd` (for Harbor-side `harbor run -a oracle` compatibility), and any provenance fields like `commit_message`, `associated_pr_url`, `task_category`.

See [`docs/msl-annotation-guideline.md`](docs/msl-annotation-guideline.md) for the exhaustive spec, including the full `parsing_script` contract and a drop-in Python converter.

## Reviewing a Task

This repo includes a `/review-task` [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill that evaluates task quality across three dimensions: spec clarity, test quality, and code patch quality. It also produces revised task specifications at three detail levels.

### Usage

In Claude Code, run:

```
/review-task my-task
```

The review will:
1. Read all task files and analyze the patch, tests, and specification
2. Run Docker-based test verification if available (checks `FAIL_TO_PASS` fails without solution, all pass with solution)
3. Rate **Spec Clarity**, **Test Quality**, and **Code Patch Quality**
4. Identify missing information in `instruction.md` and whether it's inferable from the codebase
5. Produce three revised versions of `instruction.md` (succinct, regular, verbose)

Use this to validate your task before submitting a PR. See [#4](https://github.com/codimango/swe-bench-pro-template/pull/4) for an example review output.

# SWEAP Annotation Guidance

This document specifies the exact JSONL schema annotators must produce so that the data can be consumed end-to-end by `agents/utils/sweagent/grading_spec/sweap.py` without any post-processing.

Reference file: `agents/utils/sweagent/grading_spec/sweap.py` (function `make_eval_grading_spec`).
Target registry: `MAP_REPO_VERSION_TO_SPECS` in `agents/utils/sweagent/grading_spec/sweap_constants.py` (key = `"sweap"`).
Log parser that consumes `/output.json`: `parse_log_sweaptest` in `agents/utils/sweagent/grading_spec/sweap_log_parsers.py`.

>  **Tasks must be original** — do not base them on existing commits/PRs/issues (training-data contamination). Author against a recent `base_commit`. The `commit_*`/`associated_pr_url` fields below are optional metadata, not an endorsement of historical sourcing. Policy: `team-aai:swebench-hackathon-guide`.

---

## 1. File format

- One JSON object per line (JSONL), UTF-8 encoded.
- Each line must be a complete JSON object — no trailing commas, no unescaped newlines inside strings (use `\n`).
- The top-level shape is:

```json
{
  "metadata": { ...see Section 3... },
  "dialog": [],
  "keep_loss": [],
  "version": "sample_sft_v1"
}
```

`dialog` and `keep_loss` stay empty for annotation hand-off; they are populated later during SFT pipeline runs. The top-level `version` string is a constant.

---

## 2. How `sweap.py` uses each record

`make_eval_grading_spec(instance, registry)` does the following:

1. Parses `instance_id` to derive `{owner}/{repo}`. If the derived value is not in the SWEAP registry it falls back to `instance["repo"]` (and then `instance["repo"].lower()`).
2. Looks up build/test specs for that repo under `version="sweap"` in the registry.
3. Writes `run_script` → `/run_script.sh` and `parsing_script` → `/parse_script.py` inside the container.
4. Runs a bash pipeline:
   - `cd /app`
   - `git -c core.fileMode=false diff <base_commit>` (sanity)
   - `git checkout <base_commit> <test_patch files>` (reset test files)
   - `git apply -v -` on `test_patch` (install the new tests)
   - `chmod +x /run_script.sh && /run_script.sh <selected_test_files_to_run...> 1> /stdout.txt 2> /stderror.txt`
   - `python /parse_script.py /stdout.txt /stderror.txt /output.json`
   - `cat /output.json` between `START_TEST_OUTPUT`/`END_TEST_OUTPUT` markers.
5. `parse_log_sweaptest` reads `/output.json` and expects `{"tests": [{"name": <id>, "status": "PASSED|FAILED|SKIPPED|ERROR"}, ...]}`.
6. Grading compares test names/statuses against `fail_to_pass` and `pass_to_pass`.

Any field not in this flow is passed straight through only if it matches a `CodingAgentGradingSpec` model field — see `agents/common/types/grading_specs.py`.

---

## 3. Required `metadata` fields

**All fields below are MANDATORY.** Missing any one will either crash `make_eval_grading_spec` or produce silent mis-grades.

| Field | Type | Notes |
|---|---|---|
| `instance_id` | str | Format: `aai_{owner}__{repo}-{40-to-64-hex commit \| pr_id \| issue_id}`. Example: `aai_promplate__partial-json-parser-dda8aa18114c7d97d35b6e554bbaf98f9a03f0d5`. `sweap.py` strips the `aai_` (or legacy `instance_`) prefix, strips a trailing 40-64 hex commit via regex `(-v?[0-9a-f]{40,64})+$`, and replaces `__` with `/`. Non-hex tails (PR/issue IDs) bypass the primary derivation and rely on `metadata.repo` for the registry lookup — still works, but be sure `metadata.repo` is accurate. Must be globally unique. |
| `repo` | str | `{owner}/{repo}` (e.g. `"leptonai/leptonai"`). Must match a key in `MAP_REPO_VERSION_TO_SPECS` (listed in `sweap_constants.py`). Case matters — the registry holds both canonical and lowercase/mixed-case variants. |
| `version` | str | Literal `"sweap"`. Used as the inner key for `MAP_REPO_VERSION_TO_SPECS[repo]["sweap"]`. |
| `container_type` | str | Literal `"sweap"`. Becomes `ContainerType.SWEAP`. |
| `container_mem` | str | `"4G"`, `"8G"`, or `"16G"`. Routes to FaaS tenant `async_2907157` when `"16G"` (case-insensitive), else `async_2881758`. Propagated to `container_memswap`. |
| `base_commit` | str | 40-hex SHA of the initial state the agent starts from (buggy or feature-incomplete). Must exist in the repo snapshot baked into the container image. |
| `problem_statement` | str | The issue text shown to the agent. Multi-line allowed (use `\n`). |
| `test_patch` | str | Unified `git diff` that adds/modifies test files and contains the new tests exercising the fix. Applied via `git apply -v -`. Must apply cleanly on top of `base_commit`. |
| `selected_test_files_to_run` | list[str] **or** JSON string | Test file paths relative to repo root (e.g. `["leptonai/photon/tests/test_photon.py"]`). Passed as CLI args to `/run_script.sh`. Either a real JSON array or a stringified JSON array is accepted — real arrays are preferred. |
| `fail_to_pass` | list[str] **or** JSON string | Test ids that are expected to flip FAIL → PASS after the gold fix is applied. Format must match what the parsing script produces (e.g. `"path/to/test.py::TestClass::test_method"`). |
| `pass_to_pass` | list[str] **or** JSON string | Test ids that must remain PASS (regression guard). Same id format. |
| `parsing_script` | str | Python source (see Section 4 for full spec). |
| `run_script` | str | Bash source (see Section 5). |

### Optional pass-through fields

These are not required by `sweap.py`, but they are kept automatically if they match a `CodingAgentGradingSpec` field and are useful for debugging, provenance, and spot-checks:

- `patch` / `gold_patch` — reference fix diff, useful for spot-checks and `eval_gt`.
- `hints_text` → mapped to `hints`.
- `language` — e.g. `"py"`; defaults to `"py"` when unset.
- `grading_timeout_sec` — int seconds (default `900`).
- `test_files` — list of test files in the repo.
- `before_repo_set_cmd` / `after_repo_set_cmd` — bash run before/after container reset.
- `task_category` — see `TaskCategory` enum (`"Bug Fixing"`, `"Feature Development"`, etc.).
- `commit_hash`, `commit_message`, `repo_language`, `license`, `associated_pr_url` — for provenance/filtering.

---

## 4. `parsing_script` — FULL SPEC

The parsing script is the single most important artifact. It converts raw test output into the structured JSON that grading reads.

### 4.1 What grading requires

`parse_log_sweaptest` (in `sweap_log_parsers.py`) opens `/output.json` and expects:

```json
{
  "tests": [
    {"name": "<test_id>", "status": "PASSED"},
    {"name": "<test_id>", "status": "FAILED"},
    {"name": "<test_id>", "status": "SKIPPED"},
    {"name": "<test_id>", "status": "ERROR"}
  ]
}
```

Contract details:

- Top-level key **must be** `tests`, value must be a list.
- Each element must have both `name` (str) and `status` (str). Missing either is logged as an error and that test is skipped.
- Allowed `status` values: `"PASSED"`, `"FAILED"`, `"SKIPPED"`, `"ERROR"`. Anything else will never match `FAIL_TO_PASS`/`PASS_TO_PASS` expectations.
- Duplicate names are allowed at the JSON level but the grader keeps only the last status per name (see reference script's `unique_results`).
- The `name` field **must exactly match** what you list in `FAIL_TO_PASS` / `PASS_TO_PASS`. For pytest, the canonical id is `path/to/test_file.py::TestClass::test_method` (or `path/to/test_file.py::test_func`). Whatever id scheme you use in the list is what the parser must emit.

### 4.2 Invocation contract

`sweap.py` runs:

```
python /parse_script.py /stdout.txt /stderror.txt /output.json
```

- `sys.argv[1]` — path to file containing test runner stdout.
- `sys.argv[2]` — path to file containing test runner stderr.
- `sys.argv[3]` — path to write the JSON result to (must be `/output.json` in practice).

### 4.3 Required script skeleton

Annotators should start from this canonical template and only modify the `parse_test_output` body. The outer structure and JSON writer must not change — the grader was validated against this shape.

```python
"""
Test Results Parser

Input:
    - stdout_file: Path to the file containing standard output from test execution
    - stderr_file: Path to the file containing standard error from test execution

Output:
    - JSON file containing parsed test results with structure:
      {
          "tests": [
              {"name": "test_name", "status": "PASSED|FAILED|SKIPPED|ERROR"},
              ...
          ]
      }
"""

import dataclasses
import json
import re
import sys
from enum import Enum
from pathlib import Path
from typing import List


class TestStatus(Enum):
    PASSED = 1
    FAILED = 2
    SKIPPED = 3
    ERROR = 4


@dataclasses.dataclass
class TestResult:
    name: str
    status: TestStatus


### DO NOT MODIFY THE CODE ABOVE ###
### Implement the parsing logic below ###


def parse_test_output(stdout_content: str, stderr_content: str) -> List[TestResult]:
    """Parse test runner output and return a list of TestResult."""
    results: List[TestResult] = []

    # EXAMPLE — pytest -v default output
    # matches lines like:
    #   tests/test_foo.py::TestBar::test_baz PASSED            [ 12%]
    #   tests/test_foo.py::test_qux FAILED
    pattern = re.compile(r"^(?P<name>\S+) (?P<status>PASSED|FAILED|SKIPPED|ERROR)\b")
    for line in stdout_content.splitlines():
        m = pattern.match(line.strip())
        if m:
            results.append(
                TestResult(
                    name=m.group("name"),
                    status=TestStatus[m.group("status")],
                )
            )
    return results


### Implement the parsing logic above ###
### DO NOT MODIFY THE CODE BELOW ###


def export_to_json(results: List[TestResult], output_path: Path) -> None:
    unique_results = {r.name: r for r in results}.values()
    json_results = {
        "tests": [
            {"name": r.name, "status": r.status.name} for r in unique_results
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)


def main(stdout_path: Path, stderr_path: Path, output_path: Path) -> None:
    with open(stdout_path, encoding="utf-8") as f:
        stdout_content = f.read()
    with open(stderr_path, encoding="utf-8") as f:
        stderr_content = f.read()
    results = parse_test_output(stdout_content, stderr_content)
    export_to_json(results, output_path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python parsing.py <stdout_file> <stderr_file> <output_json>")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
```

### 4.4 Parser design guidance

- **Use the default verbose output of the test runner you invoke in `run_script`.** The parser and run script are a pair — if the run script uses `pytest -v`, match pytest's format; if it uses `jest --verbose`, match jest's.
- **Prefer stdout, but consult stderr when needed.** Some runners (e.g. unittest) write results to stderr. The template exposes both. Default template parses stdout only — change this if your runner differs.
- **Capture every test the runner reports**, not just the ones in `FAIL_TO_PASS`/`PASS_TO_PASS`. Missing tests look like silent PASSes to the grader (they won't flip and will cause false negatives).
- **Never raise on unparseable lines.** The parser must tolerate warnings, pytest summary blocks, progress bars, ANSI codes. If output contains ANSI, strip with `re.sub(r"\x1b\[[0-9;]*m", "", text)` before matching.
- **Name format must be consistent with `FAIL_TO_PASS` / `PASS_TO_PASS`.** Pytest prints `tests/foo.py::TestBar::test_baz PASSED`. If your `FAIL_TO_PASS` id is `tests/foo.py::TestBar::test_baz`, the parser must output exactly that string. Do not include `[percent]` suffixes, trailing whitespace, or ANSI codes.
- **Handle parametrized tests.** Pytest parametrized ids look like `tests/foo.py::test_add[1-2-3]`. Your regex must not truncate at `[`.
- **Map runner-specific statuses:**
  - pytest: `PASSED`, `FAILED`, `SKIPPED`, `ERROR`, `XFAIL` (map to `SKIPPED`), `XPASS` (map to `PASSED` or `FAILED` depending on strict mode).
  - jest: `ok` → PASSED, `not ok` → FAILED, `# SKIP` → SKIPPED.
  - go test: `--- PASS:`, `--- FAIL:`, `--- SKIP:`.
  - unittest: `ok`, `FAIL`, `ERROR`, `skipped`.
- **De-duplicate via `unique_results`** (already in the template). Some runners emit the same test id twice (e.g. retries, pytest rerun). Keep the last status.

### 4.5 Self-validation (required before submitting)

Before handing a record over, run the pair locally:

```bash
# Run your run_script against a repo checkout at base_commit with test_patch applied:
bash /tmp/run_script.sh <selected test files> 1> /tmp/stdout.txt 2> /tmp/stderror.txt
python /tmp/parse_script.py /tmp/stdout.txt /tmp/stderror.txt /tmp/output.json
cat /tmp/output.json | python -c "
import json, sys
o = json.load(sys.stdin)
names = {t['name'] for t in o['tests']}
f2p = set(<your FAIL_TO_PASS list>)
p2p = set(<your PASS_TO_PASS list>)
missing = (f2p | p2p) - names
assert not missing, f'parser missed tests: {missing}'
print('ok', len(o['tests']), 'tests captured')
"
```

If `missing` is non-empty the parser regex is wrong — fix it before submitting.

### 4.6 Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Grader reports 0 F2P flipped even though the fix is correct | Parser name format doesn't match `FAIL_TO_PASS` strings | Print one line from `stdout.txt`, make sure regex captures the exact id used in `FAIL_TO_PASS`. |
| `/output.json` is empty / `{"tests": []}` | Regex too strict, or runner writes to stderr | Broaden regex; also parse stderr. |
| Some tests missing | Runner truncates output on failure (e.g. `pytest -x`) | Remove `-x` from `run_script`; run all selected files. |
| Status is lowercase / non-canonical | Using runner's native wording directly | Map to the 4 uppercase enum values (`PASSED/FAILED/SKIPPED/ERROR`). |
| JSON load fails in grader | Script prints extra text to output file | Use only the provided `export_to_json` — do not print to `output_path`. |

---

## 5. `run_script` — spec

`run_script` is written to `/run_script.sh`, then invoked as:

```
chmod +x /run_script.sh && /run_script.sh <selected_test_files_to_run...> 1> /stdout.txt 2> /stderror.txt
```

The `<args>` are `selected_test_files_to_run` joined by spaces (shell-expanded). No args → run the full suite.

Canonical template (adjust commands; keep the outer structure):

```bash
#!/bin/bash
### COMMON SETUP; DO NOT MODIFY ###
set -e

# --- CONFIGURE THIS SECTION ---
run_all_tests() {
  echo "Running all tests..."
  pytest -v --no-cov --disable-warnings --tb=short
}

run_selected_tests() {
  local test_files=("$@")
  echo "Running selected tests: ${test_files[@]}"
  pytest -v --no-cov "${test_files[@]}"
}
# --- END CONFIGURATION SECTION ---

### COMMON EXECUTION; DO NOT MODIFY ###
if [ $# -eq 0 ]; then
  run_all_tests
  exit $?
fi

if [[ "$1" == *","* ]]; then
  IFS=',' read -r -a TEST_FILES <<<"$1"
else
  TEST_FILES=("$@")
fi

run_selected_tests "${TEST_FILES[@]}"
```

Rules:

- Must start with `#!/bin/bash` and `set -e`.
- Must exit non-zero only on runner invocation errors, **not** on failing tests (that is signaled via the parsed output). With `pytest` this is fine because the parser reads the log regardless of exit code; but don't add `|| exit 1` wrappers that mask the log.
- Must not require network access.
- Must work from any `cwd` — `sweap.py` already `cd /app` before invoking it.
- Keep command-line flags compatible with your `parsing_script`. If you change the test runner flags here, re-validate the parser.

---

## 6. Complete example record

```json
{
  "metadata": {
    "instance_id": "instance_leptonai__leptonai-edfa644efb8bf6116cf99bb75dcb1c195ead783b",
    "repo": "leptonai/leptonai",
    "version": "sweap",
    "container_type": "sweap",
    "container_mem": "4G",
    "base_commit": "27e3002cd8fe513919bfee4f07a6034a400a91ba",
    "problem_statement": "# Title: Add Configuration for Background Tasks Concurrency\n\n...",
    "test_patch": "diff --git a/leptonai/photon/tests/test_photon.py b/leptonai/photon/tests/test_photon.py\n...",
    "selected_test_files_to_run": ["leptonai/photon/tests/test_photon.py"],
    "FAIL_TO_PASS": ["leptonai/photon/tests/test_photon.py::TestPhoton::test_background_max_concurrency"],
    "PASS_TO_PASS": ["leptonai/photon/tests/test_photon.py::TestPhoton::test_allow_origins_cors"],
    "parsing_script": "<full python source from Section 4.3>",
    "run_script": "<full bash source from Section 5>",
    "patch": "diff --git a/leptonai/photon/photon.py b/leptonai/photon/photon.py\n...",
    "language": "py"
  },
  "dialog": [],
  "keep_loss": [],
  "version": "sample_sft_v1"
}
```

---

## 7. Pre-submit validation checklist

Before shipping a batch:

1. **Repo coverage** — every `repo` appears in `MAP_REPO_VERSION_TO_SPECS` (see `sweap_constants.py`). If not, flag to eng to add it before running grading.
2. **instance_id parses** — `sweap.py::_repo_from_instance_id(instance_id)` returns the expected `{owner}/{repo}`.
3. **`test_patch` applies cleanly** on `base_commit` (`git apply --check`).
4. **`run_script` + `parsing_script` round-trip** produces a non-empty `/output.json` whose `tests` array covers every name in both `FAIL_TO_PASS` and `PASS_TO_PASS` (see Section 4.5).
5. **Without the gold patch** (clean `base_commit`): all `FAIL_TO_PASS` report `FAILED`, all `PASS_TO_PASS` report `PASSED`.
6. **With the gold patch applied**: all `FAIL_TO_PASS` and all `PASS_TO_PASS` report `PASSED`.
7. **JSONL sanity** — `for line in file: json.loads(line)` succeeds on every row.
8. **Python import smoke test**:
   ```python
   from agents.utils.sweagent.grading_spec.sweap import make_eval_grading_spec
   from agents.utils.sweagent.grading_spec.spec_registry import SpecRegistry
   from agents.common.types.grading_specs import CodingAgentGradingSpec
   # Build a CodingAgentGradingSpec from metadata and call make_eval_grading_spec.
   # If it returns without raising, the record is structurally valid.
   ```

Steps 5 and 6 together guarantee the grader can distinguish "fixed" from "not fixed" — the most common cause of silent grading failures is a `FAIL_TO_PASS` test that passes even without the fix (so the grade is always "pass") or always fails (so the grade is always "fail").
# 8. Mapping from the SWE-Bench Pro engineer template to MSL JSONL

Engineers currently author each task as a directory following `swe-bench-pro-template` (Harbor + SWE-Bench Pro format). This section defines the exact conversion from that directory to one JSONL record consumable by `sweap.py`.

## 8.1 Engineer directory → JSONL field mapping

Given a task directory `my-task/` with the standard layout:

```
my-task/
├── instruction.md              → metadata.problem_statement
├── task.toml                   → metadata.container_mem / grading_timeout_sec / task_category
├── environment/
│   └── Dockerfile              → (baked into the SWEAP container image; not in JSONL)
├── solution/
│   └── solve.sh                → metadata.solution (optional but recommended)
└── tests/
    ├── test.sh                 → (NOT used; MSL supplies its own test.sh equivalent)
    ├── config.json             → splits into several JSONL fields (see below)
    ├── run_script.sh           → metadata.run_script
    └── parser.py               → metadata.parsing_script
```

### 8.1.1 `tests/config.json` → `metadata.*`

| `config.json` key | MSL JSONL field | Transform |
|---|---|---|
| `repo` | `metadata.repo` | Must be `"{owner}/{repo}"` — check that it exists in `MAP_REPO_VERSION_TO_SPECS`. If the engineer wrote only `"partial-json-parser"`, upgrade to `"promplate/partial-json-parser"`. |
| `instance_id` | `metadata.instance_id` | Must be rewritten to MSL form: `aai_{owner}__{repo}-{40-hex commit \| pr_id \| issue_id}`. The engineer's `"hello-world-partial-json-parser"` is **not** acceptable — replace with e.g. `aai_promplate__partial-json-parser-dda8aa18114c7d97d35b6e554bbaf98f9a03f0d5`. |
| `base_commit` | `metadata.base_commit` | Copy verbatim. |
| `patch` | `metadata.patch` and `metadata.gold_patch` | Copy verbatim. Used by `eval_gt` and spot-checks. |
| `FAIL_TO_PASS` | `metadata.FAIL_TO_PASS` | Uppercase in both `config.json` and the MSL record. Test ids must match what `parser.py` emits. |
| `PASS_TO_PASS` | `metadata.PASS_TO_PASS` | Uppercase in both `config.json` and the MSL record. |
| `selected_test_files_to_run` | `metadata.selected_test_files_to_run` | Copy verbatim (accepts list or JSON-encoded string). |
| `before_repo_set_cmd` | `metadata.before_repo_set_cmd` | Copy verbatim. |

### 8.1.2 `tests/run_script.sh` → `metadata.run_script`

- Read the file as a single string and put the whole contents into `metadata.run_script`.
- Preserve `#!/bin/bash` and `set -e` at the top.
- Keep the `run_all_tests` / `run_selected_tests` dispatch pattern. MSL invokes it as `/run_script.sh <space-separated selected_test_files_to_run>` (not comma-separated by default), but the template's comma-detection branch is fine.
- Remove the `|| true` swallow if possible so real crashes surface, but leave it if the test runner exits non-zero on ordinary test failures (pytest does).

### 8.1.3 `tests/parser.py` → `metadata.parsing_script`

- Read the file as a single string and put the whole contents into `metadata.parsing_script`.
- Must follow the contract in Section 4:
  - CLI: `python parser.py <stdout> <stderr> <output_json>`
  - Writes `{"tests": [{"name": ..., "status": "PASSED|FAILED|SKIPPED|ERROR"}, ...]}`
  - `name` strings must match `FAIL_TO_PASS` / `PASS_TO_PASS` exactly.

The hello-world template's parser is MSL-compliant — engineers can reuse its structure verbatim. Only the regex/status mapping inside `parse_test_output` typically needs per-repo tuning.

### 8.1.4 `task.toml` → `metadata.*`

| `task.toml` key | MSL JSONL field |
|---|---|
| `[environment].memory_mb` | `metadata.container_mem` — round to `"4G"`/`"8G"`/`"16G"`. `memory_mb = 2048` → `"4G"` (minimum). |
| `[verifier].timeout_sec` | `metadata.grading_timeout_sec` — int seconds, default `900`. |
| `[metadata].category` | `metadata.task_category` — map to one of `TaskCategory`: `"Bug Fixing"`, `"Feature Development"`, `"Code Refactoring"`, `"System Optimization"`, `"Environment Setup and Configuration"`, `"Documentation"`, `"Maintenance"`, `"Testing"`, `"Other"`. Aliases auto-map in `_TASK_CATEGORY_ALIASES` (e.g. `"Refactoring"` → `CODE_REFACTORING`). |
| `[metadata].tags` | (drop) |
| `[task].keywords` | (drop) |

### 8.1.5 `instruction.md` → `metadata.problem_statement`

- Read verbatim as a single string and put into `metadata.problem_statement`.
- Preserve Markdown — the agent sees this text as-is.

### 8.1.6 `solution/solve.sh` → `metadata.solution`

- MSL's `CodingAgentGradingSpec.solution` is a bash script executed by `eval_gt` when `patch` is empty.
- For direct grading, `metadata.patch` (the unified diff from `config.json`) is sufficient. Keep `solution` as a belt-and-braces backup.

### 8.1.7 `environment/Dockerfile` → container image

- **Not** carried in the JSONL. MSL grades against pre-built SWEAP container images keyed by `instance_id` via `get_image_name(container_type=ContainerType.SWEAP, instance_id=...)`.
- For new repos, coordinate with MSL eng to build/register the image before grading: the image must contain `/app` at `base_commit` with all deps installed, and must NOT require network at test time. The engineer's Dockerfile is the source-of-truth for that build.

### 8.1.8 `tests/test.sh` → (not used)

- The engineer's `test.sh` is a Harbor-specific driver that writes `/logs/verifier/reward.txt`. MSL does not use it — `sweap.py` constructs its own eval pipeline (Section 2) that directly invokes `run_script.sh` and `parser.py`.
- Engineers should still keep `test.sh` so they can self-validate with `harbor run -a oracle`; it just isn't shipped to MSL.

## 8.2 Drop-in converter script

Hand this to the engineer pipeline so each task directory becomes one JSONL line.

```python
#!/usr/bin/env python3
"""Convert a swe-bench-pro-template task directory to one MSL JSONL record."""
import argparse
import json
import re
from pathlib import Path

# task_category normalization
_CATEGORY_MAP = {
    "debugging": "Bug Fixing",
    "bug-fix": "Bug Fixing",
    "feature": "Feature Development",
    "refactoring": "Code Refactoring",
    "optimization": "System Optimization",
    "env": "Environment Setup and Configuration",
    "docs": "Documentation",
    "testing": "Testing",
}


def mem_mb_to_g(mb: int) -> str:
    if mb <= 4096:
        return "4G"
    if mb <= 8192:
        return "8G"
    return "16G"


def make_instance_id(repo: str, base_commit: str) -> str:
    owner_repo = repo.replace("/", "__")
    if not re.fullmatch(r"[0-9a-f]{40,64}", base_commit):
        raise ValueError(f"base_commit must be 40-64 hex chars, got {base_commit!r}")
    return f"aai_{owner_repo}-{base_commit}"


def convert(task_dir: Path) -> dict:
    cfg = json.loads((task_dir / "tests" / "config.json").read_text())
    toml_text = (task_dir / "task.toml").read_text()

    repo = cfg["repo"]
    base_commit = cfg["base_commit"]

    # Prefer the MSL-style instance_id; upgrade engineer's short form.
    instance_id = cfg.get("instance_id", "")
    if not instance_id.startswith("instance_"):
        instance_id = make_instance_id(repo, base_commit)

    # Parse toml minimally (avoid tomllib for py<3.11 envs).
    mem_mb = 4096
    timeout_sec = 900
    category_raw = ""
    for line in toml_text.splitlines():
        line = line.strip()
        if line.startswith("memory_mb"):
            mem_mb = int(line.split("=")[1].strip())
        elif line.startswith("timeout_sec") and "verifier" in toml_text.split(line)[0][-120:]:
            timeout_sec = int(float(line.split("=")[1].strip()))
        elif line.startswith("category"):
            category_raw = line.split("=")[1].strip().strip('"')

    metadata = {
        "instance_id": instance_id,
        "repo": repo,
        "version": "sweap",
        "container_type": "sweap",
        "container_mem": mem_mb_to_g(mem_mb),
        "grading_timeout_sec": timeout_sec,
        "base_commit": base_commit,
        "problem_statement": (task_dir / "instruction.md").read_text(),
        "test_patch": cfg.get("test_patch", ""),  # add if engineer provides
        "selected_test_files_to_run": cfg["selected_test_files_to_run"],
        "FAIL_TO_PASS": cfg["FAIL_TO_PASS"],
        "PASS_TO_PASS": cfg["PASS_TO_PASS"],
        "parsing_script": (task_dir / "tests" / "parser.py").read_text(),
        "run_script": (task_dir / "tests" / "run_script.sh").read_text(),
        "patch": cfg.get("patch", ""),
        "gold_patch": cfg.get("patch", ""),
        "solution": (task_dir / "solution" / "solve.sh").read_text(),
        "language": "py",
        "task_category": _CATEGORY_MAP.get(category_raw.lower(), "Other"),
        "before_repo_set_cmd": cfg.get("before_repo_set_cmd", ""),
    }
    return {
        "metadata": metadata,
        "dialog": [],
        "keep_loss": [],
        "version": "sample_sft_v1",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.out.open("w") as f:
        for d in args.task_dirs:
            rec = convert(d)
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
```

## 8.3 Gap in the current engineer template — and what to add

The template's `config.json` schema is missing **one field that `sweap.py` requires** and that engineers commonly need to supply: `test_patch`.

- `sweap.py` runs `git apply` on `metadata.test_patch` to install the new tests against `base_commit` before running `parser.py`.
- In the current Harbor/SWE-Bench Pro template, the "new tests" live committed in the repo at a later commit, and `before_repo_set_cmd` does `git checkout <fix_commit> -- tests/...` to pull them in. That works for the engineer's local `test.sh`, but MSL grading expects a patch instead.

### Required template change

Add `test_patch` to `tests/config.json`:

```jsonc
{
  "repo": "promplate/partial-json-parser",
  "instance_id": "aai_promplate__partial-json-parser-dda8aa18114c7d97d35b6e554bbaf98f9a03f0d5",
  "base_commit": "dda8aa18114c7d97d35b6e554bbaf98f9a03f0d5",
  "patch": "diff --git a/src/...",
  "test_patch": "diff --git a/tests/test_examples.py b/tests/test_examples.py\n...",
  "FAIL_TO_PASS": ["tests/test_examples.py::test_fix_fast"],
  "PASS_TO_PASS": ["..."],
  "selected_test_files_to_run": ["tests/test_examples.py"],
  "before_repo_set_cmd": "git reset --hard dda8aa18114c7d97d35b6e554bbaf98f9a03f0d5\ngit clean -fd"
}
```

How to generate it from the existing engineer workflow:

```bash
cd /app   # repo at base_commit
# Apply the gold checkout to materialize the new tests
git checkout <fix_commit> -- tests/test_examples.py
# Capture the diff against base_commit and save as test_patch
git diff > /tmp/test_patch.diff
# Reset so the base repo stays clean
git checkout <base_commit> -- tests/test_examples.py
```

Then base64-free-embed `/tmp/test_patch.diff` into `config.json` as a JSON string (escape newlines as `\n`).

## 8.4 Updates to recommend to the template's documentation

Add this section to the template's `README.md` so engineers produce MSL-ingestible tasks from the start:

1. **`instance_id` must use MSL form**: `aai_{owner}__{repo}-{40-hex commit | pr_id | issue_id}`. The current template uses short names like `"hello-world-partial-json-parser"` — change this.
2. **Add `test_patch` to `config.json`** (per Section 8.3). Keep `before_repo_set_cmd` for `harbor run` compatibility.
3. **`FAIL_TO_PASS` / `PASS_TO_PASS` are uppercase** — this is the canonical casing in `config.json`. Harbor's `tests/test.sh` reads the uppercase keys, so no dual-case is needed.
4. **Parser name format** — `parser.py` must emit test ids that exactly equal the entries in `FAIL_TO_PASS`/`PASS_TO_PASS`. The current hello-world regex `r'(tests/[^\s]+::[^\s]+)\s+(PASSED|...)'` only matches paths beginning with `tests/`. Broaden per-repo when the test dir differs (e.g. `src/tests/`, `pkg/foo_test.go`).
5. **Containerization contract** — MSL requires `/app` as the repo root at `base_commit`, everything pre-installed, no network at test time. The template Dockerfile already matches; just confirm.
6. **Oracle double-check** — besides `harbor run -a oracle`, run the MSL-style pipeline locally to catch issues:

```bash
# Simulate sweap.py's pipeline
cd /app
git reset --hard <base_commit>
git apply /tmp/test_patch.diff
bash /path/to/run_script.sh <selected test file> > /tmp/stdout.txt 2> /tmp/stderror.txt
python /path/to/parser.py /tmp/stdout.txt /tmp/stderror.txt /tmp/output.json
cat /tmp/output.json  # should list every name in FAIL_TO_PASS and PASS_TO_PASS
```

Then apply `patch` and re-run — every required test must be `PASSED`. Without `patch`, every `FAIL_TO_PASS` must be `FAILED` and every `PASS_TO_PASS` must be `PASSED`.

7. **Remove Harbor-only fields from the MSL payload** — `task.toml` `[agent].timeout_sec`, `[environment].allow_internet`, `mcp_servers`, `keywords`, `tags` are not consumed by MSL and should be dropped by the converter.

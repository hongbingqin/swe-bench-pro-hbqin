# Test Output Contract

`tests/parser.py` must produce a JSON file with a **very specific** shape. MSL's `sweap_log_parsers.parse_log_sweaptest` reads this file to decide whether the task is graded as passed. Any deviation produces silent mis-grades.

Source of truth: [`agents/utils/sweagent/grading_spec/sweap_log_parsers.py`](https://www.internalfb.com/code/fbsource/genai/msl/agents/utils/sweagent/grading_spec/sweap_log_parsers.py) (Meta-internal).

## The exact shape

```json
{
  "tests": [
    {"name": "tests/test_examples.py::test_fix_fast",   "status": "PASSED"},
    {"name": "tests/test_examples.py::test_str",        "status": "PASSED"},
    {"name": "tests/test_examples.py::test_obj",        "status": "FAILED"},
    {"name": "tests/test_examples.py::test_skipped",    "status": "SKIPPED"},
    {"name": "tests/test_examples.py::test_bad_fixture","status": "ERROR"}
  ]
}
```

Rules (all must hold):

1. Top-level object, **exactly one** required key: `"tests"`.
2. `"tests"` is a **list** (not a dict). Empty list is legal JSON but indicates a parser bug in practice.
3. Each entry is a **dict** with exactly two required string keys: `"name"` and `"status"`.
4. `"name"` is a string. It **must exactly match** the corresponding entry in `config.json`'s `FAIL_TO_PASS` / `PASS_TO_PASS`. No trailing whitespace, no percent suffixes, no ANSI codes.
5. `"status"` is one of exactly four uppercase strings: `"PASSED"`, `"FAILED"`, `"SKIPPED"`, `"ERROR"`.
6. Duplicate names are allowed at the JSON level but the grader keeps only the last status per name — prefer de-duplicating in the parser.
7. Every test listed in `FAIL_TO_PASS` ∪ `PASS_TO_PASS` **must appear** in the output. A missing test is indistinguishable from "not passing" and will silently fail the grade.

## CLI invocation

MSL runs the parser as:

```
python /parse_script.py /stdout.txt /stderror.txt /output.json
```

- `argv[1]` — path to the test runner's stdout (captured by `run_script.sh`).
- `argv[2]` — path to the test runner's stderr.
- `argv[3]` — path where the parser **must** write the JSON result.

## Common mistakes (don't do these)

| Mistake | Why it fails |
|---|---|
| Using a flat dict: `{"tests": {"test_a": "PASSED"}}` | Grader iterates `for test in tests:` expecting list entries; accessing `test["name"]` on the string `"test_a"` raises. |
| Lowercase status: `"passed"`, `"failed"` | Only the four uppercase values match the grader's enum. Other casing is treated as unknown → never passes. |
| Custom status values: `"OK"`, `"XFAIL"`, `"ABORT"` | Same reason — unknown to the grader. Map to one of the four canonical values in the parser. |
| Test name mismatch: listing `tests/foo.py::test_a` in `FAIL_TO_PASS` but emitting `foo.py::test_a` in output | Grader string-equals the names. Any prefix, suffix, or case drift breaks the match. |
| Silently emitting `{"tests": []}` when regex fails to match any line | Grade is a guaranteed fail, but the parser itself claims success — no loud signal to the engineer. Fail the parser loudly when the output doesn't cover the expected tests. |
| Writing something other than JSON (e.g. log text before the JSON) to `argv[3]` | Grader does `json.loads(f.read())` — extra text breaks parsing. Use the provided `export_to_json()` helper, don't print to the output file. |

## Self-check before submitting

Run this against your parser's output locally, inside the oracle container:

```bash
python3 - <<'PY' /tmp/output.json
import json, sys, pathlib
o = json.loads(pathlib.Path(sys.argv[1]).read_text())
tests = o.get("tests")
assert isinstance(tests, list), f"'tests' must be a list, got {type(tests).__name__}"
valid = {"PASSED", "FAILED", "SKIPPED", "ERROR"}
for i, t in enumerate(tests):
    assert isinstance(t, dict) and "name" in t and "status" in t, f"entry {i} missing name/status: {t!r}"
    assert isinstance(t["name"], str) and isinstance(t["status"], str), f"entry {i} bad types: {t!r}"
    assert t["status"] in valid, f"entry {i} has bad status {t['status']!r}; must be one of {valid}"
print(f"OK — {len(tests)} test results")
PY
```

Then, if you have `config.json` available, check coverage:

```bash
python3 - <<'PY' tests/config.json /tmp/output.json
import json, sys, pathlib
cfg = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = json.loads(pathlib.Path(sys.argv[2]).read_text())
required = set(cfg.get("FAIL_TO_PASS", [])) | set(cfg.get("PASS_TO_PASS", []))
emitted = {t["name"] for t in out.get("tests", [])}
missing = required - emitted
if missing:
    print("PARSER CONTRACT VIOLATION: missing tests in output:")
    for m in sorted(missing):
        print(f"  - {m}")
    sys.exit(1)
print(f"OK — all {len(required)} required tests present")
PY
```

If either check fails, fix the parser before opening a PR. Silent failures here are the #1 cause of mis-graded tasks.

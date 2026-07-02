# Contamination Check Report: codimango/ignite-fairness-subgroup-ratio

## Task Identity (Check 1)
- **Path:** /Users/hbqin/AAI/swe-bench-pro-hbqin/ignite-fairness-subgroup-ratio
- **Type:** swe-bench-pro (tests/config.json present)
- **Task name:** codimango/ignite-fairness-subgroup-ratio
- **Domain:** ML (fairness metrics, PyTorch-Ignite)
- **Repo (swe-bench-pro):** pytorch/ignite
- **Instance ID (swe-bench-pro):** aai_pytorch__ignite-d5208644ab5646ee250c3b5db08f5c5e9f128354-fairness-subgroup-ratio
- **Base commit:** d5208644ab5646ee250c3b5db08f5c5e9f128354

## Internal Decontamination Table (Check 2)
- **Result:** NOT FOUND
- **Source:** Source A1 — Manifold CLI snapshot `aai_eval_baselines/tree/quality_contamination/decontaminated_delta.json`
- **Snapshot:** `exported_at=2026-07-02T03:49:05Z`, age ≈ fresh (< 48h) — not stale
- **Lookup key:** `prompt_hash = 373997454193324049` (xxHash64 of the 493-byte prompt, signed big-endian BIGINT)
  - xxh64 was computed with a pure-Python implementation (no `xxhash` module / package manager available in this env), **self-verified against canonical test vectors** `xxh64("")=0xEF46DB3751D8E999` and `xxh64("a")=0xd24ec4f1a98c6e5b` before use.
- **Details:** `by_prompt_hash` contains 49,773 entries; **0 match** this prompt hash. No legacy `tasks` index hit for the task name either. The task is not present in the decontamination table.

## Overall Contamination Risk

**Risk Level:** LOW-with-caveat

**Summary:** This is a brand-new, never-submitted task, so it has not yet been evaluated by the KNN + N-gram + Gemini decontamination pipeline and does not appear in the current (fresh) snapshot. NOT FOUND is *not* a clearance — it means "not evaluated, not cleared." The task's feature (a `reduction="ratio"` mode on ignite's own `SubgroupDifference`, `versionadded 0.5.4`) is ignite-internal and not a documented external API, which lowers a-priori recall risk, but that is a design observation, not a table verdict.

**Action Required:**
- LOW-with-caveat (NOT FOUND) → Not in the table, so NOT cleared. Submit the task to codimango GitHub so the decontamination pipeline runs and computes a real verdict. Treat as no-signal until then.

**Evidence:**
- Table verdict: NOT FOUND (0 rows for `prompt_hash=373997454193324049`).
- `snapshotExportedAt`: `2026-07-02T03:49:05Z`.

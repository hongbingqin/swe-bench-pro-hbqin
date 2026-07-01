# Contamination Check Report: codimango/ignite-precision-zero-division

## Task Identity (Check 1)
- **Path:** /Users/hbqin/AAI/swe-bench-pro-hbqin/ignite-precision-zero-division
- **Type:** swe-bench-pro (tests/config.json present)
- **Task name:** codimango/ignite-precision-zero-division
- **Domain:** ML (metrics library — classification precision/recall/fbeta)
- **Repo:** pytorch/ignite
- **Instance ID:** aai_pytorch__ignite-d5208644ab5646ee250c3b5db08f5c5e9f128354-precision-zero-division
- **Base commit:** d5208644ab5646ee250c3b5db08f5c5e9f128354

## Internal Decontamination Table (Check 2)
- **Result:** NOT FOUND
- **Source:** Source A1 — Manifold CLI snapshot
- **Snapshot:** `exported_at=2026-07-01T23:02:45Z`, `age≈0.4h` (fresh)
- **prompt_hash:** 3747351980061100895 (xxHash64 of instruction.md)
- **Details:** 0 matches in `by_prompt_hash`. No pytorch/ignite precision/metrics task in the snapshot (the "ignite" name hits are unrelated: `ignite-theme-hover-css`, codeigniter tasks). Never submitted → not yet evaluated by the pipeline.

## Overall Contamination Risk

**Risk Level:** LOW-with-caveat

**Summary:** Not in the decontamination snapshot (NOT FOUND, fresh). NOT FOUND is "not yet evaluated," not a clearance. The feature is a sklearn-parity option with a bespoke ignite implementation across average modes — not a copyable famous artifact — so expected platform contamination is LOW–MEDIUM once the pipeline runs.

**Action Required:**
- LOW-with-caveat (NOT FOUND) → submit so the pipeline computes a real verdict; treat as no-signal until then.

**Evidence:**
- `by_prompt_hash["3747351980061100895"]` → 0 rows.
- `snapshotExportedAt`: 2026-07-01T23:02:45Z (age ≈ 0.4h).

Report written to `/Users/hbqin/AAI/swe-bench-pro-hbqin/ignite-precision-zero-division/.review/contamination-report_20260701_233000.md`

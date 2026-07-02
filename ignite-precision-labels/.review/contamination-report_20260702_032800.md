# Contamination Check Report: codimango/ignite-precision-labels

## Task Identity (Check 1)
- **Type:** swe-bench-pro
- **Task name:** codimango/ignite-precision-labels
- **Domain:** ML (metrics library)
- **Repo:** pytorch/ignite
- **Base commit:** d5208644ab5646ee250c3b5db08f5c5e9f128354

## Internal Decontamination Table (Check 2)
- **Result:** NOT FOUND
- **Source:** Manifold CLI snapshot
- **Snapshot:** exported_at=2026-07-01T23:02:45Z, age≈4.4h (fresh)
- **prompt_hash:** 335340686361605791
- **Details:** 0 matches in by_prompt_hash. Never submitted → not yet evaluated.

## Overall Contamination Risk
**Risk Level:** LOW-with-caveat

**Summary:** Not in the snapshot (NOT FOUND, fresh) — not-yet-evaluated, not a clearance. Bespoke sklearn-parity `labels` implementation across ignite's average modes; not a copyable artifact → expected platform verdict LOW–MEDIUM once the pipeline runs.

**Action Required:** LOW-with-caveat → submit so the pipeline computes a verdict.

**Evidence:** by_prompt_hash["335340686361605791"] → 0 rows; snapshot 2026-07-01T23:02:45Z.

Report written to `/Users/hbqin/AAI/swe-bench-pro-hbqin/ignite-precision-labels/.review/contamination-report_20260702_032800.md`

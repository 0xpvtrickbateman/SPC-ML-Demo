# Continuation guide

Read [requirements coverage](requirements-coverage.md), [operator runbook](presentation-runbook.md) and [job setup](job-setup.md) first. Verify the current branch, source SHA and dirty state before editing. Do not assume a branch name or old pull request identifies the current deliverable. Preserve other contributors' changes.

The simple story is synthetic USCIS-related application volume, unusual activity, then analyst review. All source records and metadata are fabricated. Keep SPC rules, ML rule-label classification, anomaly scoring and actual future-count forecasting distinct. The classifier has proxy labels, not verified incidents. Report measured comparisons even when a baseline or logistic regression wins.

The training notebook has 22 executable cells. Source code and its runtime manifests are authoritative for row counts and table keys. Never copy metrics from an earlier dataset into a new deck or runbook. Preserve the event-to-daily and event-to-office reconciliation checks. Preserve chronological boundaries, future target dates, forecast origins, frozen scoring and pending human review.

Training and independent saved-model scoring are separate entrypoints; see job setup for their commands. Retain the artifact manifest and versioned model bundle used for scoring. MLflow and registry integration is optional; an independently loadable artifact is necessary for reuse. Do not describe a rerun of all training cells as inference-only.

The native dashboard default is `ml_statistical_process_controls.demo_schema`. The training notebook defaults to that Delta destination. Local execution requires `SPC_DEMO_LOCAL_TEST=1`; an analytical native run can explicitly set `OUTPUT_SCHEMA = ""`. Native import, all dataset queries, warehouse permissions and rendered widgets must be checked in the target workspace before claiming platform readiness. Contract doubles and local HTML are useful evidence but do not replace that check.

Human investigation precedes retraining. A candidate needs later-date evaluation, sufficient actuals, a baseline comparison and real approval before promotion. Retain the previous version and a tested rollback path. The repository's disposition and rollback rows are exercises; notices are prepared but never sent. No shared file should contain private discussion provenance, participant details, workspace credentials or unverified historical customer claims.

No production readiness, cloud acceptance, commit or publication should be inferred from local edits or tests. Record remaining gates explicitly in the execution receipt.

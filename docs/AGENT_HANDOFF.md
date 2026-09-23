# SPC ML Demo: Codex agent handoff

## Mission

Continue preparing the AttainX synthetic statistical process control (SPC) and machine learning notebook for a live federal RFI oral demonstration. Keep the explanation understandable to people who have never seen SPC or Databricks. Preserve the distinction between a working demonstration, an operational pilot, and USCIS results.

The durable source is [0xpvtrickbateman/SPC-ML-Demo](https://github.com/0xpvtrickbateman/SPC-ML-Demo), branch `main`. The latest verified project commit before this handoff is `9e1c5d26d98e5658644a1ee460ceb342877551f4`. Clone the GitHub repository in a new workspace.

## Deliverables already in the repo

- `notebooks/SPC_ML_Demo.py`: Databricks source notebook with ten executable cells, an explanatory markdown cell before **each** code cell, an introduction, and closing demo notes.
- `docs/SPC_ML_Demo_Walkthrough_Ready.pptx`: 22-slide presenter deck with SPC introduction, a cell-by-cell walkthrough, explanations of each technique, the observed synthetic results, the documented USCIS reference design, potential reuse by other agencies, oral-question coverage, and speaker notes.
- `README.md`: import, run, interpretation, local test, and presentation instructions.
- `docs/reference-mapping.md`, `docs/job-setup.md`, `sql/dashboard_queries.sql`, `tests/smoke_test.py`, and `requirements.txt`.

The USCIS reference notebook and real outputs were **not** committed to the public repository. The synthetic demo generates its own fictional counts.

## What the notebook actually does

1. Sets a seed and rule parameters.
2. Generates 1,260 fictional daily observations across three series.
3. Defines XmR, CUSUM, and EWMA SPC checks.
4. Computes prior-window flags and features. `signal_detected` is the OR of the three rules.
5. Trains a random forest classifier to imitate that **same historical-window** rule label. It is not future incident prediction.
6. Trains a separate random forest regressor to forecast the **next business day's count** from already known history.
7. Draws the SPC chart, held-out confusion matrix, and forecast comparison.
8. Consolidates repeated flagged windows into a pending analyst review list and a monitoring snapshot. There is no actual analyst disposition or notification integration.
9. Logs metrics and two model artifacts to MLflow when available; Unity Catalog registration is off unless explicitly configured.
10. Optionally writes six managed Delta tables to a designated demo schema; `OUTPUT_SCHEMA` defaults to empty.

The local seeded smoke test passed. It reported 915 labeled windows; 52.3% overall rule signal rate; 228 classifier test rows with a 78.1% signal rate; classifier accuracy 74.1% versus 78.1% for always predicting signal; classifier ROC AUC 0.929; one-day forecast MAE 8.85 counts versus 9.73 for a five-day trailing-mean baseline; and 479 flagged windows consolidated into 30 review episodes. These numbers describe only this synthetic run. The classifier fails a simple accuracy baseline, so recommending the direct rules is the honest decision. The forecast merits further testing; it has not been validated on real agency data.

## RFI / oral scope

The deck maps to 14 oral questions: (1) live working capability and lifecycle; (2) AI versus rules decision; (3) readiness; (4) bounded pilot; (5) governance and monitoring; (6) AI risk; (7) security; (8) interoperability; (9) vendor exit; (10) changes and rollback; (11) human oversight; (12) a verified *past* engagement where AI was declined; (13) ATO integration; and (14) staffing and knowledge transfer. The notebook demonstrates parts of these topics; the remaining governance and agency-specific answers require separate evidence and named owners.

**Outstanding oral gap:** Question 12 requests a prior customer engagement. The classifier in this demo is a present-day decision against unnecessary ML, but it does not satisfy the historical-example requirement. Obtain a verified example from AttainX leadership; do not invent one.

The supplied USCIS reference notebook documents a broader implementation: five-day forecasting, classification and anomaly detection, upstream lineage, and fixes for repeated alerts and stale retraining triggers. It is distinct from this one-day synthetic forecast. There is no supplied, verified postchange measurement of USCIS staff time saved or alert reduction; avoid claiming one.

## Next actions for the receiving agent

1. Check current `main` and review the notebook, deck notes, README, and any newer team feedback before changing code.
2. If authorized access to the AttainX Databricks workspace becomes available, pull the GitHub folder and run every notebook cell on the presentation compute. Confirm the ten markdown cells render, inspect the charts and metrics, and confirm MLflow logging. Enable Delta writes only in a dedicated writable demo catalog/schema. Record the run ID and any environment errors. Do not claim that this revision has been run in Databricks until it has.
3. If the oral demonstration needs a Lakeflow job or SQL dashboard, create and test them in the AttainX demo workspace. Their setup files are present, but execution was not verified.
4. Reconcile oral responses and deck wording against the actual one-business-day forecast, classifier baseline result, pending-only review queue, and optional nature of registry and Delta outputs. Fix any inconsistency without importing private agency data into the public repository.
5. Verify all substantive claims with the available evidence. State the sandbox/USCIS distinction at the start of the oral presentation. Preserve human review and the right to reject or withdraw an ML component.
6. Re-run `MPLBACKEND=Agg python tests/smoke_test.py` after code changes. For the slide deck, render and inspect every slide after edits. Commit reviewed changes to the GitHub repository and provide the user with an exact run sequence and remaining blockers.

## Access and limitations

The notebook revision passed a local top-to-bottom run. The deck was rendered and structurally validated. This agent did not have authenticated Databricks execution access, so MLflow writes, Unity Catalog writes, job execution, dashboard operation, and model registration remain unverified in the target workspace. The demo is suitable for *presentation validation*, not a claim of federal production authorization.

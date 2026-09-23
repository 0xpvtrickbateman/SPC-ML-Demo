# SPC ML Demo

Review the [validation report](docs/validation-report.md) for completed checks and the remaining target-workspace rehearsal steps.

An AttainX demonstration of **synthetic USCIS-related application volume → unusual activity → analyst review**. All records, counts, offices, events and lineage are fabricated. Intake queues count application-receipt events; the completion queue counts workflow-completion events. These are independent streams, not a linked applicant lifecycle. The demo accesses no USCIS data or services and makes no claim about agency outcomes.

The main [Databricks notebook](notebooks/SPC_ML_Demo.py) has 22 executable cells with explanations and intermediate dataframe previews. SPC uses statistical rules to identify unusual patterns. A separate ML experiment tests whether a classifier can copy those rule labels; a separate forecasting branch predicts actual future counts. A signal or classifier prediction is never a confirmed incident.

## Start here

- [Operator run order and fallback](docs/presentation-runbook.md)
- [Training, saved-model scoring and manual job setup](docs/job-setup.md)
- [Native dashboard setup](dashboards/README.md) and [local setup helper](dashboards/setup.html)
- [Requirement-by-requirement evidence](docs/requirements-coverage.md)
- [Reference capability mapping and limits](docs/reference-mapping.md)
- [Presenter deck](docs/SPC_ML_Demo_Walkthrough_Ready.pptx) and [talking points](docs/talking-points.md)

## What the workflow demonstrates

| Stage | Evidence in the notebook | Meaning |
| --- | --- | --- |
| Generate and reconcile | Cell 2; Cell 10 | Underlying synthetic events aggregate into daily volume and office counts. |
| Detect and explain | Cells 3–4, 8–10, 14–15 | XmR, CUSUM and EWMA feed a review queue; subgroup, lineage and event evidence support hypotheses. |
| Compare alternatives | Cells 5–6, 11–13 | Chronological evaluation compares classifiers, forecast models and simple baselines; Isolation Forest adds a separate anomaly score. |
| Monitor and decide | Cells 16–18, 21–22 | Preserve forecast origins, inspect drift and error, review candidate evidence, rehearse rollback and prepare unsent notices. |
| Retain and reuse | Cells 19–20; scoring entrypoint | Save model artifacts and metadata, score independently, and optionally retain queryable Delta history. |

Use the measured comparisons printed by the run. Random forest is a candidate, not a predetermined winner. Classification accuracy measures agreement with SPC proxy labels; forecast MAE measures error in application counts. Neither measures real incident detection or customer benefit.

## Recorded synthetic scale

The current seed-42 run materializes **1,179,830 events**, reconciled into **1,260 daily-series observations** across three queues and 420 business dates. It produces **915 historical feature/label windows**. These are different data grains: the event count is not the ML training sample size. Raw and prepared event frames remain in notebook memory; the 19 workflow and three drift Delta tables retain aggregate/results data.

## Local verification

Use Python 3.12 and an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo-lock.txt
python -m pip install duckdb==1.4.4  # test-only SQL validation dependency
MPLBACKEND=Agg python tests/smoke_test.py
MPLBACKEND=Agg python tests/mlflow_contract_test.py
MPLBACKEND=Agg python tests/delta_contract_test.py
MPLBACKEND=Agg python tests/dashboard_test.py
node tests/setup_helper_test.mjs
MPLBACKEND=Agg python tests/model_reuse_test.py
```

See the runbook for the independent scoring contract check and expected outputs. MLflow and Delta contract checks use simulated services; local dashboard SQL checks do not prove Databricks rendering. The native dashboard targets `ml_statistical_process_controls.demo_schema`; the notebook is configured to write there by default. Set `SPC_DEMO_LOCAL_TEST=1` for local execution, or explicitly set `OUTPUT_SCHEMA = ""` for an analytical run without Delta writes. This default-storage catalog requires serverless notebook compute and a serverless SQL warehouse; verify their availability and destination permissions before a native run.

## Boundaries

This is a repeatable synthetic demonstration, not a production service. It shows model training, saved-model reuse, monitoring and human decision points. Live feeds, agency validation, calibrated alert policies, durable delivery, real approval records and production promotion are separate work. Refreshing the dashboard rereads results; it does not ingest a new day, train a model or approve a decision. Actual Databricks execution, model loading on target compute, Delta permissions and native dashboard rendering require their own verification receipts.

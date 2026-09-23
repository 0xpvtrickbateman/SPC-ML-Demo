# SPC ML Demo

An AttainX demonstration of statistical process control and machine learning using **entirely synthetic data**. The notebook is the runnable product. It does not access USCIS data, services, tables, or model artifacts.

See [reference-to-demo mapping](docs/reference-mapping.md) for the specific reference ideas represented here and the capabilities intentionally left out.
The [presenter deck](docs/SPC_ML_Demo_Walkthrough_Ready.pptx) explains SPC, every executable notebook cell, the distinct ML tasks, the observed synthetic results, and the oral-question coverage. Its speaker notes provide a plain-English talk track.

## Five-minute setup in Databricks

1. In **Workspace**, open the Git folder for this repository, then open `notebooks/SPC_ML_Demo.py` as a notebook. If Git folders are not ready, import that file through **Workspace → Import**.
2. Choose available Python compute and run **Run all**. The default settings require no catalog, secret, external file, GPU, or model serving endpoint. This revision passed a local top-to-bottom check; verify it in your own Databricks workspace before the oral presentation.
3. Inspect the control chart, the forecast chart, the printed held-out metrics, the pending analyst review queue, and the MLflow run ID. Each code cell now has a short explanatory markdown cell directly above it. Allow a few minutes for compute startup.
4. If you have a writable Unity Catalog catalog and schema, set `OUTPUT_SCHEMA = "catalog.schema"` near the top and rerun to create six managed Delta tables. Leave it empty if permissions or managed storage are not yet ready.
5. If a Unity Catalog model registry is ready, set `UC_MODEL_NAME = "catalog.schema.spc_rule_classifier"` and rerun. Registration is optional. Do not point this demo at production objects.

The source file begins with `# Databricks notebook source` and contains Databricks cell markers, so the workspace recognizes it as a notebook. Running it top to bottom is the intended check.

## What the demo shows

| Step | Evidence | Interpretation |
| --- | --- | --- |
| Data | 1,260 made-up daily counts from three fictional series; seed 42 | A reproducible operational feed with planted spikes and shifts. |
| SPC | XmR, CUSUM and EWMA over 25-day windows; `signal_detected` is their logical OR | A signal is a review candidate. It does not establish cause, severity, or a real incident. The window can keep a signal active after an unusual day. |
| Rule-label classifier | Random Forest trained on rolling statistics, evaluated on later dates with a gap | Reproduces a label that the rules already calculate. This model is **not** an advance warning system. The direct rules remain the first choice for determining whether those rules fired. |
| Forward-looking model | Random Forest forecasts the next business day's count from history and known weekday/series fields | A distinct prediction task. Compare its mean absolute error against the prior count and trailing five-day mean. Synthetic results are not real-world performance estimates. |
| Oversight | Consecutive flagged windows consolidated into review episodes with triggering rules, duration, source and `Pending analyst review` | A person investigates each candidate. The demo does not email the reviewer or decide an outcome. |
| Traceability | Seed, data ID, split dates, baseline scores, MLflow run and optional Delta tables | Enough to repeat this fabricated run; actual enterprise lineage and incident audit are outside the prototype. |

## Evaluation and limits

Each example's inputs are available before its target: the rule-label classifier sees the prior 25-day window; the forecast's target is the later daily count. All series share the same date cutoff. A 25-business-day gap separates training and test windows. The code asserts these boundaries.

The generated data intentionally changes between training and test. On the verified local run, the classifier has 0.741 accuracy, while an always-signal prediction reaches 0.781 because the later set has many rule signals. Its numerical score is useful for explaining model validation and for arguing **against** deploying an unnecessary classifier. The separate forecast has 8.85 mean absolute error versus 9.73 for the trailing-mean baseline on the synthetic later period. Rerun in Databricks and use its printed results if they differ.

For a real deployment, calibrate rules with process owners, test false-alert burden, define disposition categories and owners, validate data quality and subgroups, measure drift separately from SPC process signals, review security/authorization, and demonstrate rollback. This sample proves none of those controls exist in production.

## Optional presentation extras

- **Delta tables:** `synthetic_daily`, `spc_signals`, `spc_predictions`, `count_forecasts`, `review_queue`, `monitoring_snapshot` under `OUTPUT_SCHEMA`.
- **Dashboard:** `sql/dashboard_queries.sql` contains read-only starter queries after tables exist. A notebook chart is sufficient if SQL warehouse or dashboard access is delayed.
- **Job:** create a Lakeflow Job with one notebook task targeting `notebooks/SPC_ML_Demo.py`; run it once and inspect its output before optionally scheduling it. See `docs/job-setup.md`.
- **Version control:** develop on a branch, review changes, then pull them into the Databricks Git folder. Source notebooks do not normally commit output charts; rerun after pulling.

## Local check

With Python 3.12 and packages in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
MPLBACKEND=Agg python tests/smoke_test.py
```

Local tests omit Databricks-only Delta output and MLflow if not installed. A successful local check does not replace a full Databricks run, job execution or dashboard verification.

## Before the oral presentation

After pulling the newest commit into the Databricks Git folder, verify that the introductory markdown and **Cell 1** through **Cell 10** markdown appear above their corresponding code cells. Run all cells on the presentation compute, then check the MLflow run if the workspace permits it. OUTPUT_SCHEMA remains empty unless a dedicated writable demo schema has been confirmed. Bring the presenter deck as a companion to the live notebook, and use the live notebook outputs if the displayed numbers differ from the seeded local check.

## Demonstration sequence

1. Say: “This is a fabricated AttainX version. I am showing the workflow, not USCIS data or results.”
2. Run the notebook and point to planted shifts, the rule flags and `signal_detected`.
3. Show the high signal rate and explain why the 25-day OR is a **triage input**. Point to the consolidated episode queue to show how repeated flags are grouped.
4. Show held-out classifier results **and** constant baselines; explain why the direct rule calculation is preferable for the rule label.
5. Show the separate next-day forecast, trailing-mean comparison and observed-versus-predicted chart.
6. Show a pending review row, MLflow run and optional Delta/Job evidence. Explain how a reviewer could investigate and why no decision is automatic.

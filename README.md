# SPC ML Demo

An AttainX demonstration using fictional daily counts. The Databricks notebook contains **22 executable cells, ten figures and an interactive drift dashboard**, with plain-English explanations immediately above each code cell. It accesses no USCIS data or services.

- [Notebook](notebooks/SPC_ML_Demo.py)
- [Dashboard setup helper](dashboards/setup.html), [preview](dashboards/preview.html), and [click-by-click guide](dashboards/README.md)
- [22-slide presenter deck with speaker notes](docs/SPC_ML_Demo_Walkthrough_Ready.pptx)
- [Talking points](docs/talking-points.md) and [rehearsal runbook](docs/presentation-runbook.md)
- [Reference coverage and remaining differences](docs/reference-mapping.md)

## Demonstrated workflow

| Cells | Capability |
| --- | --- |
| 1–4 | Reproducible counts, federal-holiday calendar, XmR/CUSUM/EWMA, historical labels and features |
| 5–8 | Rule classifier, separate one-day count forecast, charts, grouped pending reviews |
| 9–10 | Severity zones, distributions, correlations, office profiles and subgroup anomaly evidence |
| 11–13 | Five-fold forest/logistic comparison, feature importance, five-day ARIMA/Holt-Winters and subgroup forecasts, Isolation Forest |
| 14–15 | Five-hop fixture lineage, column mapping, freshness, event matching and investigation hypotheses |
| 16–18 | Forecast history and retraining safeguards, candidate/rollback exercise, daily notice replay and review dashboard |
| 19–20 | Four optional MLflow model artifacts, demo registry aliases/tags, 19 optional Delta tables with keyed MERGE |
| 21–22 | Input drift, error deterioration, a separate controlled exercise, interactive dashboard and three additional optional Delta tables |

## Show the dataframe transformations

Cells 2–10 now display eleven bounded previews of actual intermediate dataframes. They follow `Intake A` across the same five dates, show full row/column counts, and explain the operation and what each row represents.

Start with `daily_df`, summarize history into `window_features_df`, then join four rule-label columns to create `signals_df`. From there, show the separate classifier and count-forecast branches. Cell 8 pairs daily episode members with their grouped review row; Cell 9 joins severity; Cell 10 pairs parent counts with their office rows. Cells 3 and 7 define functions or plot existing results rather than adding analytical columns.

The tables are presentation copies: rounding and filtering affect only the previews. Full results and model inputs retain their original precision. `dataframe_stages` also keeps the displayed previews for inspection.

## Local results

September 23, 2026, seed 42 and `requirements-demo-lock.txt`:

| Comparison | Result |
| --- | --- |
| Single-holdout rule classifier | 75.0% accuracy; always-signal baseline 78.1%; AUC 0.917 |
| One-day count forecast | MAE 8.99 counts; trailing-five-day mean 9.73 |
| Five-fold mean classification accuracy | Logistic regression 81.3%; random forest 78.4% |
| Five-day pooled MAE | ARIMA 10.15; Holt-Winters 10.38; seasonal naive 12.92 |
| Untouched final forecast origin | Earlier-selected ARIMA improves MAE 22.9% versus seasonal naive; only 15 observations |

Use the direct SPC rules for their own label. Forecast results justify more evaluation on authorized real data, not operational deployment. Fold averages and single-holdout results are different experiments. Pooled five-day MAE is descriptive; only earlier origins select the model.

## Run locally

Use Python 3.12 and an isolated environment. Local tests do not install MLflow or Spark.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo-lock.txt
MPLBACKEND=Agg python tests/smoke_test.py
MPLBACKEND=Agg python tests/mlflow_contract_test.py
MPLBACKEND=Agg python tests/delta_contract_test.py
```

The smoke test executes the full notebook and checks time boundaries, forecast horizons, subgroup reconciliation, lineage cycles, retained history, retraining safeguards and replay keys. MLflow and Delta tests use simulated APIs; they do not establish workspace acceptance.

## Run in Databricks

Pull the intended reviewed branch/revision into the authorized Git folder, or import `notebooks/SPC_ML_Demo.py`. Install missing dependencies through the compute's supported environment process, including statsmodels. Open the notebook, verify Cell 1–22 markdown, select presentation compute and run all cells.

Start with `OUTPUT_SCHEMA = ""` and `UC_MODEL_NAME = ""`. MLflow still attempts experiment logging when installed and requires experiment permissions. Inspect ten figures and the printed tables. Then verify four returned model URIs if logging succeeds.

For optional writes, use a **new dedicated demo schema**. Setting `OUTPUT_SCHEMA` enables 22 keyed Delta MERGEs (19 workflow tables plus three drift tables); it does not migrate incompatible old schemas. Setting `UC_MODEL_NAME` registers four sklearn candidates with `demo_only` tags and `demo_candidate` aliases. No production alias is changed. Five-day statsmodels fits are not logged as model artifacts; predictions and assessments are retained as tables.

See [job setup](docs/job-setup.md) and [SQL starter queries](sql/dashboard_queries.sql). Real MLflow/registry writes, Spark/Delta execution, schedules and SQL dashboards remain unverified on target compute. Notifications are prepared locally and never delivered. A full run retrains the demo models; it is not a separate production inference service.

## Interpretation limits

The demo represents every major reference workflow area with selected methods at reduced depth. It is not an exact port. Lineage and events are fixtures, subgroup forecasts are independent rather than reconciled, and historical assessments replay already observed counts. The dataset is fixed, so rerunning tomorrow does not ingest a new day. Real data loaders, live metadata discovery, durable notification delivery, independent inference services and agency authorization are outside this demonstration.

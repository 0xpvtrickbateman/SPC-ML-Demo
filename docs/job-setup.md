# Training and independent scoring

Use the reviewed source revision and compatible Python environment. A Python notebook requires notebook compute; a SQL warehouse runs dashboard queries, not Python training cells. For the default-storage catalog `ml_statistical_process_controls`, use serverless notebook compute and a serverless SQL warehouse. Verify both are available to the operator and that the identities have the required catalog, schema and table permissions. Keep schedules disabled for this manual demonstration.

## 1. Train and save once

1. Open `notebooks/SPC_ML_Demo.py` from the repository Git folder, with `notebooks/model_reuse.py` beside it. If importing manually, also upload `model_reuse.py` as a Workspace file (not a notebook) beside both notebook sources. Importing only the main source file is insufficient for the companion helper.
2. The notebook defaults to `OUTPUT_SCHEMA = "ml_statistical_process_controls.demo_schema"`. Explicitly set `OUTPUT_SCHEMA = ""` and leave `UC_MODEL_NAME = ""` for a first analytical run without Delta writes. MLflow may still attempt experiment logging when installed; inspect its success or failure explicitly.
3. Set environment variable `SPC_DEMO_MODEL_DIR` to an accessible persistent directory before training if reuse must survive the compute session. The local default is `artifacts/saved_forecast`. A permitted Unity Catalog volume can use `/Volumes/ml_statistical_process_controls/demo_schema/<volume>/saved_forecast`; replace `<volume>` with an existing authorized volume.
4. Run all 22 cells. Inspect the reconciliation assertions, chronological model comparisons, future dates, pending review queue, monitoring output and artifact-save result.
5. Retain the bundle's `forecast.joblib`, `manifest.json`, `sample_inputs.json` and `expected_predictions.json` together. The manifest records feature order/types, dependency versions, integrity hash and an MLflow model URI when logging actually succeeds. This loadable bundle is required for the reuse demonstration; MLflow and registry registration are optional mechanisms.

Local-test mode disables Delta writes and MLflow logging. Training locally uses the same notebook source:

```bash
SPC_DEMO_LOCAL_TEST=1 MPLBACKEND=Agg python notebooks/SPC_ML_Demo.py
```

## 2. Score without fitting

Start a fresh process with the existing bundle:

```bash
python notebooks/SPC_Model_Scoring.py
```

The default local directory is `artifacts/saved_forecast`. Set `SPC_DEMO_MODEL_DIR` to the training bundle's directory for another location. The scoring entrypoint loads the artifact, validates its manifest/features and predicts from saved, already engineered held-out sample inputs. It prints the expected and loaded-model counts followed by `Saved-model inference passed; no model training was performed.` This is reuse/parity evidence, not a new accuracy evaluation. It must reproduce the saved expected predictions without executing training. Inspect the scorer's printed model identity, backend and parity result.

For a Databricks notebook task, open `notebooks/SPC_Model_Scoring.py` and set widgets `saved_model_directory` and `saved_model_backend`. Use the same persistent bundle directory. The default backend loads the integrity-checked local joblib artifact. If a real MLflow URI was captured and the target identity can read it, choose `mlflow`; locally this is `SPC_DEMO_SCORE_BACKEND=mlflow`. MLflow failure is not evidence of successful remote loading. Use the local bundle fallback explicitly and record which backend was verified. Load only a trusted bundle created for this demonstration.

### Optional real MLflow reuse check

The separate integration check logs the actual models to disposable local MLflow tracking and loads the returned forecast URI in a fresh process with fitting forbidden. It verifies a real library round trip, not Databricks or Unity Catalog access. Run it in a separate environment to keep the basic demonstration environment unchanged:

```bash
python3.12 -m venv .venv-mlflow
.venv-mlflow/bin/python -m pip install -r requirements-demo-lock.txt mlflow-skinny==3.16.1
MLFLOW_DISABLE_AGENT_HINT=1 MPLBACKEND=Agg .venv-mlflow/bin/python tests/mlflow_reuse_integration_test.py
```

## 3. Retain data and verify the native dashboard

1. Verify `ml_statistical_process_controls.demo_schema` exists and the notebook identity can write the dedicated demo tables. Use another approved dedicated schema only if both notebook and dashboard are rebound consistently.
2. Set `OUTPUT_SCHEMA = "ml_statistical_process_controls.demo_schema"` in the training notebook and rerun. Compare all tables against `DEMO_TABLES` and `DRIFT_TABLES`, including the selected dataset ID, row counts and composite keys. The manifests are authoritative; do not assume an old fixed table count.
3. Repeat the run and confirm no duplicate keys. Keyed MERGE retains forecast origins; it does not migrate incompatible table schemas. Resolve schema differences in the dedicated demo destination before proceeding.
4. Import `dashboards/AttainX_SPC_Demo.lvdash.json` as a draft dashboard. Its default schema is already configured. The local setup helper can generate a copy bound to another approved schema.
5. Select an existing serverless SQL warehouse the presenter can use for this default-storage catalog. Run all ten dashboard datasets in the Data tab. Verify each query uses the intended schema and dataset ID, then refresh all four content pages and the queue filter.
6. Confirm the displayed counts, forecast/error values, selected scenario, review rows and refresh/model metadata against notebook output. Record import, warehouse query and rendering evidence separately. Publishing and a live schedule are not needed for rehearsal.

If tables are missing, compare notebook `OUTPUT_SCHEMA`, dashboard SQL and successful persistence output. If reads fail, check the warehouse identity's catalog/schema/table permissions. If native import or rendering cannot be verified, use `dashboards/preview.html` and label it a local companion preview; do not claim native success.

## 4. Manual jobs and the human loop

Create separate manual tasks for the training notebook and scoring notebook. Run training deliberately when creating a candidate; run scoring against an existing bundle to demonstrate reuse. Record each task's source revision, configuration, artifact identity and result. A schedule is not created by this repository. Replaying the fixed data does not ingest a new day.

Analysts first inspect data quality, seasonality, office mix, source freshness and event context. Drift or poor errors request investigation; they do not authorize retraining or operational action. If a candidate is justified, train it separately, evaluate on later dates against the retained model and simple baselines, check sample sufficiency, and obtain real human approval. Preserve the prior model/version, feature contract and rollback route. The notebook's approval/withdrawal records are fictional exercises. External notification delivery, durable acknowledgments and real approval storage remain integration work.

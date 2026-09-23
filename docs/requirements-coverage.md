# Requirements coverage and acceptance

This map describes repository evidence, not production acceptance. **Implemented** means an inspectable artifact exists; **local check** means the named test can verify it without target services; **native gate** requires actual Databricks evidence. Record actual pass/fail results and source revision in the runbook receipt. No benchmark below is a historical customer result.

| Requirement | Precise artifact evidence | Local acceptance check | Target/cloud gate or limitation |
| --- | --- | --- | --- |
| Simple business story | README opening; notebook Cells 2–4 and 8; runbook opening | Explain synthetic volume → unusual activity → analyst review | No agency outcome or incident claim |
| Large underlying dataset | Notebook Cell 2 application records, `daily_df`; Cell 10 `subgroup_df` | Smoke test: exact record/daily/office reconciliation and reproducibility | Fixed synthetic generation, not a real loader; events are not independent training samples |
| Show dataframe transformations | Notebook Cells 2–10, `dataframe_stages` previews | Inspect common sample dates, row/column counts, grain and joins | Presentation previews are bounded; full data remain model inputs |
| Explain SPC versus ML | Notebook Cells 3–5; runbook Q&A | Rule OR labels agree with XmR/CUSUM/EWMA flags | Thresholds need operational calibration |
| Define the ML label honestly | Cell 5 `signal_detected` target; Cell 11 comparison | Verify proxy label and accuracy/baseline interpretation | No verified incident labels or sensitivity claims |
| Chronological evaluation | Cells 5–6 and 11–13; `train_df`, `test_df`, `fold_specs` | Smoke test checks ordered dates, separated measurement windows and five folds | Longer histories can overlap; no independent real validation |
| Actual algorithm alternatives | Cells 5, 11: classifier metrics; Cells 6, 12: forecast metrics | Read measured logistic/forest/baseline and forecast comparisons, not an assumed winner | Algorithm choice is dataset-dependent; no fabricated metrics |
| True future-count prediction | Cell 12 `future_forecasts_df`; Cell 6 separate one-day target | Target dates follow origins; five business-day horizons; nonnegative counts | Independent subgroup forecasts need not sum to total forecasts |
| ML anomaly scoring | Cell 13 `anomaly_df`, `iso_fold_df` | Held-out scoring and contamination bounds | Unusual pattern is not confirmed incident |
| Explain possible causes | Cells 10, 14–15: subgroup, lineage, profile, event and hypothesis frames | Reconciliation, five-hop/cycle checks and hypothesis labels | Metadata/events are fixtures; no causal proof |
| Full model lifecycle | Cells 16–20; `forecast_ledger_df`, `assessment_df`, `lifecycle_df`, `registry_exercise_df` | Preserve origins; sufficient-evidence guards; candidate/withdrawal/rollback checks | Fictional lifecycle records do not authorize promotion |
| Save a reusable model | Cell 19; `notebooks/model_reuse.py`; generated bundle `manifest.json` | Hash, metadata, feature contract and reference predictions in model reuse test | Persistent path and dependency compatibility required |
| Infer without retraining | `notebooks/SPC_Model_Scoring.py`; `tests/model_reuse_test.py` | Fresh-process artifact load and prediction parity without training | Verify target-compute load separately |
| Optional MLflow and registry | Cell 19; `tests/mlflow_contract_test.py`; scorer MLflow backend | Simulated API contracts; `tests/mlflow_reuse_integration_test.py` exercises real local MLflow logging and fresh-process URI loading | Actual URI loading, registry permissions and aliases require native evidence |
| Drift and performance monitoring | Cells 21–22; `drift_inputs_df`, `drift_performance_df`, `drift_daily_df` | Dashboard test recomputes error, thresholds, sufficiency and scenario isolation | Illustrative thresholds; fixed replay; no ongoing outcome ingestion |
| Human review and retrain approval | Cells 8, 16–18; runbook Q&A; job setup human loop | Pending queue, evidence guards, fictional disposition, rollback | Real approval and durable audit integration remain required |
| Honest notifications | Cell 18 `outbox_df`; smoke test | Prepared-not-sent statuses and suppressed repeats | No transport, delivery acknowledgments or durable cross-job service |
| Queryable Delta history | Cells 20, 22: `DEMO_TABLES`, `DRIFT_TABLES`; `tests/delta_contract_test.py` | Simulated keyed MERGE and manifest checks | Run actual writes twice, compare counts/keys, inspect Spark schema compatibility |
| Requested native schema | Native dashboard JSON; `scripts/build_dashboard.py`; SQL; job setup | Verify `ml_statistical_process_controls.demo_schema` binding | Catalog/schema permissions and actual destination must match |
| Native dashboard and warehouse | `dashboards/AttainX_SPC_Demo.lvdash.json`; dashboard guide | `tests/dashboard_test.py`: ten SQL datasets, fields and layout; setup helper test | Use serverless notebook compute and serverless SQL warehouse for the default-storage catalog; verify access, import as draft, run ten queries, render four content pages and queue filter |
| Local dashboard fallback | `dashboards/preview.html`, `reviewed-data.json` | Rebuild/inspect from current notebook output | Local companion is not native Databricks rendering |
| Presenter-ready explanations | Deck, `docs/talking-points.md`, runbook timed path and Q&A | Rehearse using recorded current results; verify notes match source | Presenter must not claim unverified platform or customer success |
| Alternate presenter can operate | Runbook ordered steps, fallback and receipt; job setup | Train → save → fresh scoring → inspect preview path | Native run needs authorized workspace identity and compute |
| Architecture and outcomes | README workflow; reference mapping; runbook Q&A | Explain data, rules/models, retained outputs, dashboard and human decision path | Demo methodology and synthetic outcomes only |
| Clear does/does-not boundary | README Boundaries; reference mapping limits; runbook Q&A | Check every claimed result against its evidence level | No live feed, production readiness, operational approval or benefit claim |
| Shared-file privacy | Synthetic narrative throughout shared docs and slides | Review for private provenance, participants and historical assertions | Keep private identifiers and execution receipts outside shared material |

## Acceptance record

The main notebook, scoring workflow, tests and native dashboard definition are inspectable implementations. Current local execution verified the main notebook, fresh-process joblib reuse and a real MLflow 3.16.1 local tracking/URI-load round trip; fitting is forbidden during the reuse checks. These do not establish Databricks or Unity Catalog success. The exact source revision and executed check results belong in the run receipt, not a timeless assertion of passing status. Actual Databricks training/scoring, MLflow URI access, registry actions, Delta writes, warehouse queries and native rendering are **unverified until recorded by a target-workspace run**. A local preview, generated JSON or mocked service test cannot close those gates.

A demonstration can use the clearly labeled local fallback while native access is unavailable. Production readiness additionally requires authorized real data, representative validation, calibrated thresholds, security/access review, durable state/delivery, real human approvals and monitored rollback. Those are outside this repository's synthetic acceptance.

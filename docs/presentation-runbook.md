# Operator and presenter runbook

“This demonstration uses synthetic USCIS-related application volumes. Statistical rules highlight unusual activity, and an analyst reviews the evidence. We separately compare models that estimate future counts. All records and metadata are fabricated.”

## Recorded local run

Dataset `attainx_applications_v3_seed42` has 1,179,830 raw events, 1,260 daily-series observations and 915 labeled windows. The classification holdout uses 615 training rows and 228 test rows; the one-day forecast uses 615 training rows and 225 test rows. Raw event rows are not independent modeling observations.

| Comparison | Recorded synthetic result |
| --- | --- |
| Single-holdout classifier | 75.0% accuracy versus 78.1% always-signal accuracy |
| One-day forecast | MAE 90.01 counts versus trailing-mean 97.26 and last-count 109.87 |
| Five-fold mean classifier accuracy | Logistic regression 81.33%; random forest 77.33% |
| Five-day lifecycle decision | ARIMA selected using earlier origins; final-origin MAE improvement 22.9% over seasonal naive, only 15 observations |

These measurements come from the current executed notebook outputs, not customer history. The classifier candidate is withdrawn; the five-day forecast is advisory. Different evaluation schemes and sample sizes must stay distinct. Recompute this record if source, seed or dependencies change. Local checks also verified fresh-process joblib inference and an actual MLflow 3.16.1 local tracking/URI-load round trip with training forbidden during scoring. This is not target-workspace evidence. There are 22 executable main cells and 19 workflow plus three drift tables. The raw event frames stay in notebook memory.

## Run in this order

1. Record `git status --short` and `git rev-parse HEAD`. Confirm the notebook, helper, scorer, dashboard and presentation come from the intended reviewed revision.
2. Create the Python 3.12 environment and run the README checks. Run `MPLBACKEND=Agg python tests/model_reuse_test.py` for independent artifact reuse. Read test output; commands in a guide are not proof of passing execution.
3. Train and save the bundle using [job setup](job-setup.md), then run the separate scoring entrypoint in a fresh process. Verify prediction parity and model identity. Retain the bundle and manifest with the run receipt.
4. For Databricks, use the repository Git folder with its helper files, serverless notebook compute for the default-storage catalog `ml_statistical_process_controls`, and a persistent artifact directory. Verify serverless availability and permissions. Run all 22 main cells, then the separate scoring notebook. Record local versus actual MLflow loading accurately.
5. Enable the dedicated Delta destination only after checking permissions. Reconcile all runtime manifests and keys, then import the native dashboard and run all ten datasets using a serverless SQL warehouse for this default-storage catalog. Inspect all four content pages and the queue filter. See [dashboard guide](../dashboards/README.md).
6. Rehearse the story below using the actual measured comparisons and artifact identities. If service verification fails, use the local preview and retained output, explicitly labeled as local evidence.

## Seven-minute path

| Time | Show | Explain |
| --- | --- | --- |
| 0:00–0:50 | Cells 1–4 and dataframe previews | Synthetic applications aggregate into daily volumes. Large event count does not equal independent daily training observations. SPC applies statistical rules. |
| 0:50–1:40 | Cells 8–10, 14–15 | A signal becomes a review episode. Office mix, source freshness and events suggest questions; none proves cause. |
| 1:40–2:40 | Cells 5–6, 11–13 | Compare classifier alternatives and baselines. The proxy label differs from future counts; forecast MAE is in count units. Show the measured winner or baseline honestly. |
| 2:40–3:30 | Cell 12 | Predictions after the last observed date are actual future-count forecasts; historical backtests estimate performance. |
| 3:30–4:20 | Cell 19 and separate scorer | A saved artifact can be loaded in another process without training. Show identity, feature contract and prediction parity. |
| 4:20–5:30 | Cells 16–17, 21–22 and dashboard | Separate input drift from worsening error. Investigate first, then evaluate a candidate, seek human approval and retain rollback. |
| 5:30–7:00 | Review queue and verified native dashboard, or labeled local fallback | Analyst owns the decision. Notices are unsent. Identify exactly which platform steps are verified and which remain pending. |

An alternate presenter should read the opening statement, follow these rows, and use the Q&A below. If compute fails, open `dashboards/preview.html`, the deck and the retained run receipt. State that the preview is generated locally and that displayed metrics are from the recorded run. Do not improvise a live success or an agency outcome.

## Technical Q&A

**Is SPC machine learning?** No. XmR, CUSUM and EWMA are statistical rules. ML is a separate experiment or forecasting method.

**What is the classifier's label?** Whether any selected SPC rule fired on a historical measurement window. It is a proxy label, not a verified incident. Learning to copy a rule does not establish business value over running the rule directly.

**What does the forecast predict?** A count on a later business date. The one-day branch uses features from preceding observations; the five-day branch retains an origin and explicit target dates. These outputs are distinct from anomaly scores and review labels.

**How is leakage controlled?** Dates split chronologically across all series, with a 25-business-day gap separating measurement windows; fold preprocessing fits on training rows. Some longer reference histories can still overlap. This is not independent real-world validation, and random row splitting would be inappropriate.

**Why random forest? What alternatives were tested?** It can represent nonlinear relationships, but that is a reason to test it, not assume it wins. Cells 5 and 11 compare classifiers with logistic regression and simple baselines. Cells 6 and 12 compare count forecasts with lag/trailing-mean or seasonal-naive baselines, ARIMA and Holt-Winters. Report the relevant table, time split, metric and sample size; do not merge different experiments.

**What do the raw records represent?** Intake A and Intake B are application-receipt events; Completions A is a separate workflow-completion stream. They are synthetic events, not linked individual cases or adjudication decisions.

**Does more than a million events mean a million training samples?** No. Underlying records reconcile into a much smaller daily-series table. Time-window features and train/test splits reduce the effective modeling sample further. Report each grain separately from the run's counts.

**What does drift mean?** Input distributions changed relative to training. It does not by itself prove worse predictions. Error monitoring needs actual outcomes and sufficient samples. The controlled deterioration exercise deliberately changes evaluation outcomes; it is not measured real performance.

**What triggers retraining?** An investigation and adequate evaluation evidence. An advisory threshold requests review. Training, candidate acceptance and promotion are separate decisions; the demo does not provide real approval.

**Can another process use the model?** Yes, through the independently loadable saved bundle and scorer. Show the verified backend. Target-compute loading and MLflow/registry access require separate evidence; a local test cannot establish them.

**What is deployed?** Only what the run receipt verifies. Source files, tests and a local preview do not prove native Delta, dashboard, scheduling or production operation. No agency data, notification delivery or automated operational decision is included.

## Execution receipt

Keep workspace identifiers and permissions evidence in an appropriate private location. Shared material should contain only sanitized results.

```text
Source revision and configuration:
Dependency/runtime versions:
Synthetic application rows / daily rows / modeling windows:
Reconciliation, calendar and chronological checks:
Measured alternatives, baselines, metrics and sample sizes:
Artifact directory / manifest identity / scoring parity:
Backend verified: local bundle or actual MLflow URI:
Databricks training/scoring run evidence, or unverified:
Delta schema / manifest row counts / duplicate-key checks, or off:
Native import / warehouse / ten queries / four pages / filter, or unverified:
Presenter fallback inspected:
Human-review exercise and rollback limits explained:
Remaining gates and responsible operator:
```

Use runtime results instead of historical numeric claims. Local checks, simulated service contracts, native platform execution and operational acceptance are separate evidence levels.

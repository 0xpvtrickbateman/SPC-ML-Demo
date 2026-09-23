# Presentation rehearsal and evidence record

“This AttainX demonstration uses fictional daily counts and fixture metadata. SPC highlights changes worth investigating. We compare models with simple baselines, inspect possible explanations, and keep decisions reviewable. These results do not measure USCIS performance.”

## Current local evidence

The expanded notebook runs all 20 code cells and produces ten figures. Dataset `attainx_demo_v2_seed42` has 1,260 counts and 915 labeled windows. Dependencies are locked in `requirements-demo-lock.txt`, including statsmodels 0.14.6. The federal-holiday calendar ends the observations on September 8, 2026; future predictions use later business dates.

| Check | Observed result |
| --- | --- |
| Classifier | 75.0% accuracy, 0.917 AUC; always-signal accuracy 78.1% |
| One-day forecast | MAE 8.9896; trailing mean 9.7262; previous count 10.9867 |
| Five-fold accuracy | Logistic regression 81.33%; forest 78.40% |
| Five-day pooled MAE | ARIMA 10.1506; Holt-Winters 10.3774; seasonal naive 12.9222 |
| Forecast coverage | 270 historical predictions, 75 future total/subgroup predictions |
| Lifecycle holdout | ARIMA selected on earlier origins; final-origin improvement 22.9%, 15 observations |
| Review grouping | 479 flagged windows, 30 episodes, latest 15 pending |
| Isolation Forest | 25 of 228 held-out windows flagged; no verified incident labels |
| Daily replay | Two prepared notices, four suppressed repeats, no delivery |
| Local checks | Full execution/invariants plus simulated MLflow 2.x/3.x and 19-table Delta contracts |
| Slides | 22 slides, updated notes, package/layout and rendered visual inspection |
| Target services | Actual Databricks run, MLflow/registry, Delta, schedule and SQL dashboard remain unverified |

Metrics from different evaluation schemes are not interchangeable. Five-day selection uses the first five origins; pooled metrics include the held-out sixth origin. Classification imitates a historical rule label. Earlier 90-day reference histories may overlap training history even though measurement windows are separated. Office forecasts are independent and need not add to the total forecast.

## Rehearsal order

1. Record `git status --short` and `git rev-parse HEAD`. Use the reviewed revision, not an assumed branch tip.
2. In the isolated local Python 3.12 environment, install `requirements-demo-lock.txt` and run the three commands in README. Confirm all pass.
3. In the authorized Databricks Git folder, pull that revision or import the source. Verify all 20 numbered explanatory markdown cells render.
4. Select presentation compute and record its runtime/packages. Install missing dependencies through its allowed process; restart Python when required.
5. Keep both optional destination strings empty initially. MLflow logging still requires experiment write access when installed. Run all 20 cells.
6. Inspect ten figures, forecast dates, baseline comparisons, pending queue, hypotheses, history decisions and outbox statuses. Use actual run numbers if they differ from this local record.
7. Open the MLflow run and all four returned sklearn model URIs. Confirm artifacts can be loaded in the intended environment. Five-day statistical model objects are not logged by this notebook.
8. If a dedicated writable demo schema is ready, enable `OUTPUT_SCHEMA` and rerun. Confirm all 19 tables match `DEMO_TABLES` row counts. Rerun again and verify duplicate keys do not appear. Old incompatible schemas need separate migration; do not assume this cell handles them.
9. Optionally enable a permitted three-part `UC_MODEL_NAME`. Inspect four registered versions, `demo_only` tags and `demo_candidate` aliases. Keep production objects out of scope.
10. Follow [job setup](job-setup.md) for a manual job run. Use [SQL queries](../sql/dashboard_queries.sql) only after table verification. A notebook dashboard is already available in Cell 18.
11. Rehearse the seven-minute path below and retain a private execution receipt. Unavailable services should be described as unverified.

## Seven-minute presentation path

| Time | Show | Say |
| --- | --- | --- |
| 0:00–0:40 | Cells 1–4; slides 1–6 | Fictional counts, explicit calendar, three checks; a signal invites review. |
| 0:40–1:40 | Cells 5–8; slides 7–9 | Classifier loses a simple baseline. Separate one-day forecast improves MAE. Keep the rules. |
| 1:40–2:40 | Cells 9–13; slides 10–13 | Zones and office shifts add context; compare models across time; show five actual future dates. |
| 2:40–3:40 | Cells 14–15; slides 14–15 | Trace five upstream hops, inspect freshness and events, label the explanation as a hypothesis. |
| 3:40–4:50 | Cells 16–17; slides 16–17 | Preserve forecast origins; reject insufficient/stale evidence; rehearse selecting and withdrawing a candidate. |
| 4:50–5:50 | Cell 18; slide 18 | Daily replay prepares one notice per episode and suppresses repeats; no message is sent. |
| 5:50–7:00 | Cells 19–20; slides 19–22 | Show verified artifacts if available. Separate local evidence from live services and agency acceptance. |

[Detailed slide-by-slide talking points](talking-points.md) repeat the deck speaker notes for quick access. The oral-question mapping follows the supplied handoff, not an independently verified RFI. A verified past-client example of declining AI is still needed for question 12. Staffing, ownership, contract rights and authorization need their own evidence.

## Private execution receipt

Keep workspace identifiers and reference exports outside the public repository.

```text
Operator / date:
Source SHA / parameter edits:
Workspace / compute / runtime (private):
Cells 1–20 completed; ten figures inspected:
Actual metrics and baseline comparisons:
MLflow run and four model URIs opened:
Delta destination and 19 row/key checks, or off:
Registered candidate versions/tags/aliases, or off:
Job run / source revision, or not demonstrated:
SQL dashboard checks, or not demonstrated:
Errors, resolutions and final full-rerun result:
Remaining gaps / presenter:
```

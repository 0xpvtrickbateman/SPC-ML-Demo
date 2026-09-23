# USCIS reference-to-demo coverage

**The ten-cell demo does not cover every capability in the supplied USCIS notebook at reduced detail.** It illustrates the central SPC workflow and two ML tasks, but omits several complete analysis and lifecycle components. Describing an omitted component in the deck does not make it executable in the demo.

## Comparison basis

Reviewed the user-supplied `SPC Predictive Modeling.py`, `.ipynb`, and `.dbc` exports on September 23, 2026. All three contain the same 46 cells after normalizing export formatting. Reference cell numbers below are one-based positions in those exports, including markdown cells. Demo numbers refer to its ten executable cells.

The user describes the saved outputs as dummy data. They illustrate the reference's behavior, not verified USCIS results or current deployments. The reference files remain outside this public repository. This review compares source capabilities and supplied outputs; it does not execute the reference against any agency system.

## Capability coverage

| Reference capability | Reference cells | Demo coverage | What is missing or different |
| --- | --- | --- | --- |
| Configuration, feature definitions and data loading | 2–10 | **Replaced**, demo 1–2 | Three fictional count series replace the staging-table loader, form/milestone definitions, incremental form rollout, source filters and detailed loader validation. There is no live data connection. |
| Business-day calendar | 7 | **Simplified**, demo 2 | Monday–Friday dates only. The reference also excludes federal holidays using a calendar source. |
| XmR, CUSUM and EWMA rule signals | 11 | **Represented**, demo 3–4 | Same rule families and an OR label over 25-day windows. Demo returns window flags and limits, rather than all reference signal indexes and richer control-run fields. This is an illustration, not an exact port. |
| Zone severity and pattern checks | 11 | **Omitted** | Reference code defines and calls `compute_zone_tests`, producing current-point and window severity plus breach reasons. The demo has no NORMAL/WATCH/WARNING/CRITICAL grading or zone-pattern checks. |
| Exploratory analysis | 12–13 | **Partial**, demo 4 and 7 | Demo prints overall/per-rule signal rates and three figures. It omits the full per-feature chart grid, histograms, cross-feature correlations and dimension cardinality/volume profiles. |
| Feature engineering | 14–15 | **Simplified**, demo 4 and 6 | Demo includes window statistics, trend and recent values. It omits much of the reference's moving-range detail, limit-distance fields, sigma-band proportions, active subdimension counts and concentration features. |
| Five-day time-series forecasts | 16–18 | **Replaced**, demo 6 | One-day random forest regression replaces ARIMA and Holt-Winters, model comparison, rolling-origin evaluation and five-day forecasts generated beyond the latest observation. The demo evaluates next-day estimates on held-out historical dates; it does not create a new forecast beyond the generated dataset. |
| Forecasts for subgroups | 19–20 | **Omitted** | No service-center-level forecasting or subgroup forecast comparison. Three fictional series are not a hierarchical subgroup analysis. |
| Rule-label classification | 21–22 | **Simplified**, demo 5 | One random forest and one chronological holdout with a gap. No logistic regression comparison, five-fold expanding-window validation, pooled fold analysis or feature-importance display. This demo imitates historical rule labels; it does not forecast future incidents. |
| ML anomaly detection | 23–24 | **Omitted** | No Isolation Forest, adaptive contamination, anomaly score, walk-forward comparison with SPC labels, or service-center anomaly analysis. SPC flags and a classifier that copies them do not substitute for this third modeling approach. |
| Upstream table and column lineage | 25–28 and 33 | **Omitted** | No dependency traversal, column mapping, lineage tree, dynamic upstream-role inference, Delta history/freshness profiling or persisted upstream profiles. `source_name` and `dataset_id` are provenance labels only. The active reference configuration allows five hops. |
| Root-cause hypotheses | 29–31 | **Omitted** | No linkage of signal/anomaly periods to upstream activity or subgroup evidence. No directional service-center evidence, bulk-action/mix-shift classification, hypothesis reason/confidence fields or root-cause persistence. These reference outputs are hypotheses, not proof of causality. |
| Deeper subdimension analysis | 32–34 and 37–38 | **Omitted** | No dimension-value z-score analysis, collinearity checks, high-cardinality filtering, null-equivalent category filtering or comparisons by office, milestone, source system, filing/status/class dimensions. |
| Operational-event correlation | 35–36 | **Omitted** | No event log, nearby-event matching, affected-feature filtering or enrichment of hypotheses using events such as outages or staffing changes. Reference code supports this, although the supplied saved run reports no event correlations. |
| Model scoring and lifecycle recommendations | 39–40 | **Omitted** | Demo compares baselines and recommends direct rules. It does not assign advisory/staging/production recommendations using per-model thresholds. Those reference labels are model-management recommendations, not agency authorization. |
| MLflow and conditional registration | 41–42, plus modeling cells | **Partial**, demo 9 | Demo logs two models when MLflow is available and optionally registers the classifier. It omits multi-model registration widgets, lifecycle caps, aliases/tags, the registry ledger and champion/challenger gating with an improvement threshold. Actual demo workspace writes remain unverified. |
| Historical forecast assessment and retraining advice | 43–44 | **Omitted** | Demo calculates MAE once on a held-out sample. It does not join historical stored forecasts to later actuals, track sMAPE/RMSE/bias/direction accuracy, compare assessments across runs, or produce confidence/trend/retraining recommendations. No insufficient-data, superseded-run or architecture-review decisions. |
| Persistent history | 18, 28, 31, 33, 36, 38, 42 and 44 | **Simplified**, demo 10 | Six optional tables use overwrite. No high-water-mark MERGE/upsert history for forecasts, assessments, lineage, hypotheses or registry decisions. Demo writes require a dedicated writable schema. |
| Repeated signals and human review | 30; discussion in 45 | **Narrow illustration**, demo 8 | Demo groups consecutive flagged windows into pending review episodes and shows the 15 latest. It does not implement the reference's assessment-driven retraining safeguards or a complete notification, suppression or analyst-disposition system. |
| Scheduled inference, retraining and dashboards | Described in 43, 45–46 | **External/not implemented in demo** | The reference describes a separate inference notebook, jobs and SQL alerts. Those operational components are not fully supplied by these three exports. Demo provides a manual job guide and starter SQL only; no tested schedule, inference service, notification delivery or working dashboard is established. |

## Code takes precedence over stale reference prose

- **Zone checks exist in the supplied code.** Cell 11 defines `compute_zone_tests`, calls it for current and window readings, and stores severity/breach fields. Later roadmap markdown still describes zone checks as future work. For this comparison, they count as an implemented reference feature missing from the demo.
- **Lineage is configured for five hops.** Some introductory and closing text still says four. The demo has no traversal at either depth.
- **Training cadence is inconsistent in the reference prose.** It describes both daily and weekly training. No live schedule was verified, so this comparison makes no deployed-cadence claim.
- **Alert consolidation is a separate issue.** Grouping signal periods for analysis is present in the reference, while its notes still list downstream notification consolidation and persistent-alert suppression as future work. The supplied separate inference implementation is unavailable for verification.
- **Saved results are not complete operational proof.** The example event-correlation cell reports no correlations; saved forecast-assessment rows include insufficient-data outcomes. That does not remove the corresponding code capability, but it limits what those outputs demonstrate.

## What can be said accurately

“This demo shows the core SPC rules, a classifier that imitates their historical labels, a separate one-day count-forecast experiment, a pending review list, and optional tracking and table outputs. The USCIS reference contains additional forecasting, anomaly detection, lineage, root-cause and model-lifecycle capabilities that this demo does not implement.”

A compact demo covering every major implemented reference area would need representative examples of zone severity, five-day forecasting and subgroup analysis, Isolation Forest, lineage and root-cause/event investigation, lifecycle/champion-challenger decisions, and historical forecast assessment with retraining safeguards. A complete implementation of those additions is outside this wording-and-comparison change.

# Capability mapping

This synthetic example demonstrates the major SPC and predictive-modeling workflow areas at bounded demo depth. It is not an exact implementation of an agency system. The mapping below refers to repository artifacts, not private source exports or historical customer results.

| Capability | Demonstration evidence | Limit |
| --- | --- | --- |
| Data and business calendar | Cell 2: synthetic application data, daily totals, validation, federal-business-day calendar | Fixed fabricated observations; no authorized live loader or incremental ingestion |
| SPC detection | Cells 3–4: XmR, CUSUM, EWMA and OR label | Statistical rules, not ML; illustrative thresholds |
| Severity and exploration | Cells 9–10: zones, distributions, correlations and office profiles | Reduced dimensions and rule set |
| Feature engineering | Cells 4, 6, 10–11: rolling history, lagged counts, trends and subgroup evidence | Feature choices require new validation on real data |
| Classification | Cells 5, 11: random forest, logistic regression, time folds and simple baselines | Predicts a historical SPC proxy label, not incidents |
| Future-count forecasting | Cells 6, 12: one-day regression; five-day ARIMA, Holt-Winters and seasonal naive | Historical backtests and explicit future dates; modest horizon and fixed settings |
| Subgroup forecasting | Cell 12: office forecasts | Independently fitted forecasts need not sum to total forecasts; observed office counts do reconcile |
| Unsupervised anomalies | Cell 13: Isolation Forest and disagreement evidence | An unusual pattern does not confirm a problem |
| Lineage and freshness | Cell 14: five-hop traversal, cycle protection, mappings and profiles | Fabricated metadata; no live catalog discovery |
| Investigation | Cell 15: rules, office changes, freshness and nearby events | Unverified hypotheses, not causal findings |
| Evaluation and retraining | Cell 16: retained forecast origins, actuals, error metrics and evidence guards | Historical replay; no ongoing actuals feed or automatic retraining |
| Candidate lifecycle | Cell 17: comparisons, withdrawal and rollback exercise | Does not grant human approval or change production aliases |
| Review and notices | Cells 8, 18: pending queue, simulated disposition and replay outbox | No real adjudication or external message delivery |
| Artifacts and scoring | Cell 19 and independent scoring entrypoint | Local artifact reuse is separate from target-compute or registry acceptance |
| Queryable history | Cells 20, 22: table manifests and keyed MERGE | Requires authorized Spark/Delta destination; no general schema migration |
| Drift and error monitoring | Cells 21–22: frozen forecast model, separate input/error metrics and controlled exercise | Illustrative thresholds; exercise outcomes are not observed performance |
| Dashboard and jobs | Native dashboard JSON, SQL and job setup guide | Local query/layout checks do not prove a native warehouse run or schedule |

The correct claim is a runnable synthetic workflow with explicit simulations and measurable local evidence. Real integrations, operational validation, human approval and platform acceptance remain separate gates. See [requirements coverage](requirements-coverage.md) for the exact evidence and status to inspect.

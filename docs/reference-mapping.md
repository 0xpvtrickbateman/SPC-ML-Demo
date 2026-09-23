# USCIS reference-to-demo coverage

The expanded 20-cell notebook now represents **every major workflow area** identified in the supplied reference, using selected methods at demo depth. It is not feature-for-feature production parity. Live integrations, the full agency dimensional model and several detailed algorithms remain different or omitted, as listed below.

The supplied Python, Jupyter and DBC exports contain the same 46 cells after export normalization. Reference numbers include markdown; demo numbers identify executable cells. The user describes saved reference outputs as dummy data. They are not evidence of USCIS outcomes or deployment. Reference exports remain outside this repository and were not executed against agency systems.

| Reference capability | Reference cells | Expanded demo | Remaining difference |
| --- | --- | --- | --- |
| Configuration and loader | 2–10 | 1–2: seeded fictional series, validation and dataset ID | No staging loader, agency forms/milestones, live filters or incremental rollout |
| Business calendar | 7 | 2: weekdays and U.S. federal holidays | No agency-specific exceptions or live calendar source |
| XmR/CUSUM/EWMA | 11 | 3–4: three checks and OR labels | Representative calculations, not identical reference control-run fields or thresholds |
| Zones | 11 | 9: normal/watch/warning/critical; point, consecutive and same-side rules | Compact rule set and window severity; not all reference current/window fields |
| Exploration | 12–13 | 9–10: distributions, correlations, office shares and profiles | Smaller chart set and three parent series |
| Feature engineering | 14–15 | 4, 6, 10–11: window statistics, trends, lagged counts, office z-scores and importance | Not every sigma-band, moving-range, concentration or limit-distance feature |
| Five-day forecasting | 16–18 | 12: ARIMA, Holt-Winters, seasonal baseline, six origins and future dates | Fixed lightweight model settings; no extensive tuning or uncertainty intervals |
| Subgroup forecasting | 19–20 | 12: future forecasts for six office/series groups | Independently fit; not reconciled to totals; no separate subgroup model-selection backtest |
| Classification | 21–22 | 5, 11: forest/logistic, five chronological folds, gap, baselines and permutation importance | Different features/settings; no claim of identical scores or independent histories |
| ML anomaly detection | 23–24 | 13: Isolation Forest, training-derived contamination per fold, score/disagreement | Office anomalies use lagged z-scores rather than separate office Isolation Forests; labels are rule proxies |
| Lineage and freshness | 25–28, 33 | 14: five-hop branching traversal with cycle protection, column mappings, role/freshness profiles | Fixture metadata; no live UC discovery, Delta operation inspection or complete lineage visualization |
| Root-cause hypotheses | 29–31 | 15: episode/rule, office deviation, relevant late source and event evidence | Heuristic unverified hypotheses; no full bulk-action/mix-shift taxonomy or causal proof |
| Subdimensions | 32–34, 37–38 | 10, 15: cardinality/null/placeholder profiles, redundant channel exclusion, office shares/z-scores | No full form/status/milestone dimension suite or complete automatic filtering pipeline |
| Events | 35–36 | 15: fictional event log, nearby-date and affected-series matching | No live event ingestion or verified event causation |
| Lifecycle | 39–40 | 17: baseline/candidate check, 2% improvement criterion, advisory cap, withdrawal and rollback exercise | Compact lifecycle policy; no real reviewer approval or production promotion |
| MLflow/registry | 41–42 | 19: four sklearn model artifacts, metrics, optional four registrations/tags/demo aliases | Five-day statsmodels model objects are not logged; no registry widget suite or live champion alias transaction |
| Assessment/retraining | 43–44 | 16: origin-preserving ledger, MAE/RMSE/sMAPE/bias/direction, previous/current comparison and guards | Retrospective replay of known actuals; no live join to newly arrived actuals or automatic retraining job |
| Persistence | 18, 28, 31, 33, 36, 38, 42, 44 | 20: 19 keyed MERGE tables, distinct forecast origins retained | Actual Spark execution unverified; latest state tables do not retain every prior lifecycle/profile snapshot; no schema migration |
| Signals and oversight | 30, 45 | 8, 18: pending episodes, notice preparation/suppression and separate fictional disposition | No durable delivery service, real disposition or operational action |
| Scheduling and dashboards | 43, 45–46 | 18: executable daily replay and notebook dashboard; manual job/SQL instructions | No verified live schedule, independent inference notebook, SQL dashboard service or external notifications |

Reference code takes precedence over stale prose: zone checks are implemented; lineage allows five hops. The reference prose describes inconsistent training cadences, so no deployed schedule is inferred. Downstream notice suppression is described as planned in the reference; the new local replay demonstrates a selected approach without claiming the reference deployed it.

The correct presentation claim is: “We have a runnable, compact example of the major reference workflow areas, with explicit local simulations for operational components. Detailed agency logic and live platform integrations require separate implementation and verification.”

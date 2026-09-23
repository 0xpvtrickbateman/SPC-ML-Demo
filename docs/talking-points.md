# Presenter talking points

Use the seven-minute route in the presentation runbook. The notes below match the 22-slide deck and cover all 20 notebook cells. Numbers are from the locked local run, not agency results.

## Slide 1: Statistical process control + machine learning

Opening: These are fictional counts and fixture metadata. We demonstrate decisions and execution, not USCIS outcomes. Start with the rules, compare forecasts, then investigate and review. Twenty code cells are grouped across these 22 slides.

## Slide 2: What statistical process control asks

SPC asks whether observed movement deserves investigation. This editable chart is an illustrative example retained from the deck. Current run charts are in the notebook. A signal does not establish cause.

## Slide 3: Three checks, plus severity zones

Cell 3 computes the three original checks. Cell 9 adds critical, warning, watch and normal zones from simple point and run rules. Zone thresholds are demo settings requiring calibration; they are not universal alert policies.

## Slide 4: Detect, forecast, investigate, review

We separate a rule label, an unusualness score, a future workload estimate and an investigation hypothesis. Those outputs answer different questions. None authorizes operational action.

## Slide 5: Fictional counts, explicit calendar

Show configuration then daily_df. The dataset ID is attainx_demo_v2_seed42. Last observed date is September 8, 2026. Installing the locked dependencies reproduces this local environment; target compute still requires a full rehearsal.

## Slide 6: Historical rules become labels

Show signals_df, the three flags, logical OR, and feature columns. The overall rule-positive share is 52.3%. Window boundaries are separated in evaluation, while earlier reference histories may overlap. Explain that distinction before citing metrics.

## Slide 7: Keep the direct rules

The holdout contains 46 true negatives, 4 false positives, 53 false negatives and 125 true positives relative to the rule label. It does not represent confirmed operational incidents. Class mix explains why accuracy needs a baseline.

## Slide 8: Forecast the next business day

Use the actual-versus-predicted chart. Previous-count baseline MAE is 10.99. The federal-holiday calendar changed weekday features, so the updated result is 8.99 rather than the old 8.85. This is separate from the five-day statistical forecasts.

## Slide 9: Visual evidence and review episodes

Use the three initial figures. Consecutive rule-positive dates form episodes. Later cells add a separate fictional reviewer exercise and prepared notice outbox; they do not turn this queue into completed operational decisions.

## Slide 10: Severity, distribution and subgroups

Show zone_df and both exploration figures. The office split is fictional, with a planted North workload shift. Lagged subgroup means and standard deviations support anomaly evidence. Office forecasts later are independent and need not reconcile to the total.

## Slide 11: Compare models across five time folds

Fold averages differ from Cell 5's single holdout and must not be presented as the same experiment. Labels remain direct-rule labels. Logistic regression is useful as a simpler comparison, not a reason to replace rules. Feature importance is predictive association.

## Slide 12: Five-day forecasts and future dates

ARIMA is selected using only the first five origins. The final origin is held out for lifecycle advice; pooled chart metrics include all six origins and are descriptive, not the selection score. Forecast dates exclude U.S. federal holidays. Office predictions are independent, not reconciled.

## Slide 13: A separate unusualness signal

Show scores and rule/model disagreement. Disagreement can focus analyst review. The final model uses a fixed 8% contamination setting and the time folds explore bounded settings. It does not discover an objectively correct anomaly rate.

## Slide 14: Follow upstream dependencies

Show lineage_df, column_lineage_df and upstream_profile_df. Trace the daily-count table to its source export and office lookup branch. A late upstream table provides a possible explanation to investigate, not proof that the count signal is caused by ingestion.

## Slide 15: Make a hypothesis, then verify it

Show the generated operational events and hypotheses_df. Time matching prevents a recent freshness issue from explaining every older episode. Explain alternative causes and the need to inspect the source before recording a real disposition.

## Slide 16: Keep history; assess only mature evidence

Show forecast_ledger_df and assessment_df. A one-observation partial result is explicitly insufficient. Only the latest origin can drive current advice. A very poor worsening result suggests architecture review, while improvement should not trigger retraining. Direction compares target movement with its origin count.

## Slide 17: Candidate review and rollback exercise

Show lifecycle_df and the registry exercise. The candidate was chosen before this final forecast origin. This small example demonstrates the decision record and a rollback sequence, not a verified registry transaction or real human approval.

## Slide 18: Replay daily review and suppress repeats

Show outbox_df: two prepared notices and four suppressed repeats. Show operator_disposition_df and the dashboard figure. Replay deduplication is local to this run; durable delivery idempotency and live scheduling need platform implementation and verification.

## Slide 19: Record models and retain Delta history

Four model artifacts: rule classifier, one-day regressor, logistic classifier and Isolation Forest. Five-day statsmodels fits are not logged as model artifacts; their predictions and assessments are tables. Local stub tests verify API and SQL contracts, not actual service acceptance.

## Slide 20: Questions 1–7: what to show and explain

These questions follow the supplied oral handoff, not an independently verified solicitation. Local execution is demonstrated. A pilot still needs owners, approved real data and prospective success and stop criteria.

## Slide 21: Questions 8–14: what remains to establish

Do not invent a historical engagement for question 12. The current classifier decision is only a current demo example. No agency approval, staffing assignment, contract term or production deployment is established here.

## Slide 22: The useful decisions are visible

Close: The expanded demo now represents the major reference workflow areas. The remaining differences are depth, exact agency algorithms and live integrations. Do not claim identical results, complete production parity or verified operational benefit.


## Optional dataframe walkthrough in Cells 2–10

Use the tables directly in the notebook: “Here are the original counts. We summarize the previous 25 business days, then add the rule flags. From those measurements we branch into classification and forecasting. We also group repeated flags for review and examine office-level detail.”

Follow the same Intake A dates through Cells 2, 4, 5, 6 and 9. Explain that `last_value` belongs to `window_end`, while the observed `daily_count` belongs to `run_date`. Cell 8 explicitly switches to one review episode and its contributing daily rows. Cell 10 shows three parent dates expanding to six office rows whose counts reconcile. A row does not mean the same thing after grouping or splitting.

The earlier slide-by-slide notes still apply; these tables provide an additional live explanation within the existing 20 cells.

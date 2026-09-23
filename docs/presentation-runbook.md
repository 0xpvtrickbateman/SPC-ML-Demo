# Presentation rehearsal and evidence record

## Opening statement

“This is an AttainX demonstration using fictional daily counts. Statistical process control, or SPC, checks whether a work process has changed enough to deserve a closer look. Databricks is the workspace where we run the Python notebook, inspect charts, and record the experiment. Today's synthetic results do not measure USCIS performance or establish production authorization.”

A working demonstration shows that code executes and produces inspectable outputs. An operational pilot would add authorized real data, process owners, analyst dispositions, prospective acceptance criteria, and measured outcomes. Any USCIS result requires its own source and verification.

## Local evidence, September 23, 2026

Reviewed GitHub base: `a23afbd08a4761d3e0ccbdbd3b6bb0afc5026695`. This continuation changes explanations, the deck, and chart-caption spacing; it preserves the modeling calculations. The exact revision to rehearse is the commit containing this runbook, or a subsequently reviewed revision. Record its SHA before presenting.

Local environment: Python 3.12.14, NumPy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1, SciPy 1.18.1, Matplotlib 3.11.2. `requirements-demo-lock.txt` records the full tested package snapshot. The seed is 42. These versions are a local reproduction record, not certification of a Databricks runtime.

| Check | Observed local result |
| --- | --- |
| Complete source notebook smoke test | Passed; ten code cells, each preceded by its numbered explanatory markdown cell |
| Synthetic data and labels | 1,260 counts; 915 labeled windows; 52.3% rule signal rate |
| Classifier holdout | 228 rows, May 1–August 14, 2026; 78.1% rule-positive |
| Classifier accuracy / ROC AUC | 75.0% / 0.917 |
| Constant baselines | Always signal: 78.1%; training-majority class (no signal): 21.9% |
| Confusion matrix | 46 true negatives, 4 false positives, 53 false negatives, 125 true positives, relative to the SPC rule label |
| Forecast holdout | 225 rows, May 4–August 14, 2026 |
| Forecast MAE | 8.8544 counts; trailing-five-day-mean MAE 9.7262; previous-count MAE 10.9867 |
| Review grouping | 479 flagged windows form 30 episodes; the queue retains the 15 with latest signal dates |
| Local MLflow contract test | Passed with simulated 2.x and 3.x APIs; no real experiment or registry writes |
| Visual review | Three notebook charts and all 22 deck slides rendered and inspected; deck package/layout checks passed; original editable chart and embedded workbook preserved |
| Databricks / Delta / Job / dashboard / registry | Not executed or verified in this continuation |

The original handoff recorded 74.1% classifier accuracy and 0.929 AUC. A fresh installation within the allowed dependency ranges produces the results above. Both runs support the same decision: use the direct rules for their own label. The forecast merits additional testing, with no claim of real-agency effectiveness.

The chronological split separates the 25-day measurement windows. Earlier 90-day reference histories can overlap training history. All features use past information, but this test does not demonstrate performance on independent histories, new agencies, or unseen series. XmR calculates limits from the 25-day window; CUSUM and EWMA use the earlier 90 days. Business days here mean Monday through Friday, without an agency holiday calendar.

## Exact local run sequence

From the repository root, with Python 3.12 available:

```bash
git status --short
git rev-parse HEAD
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo-lock.txt
MPLBACKEND=Agg python tests/smoke_test.py
MPLBACKEND=Agg python tests/mlflow_contract_test.py
```

The Agg backend checks execution without opening chart windows. Open the notebook on the presentation compute for the live chart walkthrough. The local tests do not install MLflow or Spark. If another environment already contains MLflow, the notebook attempts experiment logging there; use the isolated environment above for the local check.

## Exact Databricks rehearsal sequence

1. Open the authorized **AttainX** workspace and the repository's Git folder. Confirm the repository URL and branch `main`. Review any local changes before pulling. Pull the intended reviewed revision and record its commit SHA. If Git folders are unavailable, import `notebooks/SPC_ML_Demo.py` and separately record the source SHA.
2. Open the notebook. Confirm the introduction, **Cell 1** through **Cell 10** explanations, and closing demo notes render. These are markdown cells; the following Python cells do the work.
3. Select the presentation compute. Record its runtime or serverless environment version and installed package versions. Install missing dependencies only through the workspace's allowed environment process, then restart Python if required. Do not assume the local package snapshot matches the runtime.
4. Keep `OUTPUT_SCHEMA = ""` and `UC_MODEL_NAME = ""` for the initial rehearsal. This disables Delta writes and model registration. Cell 9 still needs experiment write access when MLflow is installed.
5. Select **Run all**. Confirm all ten code cells complete. If a cell fails, record the cell, error and environment, correct the cause, and rerun from the top. A failed logging cell does not establish a complete target-workspace run.
6. Inspect the three charts in Cell 7. Read circles as signals from the preceding window, not necessarily a limit breach by the plotted point. Read the confusion matrix relative to the rule label. Compare the forecast against its trailing-mean baseline.
7. In Cell 8, inspect the monitoring counts and all 15 pending review rows. Explain that grouping windows does not measure alerts avoided, cases resolved, or staff time saved. No disposition or notification integration exists.
8. In Cell 9, record the MLflow run ID and both returned model URIs. Open the run, verify metrics and both model artifacts, and record whether they are accessible. Simulated local API checks do not satisfy this step.
9. Only when a dedicated writable demo schema is confirmed, set `OUTPUT_SCHEMA = "demo_catalog.spc_demo"` using the actual approved names and rerun. This overwrites six named tables. Verify the destination and row counts: 1,260 source, 915 signals, 228 classifier predictions, 225 forecasts, 15 review rows and one monitoring row for this seed/configuration. Leave writes off if the schema is not ready.
10. Registry registration is a separate optional rehearsal. Leave `UC_MODEL_NAME` empty unless a permitted three-part name is confirmed. If enabled, inspect the returned classifier model version. Registration demonstrates the API path; it does not endorse this classifier for deployment or automatically promote it.
11. If the oral agenda needs a job, follow [job setup](job-setup.md), leave the schedule off, run once, and retain the successful Job run ID and source revision. If it needs a SQL dashboard, substitute the approved schema in [starter queries](../sql/dashboard_queries.sql), run each query, create the necessary visualizations, and verify displayed values. Neither is a prerequisite for the notebook walkthrough.
12. Complete the private execution receipt below. Rehearse the oral sequence with actual outputs. If numbers differ, update the talk track and deck together. Use clearly labeled local screenshots or the deck as a contingency if target execution is unavailable.

Databricks documentation: [Git folder operations](https://docs.databricks.com/aws/en/repos/git-operations-with-repos), [MLflow experiments and permissions](https://docs.databricks.com/aws/en/mlflow/experiments), and [Git sources for Jobs](https://docs.databricks.com/aws/en/jobs/git). Menus and access depend on the workspace configuration.

## Seven-minute talk track

| Time | Show | Explain |
| --- | --- | --- |
| 0:00–0:45 | Introduction and Cell 1 | State the synthetic scope. SPC identifies changes worth investigating. |
| 0:45–1:30 | Cells 2–4 | Fictional counts become three rule flags. Any flag makes the historical window a review candidate. |
| 1:30–2:45 | Cell 5 and confusion matrix | The classifier imitates rules we already know. Its 75.0% accuracy loses to 78.1% always-signal accuracy in this local run. Keep the rules. |
| 2:45–4:00 | Cell 6 and forecast chart | A separate model estimates the next business day's count. MAE is average error in counts. Compare 8.85 with 9.73 on synthetic data only. |
| 4:00–5:00 | Cell 8 | Repeated windows become pending episodes. A person would investigate and record a disposition in a pilot. |
| 5:00–6:00 | Cell 9; optional outputs if verified | Show the actual MLflow evidence. Describe unexecuted optional components as unverified. |
| 6:00–7:00 | Reference and closing slides | The reference documents lineage and retraining fixes; downstream alert grouping remains planned. Further testing requires owners and evidence. |

## Oral evidence still needed

The deck maps to the 14 questions summarized in the supplied handoff. The original RFI and any newer team response packet were not supplied in this continuation, so this is a coverage aid rather than independent verification of solicitation wording.

| Question | Supported demonstration or proposed answer | Evidence / role still needed |
| --- | --- | --- |
| 1. Working capability and lifecycle | Ten-cell local execution and inspectable outputs | Presentation-compute run receipt; demo operator |
| 2. AI versus rules | Reject the unnecessary classifier; separately evaluate the forecast | Mission-specific benefit criteria; process owner |
| 3. Readiness | Identify data, access and measurement needs | Approved definitions, owners, permissions and readiness decision; pilot lead |
| 4. Bounded pilot | Propose limited series, duration and human review | Agreed scope, baseline, cost limit, success/stop criteria; sponsor |
| 5. Governance and monitoring | Local snapshot exposes error and review volume | Monitoring cadence, named owner and retraining decisions; model/process owners |
| 6. AI risk | Rule imitation can inherit poor rule choices; synthetic evaluation is limited | Data-quality, subgroup and false-alert assessment; risk owner |
| 7. Security | Demo reads fictional data only | Authorized boundary, access, retention and incident procedures; security lead |
| 8. Interoperability | Python source and tabular outputs | Actual interfaces, schemas and migration test; platform lead |
| 9. Vendor exit | Source and model artifacts can support transfer | Contractual ownership, export rights and tested replacement path; contract/platform owners |
| 10. Change and rollback | Source versions allow review | Tested release, rollback and model withdrawal procedure; release owner |
| 11. Human oversight | Queue remains pending | Recorded dispositions, override and escalation exercise; analyst lead |
| 12. Prior customer example declining AI | This demo is only a present-day decision | Verified historical engagement, alternatives, decision, outcome and approval to disclose; AttainX leadership |
| 13. ATO integration | Prototype is not authorization evidence | Agency-specific boundary, control evidence and approval process; agency authorization/security owners |
| 14. Staffing and knowledge transfer | Notebook explanations and runbook support teaching | Named staffing plan and receiving-team rerun/rollback exercise; delivery lead |

Roles above are proposed responsibilities, not assigned people. No one was contacted on the user's behalf. Do not invent the Question 12 engagement or claim that the synthetic example satisfies it.

## Private execution receipt template

Keep the completed receipt in an approved private location. Do not add workspace identifiers, private reference exports or agency outputs to this public repository.

```text
Operator / date:
Source commit SHA / any local parameter changes:
Workspace / notebook / compute identifiers (private):
Runtime or environment version / package versions:
Cells 1–10 complete; explanatory markdown rendered:
Charts inspected; live metrics and baseline comparisons:
MLflow run ID / classifier URI / forecast URI:
MLflow artifacts opened successfully:
Delta destination / six table counts, or left off:
Registry model version, or left off:
Job run ID / source revision, or not demonstrated:
Dashboard queries and displayed values checked, or not demonstrated:
Errors, resolution and final full-rerun result:
Presenter / reviewer and remaining blockers:
```

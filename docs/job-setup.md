# Optional Databricks job and demo checklist

The notebook works without a job. Once it succeeds interactively:

1. Open **Jobs & Pipelines → Create job**. Name it `SPC Synthetic Demo`.
2. Add one **Notebook** task named `spc_demo`, pointing to `notebooks/SPC_ML_Demo.py` in the Git folder or the imported workspace notebook.
3. Choose the compute already used successfully for the interactive run. Leave the schedule off for the oral demonstration; use **Run now**.
4. Open the run details and confirm success, cell outputs, the MLflow run ID, and (only if configured) Delta tables in the intended demo schema.
5. After the demo, a daily run may be configured for the synthetic workflow. This project does not automatically retrain weekly or deliver notifications. Describe those as production design steps unless separately implemented and tested.

For the orals, show the notebook first, then a completed job run if available. A pre-recorded successful run and saved screenshots are useful contingency evidence, but represent the actual version and environment they came from.

## Acceptance check

- The source is fictional; no USCIS table or connection appears in the code.
- Rules, `signal_detected`, and both charts display after **Run all**.
- Both ML tasks show chronological holdout metrics and simple baselines.
- The review queue is pending, and no automatic action occurs.
- The experiment records metrics and model artifacts when MLflow is available.
- Optional UC writes and registry are enabled only after permissions are verified.
- The presenter explicitly distinguishes a local synthetic demonstrator from a production deployment.

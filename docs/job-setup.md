# Optional Databricks job and dashboard

The notebook already replays three days and renders a local operational dashboard. This guide prepares a manual platform demonstration; no live schedule or external notification delivery is enabled by the repository.

1. Run all 20 cells successfully on presentation compute first.
2. Create a job named `SPC Demo` with one notebook task for `notebooks/SPC_ML_Demo.py`, using the reviewed source revision and compatible compute/environment. Install the locked dependencies through the supported environment process.
3. Leave its schedule off and select **Run now**. Inspect the completed run, notebook outputs and MLflow artifacts. Record the run ID and source SHA privately.
4. If `OUTPUT_SCHEMA` is enabled, verify the dedicated schema and all 19 table counts/keys, including forecast origins. Repeat the run and check key uniqueness. The notebook uses MERGE, not a complete history replacement.
5. Run the read-only SQL starter queries and configure time-series, scorecard and review-table visualizations. Filter every chart to the intended dataset ID. Verify the dashboard against the notebook tables.
6. A future daily schedule would rerun the entire fixed-data demonstration, including fitting models. It is not new-data ingestion or an inference-only service. Use a live feed, stored model loading, an explicit evaluation cadence and durable review/outbox state before an operational pilot.

Cell 18 prepares notices and suppresses repeats within its replay. It never sends messages. Durable cross-job deduplication, retries, delivery acknowledgments and actual recipient configuration remain integration work. Retraining advice in Cell 16 never automatically retrains or promotes a deployed model.

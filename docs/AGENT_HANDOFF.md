# SPC ML Demo continuation

The current deliverable is the expanded **20-cell notebook**, 22-slide deck with notes, and [talking points](talking-points.md). See [runbook](presentation-runbook.md) for evidence and [reference mapping](reference-mapping.md) for exact simplifications. Those documents supersede the earlier ten-cell handoff.

Repository: `0xpvtrickbateman/SPC-ML-Demo`. Work is on `codex/presentation-validation`, with draft PR #1; verify branch and SHA before pulling. Do not assume `main` contains draft changes.

The demonstration uses fictional counts and fixture metadata. Original USCIS reference exports remain private and are not executed or committed. All major reference workflow areas now have a compact representation. Exact agency loaders/dimensions, live metadata, separately deployed inference, delivery and production registry transactions remain out of scope.

Run the three tests listed in README in the locked Python 3.12 environment. The smoke test executes all models, history/retraining and review exercises. MLflow and Delta checks use simulated APIs. A successful target Databricks run, actual artifacts, tables, jobs and SQL dashboard still need workspace verification.

Preserve the distinction between historical rule imitation, count forecasting and anomaly scoring. The classifier loses the simple accuracy baseline; use direct rules. Forecast advantages are local demo evidence only. Lifecycle and reviewer records are exercises, not real approval. Notifications are never sent.

Speaker notes and the separate talking-points file should remain synchronized. The full notebook still fits a short oral by using the seven-minute path, with detailed cells available for questions. The 14-question mapping is from the supplied handoff; a verified past-client example of declining AI is still needed for question 12.

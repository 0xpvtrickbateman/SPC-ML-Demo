# Reference-to-demo mapping

This project is a small synthetic illustration of the workflow shown in the supplied SPC predictive modeling export. It does not reproduce its datasets, environment, 46-cell implementation, or reported metrics.

| Reference workflow | This demo | Difference to say aloud |
| --- | --- | --- |
| Underlying daily staging counts | Reproducible fictional counts for three series | No external connection or federal records. |
| XmR, CUSUM, EWMA control calculations | Same three categories of SPC rules on 25-day windows | Parameters and generated data are demonstration choices; they have not been calibrated by a program owner. |
| Five-day ARIMA/Holt-Winters forecasts | One-day Random Forest count forecast | A simpler forward-looking example for the orals, with a trailing-mean baseline. This is not a port of the reference forecasters. |
| Random Forest and Logistic Regression rule-label classifiers | One Random Forest rule-label classifier | Explicitly compares with constant predictions and recommends direct rule evaluation if the classifier adds no value. |
| Isolation Forest anomaly detection | Omitted | A third modeling approach would obscure the small demonstration and require careful contamination calibration. |
| Service-center and other subdimension exploration | Three fictional series | These are invented IDs, not operational dimensions. |
| Unity Catalog lineage and upstream event correlation | `source_name` and `dataset_id` provenance fields | This is data provenance, not discovered table/column lineage or proven root cause. |
| MLflow experiment and Unity Catalog model lifecycle | MLflow logging, optional registry registration | Registry is off until catalog permissions and intended model name are verified. No automatic production promotion. |
| Daily inference, weekly retraining, persisted forecast assessment | One runnable notebook and an optional manual Lakeflow Job | Scheduling, historical assessment, retraining triggers and champion/challenger decisions are future work. |
| Alert-fatigue mitigation | Consecutive flagged windows grouped into episodes | One row per episode for review; no notification delivery, suppression policy or full incident workflow. |

The reference's alert-fatigue section is especially relevant: repeated window-level flags should not be turned into dozens of daily emails. The episode table makes that design lesson visible without pretending to implement a production alerting system.

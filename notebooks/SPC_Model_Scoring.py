# Databricks notebook source
# MAGIC %md
# MAGIC # Load a saved forecast and score without training
# MAGIC Run this notebook in a fresh Python session after the main demo saved its bundle.
# MAGIC It loads the fitted random forest, validates the saved feature contract and scores
# MAGIC the saved synthetic application-volume sample. It never runs the training notebook.
# MAGIC Keep `model_reuse.py` beside this notebook in a Databricks Git folder or local checkout.

# COMMAND ----------
# DBTITLE 1,Choose the saved bundle and loading backend
import os
from pathlib import Path
import sys

# Local: set SPC_DEMO_MODEL_DIR and SPC_DEMO_SCORE_BACKEND in the environment.
# Databricks: set the widgets to a persistent bundle path (for example a permitted
# /Volumes/ml_statistical_process_controls/demo_schema/<volume>/saved_forecast).
# Choose mlflow to load the actual logged model URI recorded by the training run.
# An ephemeral /tmp bundle will not survive a different cluster or compute session.
BUNDLE_DIR = os.environ.get("SPC_DEMO_MODEL_DIR", "artifacts/saved_forecast")
BACKEND = os.environ.get("SPC_DEMO_SCORE_BACKEND", "local")
if "dbutils" in globals():
    dbutils.widgets.text("saved_model_directory", BUNDLE_DIR)
    dbutils.widgets.dropdown("saved_model_backend", BACKEND, ["local", "mlflow"])
    BUNDLE_DIR = dbutils.widgets.get("saved_model_directory")
    BACKEND = dbutils.widgets.get("saved_model_backend")

helper_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (helper_dir / "model_reuse.py").exists():
    helper_dir = Path.cwd() / "notebooks"
sys.path.insert(0, str(helper_dir))
from model_reuse import load_forecast_bundle, read_sample_inputs, score_inputs
import json
import numpy as np
import pandas as pd

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load the exact saved artifact
# MAGIC Local mode verifies the model file's SHA-256 before loading. MLflow mode uses
# MAGIC `mlflow.sklearn.load_model` on the recorded model URI, not an in-memory model.
# MAGIC Only load a bundle created by a trusted training run. Use the same package versions.

# COMMAND ----------
model, manifest, artifact_identity = load_forecast_bundle(BUNDLE_DIR, backend=BACKEND)
print("Loaded artifact:", artifact_identity)
print("Feature order:", manifest["feature_columns"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## Score and compare with the saved receipt
# MAGIC These are held-out, already engineered inputs. Their lag/window features use only
# MAGIC earlier days; queue encoding, feature order, dtypes and nonnegative clipping are
# MAGIC preserved. `prepare_inputs` in the helper also encodes newly engineered rows with
# MAGIC the saved queue vocabulary and rejects unknown queues. This is a model reuse check,
# MAGIC not a new accuracy evaluation or a production forecasting service.

# COMMAND ----------
sample_inputs = read_sample_inputs(BUNDLE_DIR, manifest)
loaded_predictions = score_inputs(model, sample_inputs, manifest)
expected_predictions = np.asarray(json.loads((Path(BUNDLE_DIR) / "expected_predictions.json").read_text()))
np.testing.assert_allclose(loaded_predictions, expected_predictions, rtol=1e-12, atol=1e-9)
scoring_results = pd.DataFrame({"expected_count": expected_predictions, "loaded_model_count": loaded_predictions})
if "display" in globals():
    display(scoring_results)
else:
    print(scoring_results.to_string(index=False))
print("Saved-model inference passed; no model training was performed.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Next step
# MAGIC Supply reviewed historical feature rows under the same contract for another batch.
# MAGIC Scoring here creates no endpoint, training job, table write or model promotion.

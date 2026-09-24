"""Portable, inference-only contract for the synthetic one-day count forecast."""
from hashlib import sha256
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn


def validate_inputs(frame, manifest):
    """Restore saved feature order; reject missing, extra, nonnumeric or invalid values."""
    columns = manifest["feature_columns"]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Scoring requires a nonempty pandas DataFrame")
    if not frame.columns.is_unique or set(frame.columns) != set(columns):
        raise ValueError("Scoring columns must exactly match the saved feature schema")
    ordered = frame.loc[:, columns].copy()
    for name, dtype in manifest["feature_dtypes"].items():
        if str(ordered[name].dtype) != dtype:
            raise ValueError(f"Wrong dtype for {name}: expected {dtype}, got {ordered[name].dtype}")
    if not np.isfinite(ordered.to_numpy(dtype=float)).all():
        raise ValueError("Scoring inputs must be finite and nonmissing")
    return ordered


def prepare_inputs(engineered_frame, manifest):
    """Encode an already historical-only feature row using the saved queue vocabulary.

    The caller must calculate lag/window features from dates before the prediction
    date, exactly as the training notebook does. This function never fits anything.
    """
    raw_columns = manifest["raw_feature_columns"]
    if not engineered_frame.columns.is_unique or set(engineered_frame.columns) != set(raw_columns):
        raise ValueError("Raw scoring columns must exactly match the saved feature contract")
    if not engineered_frame["series_id"].isin(manifest["series_categories"]).all():
        raise ValueError("Unknown or missing series_id")
    frame = engineered_frame.loc[:, raw_columns].copy()
    frame["series_id"] = pd.Categorical(frame["series_id"], categories=manifest["series_categories"])
    encoded = pd.get_dummies(frame, columns=["series_id"], dtype=float)
    return validate_inputs(encoded, manifest)


def save_forecast_bundle(model, sample_inputs, directory, model_uri=None):
    """Save an already fitted estimator and an inspectable inference receipt."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_file = directory / "forecast.joblib"
    columns = list(model.feature_names_in_)
    if list(sample_inputs.columns) != columns:
        raise ValueError("Save sample must match fitted feature order")
    categories = [name.removeprefix("series_id_") for name in columns if name.startswith("series_id_")]
    if not categories:
        raise ValueError("Expected the forecast's one-hot series_id features")
    manifest = {
        "format_version": 1,
        "purpose": "synthetic_application_volume_one_day_forecast",
        "model_file": "forecast.joblib",
        "model_uri": model_uri,
        "feature_columns": columns,
        "feature_dtypes": {name: str(sample_inputs[name].dtype) for name in columns},
        "raw_feature_columns": ["series_id"] + [name for name in columns if not name.startswith("series_id_")],
        "series_categories": categories,
        "preprocessing": "Historical features from the preceding 25 business days; pandas one-hot series_id; saved order and dtypes; no scaling",
        "postprocessing": "maximum(prediction, 0)",
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    sample = validate_inputs(sample_inputs, manifest)
    joblib.dump(model, model_file)
    manifest["model_sha256"] = sha256(model_file.read_bytes()).hexdigest()
    sample.to_json(directory / "sample_inputs.json", orient="table", double_precision=15, index=False)
    expected = np.maximum(0, model.predict(sample)).tolist()
    (directory / "expected_predictions.json").write_text(json.dumps(expected) + "\n")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def record_model_uri(directory, model_uri):
    """Attach a successfully logged MLflow URI to an existing bundle; the model digest is unchanged."""
    path = Path(directory) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["model_uri"] = model_uri
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_sample_inputs(directory, manifest):
    """Restore pandas integer widths lost by the JSON table format."""
    frame = pd.read_json(Path(directory) / "sample_inputs.json", orient="table")
    frame = frame.astype(manifest["feature_dtypes"])
    return validate_inputs(frame, manifest)


def load_forecast_bundle(directory, backend="local"):
    """Load only a trusted model artifact; pickle/joblib is executable content."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["format_version"] != 1 or manifest["sklearn_version"] != sklearn.__version__:
        raise ValueError("Bundle version or scikit-learn version mismatch; use the training environment")
    if backend == "local":
        model_file = directory / "forecast.joblib"
        if sha256(model_file.read_bytes()).hexdigest() != manifest["model_sha256"]:
            raise ValueError("Model artifact digest mismatch")
        model = joblib.load(model_file)
        identity = "sha256:" + manifest["model_sha256"]
    elif backend == "mlflow":
        if not manifest.get("model_uri"):
            raise ValueError("No MLflow model URI was recorded for this bundle")
        import mlflow.sklearn
        model = mlflow.sklearn.load_model(manifest["model_uri"])
        identity = manifest["model_uri"]
    else:
        raise ValueError("Choose local or mlflow backend")
    if list(model.feature_names_in_) != manifest["feature_columns"]:
        raise ValueError("Loaded model feature contract does not match the manifest")
    return model, manifest, identity


def score_inputs(model, frame, manifest):
    return np.maximum(0, model.predict(validate_inputs(frame, manifest)))

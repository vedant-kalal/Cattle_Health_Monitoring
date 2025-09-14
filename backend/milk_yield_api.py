import os
import pickle
import numpy as np
import joblib
from collections.abc import Mapping, Iterable
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Path to the pickle model (can be overridden via env)
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'milk_yeild_prediction', 'model', 'milk_yield_pipeline.pkl')
MODEL_PATH = os.getenv("MILK_MODEL_PATH", DEFAULT_MODEL_PATH)

# List of features required by the model

FEATURES = [
    "milk_yesterday",
    "milk_7day_avg",
    "days_in_milk",
    "parity",
    "breed",
    "feed_concentrate_qty",
    "green_fodder_qty",
    "milking_frequency",
    "health_flag",
    "ambient_temp",
    "humidity",
    "season"
]

class MilkYieldInput(BaseModel):
    milk_yesterday: float
    milk_7day_avg: float
    days_in_milk: int
    parity: int
    breed: str
    feed_concentrate_qty: float
    green_fodder_qty: float
    milking_frequency: int
    health_flag: str
    ambient_temp: float
    humidity: float
    season: str

router = APIRouter()

# Lazy-loaded model to avoid startup crash if file is missing
model = None

def _find_predict_object(obj, _seen=None):
    """Recursively search for an object with a 'predict' method in common containers."""
    if _seen is None:
        _seen = set()
    if obj is None:
        return None
    try:
        oid = id(obj)
        if oid in _seen:
            return None
        _seen.add(oid)
    except Exception:
        pass

    # Direct hit
    if hasattr(obj, 'predict'):
        return obj

    # Bytes payload that might be a pickled object
    if isinstance(obj, (bytes, bytearray, memoryview)):
        try:
            inner = pickle.loads(bytes(obj))
            return _find_predict_object(inner, _seen)
        except Exception:
            return None

    # Numpy scalar/record wrappers
    if isinstance(obj, np.void):
        try:
            return _find_predict_object(obj.item(), _seen)
        except Exception:
            pass

    # Dict container
    if isinstance(obj, Mapping):
        # Prefer common keys first
        for k in ("pipeline", "model", "estimator", "regressor"):
            if k in obj:
                found = _find_predict_object(obj[k], _seen)
                if found is not None:
                    return found
        # Fallback: scan all values
        for v in obj.values():
            found = _find_predict_object(v, _seen)
            if found is not None:
                return found

    # Iterable containers
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            found = _find_predict_object(v, _seen)
            if found is not None:
                return found
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray)):
        try:
            for v in obj:
                found = _find_predict_object(v, _seen)
                if found is not None:
                    return found
        except Exception:
            pass

    # Numpy arrays (including object arrays and 0-d wrappers)
    if isinstance(obj, np.ndarray):
        try:
            if obj.ndim == 0:
                return _find_predict_object(obj.item(), _seen)
            # Flatten and traverse; prioritize object dtype
            arr = obj.ravel()
            # If elements are bytes, try unpickling each
            for v in arr:
                if isinstance(v, (bytes, bytearray, memoryview)):
                    try:
                        inner = pickle.loads(bytes(v))
                        found = _find_predict_object(inner, _seen)
                        if found is not None:
                            return found
                    except Exception:
                        continue
                else:
                    found = _find_predict_object(v, _seen)
                    if found is not None:
                        return found
        except Exception:
            pass

    return None

def _ensure_model_loaded():
    global model
    if model is None:
        if not os.path.isfile(MODEL_PATH):
            raise HTTPException(status_code=503, detail=f"Milk model not found at '{MODEL_PATH}'. Set MILK_MODEL_PATH or place the .pkl file correctly.")
        try:
            # Prefer joblib.load for joblib-produced artifacts; fallback to pickle
            loaded = None
            try:
                loaded = joblib.load(MODEL_PATH)
            except Exception:
                with open(MODEL_PATH, 'rb') as f:
                    loaded = pickle.load(f)
            # Recursively look for an estimator with 'predict'
            candidate = _find_predict_object(loaded)
            if candidate is None:
                dtype = getattr(loaded, 'dtype', None)
                shape = getattr(loaded, 'shape', None)
                raise RuntimeError(f"Loaded object has no 'predict' method in known containers (expected sklearn Pipeline). Got type={type(loaded).__name__}, dtype={dtype}, shape={shape}")
            model = candidate
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load milk model: {e}")

@router.get("/milk_status")
def milk_status():
    """Return diagnostic info for the milk-yield pipeline without mutating state."""
    exists = os.path.isfile(MODEL_PATH)
    loaded = model is not None
    versions = {}
    try:
        import sklearn as _sk
        versions["sklearn"] = getattr(_sk, "__version__", None)
    except Exception:
        versions["sklearn"] = None
    for lib in ("xgboost", "lightgbm", "catboost"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", None)
        except Exception:
            versions[lib] = None

    preproc_cols = {"numeric": None, "categorical": None}
    try:
        if loaded and hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            pre = model.named_steps["preprocessor"]
            # After fit, ColumnTransformer has transformers_ with (name, transformer, columns)
            numeric_cols = []
            categorical_cols = []
            if hasattr(pre, "transformers_"):
                for name, transformer, cols in pre.transformers_:
                    if name == "num":
                        numeric_cols = list(cols) if cols is not None else []
                    elif name == "cat":
                        categorical_cols = list(cols) if cols is not None else []
            preproc_cols = {"numeric": numeric_cols, "categorical": categorical_cols}
    except Exception:
        # Best-effort only
        pass

    return {
        "model_path": MODEL_PATH,
        "exists": exists,
        "loaded": loaded,
        "model_type": type(model).__name__ if loaded else None,
        "versions": versions,
        "preprocessor_expected_columns": preproc_cols,
        "api_features": FEATURES,
    }

@router.post("/predict_milk_yield/")
def predict_milk_yield(input: MilkYieldInput):
    try:
        _ensure_model_loaded()
        import pandas as pd
        # Convert input to DataFrame (avoid deprecated .dict())
        input_dict = input.model_dump()
        df = pd.DataFrame([input_dict])
        # Add engineered features (same as feature_engineering.py)
        df["heat_index"] = (df["ambient_temp"] * df["humidity"]) / 100.0
        if df["milk_7day_avg"].iloc[0] != 0:
            df["milk_ratio"] = df["milk_yesterday"] / (df["milk_7day_avg"] + 1e-5)
        else:
            df["milk_ratio"] = 0.0
        # Ensure all expected columns from the trained preprocessor exist
        try:
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                pre = model.named_steps["preprocessor"]
                if hasattr(pre, "transformers_"):
                    numeric_cols = []
                    categorical_cols = []
                    for name, transformer, cols in pre.transformers_:
                        if name == "num":
                            numeric_cols = list(cols) if cols is not None else []
                        elif name == "cat":
                            categorical_cols = list(cols) if cols is not None else []
                    for col in numeric_cols:
                        if col not in df.columns:
                            df[col] = 0.0
                    for col in categorical_cols:
                        if col not in df.columns:
                            df[col] = "unknown"
        except Exception:
            # Best effort only
            pass
        # Predict
        pred = model.predict(df)[0]
        return {"predicted_milk_yield": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
Fall detection XGBoost retraining script with MLflow tracking.

Reads labelled data from Postgres (pre-computed features + patient confirmation),
trains a new XGBoost classifier, logs everything to MLflow, and saves the .pkl
where the inference server's hot-swap endpoint can load it.

Usage (from _6G_Integration_v2_mqtt/ directory):

    # Minimal: retrain v0 model on whatever is in Postgres
    python -m retrain.retrain

    # Specify base model + dataset tag
    python -m retrain.retrain --model-version v3 --dataset our_data

    # Register in MLflow Model Registry after training
    python -m retrain.retrain --model-version v3 --register

    # Dry run: just print dataset stats, don't train
    python -m retrain.retrain --dry-run

Environment variables (read from .env):
    DATABASE_URL          — Postgres or SQLite (default: sqlite:///./caregiver.db)
    MLFLOW_TRACKING_URI   — MLflow server URL (default: ./mlruns — local file store)

MLflow UI:
    mlflow ui --backend-store-uri ./mlruns   (local mode)
    or just open MLFLOW_TRACKING_URI in browser (remote mode)
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Load .env before any config imports
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGISTERED_MODEL_NAME = "fall-detection-xgboost"  # MLflow Model Registry name

DEFAULT_XGB_PARAMS = {
    "n_estimators":   200,
    "max_depth":      4,
    "learning_rate":  0.1,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "scale_pos_weight": 1,   # updated dynamically to balance classes
    "eval_metric":    "logloss",
    "random_state":   42,
}


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train(
    model_version:     str = "v0",
    dataset_tag:       str = "our_data",
    output_dir:        Optional[str] = None,
    register:          bool = False,
    min_samples:       int = 10,
    threshold:         float = 0.5,
    dry_run:           bool = False,
) -> Optional[str]:
    """
    Load data from Postgres, train XGBoost, log to MLflow.

    Returns the MLflow run_id on success, None if skipped.

    Parameters
    ----------
    model_version   : which feature set to use ('v0', 'v3', ...). This determines
                      which feature columns are passed to XGBoost. If the DB contains
                      mixed model versions, only rows matching this version are used.
    dataset_tag     : MLflow tag value for 'dataset' key ('our_data' or 'charite')
    output_dir      : where to save the .pkl (default: model/model_{version}_retrained/)
    register        : if True, register best model in MLflow Model Registry
    min_samples     : abort training if fewer than this many labelled samples exist
    threshold       : classification threshold written into model metadata
    dry_run         : print dataset stats only, skip training
    """
    import numpy as np
    import mlflow
    import mlflow.xgboost
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )
    import xgboost as xgb

    from retrain.data_pipeline import load_labelled_dataset
    from app.core.model_config import MODEL_CONFIGS
    from app.core.model_registry import get_model_name

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info(f"Loading labelled dataset  model_version={model_version}  dataset={dataset_tag}")
    df, meta = load_labelled_dataset(model_version=model_version)

    n_total = len(df)
    n_pos   = meta["n_positive"]
    n_neg   = meta["n_negative"]

    print(f"\nDataset summary:")
    print(f"  Total labelled rows : {n_total}")
    print(f"  Positive (fall=yes) : {n_pos}")
    print(f"  Negative            : {n_neg}")
    print(f"  Features            : {len(meta['feature_names'])}")
    print(f"  Model versions      : {meta['model_versions']}")

    if n_total < min_samples:
        logger.warning(
            f"Only {n_total} labelled rows — need at least {min_samples}. "
            f"Run seed_test_data.py first, or wait for more production data."
        )
        return None

    if dry_run:
        print("\n(dry-run — skipping training)")
        return None

    # ── 2. Build feature matrix ────────────────────────────────────────────
    # Use the feature order from model_config so the trained model's feature
    # list matches what the inference server sends at prediction time.
    try:
        model_type    = get_model_name(model_version)
        config        = MODEL_CONFIGS[model_type]
        ordered_features = config.acc_feature_names + config.baro_feature_names
        # Keep only features that exist in the DB (subset if model_version mixed)
        available = set(meta["feature_names"])
        ordered_features = [f for f in ordered_features if f in available]
    except Exception:
        # Fallback: use sorted feature names from DB
        ordered_features = sorted(meta["feature_names"])

    X = df[ordered_features].values.astype(float)
    y = df["label"].values.astype(int)

    logger.info(f"Feature matrix: {X.shape}  positives: {int(y.sum())} / {len(y)}")

    # ── 3. Train / test split ──────────────────────────────────────────────
    # Stratified split to preserve class ratio in both sets.
    # If too few samples, skip split and train on full set.
    if n_total >= 20 and n_pos >= 2 and n_neg >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        logger.warning("Too few samples for a held-out test set — training on full dataset.")
        X_train, y_train = X, y
        X_test, y_test   = X, y   # report train metrics only

    # ── 4. Configure XGBoost ──────────────────────────────────────────────
    n_neg_train = int((y_train == 0).sum())
    n_pos_train = int((y_train == 1).sum())
    spw = max(1, n_neg_train // max(1, n_pos_train))  # balance minority class

    params = {**DEFAULT_XGB_PARAMS, "scale_pos_weight": spw}

    # ── 5. MLflow run ─────────────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"fall-detection-{model_version}")

    run_name = f"retrain_{model_version}_{dataset_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # ── Log parameters ────────────────────────────────────────────────
        mlflow.log_params({
            "model_version":    model_version,
            "n_features":       len(ordered_features),
            "n_train":          len(X_train),
            "n_test":           len(X_test),
            "n_positive_train": n_pos_train,
            "n_negative_train": n_neg_train,
            "scale_pos_weight": spw,
            "threshold":        threshold,
            **{k: v for k, v in params.items()
               if k not in ("eval_metric", "random_state")},
        })

        # ── Log dataset tags ──────────────────────────────────────────────
        mlflow.set_tags({
            "dataset":          dataset_tag,
            "model_version":    model_version,
            "feature_set":      "_".join(ordered_features[:3]) + "...",
            "window_seconds":   "9",
            "sample_rate_hz":   "50",
        })

        # ── Train ─────────────────────────────────────────────────────────
        logger.info("Training XGBoost...")
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Evaluate ──────────────────────────────────────────────────────
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = float("nan")   # only one class in test set

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, int(y_test.sum()))

        mlflow.log_metrics({
            "accuracy":  round(acc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "auc":       round(float(auc), 4) if auc == auc else 0.0,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

        print(f"\nEvaluation results (threshold={threshold}):")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}   ← most important for fall detection")
        print(f"  F1        : {f1:.4f}")
        print(f"  AUC       : {auc:.4f}")
        print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

        # ── Log model ─────────────────────────────────────────────────────
        # Also save feature_names in the signature so the model is self-describing
        from mlflow.models.signature import infer_signature
        import pandas as pd
        signature = infer_signature(
            pd.DataFrame(X_train, columns=ordered_features),
            model.predict(X_train),
        )

        mlflow.xgboost.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME if register else None,
            signature=signature,
            input_example=pd.DataFrame(X_train[:2], columns=ordered_features),
        )

        # ── Save .pkl locally ─────────────────────────────────────────────
        # The inference server's hot-swap endpoint loads models from the
        # model/ directory by file path. Save here so /model/switch can pick it up.
        import joblib
        out_dir = output_dir or f"model/model_{model_version}_retrained"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pkl_path = f"{out_dir}/model_{model_version}_retrained.pkl"
        joblib.dump(model, pkl_path)
        logger.info(f"Model saved to {pkl_path}")

        # Log feature list as artifact for reproducibility
        feat_path = f"{out_dir}/feature_names.txt"
        with open(feat_path, "w") as fh:
            fh.write("\n".join(ordered_features))
        mlflow.log_artifact(feat_path, artifact_uri="model")
        mlflow.log_artifact(pkl_path, artifact_uri="model")

        print(f"\nMLflow run_id : {run_id}")
        print(f"Tracking URI  : {tracking_uri}")
        print(f"Model saved   : {pkl_path}")
        print(f"\nTo view results:")
        print(f"  mlflow ui --backend-store-uri {tracking_uri}")

    return run_id


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain fall detection XGBoost model from Postgres data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-version", default="v0",
                        help="Base model version to retrain (default: v0)")
    parser.add_argument("--dataset", default="our_data",
                        choices=["our_data", "charite"],
                        help="MLflow dataset tag (default: our_data)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save .pkl (default: model/model_{version}_retrained/)")
    parser.add_argument("--register", action="store_true",
                        help="Register model in MLflow Model Registry after training")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Minimum labelled samples required (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print dataset stats only — skip training")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args()
    train(
        model_version = args.model_version,
        dataset_tag   = args.dataset,
        output_dir    = args.output_dir,
        register      = args.register,
        min_samples   = args.min_samples,
        threshold     = args.threshold,
        dry_run       = args.dry_run,
    )

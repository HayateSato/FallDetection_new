"""
Retraining data pipeline — reads labelled inference data from Postgres.

Why Postgres, not InfluxDB?
  The inference server already stores pre-computed features in `feature_snapshot`
  (one row per feature per /predict call).  `fall_history` carries the ground-truth
  label: patient_confirmed = 'yes' | 'no' | 'not_answered'.

  This means retraining is a single SQL JOIN — no re-running of the feature
  extraction pipeline, no InfluxDB access, no raw sensor data needed.

Retraining JOIN:
  inference_log  (il)  ← one row per /predict
       ↓ il.id = fs.inference_id
  feature_snapshot (fs) ← N rows per inference (one per feature)
       ↓ il.observation_id = fh.observation_id
  fall_history    (fh)  ← one row per confirmed MQTT alert

Label assignment:
  label = 1  if  fall_detected=True  AND  patient_confirmed='yes'
  label = 0  if  fall_detected=False  OR  patient_confirmed='no'
  excluded   if  fall_detected=True   AND  patient_confirmed='not_answered'
             (ambiguous — model said fall but patient never responded)
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_labelled_dataset(
    model_version:  Optional[str] = None,
    include_all_negatives: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Load a labelled training DataFrame from Postgres.

    Parameters
    ----------
    model_version : optional filter — only load inferences from this model version
                    (e.g. 'v3'). None = all versions.
    include_all_negatives : if True, include all non-fall inferences as negative
                            examples even if there is no fall_history row.
                            Recommended: True — real-world data has many more
                            negatives than positives.

    Returns
    -------
    df : DataFrame with columns
           [feature_name_1 ... feature_name_N, 'label',
            'patient_id', 'model_version', 'detection_time', 'observation_id']
         label = 1 (fall confirmed) or 0 (no fall / patient said no)

    meta : dict with keys 'n_positive', 'n_negative', 'feature_names', 'model_versions'
    """
    from shared_db.db.session import SessionLocal

    db = SessionLocal()
    try:
        return _build_dataset(db, model_version, include_all_negatives)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_dataset(
    db,
    model_version: Optional[str],
    include_all_negatives: bool,
) -> Tuple[pd.DataFrame, dict]:

    # ── Step 1: fetch the long-format join result ─────────────────────────
    version_filter = "AND il.model_version = :mv" if model_version else ""
    sql = text(f"""
        SELECT
            il.id              AS inference_id,
            il.observation_id,
            il.patient_id,
            il.model_version,
            il.fall_detected,
            il.detection_time,
            fs.feature_name,
            fs.feature_value,
            fh.patient_confirmed,
            fh.needs_help
        FROM inference_log il
        JOIN feature_snapshot fs ON fs.inference_id = il.id
        LEFT JOIN fall_history fh ON fh.observation_id = il.observation_id
        WHERE 1=1 {version_filter}
        ORDER BY il.id
    """)

    params = {"mv": model_version} if model_version else {}
    rows = db.execute(sql, params).fetchall()

    if not rows:
        logger.warning("No data found in Postgres — run seed_test_data.py first.")
        return pd.DataFrame(), {"n_positive": 0, "n_negative": 0,
                                 "feature_names": [], "model_versions": []}

    long_df = pd.DataFrame(rows, columns=[
        "inference_id", "observation_id", "patient_id", "model_version",
        "fall_detected", "detection_time", "feature_name", "feature_value",
        "patient_confirmed", "needs_help",
    ])

    # ── Step 2: pivot long → wide  (one row per inference, one col per feature) ──
    meta_cols  = ["inference_id", "observation_id", "patient_id", "model_version",
                  "fall_detected", "detection_time", "patient_confirmed", "needs_help"]
    meta_df    = long_df[meta_cols].drop_duplicates("inference_id").set_index("inference_id")

    feature_df = (
        long_df[["inference_id", "feature_name", "feature_value"]]
        .pivot_table(index="inference_id", columns="feature_name",
                     values="feature_value", aggfunc="first")
    )

    wide_df = meta_df.join(feature_df, how="left").reset_index()

    # ── Step 3: assign labels ──────────────────────────────────────────────
    # Positive: model said fall AND patient confirmed it
    pos_mask = (wide_df["fall_detected"] == True) & (wide_df["patient_confirmed"] == "yes")

    # Negative: model said no fall, OR patient explicitly denied
    neg_mask = (
        (wide_df["fall_detected"] == False) |
        (wide_df["patient_confirmed"] == "no")
    )
    # Negatives without a fall_history row (model said no fall, no MQTT alert)
    if include_all_negatives:
        no_alert_mask = (wide_df["fall_detected"] == False) & (wide_df["patient_confirmed"].isna())
        neg_mask = neg_mask | no_alert_mask

    # Ambiguous: model said fall but patient never answered → exclude from training
    ambiguous_mask = (
        (wide_df["fall_detected"] == True) &
        (wide_df["patient_confirmed"].fillna("not_answered") == "not_answered")
    )

    labelled = wide_df[pos_mask | neg_mask].copy()
    labelled["label"] = labelled.apply(
        lambda r: 1 if (r["fall_detected"] and r["patient_confirmed"] == "yes") else 0,
        axis=1,
    )

    n_ambiguous = int(ambiguous_mask.sum())
    if n_ambiguous > 0:
        logger.info(f"Excluded {n_ambiguous} ambiguous rows (fall detected but patient never responded)")

    # ── Step 4: identify feature columns ──────────────────────────────────
    non_feature_cols = {"inference_id", "observation_id", "patient_id", "model_version",
                        "fall_detected", "detection_time", "patient_confirmed",
                        "needs_help", "label"}
    feature_names = sorted([c for c in labelled.columns if c not in non_feature_cols])

    # ── Step 5: clean up output ────────────────────────────────────────────
    output_cols = feature_names + ["label", "patient_id", "model_version",
                                    "detection_time", "observation_id"]
    result_df = labelled[output_cols].reset_index(drop=True)

    # Fill missing features with 0 (can happen if different model versions coexist)
    result_df[feature_names] = result_df[feature_names].fillna(0.0)

    n_positive = int((result_df["label"] == 1).sum())
    n_negative = int((result_df["label"] == 0).sum())
    model_versions = sorted(result_df["model_version"].dropna().unique().tolist())

    logger.info(
        f"Dataset loaded: {len(result_df)} rows  "
        f"(+{n_positive} falls / -{n_negative} non-falls)  "
        f"features={len(feature_names)}  model_versions={model_versions}"
    )

    meta = {
        "n_positive":    n_positive,
        "n_negative":    n_negative,
        "feature_names": feature_names,
        "model_versions": model_versions,
    }
    return result_df, meta

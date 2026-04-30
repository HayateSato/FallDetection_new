Item 11.10 is describing the **operational workflow** for promoting a newly retrained model to production — not a coding task. Here's exactly what it means and what you'd actually do:

---

**The scenario:** You run `retrain.py --register` and a new model version (e.g., version 7) appears in the registry. It has no alias yet — it's not Production, not Staging.

**The "Staging → evaluate → Production" steps:**

1. Open MLflow UI:
    
    `mlflow ui --backend-store-uri ./mlruns
    # → http://localhost:5000`
    
2. Go to **Models → fall-detection-xgboost → Version 7**. Click "Add alias" → type `Staging` → save.
3. **Evaluate:** Look at the logged metrics for that version (F1, AUC, precision, recall) and compare against the current Production version. You can see both side-by-side in the MLflow UI.
4. If the new version looks good, click that version again → "Edit alias" → add `Production` (or change the Production alias from version 6 to version 7).
5. Now call the hot-swap:
    
    `.\switch_model.ps1 -Stage Production`
    
    The inference server loads the new model from MinIO without restarting.
    

---

**Why "no code needed":** All the infrastructure is already done — 11.9 registers the model, 11.11 wires `/model/switch` to load by alias. Step 11.10 is just the human decision point in the middle.

**Why the checkbox is still open:** It's not a one-time implementation task — it's a repeating operational step you'll do every time you retrain. You can mark it done now if you want; it just means you understand and accept this as the manual promotion workflow rather than automating it with e.g. an auto-promotion script.
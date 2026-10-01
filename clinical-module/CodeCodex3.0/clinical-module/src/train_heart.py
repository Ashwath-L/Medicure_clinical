import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Import local preprocessing helper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import create_heart_groups, HEART_FEATURE_COLS, HEART_TARGET_COL

def evaluate_model_cv(model_name, pipeline, X, y, groups, cv):
    """
    Evaluates a model pipeline using StratifiedGroupKFold cross-validation.
    Ensures group isolation between train and validation folds.
    """
    print(f"\n--- Evaluating {model_name} with StratifiedGroupKFold (5 Folds) ---")
    
    oof_preds = np.zeros(len(y))
    oof_probs = np.zeros(len(y))
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])
        
        # Verify strict group separation
        overlap = train_groups.intersection(val_groups)
        assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping group IDs in fold {fold}: {overlap}"
        
        # Fit model pipeline on training fold
        pipeline.fit(X_train, y_train)
        
        # Predict on validation fold
        val_preds = pipeline.predict(X_val)
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        
        oof_preds[val_idx] = val_preds
        oof_probs[val_idx] = val_probs
        
        fold_acc = accuracy_score(y_val, val_preds)
        fold_prec = precision_score(y_val, val_preds, zero_division=0)
        fold_rec = recall_score(y_val, val_preds, zero_division=0)
        fold_f1 = f1_score(y_val, val_preds, zero_division=0)
        fold_auc = roc_auc_score(y_val, val_probs)
        
        fold_metrics.append({
            'fold': fold,
            'accuracy': float(fold_acc),
            'precision': float(fold_prec),
            'recall': float(fold_rec),
            'f1': float(fold_f1),
            'roc_auc': float(fold_auc)
        })
        print(f"Fold {fold}: Acc={fold_acc:.4f} | Prec={fold_prec:.4f} | Rec={fold_rec:.4f} | F1={fold_f1:.4f} | AUC={fold_auc:.4f}")
        
    # Overall Out-Of-Fold (OOF) Metrics
    oof_acc = accuracy_score(y, oof_preds)
    oof_prec = precision_score(y, oof_preds, zero_division=0)
    oof_rec = recall_score(y, oof_preds, zero_division=0)
    oof_f1 = f1_score(y, oof_preds, zero_division=0)
    oof_auc = roc_auc_score(y, oof_probs)
    cm = confusion_matrix(y, oof_preds)
    
    summary = {
        'model_name': model_name,
        'oof_accuracy': float(oof_acc),
        'oof_precision': float(oof_prec),
        'oof_recall': float(oof_rec),
        'oof_f1': float(oof_f1),
        'oof_roc_auc': float(oof_auc),
        'confusion_matrix': cm.tolist(),
        'fold_metrics': fold_metrics,
        'mean_fold_accuracy': float(np.mean([m['accuracy'] for m in fold_metrics])),
        'std_fold_accuracy': float(np.std([m['accuracy'] for m in fold_metrics])),
        'mean_fold_roc_auc': float(np.mean([m['roc_auc'] for m in fold_metrics])),
        'std_fold_roc_auc': float(np.std([m['roc_auc'] for m in fold_metrics]))
    }
    
    print(f"\n==> {model_name} Overall OOF Performance:")
    print(f"    Accuracy  : {oof_acc:.4f}")
    print(f"    Precision : {oof_prec:.4f}")
    print(f"    Recall    : {oof_rec:.4f}")
    print(f"    F1 Score  : {oof_f1:.4f}")
    print(f"    ROC-AUC   : {oof_auc:.4f}")
    print(f"    Confusion Matrix:\n{cm}")
    
    return summary

def main():
    print("=" * 70)
    print("MEDSCOPE AI - PHASE 2: HEART DISEASE ML PIPELINE TRAINING")
    print("=" * 70)
    
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded heart.csv dataset with shape: {df.shape}")
    assert len(df) == 1025, f"Expected 1025 rows, found {len(df)}"
    
    X = df[HEART_FEATURE_COLS]
    y = df[HEART_TARGET_COL]
    
    # 2. Create Duplicate Groups (Features ONLY, target strictly excluded)
    groups = create_heart_groups(df, feature_cols=HEART_FEATURE_COLS)
    num_unique_groups = groups.nunique()
    print(f"Extracted {num_unique_groups} unique duplicate feature groups across all {len(df)} rows.")
    
    # 3. Setup StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 4. Define Candidate Pipelines
    pipelines = {
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
        ])
    }
    
    # 5. Evaluate Candidate Models
    results = {}
    for name, pipeline in pipelines.items():
        results[name] = evaluate_model_cv(name, pipeline, X, y, groups, cv)
        
    # 6. Model Selection (emphasizing balanced metrics: ROC-AUC and F1)
    lr_score = results["Logistic Regression"]["oof_roc_auc"] + results["Logistic Regression"]["oof_f1"]
    rf_score = results["Random Forest"]["oof_roc_auc"] + results["Random Forest"]["oof_f1"]
    
    if rf_score >= lr_score:
        selected_model_name = "Random Forest"
    else:
        selected_model_name = "Logistic Regression"
        
    print("\n" + "=" * 70)
    print(f"MODEL SELECTION SUMMARY")
    print("=" * 70)
    print(f"Logistic Regression Score (AUC + F1): {lr_score:.4f} (AUC: {results['Logistic Regression']['oof_roc_auc']:.4f}, F1: {results['Logistic Regression']['oof_f1']:.4f})")
    print(f"Random Forest Score (AUC + F1)      : {rf_score:.4f} (AUC: {results['Random Forest']['oof_roc_auc']:.4f}, F1: {results['Random Forest']['oof_f1']:.4f})")
    print(f"SELECTED BEST MODEL: {selected_model_name}")
    print("=" * 70)
    
    # 7. Train Final Model on Full Dataset (All 1,025 rows)
    final_pipeline = pipelines[selected_model_name]
    final_pipeline.fit(X, y)
    
    # 8. Save Pipeline & Model
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, 'heart_pipeline.pkl')
    joblib.dump(final_pipeline, model_save_path)
    print(f"\nFinal full-dataset trained pipeline saved to: {model_save_path}")
    
    # Save model evaluation report
    reports_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    heart_report = {
        "dataset": "heart.csv",
        "total_rows": len(df),
        "unique_feature_groups": num_unique_groups,
        "evaluation_strategy": "StratifiedGroupKFold (5 splits, grouped by feature hash)",
        "selected_model": selected_model_name,
        "model_results": results
    }
    
    report_save_path = os.path.join(reports_dir, 'heart_model_results.json')
    with open(report_save_path, 'w') as f:
        json.dump(heart_report, f, indent=2)
    print(f"Detailed Heart evaluation report saved to: {report_save_path}")
    
    # Also save to master model_results.json
    master_report_path = os.path.join(reports_dir, 'model_results.json')
    master_data = {}
    if os.path.exists(master_report_path):
        try:
            with open(master_report_path, 'r') as f:
                master_data = json.load(f)
        except Exception:
            master_data = {}
    master_data['heart'] = heart_report
    with open(master_report_path, 'w') as f:
        json.dump(master_data, f, indent=2)
    print(f"Master evaluation report updated at: {master_report_path}")

if __name__ == '__main__':
    main()

# src/models/train_and_evaluate.py
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import json

# optional xgboost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# optional mlflow
try:
    import mlflow, mlflow.sklearn
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

from src.preprocess import build_preprocessor

TARGET = 'Premium Amount'
ID_COL = 'id'

def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0) + 1e-9
    y_pred = np.maximum(y_pred, 0) + 1e-9
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2)))

def evaluate_metrics(y_true, y_pred):
    return {
        'rmsle': rmsle(y_true, y_pred),
        'rmse' : float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae'  : float(mean_absolute_error(y_true, y_pred)),
        'r2'   : float(r2_score(y_true, y_pred))
    }

def _to_serial(obj):
    import numpy as _np
    if isinstance(obj, (_np.integer, _np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def main(train_path, test_path, sample_path, out_path, quick=False, use_mlflow=False):
    warnings.filterwarnings("ignore")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    # normalize column names
    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]
    sample.columns = [c.strip() for c in sample.columns]

    # drop rows without target
    train = train.dropna(subset=[TARGET])

    # keep ids
    test_ids = test[ID_COL] if ID_COL in test.columns else pd.Series(range(len(test)), name=ID_COL)

    X = train.drop(columns=[TARGET, ID_COL], errors='ignore')
    y = train[TARGET]

    # Build preprocessor
    preproc = build_preprocessor()
    X_proc = preproc.fit_transform(X)

    # train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(X_proc, y, test_size=0.2, random_state=42)

    # ==============================
    # Define candidate models + grids
    # ==============================
    models = []

    # Linear Regression (no grid)
    models.append(('linreg', LinearRegression(), {}))

    # Ridge
    models.append(('ridge', Ridge(random_state=42), {'alpha': [0.1,1.0,10.0,50.0]}))

    # RandomForest
    if quick:
        rf_grid = {'n_estimators': [50], 'max_depth': [5]}
    else:
        rf_grid = {'n_estimators': [150, 300], 'max_depth': [None, 16]}
    models.append(('rf', RandomForestRegressor(random_state=42, n_jobs=-1), rf_grid))

# XGBoost (optional)
    if HAS_XGB:
        if quick:
            xgb_grid = {'n_estimators': [50], 'max_depth': [3], 'learning_rate': [0.1]}
        else:
            xgb_grid = {'n_estimators': [200, 400], 'max_depth': [4, 8], 'learning_rate': [0.05, 0.1]}
        models.append(('xgb', XGBRegressor(random_state=42, n_jobs=4, verbosity=0), xgb_grid))


    # ==============================
    # Train & Evaluate
    # ==============================
    best_model = None
    best_name = None
    best_score = float('inf')
    results = {}

    if use_mlflow and HAS_MLFLOW:
        mlflow.set_experiment("SmartPremium")

    for name, estimator, grid in models:
        print(f"\nTraining {name} ...")

        if grid:
            gs = GridSearchCV(estimator, grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
            gs.fit(X_tr, y_tr)
            est = gs.best_estimator_
            best_params = gs.best_params_
        else:
            est = estimator.fit(X_tr, y_tr)
            best_params = {}

        pred_val = est.predict(X_val)
        metrics = evaluate_metrics(y_val, pred_val)
        print(f"{name} metrics: {metrics}, best_params: {best_params}")

        results[name] = {
            'metrics': metrics,
            'params': {k: (v.item() if hasattr(v, 'item') else v) for k,v in best_params.items()}
        }

        if use_mlflow and HAS_MLFLOW:
            with mlflow.start_run(run_name=name):
                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(est, artifact_path='model')

        if metrics['rmse'] < best_score:
            best_score = metrics['rmse']
            best_model = est
            best_name = name


    print(f"\nBest model: {best_name} (RMSE={best_score:.3f})")

    # Save pipeline
    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'preprocessor': preproc, 'model': best_model}, out_dir / 'smartpremium_pipeline.joblib')
    print("Saved pipeline to outputs/smartpremium_pipeline.joblib")

    # Final evaluation on val
    final_metrics = {}
    if best_model is not None:
        pred_val_best = best_model.predict(X_val)
        final_metrics = evaluate_metrics(y_val, pred_val_best)
        print("\nFinal evaluation on validation set (best model):")
        for k,v in final_metrics.items():
            print(f"  {k}: {v:.4f}")

    # Predict on test
    X_test = test.drop(columns=[ID_COL], errors='ignore')
    X_test_proc = preproc.transform(X_test)
    preds = best_model.predict(X_test_proc)
    preds = np.maximum(preds, 0.0)

    submission = pd.DataFrame({ID_COL: test_ids, 'Premium Amount': preds})
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")

    # Save metrics summary
    metrics_summary = {
        'results': results,
        'best_model': best_name,
        'best_score_rmse': best_score,
        'final_metrics': final_metrics
    }
    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf8') as f:
        json.dump(metrics_summary, f, indent=2, default=_to_serial)
    print(f"Saved metrics summary to {metrics_path}")

    return metrics_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='src/data/train.csv')
    parser.add_argument('--test', default='src/data/test.csv')
    parser.add_argument('--sample', default='src/data/sample_submission.csv')
    parser.add_argument('--out', default='outputs/submission.csv')
    parser.add_argument('--quick', action='store_true', help='quick mode (faster, smaller grid)')
    parser.add_argument('--mlflow', action='store_true', help='log experiments to MLflow (if installed)')
    args = parser.parse_args()
    print("Running with:")
    print(vars(args))
    main(args.train, args.test, args.sample, args.out, quick=args.quick, use_mlflow=args.mlflow)

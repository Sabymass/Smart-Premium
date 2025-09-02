# run.py
import argparse
import sys
from pathlib import Path
import importlib
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Run EDA -> Train & Evaluate pipeline (SmartPremium)")
    parser.add_argument('--train', default='data/train.csv')
    parser.add_argument('--test', default='data/test.csv')
    parser.add_argument('--sample', default='data/sample_submission.csv')
    parser.add_argument('--out', default='outputs/submission.csv')
    parser.add_argument('--quick', action='store_true', help='Quick mode (smaller grids)')
    parser.add_argument('--mlflow', action='store_true', help='Enable MLflow logging (if installed)')
    args = parser.parse_args()

    # ✅ ensure project root is on sys.path
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # ---------------- Step 1: EDA ----------------
    print("🔹 Step 1: Running EDA (13 artifacts)...")
    try:
        eda_mod = importlib.import_module('src.SmartPremium_EDA')
        if hasattr(eda_mod, "run_eda"):
            eda_mod.run_eda(
                train_path=args.train,
                out_dir='outputs/eda',
                target_plots=13
            )
        print(f"✅ EDA finished. Artifacts -> {Path('outputs/eda').resolve()}")
    except ModuleNotFoundError:
        print("⚠️ No EDA module found (skipping)...")
    print()

    # ---------------- Step 2: Training ----------------
    print("🔹 Step 2: Training & Evaluating models (LinearRegression, Ridge, RandomForest, XGBoost(if available))...")
    train_mod = importlib.import_module('src.train_and_evaluate')
    results = train_mod.main(
        train_path=args.train,
        test_path=args.test,
        sample_path=args.sample,
        out_path=args.out,
        quick=args.quick,
        use_mlflow=args.mlflow
    )

    # ---------------- Metrics summary ----------------
    if isinstance(results, dict) and 'final_metrics' in results:
        fm = results['final_metrics']
        print("\n📌 Final evaluation (best model on hold-out validation):")
        print(f"  RMSE : {fm['rmse']:.4f}")
        print(f"  R2   : {fm['r2']:.4f}")
        print(f"  MAE  : {fm['mae']:.4f}")
        print(f"  RMSLE: {fm.get('rmsle', 'N/A'):.5f}")
    else:
        print("\n⚠️ Final metrics not returned by train_and_evaluate.")

    # ---------------- Wrap up ----------------
    print("\nAll steps completed.")
    print(f"Artifacts : {Path('outputs/eda').resolve()}")
    print(f"Pipeline  : {Path('outputs/smartpremium_pipeline.joblib').resolve() if Path('outputs/smartpremium_pipeline.joblib').exists() else 'NOT FOUND'}")
    print(f"Submission: {Path(args.out).resolve()}")

    # ---------------- Launch Streamlit ----------------
    print("\n🚀 Launching Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/streamlit_app.py"])


if __name__ == "__main__":
    main()

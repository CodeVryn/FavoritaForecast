import argparse
import glob
import json
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import polars as pl

# --- Constants ---
DATASET_DIR = Path("../dataset")
RESULTS_DIR = Path("../results")
PREPARED_DIR = DATASET_DIR / "prepared"
PREPARED_TRAIN_PATTERN = PREPARED_DIR / "train_batch_*.parquet"
PREPARED_TEST_PATTERN = PREPARED_DIR / "test_batch_*.parquet"
MODEL_DIR = Path("../models")
EVAL_RESULTS_PATH = RESULTS_DIR / "results_catboost_eval_data.json"
TEST_RESULTS_PATH = RESULTS_DIR / "results_catboost_test_data.json"
FORECAST_EVAL_PATH = DATASET_DIR / "forecast_CatBoost_v1_eval.parquet"
FORECAST_TEST_PATH = DATASET_DIR / "forecast_CatBoost_v1_test.parquet"
TEST_PATH = DATASET_DIR / "test.csv"
DATE_FROM = "2017-01-01"
SUBMISSION_PATH = DATASET_DIR / "submission_CatBoost_v1.csv"
TRAIN_DIR = Path("../tensorboard_logs")

HORIZON = 16
RANDOM_SEED = 42


def load_prepared_data(pattern: str | Path | None = None) -> pd.DataFrame:
    """Load all prepared parquet files (Polars for speed, then to pandas)."""
    pattern = pattern or PREPARED_TRAIN_PATTERN
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError("No prepared data found. Run prepare_features.py first")

    # Загрузим данные, и отфильтруем по дате
    df = (
        pl.scan_parquet(files)
        .filter(pl.col("date") >= pl.lit(DATE_FROM).str.strptime(pl.Date))
        .collect()
    )
    return df.to_pandas()


def nwrmsle_flat(
    y_true: np.ndarray, y_pred: np.ndarray, perishable: np.ndarray
) -> float:
    """NWRMSLE on unit_sales; weights: perishable*0.25 + 1."""
    w = perishable.astype(float) * 0.25 + 1.0
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    sq = (log_true - log_pred) ** 2
    return float(np.sqrt((sq * w).sum() / w.sum()))


def invert_log1p(y: np.ndarray) -> np.ndarray:
    return np.expm1(y)


MODEL_PATH = MODEL_DIR / "catboost_favorita.cbm"


def _json_serial(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run_eval():
    """Load train, gridsearch, save model and results."""
    print("Loading prepared train data...")
    train_df = load_prepared_data(PREPARED_TRAIN_PATTERN)
    print("dtype of train_df:", train_df.dtypes)

    # Train/test split
    print(f"Loaded {len(train_df):,} rows, splitting into train and test...")
    test_start = train_df["date"].max() - timedelta(days=HORIZON)
    mask = train_df["date"] < test_start
    Xtrain = train_df.loc[mask].drop(columns=["unit_sales", "unit_sales_log1p"])
    Xtest = train_df.loc[~mask].drop(columns=["unit_sales", "unit_sales_log1p"])
    ytrain = train_df.loc[mask, ["series_id", "date", "unit_sales", "unit_sales_log1p"]]
    ytest = train_df.loc[~mask, ["series_id", "date", "unit_sales", "unit_sales_log1p"]]

    del train_df

    # Уберем из test ряды, которых нет в train
    print("Removing test rows with series_ids not in train...")
    train_series_ids = set(Xtrain["series_id"].unique())
    Xtest = Xtest[Xtest["series_id"].isin(train_series_ids)].copy()
    ytest = ytest[ytest["series_id"].isin(train_series_ids)].copy()

    print("Calculating categorical features...")
    catboost_cols = list(sorted(set(Xtest.columns) - {"series_id", "date"}))
    CAT_FEATURES = [
        c
        for c in catboost_cols
        if str(Xtest[c].dtype) in ("category", "object", "bool")
    ]

    print("Creating train/test pools...")
    train_pool = Pool(
        data=Xtrain[catboost_cols],
        label=ytrain["unit_sales_log1p"],
        cat_features=CAT_FEATURES,
    )
    valid_pool = Pool(
        data=Xtest[catboost_cols],
        label=ytest["unit_sales_log1p"],
        cat_features=CAT_FEATURES,
    )

    task_type = "GPU"
    param_grid = [
        {"depth": 6, "learning_rate": 0.05, "n_estimators": 1000, "subsample": 0.8},
        {"depth": 6, "learning_rate": 0.1, "n_estimators": 800, "subsample": 0.8},
        {"depth": 7, "learning_rate": 0.05, "n_estimators": 800, "subsample": 0.8},
    ]

    print("Running a grid search...")

    best_model = None
    best_params = None
    best_nwrmsle = float("inf")
    perishable_arr = Xtest["perishable"].fillna(False).astype(int).to_numpy()
    y_test_lin = ytest["unit_sales"].to_numpy(dtype=float)

    for i, params in enumerate(param_grid):
        print(f"\n--- Config {i + 1}/{len(param_grid)}: {params} ---")
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            n_estimators=params["n_estimators"],
            subsample=params["subsample"],
            random_seed=RANDOM_SEED,
            task_type=task_type,
            max_bin=128,
            grow_policy="Lossguide",
            bootstrap_type="Bernoulli",
            early_stopping_rounds=150,
            train_dir=TRAIN_DIR / f"config_{i + 1}",
            verbose=200,
        )
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
        )
        pred_log = model.predict(Xtest[catboost_cols])
        pred_lin = invert_log1p(pred_log)
        nwrmsle = nwrmsle_flat(
            y_true=y_test_lin,
            y_pred=pred_lin,
            perishable=perishable_arr,
        )
        print(f"Validation NWRMSLE: {nwrmsle:.4f}")
        if nwrmsle < best_nwrmsle:
            best_nwrmsle = nwrmsle
            best_model = model
            best_params = params.copy()

    print(f"\nBest params: {best_params}")
    print(f"Best validation NWRMSLE: {best_nwrmsle:.4f}")

    pred_log = best_model.predict(Xtest[catboost_cols])
    pred_lin = invert_log1p(pred_log)
    perishable = Xtest["perishable"].fillna(False).astype(int).to_numpy()
    nwrmsle = nwrmsle_flat(
        y_true=ytest["unit_sales"].to_numpy(dtype=float),
        y_pred=pred_lin,
        perishable=perishable,
    )
    print(f"Test NWRMSLE: {nwrmsle:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model.save_model(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}")

    forecast_df = Xtest[["series_id", "date"]].copy()
    forecast_df["pred_log1p"] = pred_log
    forecast_df["pred"] = pred_lin
    forecast_df.to_parquet(FORECAST_EVAL_PATH, compression="zstd", index=False)
    print(f"Forecast saved to {FORECAST_EVAL_PATH}")

    results = {
        "nwrmsle": nwrmsle,
        "best_params": best_params,
        "model_path": str(MODEL_PATH),
        "forecast_path": str(FORECAST_EVAL_PATH),
    }
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {EVAL_RESULTS_PATH}")


def run_test():
    """Load model, load test batches, predict, save forecast and metrics."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run with --mode=eval first."
        )

    print("Loading model...")
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))

    print("Loading prepared test data...")
    test_df = load_prepared_data(pattern=PREPARED_TEST_PATTERN)
    print(f"Loaded {len(test_df):,} test rows, preparing data...")

    Xtest = test_df.drop(columns=["unit_sales", "unit_sales_log1p"], errors="ignore")

    print("Calculating categorical features...")
    catboost_cols = list(sorted(set(Xtest.columns) - {"series_id", "date"}))

    print("Making predictions for the test data...")
    pred_log = model.predict(Xtest[catboost_cols])
    pred_lin = invert_log1p(pred_log)

    forecast_df = Xtest[["series_id", "date", "store_nbr", "item_nbr"]].copy()
    forecast_df["pred_log1p"] = pred_log
    forecast_df["pred"] = pred_lin

    # Save parquet forecast (internal)
    forecast_df[["series_id", "date", "pred_log1p", "pred"]].to_parquet(
        FORECAST_TEST_PATH, compression="zstd", index=False
    )
    print(f"Forecast saved to {FORECAST_TEST_PATH}")

    # Match each prediction to id from test.csv, save submission (id, unit_sales)
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"{TEST_PATH} not found. Required to match predictions to submission ids."
        )
    test_ids = pd.read_csv(
        TEST_PATH,
        usecols=["id", "date", "store_nbr", "item_nbr"],
        dtype={"id": np.int64, "store_nbr": np.int16, "item_nbr": np.int32},
        parse_dates=["date"],
    )
    # Normalize dates for reliable merge
    pred_map = forecast_df[["date", "store_nbr", "item_nbr", "pred"]].copy()
    pred_map["date"] = pd.to_datetime(pred_map["date"]).dt.normalize()
    test_ids["date"] = pd.to_datetime(test_ids["date"]).dt.normalize()
    merged = test_ids.merge(pred_map, on=["date", "store_nbr", "item_nbr"], how="left")
    merged["unit_sales"] = (
        merged["pred"].fillna(0).round().clip(0, None).astype(np.int64)
    )
    submission = merged[["id", "unit_sales"]]
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH} (id, unit_sales)")

    results = {
        "mode": "test",
        "forecast_path": str(FORECAST_TEST_PATH),
        "submission_path": str(SUBMISSION_PATH),
        "n_test_rows": len(test_df),
    }

    with open(TEST_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {TEST_RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train or run inference with CatBoost")
    parser.add_argument(
        "--mode",
        choices=["eval", "test"],
        default="eval",
        help="eval: train and save model; test: load model and predict on test batches",
    )
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent)

    if args.mode == "eval":
        run_eval()
    else:
        run_test()


if __name__ == "__main__":
    main()

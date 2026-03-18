import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# --- Constants ---
DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
ITEMS_PATH = DATA_DIR / "items.csv"
STORES_PATH = DATA_DIR / "stores.csv"
HOLIDAYS_PATH = DATA_DIR / "holidays_events.csv"
TRANSACTIONS_PATH = DATA_DIR / "transactions.csv"
PREPARED_DIR = DATA_DIR / "prepared"
PREPARED_TRAIN_PATTERN = PREPARED_DIR / "train_batch_*.parquet"
PREPARED_TEST_PATTERN = PREPARED_DIR / "test_batch_*.parquet"
PREPARED_META_PATH = PREPARED_DIR / "meta.json"

DATE_FROM = "2016-01-01"
HORIZON = 16
BATCH_ITEMS = 500  # items per batch for lag computation
LAG_WINDOWS = [16, 21, 30, 60, 120]


def load_metadata():
    """Load holidays, stores, items, transactions (small tables)."""
    print("Loading metadata...")
    holidays_events_df = pd.read_csv(
        HOLIDAYS_PATH,
        parse_dates=["date"],
        dtype={
            "type": "category",
            "locale": "category",
            "locale_name": "category",
            "description": "category",
            "transferred": bool,
        },
    )

    stores_df = pd.read_csv(
        STORES_PATH,
        dtype={
            "store_nbr": np.int8,
            "city": "category",
            "state": "category",
            "type": "category",
            "cluster": np.int8,
        },
    )

    items_df = pd.read_csv(
        ITEMS_PATH,
        dtype={
            "item_nbr": np.int32,
            "family": "category",
            "class": np.int16,
            "perishable": bool,
        },
    )

    transactions_df = pd.read_csv(
        TRANSACTIONS_PATH,
        parse_dates=["date"],
        dtype={"store_nbr": np.int16, "transactions": np.int32},
    )

    return holidays_events_df, stores_df, items_df, transactions_df


def prepare_transactions(
    transactions_df: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Fill missing transactions per store, handle store 52."""
    transactions_df_from_2016 = transactions_df[
        transactions_df["date"] >= DATE_FROM
    ].copy()

    for store in transactions_df["store_nbr"].unique():
        mask = transactions_df_from_2016["store_nbr"] == store
        store_dates = transactions_df_from_2016.loc[mask, "date"].sort_values().unique()
        missing_dates = set(dates) - set(store_dates)

        if len(missing_dates) == 0:
            continue

        new_rows = pd.DataFrame(
            [
                {"store_nbr": store, "date": d, "transactions": np.nan}
                for d in sorted(missing_dates)
            ]
        )
        new_rows["date"] = pd.to_datetime(new_rows["date"])
        transactions_df_from_2016 = pd.concat(
            [transactions_df_from_2016, new_rows], ignore_index=True
        )
        transactions_df_from_2016 = transactions_df_from_2016.sort_values(
            by=["store_nbr", "date"]
        )

        mask = transactions_df_from_2016["store_nbr"] == store
        transactions_df_from_2016.loc[mask, "transactions"] = (
            transactions_df_from_2016.loc[mask, "transactions"]
            .interpolate(method="linear")
            .bfill()
            .ffill()
        )

    return transactions_df_from_2016.reset_index(drop=True)


def build_store_tx_lags(transactions_df_from_2016: pd.DataFrame) -> pd.DataFrame:
    """Precompute transactions_lag_* per (date, store_nbr)."""
    store_tx = (
        transactions_df_from_2016[["date", "store_nbr", "transactions"]]
        .drop_duplicates()
        .sort_values(["store_nbr", "date"])
        .reset_index(drop=True)
    )
    g_tx = store_tx.groupby("store_nbr", sort=False)
    for lag in LAG_WINDOWS:
        store_tx[f"transactions_lag_{lag}"] = (
            g_tx["transactions"].shift(lag).astype(np.float32)
        )
    return store_tx.drop(columns=["transactions"])


def build_holiday_tables(
    holidays_events_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build national, regional, local holiday/event flags."""
    h = holidays_events_df.copy()
    h["transferred"] = h["transferred"].fillna(False).astype(bool)
    h = h.loc[~h["transferred"]].copy()
    holiday_like = {"Holiday", "Additional", "Bridge", "Transfer"}

    def aggregate_flags(group_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        grouped = group_df.groupby(group_cols, observed=True)
        return (
            grouped["type"]
            .agg(
                is_holiday=lambda s: int(s.isin(holiday_like).any()),
                is_event=lambda s: int((s == "Event").any()),
                is_work_day=lambda s: int((s == "Work Day").any()),
            )
            .reset_index()
        )

    national = aggregate_flags(h.loc[h["locale"] == "National"], ["date"]).rename(
        columns={
            "is_holiday": "holiday_national",
            "is_event": "event_national",
            "is_work_day": "workday_national",
        }
    )
    regional = (
        h.loc[h["locale"] == "Regional"]
        .rename(columns={"locale_name": "state"})
        .pipe(lambda d: aggregate_flags(d, ["date", "state"]))
        .rename(
            columns={
                "is_holiday": "holiday_regional",
                "is_event": "event_regional",
                "is_work_day": "workday_regional",
            }
        )
    )
    local = (
        h.loc[h["locale"] == "Local"]
        .rename(columns={"locale_name": "city"})
        .pipe(lambda d: aggregate_flags(d, ["date", "city"]))
        .rename(
            columns={
                "is_holiday": "holiday_local",
                "is_event": "event_local",
                "is_work_day": "workday_local",
            }
        )
    )
    return national, regional, local


def get_item_batches(train_scan: pl.LazyFrame) -> list[list[int]]:
    """Get list of item_nbr batches (each batch has BATCH_ITEMS items)."""
    all_items = (
        train_scan.filter(pl.col("date") >= pl.lit(DATE_FROM).str.strptime(pl.Date))
        .select("item_nbr")
        .unique()
        .collect(engine="streaming")
    )
    items_list = all_items["item_nbr"].to_list()
    batches = [
        items_list[i : i + BATCH_ITEMS] for i in range(0, len(items_list), BATCH_ITEMS)
    ]
    return batches


def process_batch(
    batch_items: list[int],
    train_scan: pl.LazyFrame,
    store_tx_lags: pd.DataFrame,
    national: pd.DataFrame,
    regional: pd.DataFrame,
    local: pd.DataFrame,
    stores_df: pd.DataFrame,
    items_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    test_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one batch of items, add all features, return (train_df, test_df)."""
    train_batch = (
        train_scan.filter(pl.col("date") >= pl.lit(DATE_FROM).str.strptime(pl.Date))
        .filter(pl.col("item_nbr").is_in(batch_items))
        .collect(engine="streaming")
    )
    df = train_batch.to_pandas().drop(columns=["id"], errors="ignore")

    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    df["unit_sales_log1p"] = df["unit_sales"].apply(
        lambda x: np.log1p(x) if x > 0 else 0.0
    )
    df["series_id"] = df["item_nbr"].astype(str) + "_" + df["store_nbr"].astype(str)

    # Fill missing
    all_dates = pd.DataFrame({"date": pd.to_datetime(dates)})
    series_meta = (
        df.groupby("series_id", sort=False)
        .agg(
            first_date=("date", "min"),
            store_nbr=("store_nbr", "first"),
            item_nbr=("item_nbr", "first"),
        )
        .reset_index()
    )
    full_grid = (
        series_meta.merge(all_dates, how="cross")
        .query("date >= first_date")
        .drop(columns="first_date")
    )
    result = full_grid.merge(
        df.drop(columns=["store_nbr", "item_nbr"]),
        on=["series_id", "date"],
        how="left",
    )
    result["unit_sales"] = (
        result["unit_sales"].fillna(0.0).astype(df["unit_sales"].dtype)
    )
    result["unit_sales_log1p"] = (
        result["unit_sales_log1p"].fillna(0.0).astype(df["unit_sales_log1p"].dtype)
    )
    result["onpromotion"] = result["onpromotion"].fillna(False).astype(bool)
    df = result

    # Holidays
    df = df.merge(stores_df[["store_nbr", "city", "state"]], on="store_nbr", how="left")
    df = df.merge(national, on="date", how="left")
    df = df.merge(regional, on=["date", "state"], how="left")
    df = df.merge(local, on=["date", "city"], how="left")
    flag_cols = [
        "holiday_national",
        "event_national",
        "workday_national",
        "holiday_regional",
        "event_regional",
        "workday_regional",
        "holiday_local",
        "event_local",
        "workday_local",
    ]
    df[flag_cols] = df[flag_cols].fillna(0).astype(np.int8)
    df["is_holiday"] = (
        df[["holiday_national", "holiday_regional", "holiday_local"]].max(axis=1)
    ).astype(np.int8)
    df = df.drop(columns=["city", "state"])

    # Calendar
    df["day_of_week"] = df["date"].dt.dayofweek.astype(np.int8)
    df["day_of_month"] = df["date"].dt.day.astype(np.int8)
    df["day_of_year"] = df["date"].dt.dayofyear.astype(np.int16)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int16)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["quarter"] = df["date"].dt.quarter.astype(np.int8)
    df["year"] = df["date"].dt.year.astype(np.int16)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(np.int8)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(np.int8)
    df["is_payday_1"] = (df["day_of_month"] == 1).astype(np.int8)
    df["is_payday_15"] = (df["day_of_month"] == 15).astype(np.int8)
    df["is_payday_1_15"] = (
        (df["day_of_month"] == 1) | (df["day_of_month"] == 15)
    ).astype(np.int8)
    df["days_to_month_end"] = (df["date"].dt.days_in_month - df["day_of_month"]).astype(
        np.int8
    )

    # Lag features
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    df["_promo_int"] = df["onpromotion"].astype(int)
    g = df.groupby("series_id", sort=False)
    for lag in LAG_WINDOWS:
        df[f"sales_lag_{lag}"] = g["unit_sales"].shift(lag).astype(np.float32)
        df[f"promo_lag_{lag}"] = g["_promo_int"].shift(lag).astype(np.float32)
    df = df.drop(columns=["_promo_int"])
    df = df.merge(store_tx_lags, on=["date", "store_nbr"], how="left")

    # Item/store
    df = df.merge(
        items_df[["item_nbr", "family", "class", "perishable"]],
        on="item_nbr",
        how="left",
    )
    df = df.merge(
        stores_df[["store_nbr", "type", "cluster"]],
        on="store_nbr",
        how="left",
    )
    df["perishable"] = df["perishable"].fillna(False).astype(bool)

    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()
    train_df = train_df.dropna()
    if len(test_df) > 0:
        for c in test_df.select_dtypes(include=[np.floating]).columns:
            test_df[c] = test_df[c].fillna(0).astype(np.float32)
    return train_df, test_df


def main():
    os.chdir(Path(__file__).resolve().parent)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    holidays_events_df, stores_df, items_df, transactions_df = load_metadata()

    train_scan = pl.scan_csv(
        TRAIN_PATH,
        has_header=True,
        try_parse_dates=True,
        schema_overrides={
            "date": pl.Date,
            "store_nbr": pl.Int16,
            "item_nbr": pl.Int32,
            "unit_sales": pl.Float32,
            "onpromotion": pl.Boolean,
        },
    )

    if TEST_PATH.exists():
        test_scan = pl.scan_csv(
            TEST_PATH,
            has_header=True,
            try_parse_dates=True,
            schema_overrides={
                "date": pl.Date,
                "store_nbr": pl.Int16,
                "item_nbr": pl.Int32,
                "onpromotion": pl.Boolean,
            },
        )
        test_cols = test_scan.collect_schema().names()
        if "unit_sales" not in test_cols:
            test_scan = test_scan.with_columns(
                pl.lit(0.0).cast(pl.Float32).alias("unit_sales")
            )
        cols = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
        train_scan = train_scan.select(cols)
        test_scan = test_scan.select(cols)
        train_scan = pl.concat([train_scan, test_scan])
        print("Concatenated train + test")

    max_date_val = train_scan.select(pl.col("date").max()).collect().item(0, 0)
    max_date = (
        pd.Timestamp(max_date_val) if max_date_val is not None else pd.Timestamp.now()
    )
    dates = pd.date_range(DATE_FROM, max_date, freq="D")

    if TEST_PATH.exists():
        test_dates = pl.scan_csv(
            TEST_PATH, has_header=True, try_parse_dates=True
        ).select("date")
        test_min_val = test_dates.collect()["date"].min()
        test_start = (
            pd.Timestamp(test_min_val)
            if test_min_val is not None and pd.notna(test_min_val)
            else max_date - pd.Timedelta(days=15)
        )
    else:
        test_start = max_date - pd.Timedelta(days=15)
    transactions_df_from_2016 = prepare_transactions(transactions_df, dates)
    store_tx_lags = build_store_tx_lags(transactions_df_from_2016)
    national, regional, local = build_holiday_tables(holidays_events_df)

    batches = get_item_batches(train_scan)
    print(
        f"Processing {len(batches)} batches ({sum(len(b) for b in batches)} items total)"
    )

    for i, batch_items in enumerate(batches):
        print(f"Batch {i + 1}/{len(batches)} ({len(batch_items)} items)...")
        train_df, test_df = process_batch(
            batch_items=batch_items,
            train_scan=train_scan,
            store_tx_lags=store_tx_lags,
            national=national,
            regional=regional,
            local=local,
            stores_df=stores_df,
            items_df=items_df,
            dates=dates,
            test_start=test_start,
        )
        if len(train_df) == 0 and len(test_df) == 0:
            continue
        if len(train_df) > 0:
            train_path = PREPARED_DIR / f"train_batch_{i:04d}.parquet"
            train_df.to_parquet(train_path, compression="zstd", index=False)
            print(f"  Saved train {len(train_df)} rows -> {train_path}")
        if len(test_df) > 0:
            test_path = PREPARED_DIR / f"test_batch_{i:04d}.parquet"
            test_df.to_parquet(test_path, compression="zstd", index=False)
            print(f"  Saved test {len(test_df)} rows -> {test_path}")

    meta = {
        "n_batches": len(batches),
        "date_from": DATE_FROM,
        "test_start": str(test_start.date()),
        "horizon": HORIZON,
        "lag_windows": LAG_WINDOWS,
    }
    with open(PREPARED_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta saved to {PREPARED_META_PATH}")


if __name__ == "__main__":
    main()

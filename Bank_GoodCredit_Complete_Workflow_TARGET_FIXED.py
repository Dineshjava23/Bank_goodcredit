"""
Bank GoodCredit - DataMites PM-PR-0015
End-to-end credit risk modelling workflow.

Source requirements followed:
- Target: Bad_label (0 = good credit history, 1 = bad / 30+ DPD)
- Tables: Cust_Account, Cust_Enquiry, Cust_Demographics
- Deliverables: EDA insights, feature matrix with gain/importance,
  model evaluation using Gini and decile rank ordering.
- Benchmark Gini: 37.9%

IMPORTANT:
1) Keep the database password private. Prefer environment variable DB_PASSWORD.
2) The exact encoding/order of paymenthistory1/paymenthistory2 and enquiry purpose
   codes is NOT documented in the supplied project PDF. The parser below handles
   common 3-character bureau history tokens conservatively. Review the data before
   treating those engineered variables as final business definitions.
"""

from __future__ import annotations

import os
import re
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -----------------------------
# 0. CONFIGURATION
# -----------------------------
RANDOM_STATE = 42
OUTPUT_DIR = Path("bank_goodcredit_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "18.136.157.135"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "dm_team1"),
    # Uses the supplied project password as a fallback so the script runs directly.
    # For GitHub/submission, set DB_PASSWORD in your environment and remove the fallback.
    "password": os.getenv("DB_PASSWORD", "DM!$Team&279@20!"),
    "database": os.getenv("DB_NAME", "project_banking"),
}

TABLES = {
    "account": "Cust_Account",
    "enquiry": "Cust_Enquiry",
    "demographics": "Cust_Demographics",
}

TARGET = "Bad_label"
CUSTOMER_KEY_CANDIDATES = ["customer_no", "customer no", "customer_number", "customerid", "customer_id"]

# Optional mapping. Fill this only if your mentor/data dictionary gives numeric enquiry-purpose codes.
UNSECURED_PURPOSE_CODES = set()  # example: {"05", "10"} only after confirming business meaning

# -----------------------------
# 1. PACKAGE CHECK
# -----------------------------
def print_package_help():
    print("""
Install required packages once:

pip install pandas numpy sqlalchemy pymysql scikit-learn matplotlib joblib openpyxl

Optional stronger models:
pip install xgboost lightgbm
""")


# -----------------------------
# 2. DATABASE CONNECTION / LOAD
# -----------------------------
def make_engine():
    from sqlalchemy import create_engine
    from urllib.parse import quote_plus

    password = quote_plus(DB_CONFIG["password"])
    url = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{password}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        "?charset=utf8mb4"
    )
    # pool_pre_ping catches dead connections; connect_timeout avoids hanging forever.
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={
            "connect_timeout": 30,
            "read_timeout": 300,
            "write_timeout": 300,
        },
    )


def test_connection(engine):
    from sqlalchemy import text
    with engine.connect() as conn:
        version = conn.execute(text("SELECT VERSION()")).scalar()
        current_db = conn.execute(text("SELECT DATABASE()")).scalar()
        print(f"MySQL connected successfully. Version={version}, Database={current_db}")


def show_tables(engine):
    from sqlalchemy import text
    q = text("SHOW TABLES")
    with engine.connect() as conn:
        rows = conn.execute(q).fetchall()
    tables = [r[0] for r in rows]
    print("Available tables:", tables)
    return tables


def read_table_case_insensitive(
    engine,
    requested_name: str,
    batch_size: int = 20000,
    max_retries: int = 4,
) -> pd.DataFrame:
    """
    Find a table ignoring case and load it in short LIMIT/OFFSET batches.

    Why this is used instead of one SELECT *:
    The project tables (especially Cust_Account) can be large. A single long
    result transfer may make the remote MySQL server close the connection with
    error 2013: 'Lost connection to MySQL server during query'. Each batch here
    uses a fresh connection and has retry logic, so one dropped connection does
    not force the whole table to be downloaded again.
    """
    from sqlalchemy import text
    import time

    available = show_tables(engine)
    mapping = {t.lower(): t for t in available}
    actual = mapping.get(requested_name.lower())
    if not actual:
        raise KeyError(f"Table '{requested_name}' not found. Available: {available}")

    # First get the row count with a tiny query.
    with engine.connect() as conn:
        total_rows = int(conn.execute(text(f"SELECT COUNT(*) FROM `{actual}`")).scalar() or 0)

    print(f"Loading {actual}: {total_rows:,} rows in batches of {batch_size:,} ...")
    if total_rows == 0:
        return pd.read_sql(f"SELECT * FROM `{actual}` LIMIT 0", engine)

    chunks = []
    offset = 0

    while offset < total_rows:
        limit = min(batch_size, total_rows - offset)
        sql = f"SELECT * FROM `{actual}` LIMIT {limit} OFFSET {offset}"

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                # dispose any stale pooled sockets before retrying
                if attempt > 1:
                    engine.dispose()
                    time.sleep(min(2 * attempt, 8))

                with engine.connect() as conn:
                    chunk = pd.read_sql_query(sql, conn)

                chunks.append(chunk)
                offset += len(chunk)
                print(f"  {actual}: {offset:,}/{total_rows:,} rows loaded")

                # Safety: if fewer rows than requested arrive, stop rather than loop forever.
                if len(chunk) < limit:
                    offset = total_rows
                break

            except Exception as exc:
                last_error = exc
                print(
                    f"  Batch starting at row {offset:,} failed "
                    f"(attempt {attempt}/{max_retries}): {exc}"
                )
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Could not load {actual} after {max_retries} retries. "
                        f"Last error: {last_error}"
                    ) from exc

    out = pd.concat(chunks, ignore_index=True)
    print(f"Finished {actual}: {out.shape}")
    return out


def load_all_tables(engine) -> Dict[str, pd.DataFrame]:
    data = {}
    for logical_name, db_name in TABLES.items():
        data[logical_name] = read_table_case_insensitive(engine, db_name)
        print(f"{logical_name}: {data[logical_name].shape}")
    return data


# -----------------------------
# 3. COLUMN NORMALIZATION
# -----------------------------
def normalize_column_name(c: str) -> str:
    c = str(c).strip()
    c = re.sub(r"[\s\-]+", "_", c)
    c = re.sub(r"[^0-9a-zA-Z_]+", "", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c.lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    return out


def resolve_customer_key(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    normalized_candidates = [normalize_column_name(x) for x in CUSTOMER_KEY_CANDIDATES]
    for c in normalized_candidates:
        if c in cols:
            return c
    # fallback heuristic
    for c in df.columns:
        if "customer" in c and ("no" in c or "id" in c):
            return c
    raise KeyError(f"Could not identify customer key in columns: {list(df.columns)}")



def normalize_bad_label(series: pd.Series) -> pd.Series:
    """Normalize the project target to nullable integers 0/1.

    Handles numeric values and common string representations returned by MySQL
    such as '0', '1', '0.0', '1.0', and surrounding whitespace.  Text labels
    are mapped only when their meaning is unambiguous from the project brief.
    Unknown values become <NA> and are reported instead of silently changing them.
    """
    raw = series.copy()

    # Decode bytes safely if the DB driver returns byte strings.
    def _decode(v):
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode('utf-8', errors='ignore')
            except Exception:
                return str(v)
        return v

    cleaned = raw.map(_decode)
    text = cleaned.astype('string').str.strip().str.lower()

    # First accept numeric encodings 0, 1, 0.0, 1.0.
    numeric = pd.to_numeric(text, errors='coerce')
    out = pd.Series(pd.NA, index=series.index, dtype='Int64')
    out.loc[numeric.eq(0)] = 0
    out.loc[numeric.eq(1)] = 1

    # Conservative textual aliases consistent with the project definition.
    aliases = {
        'good': 0, 'good credit': 0, 'good credit history': 0,
        'bad': 1, 'bad credit': 1, 'bad credit history': 1,
    }
    for label, value in aliases.items():
        out.loc[text.eq(label)] = value

    return out


def print_target_diagnostics(series: pd.Series, stage: str) -> None:
    """Print raw/normalized target diagnostics without hiding unexpected labels."""
    print(f"\nTARGET DIAGNOSTICS - {stage}")
    print('-' * 80)
    print('dtype:', series.dtype)
    print('missing:', int(series.isna().sum()), 'of', len(series))
    print('value counts (including missing):')
    print(series.value_counts(dropna=False).head(20))

# -----------------------------
# 4. DATA AUDIT / EDA
# -----------------------------
def audit_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"AUDIT: {name}")
    print("=" * 80)
    print("Shape:", df.shape)
    print("Duplicates:", int(df.duplicated().sum()))
    print("\nDtypes:")
    print(df.dtypes.value_counts())

    audit = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_pct": [float(df[c].isna().mean() * 100) for c in df.columns],
        "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    }).sort_values(["missing_pct", "n_unique"], ascending=[False, True])

    print("\nTop missing columns:")
    print(audit.head(20).to_string(index=False))
    audit.to_csv(OUTPUT_DIR / f"audit_{name}.csv", index=False)
    return audit


def save_target_summary(demo: pd.DataFrame):
    if TARGET.lower() not in demo.columns:
        raise KeyError(f"Target '{TARGET}' not found after normalization.")
    t = TARGET.lower()
    summary = demo[t].value_counts(dropna=False).rename_axis(t).reset_index(name="count")
    summary["pct"] = summary["count"] / len(demo) * 100
    print("\nTarget distribution:")
    print(summary)
    summary.to_csv(OUTPUT_DIR / "target_distribution.csv", index=False)


# -----------------------------
# 5. DATE / NUMERIC HELPERS
# -----------------------------
def to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def to_numeric_safe(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    cleaned = (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d\.\-\+eE]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def safe_divide(a, b):
    if np.isscalar(b):
        return np.nan if b in (0, None) else a / b
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a.div(b.replace(0, np.nan))


# -----------------------------
# 6. PAYMENT HISTORY PARSING
# -----------------------------
DPD_MAP = {
    "STD": 0,
    "SMA": 30,
    "SUB": 60,
    "DBT": 90,
    "LSS": 120,
    "XXX": np.nan,
}

def payment_history_tokens(value) -> List[float]:
    """
    Conservatively parses common bureau strings split in 3-char chunks.
    Numeric 000..999 => DPD number.
    STD/SMA/SUB/DBT/LSS => mapped approximate delinquency severity.
    Unknown tokens => NaN.

    IMPORTANT: The PDF does not define token order. We do not claim calendar
    direction unless verified from the raw dataset/data dictionary.
    """
    if pd.isna(value):
        return []
    s = re.sub(r"[\s,;|/\-]+", "", str(value).upper())
    if not s:
        return []
    chunks = [s[i:i+3] for i in range(0, len(s), 3)]
    out = []
    for x in chunks:
        if len(x) != 3:
            continue
        if x.isdigit():
            out.append(float(int(x)))
        elif x in DPD_MAP:
            out.append(float(DPD_MAP[x]) if pd.notna(DPD_MAP[x]) else np.nan)
        else:
            out.append(np.nan)
    return out


def combine_payment_history(row) -> List[float]:
    tokens = []
    for col in ["paymenthistory1", "paymenthistory2"]:
        if col in row.index:
            tokens.extend(payment_history_tokens(row[col]))
    return tokens


def add_payment_history_account_features(account: pd.DataFrame) -> pd.DataFrame:
    out = account.copy()
    if not any(c in out.columns for c in ["paymenthistory1", "paymenthistory2"]):
        return out

    histories = out.apply(combine_payment_history, axis=1)

    def arr(h):
        return np.array([x for x in h if pd.notna(x)], dtype=float)

    out["ph_months_observed"] = histories.map(lambda h: len([x for x in h if pd.notna(x)]))
    out["ph_mean_dpd"] = histories.map(lambda h: np.nanmean(arr(h)) if len(arr(h)) else np.nan)
    out["ph_max_dpd"] = histories.map(lambda h: np.nanmax(arr(h)) if len(arr(h)) else np.nan)
    out["ph_count_0_29"] = histories.map(lambda h: int(np.sum((arr(h) >= 0) & (arr(h) <= 29))) if len(arr(h)) else 0)
    out["ph_count_30_plus"] = histories.map(lambda h: int(np.sum(arr(h) >= 30)) if len(arr(h)) else 0)
    out["ph_count_60_plus"] = histories.map(lambda h: int(np.sum(arr(h) >= 60)) if len(arr(h)) else 0)
    out["ph_count_90_plus"] = histories.map(lambda h: int(np.sum(arr(h) >= 90)) if len(arr(h)) else 0)
    out["ph_ratio_0_29"] = safe_divide(out["ph_count_0_29"], out["ph_months_observed"])
    out["ph_ratio_30_plus"] = safe_divide(out["ph_count_30_plus"], out["ph_months_observed"])

    # Position of first observed 30+ token. Treat only as a sequence index until
    # data dictionary confirms whether the string is newest->oldest or oldest->newest.
    def first_30_plus_position(h):
        vals = [x for x in h if pd.notna(x)]
        idx = [i + 1 for i, x in enumerate(vals) if x >= 30]
        return min(idx) if idx else np.nan

    out["ph_first_30_plus_position"] = histories.map(first_30_plus_position)
    return out


# -----------------------------
# 7. LEAKAGE-SAFE SNAPSHOT JOIN
# -----------------------------
def prepare_snapshot_dates(demo, account, enquiry):
    demo = demo.copy()
    account = account.copy()
    enquiry = enquiry.copy()

    for df in [demo, account, enquiry]:
        for c in df.columns:
            if c.endswith("_dt") or c in {"dt_opened", "entry_time", "opened_dt", "closed_dt", "reporting_dt"}:
                df[c] = to_datetime_safe(df[c])

    return demo, account, enquiry


def filter_records_to_application_date(
    demo: pd.DataFrame,
    child: pd.DataFrame,
    child_date_col: str,
    suffix: str
) -> pd.DataFrame:
    """
    Keeps child records occurring on/before the customer's demographic/application dt_opened.
    If application date or child date is missing, the row is retained but a warning is printed.
    """
    dkey = resolve_customer_key(demo)
    ckey = resolve_customer_key(child)

    if "dt_opened" not in demo.columns or child_date_col not in child.columns:
        print(f"WARNING: Cannot time-filter {suffix}; missing dt_opened or {child_date_col}.")
        return child.copy()

    ref = demo[[dkey, "dt_opened"]].rename(columns={dkey: "_cust", "dt_opened": "_application_dt"})
    c = child.rename(columns={ckey: "_cust"}).merge(ref, on="_cust", how="left")
    before = len(c)

    keep = (
        c["_application_dt"].isna()
        | c[child_date_col].isna()
        | (c[child_date_col] <= c["_application_dt"])
    )
    c = c.loc[keep].copy()
    print(f"{suffix}: removed {before - len(c):,} possible future records using application-date filter.")
    c = c.drop(columns=["_application_dt"]).rename(columns={"_cust": ckey})
    return c


# -----------------------------
# 8. ACCOUNT FEATURE ENGINEERING
# -----------------------------
def engineer_account_features(account: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    account = account.copy()
    key = resolve_customer_key(account)

    # Numeric conversion for documented amount/rate columns
    numeric_cols = [
        "high_credit_amt", "cur_balance_amt", "amt_past_due",
        "creditlimit", "cashlimit", "rateofinterest",
        "paymentfrequency", "actualpaymentamount",
    ]
    for c in numeric_cols:
        if c in account.columns:
            account[c] = to_numeric_safe(account[c])

    # Date conversions
    date_cols = ["dt_opened", "upload_dt", "opened_dt", "last_paymt_dt",
                 "closed_dt", "reporting_dt", "paymt_str_dt", "paymt_end_dt"]
    for c in date_cols:
        if c in account.columns:
            account[c] = to_datetime_safe(account[c])

    # Filter records that are clearly after current application.
    if "opened_dt" in account.columns:
        account = filter_records_to_application_date(demo, account, "opened_dt", "account")

    # Payment history features at account row level
    account = add_payment_history_account_features(account)

    # Account-level durations
    if {"last_paymt_dt", "opened_dt"}.issubset(account.columns):
        account["diff_lastpaymt_opened_days"] = (
            account["last_paymt_dt"] - account["opened_dt"]
        ).dt.days

    if {"closed_dt", "opened_dt"}.issubset(account.columns):
        account["account_lifetime_days"] = (
            account["closed_dt"] - account["opened_dt"]
        ).dt.days

    if {"reporting_dt", "opened_dt"}.issubset(account.columns):
        account["account_age_at_reporting_days"] = (
            account["reporting_dt"] - account["opened_dt"]
        ).dt.days

    # Row-level utilization
    if "cur_balance_amt" in account.columns and "creditlimit" in account.columns:
        account["row_utilisation"] = safe_divide(account["cur_balance_amt"], account["creditlimit"])

    # Base customer count
    grouped = account.groupby(key, dropna=False)
    feat = grouped.size().rename("account_count").to_frame()

    # Numeric aggregations
    candidate_numeric = [
        "high_credit_amt", "cur_balance_amt", "amt_past_due",
        "creditlimit", "cashlimit", "rateofinterest",
        "actualpaymentamount", "diff_lastpaymt_opened_days",
        "account_lifetime_days", "account_age_at_reporting_days",
        "row_utilisation", "ph_months_observed", "ph_mean_dpd", "ph_max_dpd",
        "ph_count_0_29", "ph_count_30_plus", "ph_count_60_plus",
        "ph_count_90_plus", "ph_ratio_0_29", "ph_ratio_30_plus",
        "ph_first_30_plus_position",
    ]
    agg_spec = {}
    for c in candidate_numeric:
        if c in account.columns:
            agg_spec[c] = ["sum", "mean", "min", "max"]

    if agg_spec:
        a = grouped.agg(agg_spec)
        a.columns = [f"acct_{c}_{stat}" for c, stat in a.columns]
        feat = feat.join(a, how="left")

    # Counts of past due / open accounts
    if "amt_past_due" in account.columns:
        feat["accounts_with_past_due"] = grouped["amt_past_due"].apply(lambda s: int((s.fillna(0) > 0).sum()))
        feat["ratio_accounts_with_past_due"] = safe_divide(feat["accounts_with_past_due"], feat["account_count"])

    if "closed_dt" in account.columns:
        feat["open_account_count"] = grouped["closed_dt"].apply(lambda s: int(s.isna().sum()))
        feat["closed_account_count"] = grouped["closed_dt"].apply(lambda s: int(s.notna().sum()))

    # Requested/suggested project feature names
    if "ph_count_0_29" in account.columns:
        feat["payment_history_avg_dpd_0_29_bucket"] = grouped["ph_count_0_29"].mean()

    if "diff_lastpaymt_opened_days" in account.columns:
        feat["total_diff_lastpaymt_opened_dt"] = grouped["diff_lastpaymt_opened_days"].sum(min_count=1)
        feat["mean_diff_lastpaymt_opened_dt"] = grouped["diff_lastpaymt_opened_days"].mean()

    if "ph_first_30_plus_position" in account.columns:
        feat["min_months_last_30_plus"] = grouped["ph_first_30_plus_position"].min()

    if "ph_months_observed" in account.columns:
        feat["payment_history_mean_length"] = grouped["ph_months_observed"].mean()

    # Customer-level ratio_currbalance_creditlimit
    if {"cur_balance_amt", "creditlimit"}.issubset(account.columns):
        total_bal = grouped["cur_balance_amt"].sum(min_count=1)
        total_cl = grouped["creditlimit"].sum(min_count=1)
        feat["ratio_currbalance_creditlimit"] = safe_divide(total_bal, total_cl)

    # utilisation_trend per formula in PDF:
    # [total cur_bal_amt / total credit limit] /
    # [mean cur_bal_amt / (mean credit limit + mean cashlimit)]
    needed = {"cur_balance_amt", "creditlimit", "cashlimit"}
    if needed.issubset(account.columns):
        total_bal = grouped["cur_balance_amt"].sum(min_count=1)
        total_cl = grouped["creditlimit"].sum(min_count=1)
        mean_bal = grouped["cur_balance_amt"].mean()
        mean_cl = grouped["creditlimit"].mean()
        mean_cash = grouped["cashlimit"].mean()
        numerator = safe_divide(total_bal, total_cl)
        denominator = safe_divide(mean_bal, mean_cl + mean_cash)
        feat["utilisation_trend"] = safe_divide(numerator, denominator)

    # Type diversity
    for cat_col in ["acct_type", "owner_indic"]:
        if cat_col in account.columns:
            feat[f"{cat_col}_nunique"] = grouped[cat_col].nunique(dropna=True)
            feat[f"{cat_col}_mode"] = grouped[cat_col].agg(
                lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
            )

    return feat.reset_index()


# -----------------------------
# 9. ENQUIRY FEATURE ENGINEERING
# -----------------------------
def purpose_is_unsecured(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().upper()
    if s in {str(v).upper() for v in UNSECURED_PURPOSE_CODES}:
        return True
    # Conservative text-only matching when descriptions are textual.
    patterns = ["PERSONAL", "CREDIT CARD", "CONSUMER", "UNSECURED"]
    return any(p in s for p in patterns)


def engineer_enquiry_features(enquiry: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    enquiry = enquiry.copy()
    ekey = resolve_customer_key(enquiry)
    dkey = resolve_customer_key(demo)

    if "enquiry_dt" in enquiry.columns:
        enquiry["enquiry_dt"] = to_datetime_safe(enquiry["enquiry_dt"])
    for c in ["dt_opened", "upload_dt"]:
        if c in enquiry.columns:
            enquiry[c] = to_datetime_safe(enquiry[c])
    if "enq_amt" in enquiry.columns:
        enquiry["enq_amt"] = to_numeric_safe(enquiry["enq_amt"])

    # Bring application/reference date from demographics
    ref = demo[[dkey, "dt_opened"]].rename(columns={dkey: "_cust", "dt_opened": "_application_dt"})
    e = enquiry.rename(columns={ekey: "_cust"}).merge(ref, on="_cust", how="left")

    if "enquiry_dt" in e.columns:
        # avoid future leakage
        keep = (
            e["_application_dt"].isna()
            | e["enquiry_dt"].isna()
            | (e["enquiry_dt"] <= e["_application_dt"])
        )
        removed = (~keep).sum()
        e = e.loc[keep].copy()
        print(f"enquiry: removed {removed:,} possible future records.")

        e["days_before_application"] = (e["_application_dt"] - e["enquiry_dt"]).dt.days
        e["is_enquiry_30d"] = e["days_before_application"].between(0, 30, inclusive="both").astype(int)
        e["is_enquiry_90d"] = e["days_before_application"].between(0, 90, inclusive="both").astype(int)
        e["is_enquiry_180d"] = e["days_before_application"].between(0, 180, inclusive="both").astype(int)
        e["is_enquiry_365d"] = e["days_before_application"].between(0, 365, inclusive="both").astype(int)

    grouped = e.groupby("_cust", dropna=False)
    feat = grouped.size().rename("enquiry_count").to_frame()

    if "enq_amt" in e.columns:
        for stat in ["sum", "mean", "min", "max", "median"]:
            feat[f"enq_amt_{stat}"] = getattr(grouped["enq_amt"], stat)()

    if "days_before_application" in e.columns:
        feat["mean_diff_open_enquiry_dt"] = grouped["days_before_application"].mean()
        feat["min_days_since_enquiry"] = grouped["days_before_application"].min()
        feat["max_days_since_enquiry"] = grouped["days_before_application"].max()
        for d in [30, 90, 180, 365]:
            feat[f"count_enquiry_recency_{d}"] = grouped[f"is_enquiry_{d}d"].sum()

    if "enq_purpose" in e.columns:
        feat["enq_purpose_nunique"] = grouped["enq_purpose"].nunique(dropna=True)
        feat["max_freq_enquiry"] = grouped["enq_purpose"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        )

        e["is_unsecured_purpose"] = e["enq_purpose"].map(purpose_is_unsecured).astype(int)
        unsecured = e.groupby("_cust")["is_unsecured_purpose"].sum()
        feat["perc_unsecured_others"] = safe_divide(unsecured, feat["enquiry_count"])

    return feat.reset_index().rename(columns={"_cust": ekey})


# -----------------------------
# 10. DEMOGRAPHIC PREPARATION / MASTER MATRIX
# -----------------------------
def add_demographic_date_features(demo: pd.DataFrame) -> pd.DataFrame:
    d = demo.copy()
    for c in ["dt_opened", "entry_time"]:
        if c in d.columns:
            d[c] = to_datetime_safe(d[c])
            d[f"{c}_year"] = d[c].dt.year
            d[f"{c}_month"] = d[c].dt.month
            d[f"{c}_quarter"] = d[c].dt.quarter
            d[f"{c}_dayofweek"] = d[c].dt.dayofweek
    return d


def build_master_dataset(
    demographics: pd.DataFrame,
    account_features: pd.DataFrame,
    enquiry_features: pd.DataFrame
) -> pd.DataFrame:
    demo = add_demographic_date_features(demographics)
    dkey = resolve_customer_key(demo)
    akey = resolve_customer_key(account_features)
    ekey = resolve_customer_key(enquiry_features)

    master = demo.copy()
    master = master.merge(
        account_features.rename(columns={akey: dkey}),
        on=dkey,
        how="left",
        validate="one_to_one",
    )
    master = master.merge(
        enquiry_features.rename(columns={ekey: dkey}),
        on=dkey,
        how="left",
        validate="one_to_one",
    )

    # Drop raw datetime columns before ML. Keep application dt separately for OOT split.
    print("Master shape:", master.shape)
    return master


# -----------------------------
# 11. CLEANING / FEATURE FILTERS
# -----------------------------
def coerce_numeric_like_columns(df: pd.DataFrame, threshold=0.95) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            nonnull = out[c].dropna()
            if len(nonnull) == 0:
                continue
            converted = to_numeric_safe(nonnull)
            success = converted.notna().mean()
            if success >= threshold:
                out[c] = to_numeric_safe(out[c])
    return out


def drop_unusable_features(
    X: pd.DataFrame,
    missing_threshold=0.95,
    max_category_ratio=0.90
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    X = X.copy()
    report = {"all_missing": [], "high_missing": [], "constant": [], "id_like": []}

    for c in X.columns:
        miss = X[c].isna().mean()
        nunique = X[c].nunique(dropna=True)

        if miss == 1:
            report["all_missing"].append(c)
        elif miss >= missing_threshold:
            report["high_missing"].append(c)
        elif nunique <= 1:
            report["constant"].append(c)
        elif X[c].dtype == "object":
            # high-cardinality strings that are nearly row IDs often hurt models
            nonnull_n = max(1, X[c].notna().sum())
            if nunique / nonnull_n >= max_category_ratio and nunique > 50:
                report["id_like"].append(c)

    drop_cols = sorted(set(sum(report.values(), [])))
    X = X.drop(columns=drop_cols, errors="ignore")
    print("Dropped unusable features:", {k: len(v) for k, v in report.items()})
    return X, report


# -----------------------------
# 12. TRAIN / VALIDATION / TEST SPLIT
# -----------------------------
def make_splits(master: pd.DataFrame):
    from sklearn.model_selection import train_test_split

    target = TARGET.lower()
    key = resolve_customer_key(master)

    work = master.copy()

    # Robustly normalize target before filtering. MySQL/pandas may return the
    # same 0/1 labels as strings, floats, decimals, or byte strings.
    print_target_diagnostics(work[target], "before normalization in make_splits")
    normalized_target = normalize_bad_label(work[target])
    print_target_diagnostics(normalized_target, "after normalization in make_splits")

    invalid_mask = normalized_target.isna() & work[target].notna()
    if invalid_mask.any():
        examples = (
            work.loc[invalid_mask, target]
            .astype(str)
            .value_counts(dropna=False)
            .head(20)
            .to_dict()
        )
        print(f"WARNING: Dropping {int(invalid_mask.sum()):,} rows with unsupported Bad_label values: {examples}")

    work[target] = normalized_target
    work = work.loc[work[target].isin([0, 1])].copy()
    work[target] = work[target].astype(int)

    # Save application date only for possible out-of-time split
    app_dt = work["dt_opened"].copy() if "dt_opened" in work.columns else pd.Series(pd.NaT, index=work.index)

    drop_cols = [target, key]
    # raw dates are dropped; engineered date parts stay
    for c in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[c]):
            drop_cols.append(c)

    X = work.drop(columns=list(set(drop_cols)), errors="ignore")
    y = work[target]

    X = coerce_numeric_like_columns(X)
    X, dropped = drop_unusable_features(X)

    # Prefer out-of-time split only when application dates are sufficiently available
    valid_dates = app_dt.notna().mean()
    unique_dates = app_dt.nunique(dropna=True)

    use_oot = valid_dates >= 0.80 and unique_dates >= 30

    if use_oot:
        order = np.argsort(app_dt.fillna(app_dt.min()).values)
        Xo = X.iloc[order].reset_index(drop=True)
        yo = y.iloc[order].reset_index(drop=True)

        n = len(Xo)
        i1 = int(n * 0.70)
        i2 = int(n * 0.85)

        X_train, y_train = Xo.iloc[:i1], yo.iloc[:i1]
        X_valid, y_valid = Xo.iloc[i1:i2], yo.iloc[i1:i2]
        X_test, y_test = Xo.iloc[i2:], yo.iloc[i2:]

        # AUC/Gini and binary classifiers require both 0 and 1 to be present.
        # Credit-risk data can be time-clustered, so a pure chronological split
        # may accidentally create a one-class train/validation/test sample.
        split_has_both_classes = all(
            pd.Series(part).nunique(dropna=True) == 2
            for part in (y_train, y_valid, y_test)
        )

        if split_has_both_classes:
            split_type = "out_of_time"
        else:
            print(
                "WARNING: Out-of-time split produced a one-class partition. "
                "Falling back to stratified random split so binary models and "
                "AUC/Gini can be evaluated correctly."
            )
            use_oot = False

    if not use_oot:
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            raise ValueError(
                "Target Bad_label contains only one class after cleaning. "
                "Both 0 and 1 are required for credit-risk modelling."
            )
        if class_counts.min() < 4:
            raise ValueError(
                f"Too few observations in the minority class for train/validation/test "
                f"splitting: {class_counts.to_dict()}. Need at least 4 minority-class rows."
            )

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
        )
        split_type = "stratified_random"

    print(f"Split type: {split_type}")
    print("Train:", X_train.shape, "Valid:", X_valid.shape, "Test:", X_test.shape)
    print("Train bad rate:", y_train.mean())
    print("Valid bad rate:", y_valid.mean())
    print("Test bad rate:", y_test.mean())

    with open(OUTPUT_DIR / "dropped_feature_report.json", "w") as f:
        json.dump(dropped, f, indent=2)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, split_type


# -----------------------------
# 13. PREPROCESSOR + MODELS
# -----------------------------
def build_preprocessor(X_train: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_cols = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor, num_cols, cat_cols


def get_models(preprocessor):
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

    models = {
        "LogisticRegression": Pipeline([
            ("prep", preprocessor),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=0.5,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )),
        ]),
        "RandomForest": Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }

    # Optional XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("prep", preprocessor),
            ("model", XGBClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=10,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="auc",
                reg_lambda=2.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])
    except Exception:
        print("XGBoost not installed; skipping.")

    # Optional LightGBM
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = Pipeline([
            ("prep", preprocessor),
            ("model", LGBMClassifier(
                n_estimators=600,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            )),
        ])
    except Exception:
        print("LightGBM not installed; skipping.")

    return models


# -----------------------------
# 14. METRICS / GINI / RANK ORDERING
# -----------------------------
def gini_from_auc(auc: float) -> float:
    return 2 * auc - 1


def evaluate_predictions(y_true, prob, threshold=0.5) -> Dict[str, float]:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score,
        f1_score, log_loss, brier_score_loss
    )
    pred = (prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, prob)
    return {
        "auc": auc,
        "gini": gini_from_auc(auc),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "logloss": log_loss(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
    }


def decile_table(y_true, prob, n_bins=10) -> pd.DataFrame:
    """
    Decile 10 = highest predicted risk, Decile 1 = lowest predicted risk.
    Uses rank before qcut so tied probabilities still distribute consistently.
    """
    df = pd.DataFrame({"actual": np.asarray(y_true), "prob": np.asarray(prob)})
    df["risk_rank"] = df["prob"].rank(method="first", ascending=True)
    df["decile"] = pd.qcut(df["risk_rank"], q=n_bins, labels=range(1, n_bins + 1)).astype(int)

    overall_bad_rate = df["actual"].mean()
    tab = (
        df.groupby("decile", as_index=False)
        .agg(
            customers=("actual", "size"),
            bads=("actual", "sum"),
            avg_predicted_bad_rate=("prob", "mean"),
            min_score=("prob", "min"),
            max_score=("prob", "max"),
        )
    )
    tab["goods"] = tab["customers"] - tab["bads"]
    tab["actual_bad_rate"] = safe_divide(tab["bads"], tab["customers"])
    tab["lift"] = safe_divide(tab["actual_bad_rate"], overall_bad_rate)

    total_bads = tab["bads"].sum()
    # cumulative bad capture from highest risk to lowest
    temp = tab.sort_values("decile", ascending=False).copy()
    temp["cumulative_bads"] = temp["bads"].cumsum()
    temp["cumulative_bad_capture"] = safe_divide(temp["cumulative_bads"], total_bads)
    tab = tab.merge(
        temp[["decile", "cumulative_bad_capture"]],
        on="decile",
        how="left"
    )
    return tab.sort_values("decile", ascending=False).reset_index(drop=True)


def ks_table(y_true, prob, n_bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"actual": np.asarray(y_true), "prob": np.asarray(prob)})
    df["rank"] = df["prob"].rank(method="first", ascending=True)
    df["decile"] = pd.qcut(df["rank"], q=n_bins, labels=range(1, n_bins + 1)).astype(int)
    g = df.groupby("decile").agg(total=("actual", "size"), bad=("actual", "sum")).reset_index()
    g["good"] = g["total"] - g["bad"]
    g = g.sort_values("decile", ascending=False)
    g["cum_bad_pct"] = g["bad"].cumsum() / max(1, g["bad"].sum())
    g["cum_good_pct"] = g["good"].cumsum() / max(1, g["good"].sum())
    g["ks"] = (g["cum_bad_pct"] - g["cum_good_pct"]).abs()
    return g


# -----------------------------
# 15. MODEL TRAINING / SELECTION
# -----------------------------
def train_and_compare(
    X_train, X_valid, y_train, y_valid
):
    import joblib

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    models = get_models(preprocessor)
    rows = []
    fitted = {}

    for name, model in models.items():
        print("\nTraining:", name)
        try:
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_valid)[:, 1]
            metrics = evaluate_predictions(y_valid, prob)
            rows.append({"model": name, **metrics})
            fitted[name] = model
            print(
                f"{name}: AUC={metrics['auc']:.4f}, "
                f"Gini={metrics['gini']:.4f} ({metrics['gini']*100:.2f}%)"
            )
        except Exception as e:
            print(f"{name} failed: {e}")

    if not rows:
        raise RuntimeError(
            "No model trained successfully. Review the individual model errors printed above. "
            "Also verify that train/validation data contain both Bad_label classes (0 and 1)."
        )

    results = pd.DataFrame(rows).sort_values("gini", ascending=False)
    results.to_csv(OUTPUT_DIR / "validation_model_comparison.csv", index=False)
    print("\nValidation comparison:")
    print(results.to_string(index=False))

    best_name = results.iloc[0]["model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, OUTPUT_DIR / f"best_model_validation_{best_name}.joblib")
    return best_name, best_model, results


def refit_best_and_test(
    best_name,
    X_train, X_valid, X_test,
    y_train, y_valid, y_test
):
    import joblib

    X_tv = pd.concat([X_train, X_valid], axis=0)
    y_tv = pd.concat([y_train, y_valid], axis=0)

    preprocessor, _, _ = build_preprocessor(X_tv)
    all_models = get_models(preprocessor)
    if best_name not in all_models:
        raise KeyError(f"Best model {best_name} unavailable during refit.")

    model = all_models[best_name]
    model.fit(X_tv, y_tv)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_predictions(y_test, prob)
    print("\nFINAL TEST RESULTS")
    print(json.dumps(metrics, indent=2))

    benchmark_gini = 0.379
    print(
        f"\nBenchmark Gini: {benchmark_gini:.3f} ({benchmark_gini*100:.1f}%)"
        f"\nTest Gini:      {metrics['gini']:.3f} ({metrics['gini']*100:.2f}%)"
        f"\nBeat benchmark: {metrics['gini'] > benchmark_gini}"
    )

    pd.DataFrame([{"model": best_name, **metrics}]).to_csv(
        OUTPUT_DIR / "final_test_metrics.csv", index=False
    )

    dec = decile_table(y_test, prob)
    dec.to_csv(OUTPUT_DIR / "test_decile_rank_ordering.csv", index=False)

    ks = ks_table(y_test, prob)
    ks.to_csv(OUTPUT_DIR / "test_ks_table.csv", index=False)

    pred_out = pd.DataFrame({
        "actual_bad_label": np.asarray(y_test),
        "predicted_bad_probability": prob,
    })
    pred_out.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    joblib.dump(model, OUTPUT_DIR / f"final_model_{best_name}.joblib")
    return model, metrics, dec, ks


# -----------------------------
# 16. FEATURE IMPORTANCE / GAIN
# -----------------------------
def extract_feature_importance(pipeline, top_n=100) -> pd.DataFrame:
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    try:
        names = prep.get_feature_names_out()
    except Exception:
        names = np.array([f"f{i}" for i in range(getattr(model, "n_features_in_", 0))])

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        importance_type = "feature_importance"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        values = np.abs(coef)
        importance_type = "abs_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "gain_or_importance", "importance_type"])

    n = min(len(names), len(values))
    imp = pd.DataFrame({
        "feature": names[:n],
        "gain_or_importance": values[:n],
        "importance_type": importance_type,
    }).sort_values("gain_or_importance", ascending=False)

    # normalized gain/importance for easier reading
    denom = imp["gain_or_importance"].sum()
    imp["normalized_gain"] = imp["gain_or_importance"] / denom if denom > 0 else np.nan
    imp = imp.head(top_n)
    imp.to_csv(OUTPUT_DIR / "feature_matrix_with_gain.csv", index=False)
    return imp


# -----------------------------
# 17. PLOTS
# -----------------------------
def save_plots(y_test, test_prob, decile_df, feature_imp):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    # ROC
    RocCurveDisplay.from_predictions(y_test, test_prob)
    plt.title("Bank GoodCredit - ROC Curve")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=160)
    plt.close()

    # Rank ordering
    plt.figure(figsize=(8, 5))
    plot_df = decile_df.sort_values("decile")
    plt.plot(plot_df["decile"], plot_df["actual_bad_rate"], marker="o")
    plt.xlabel("Decile (1 = lowest risk, 10 = highest risk)")
    plt.ylabel("Actual Bad Rate")
    plt.title("Decile Rank Ordering")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decile_rank_ordering.png", dpi=160)
    plt.close()

    # Feature importance
    if not feature_imp.empty:
        top = feature_imp.head(20).sort_values("gain_or_importance", ascending=True)
        plt.figure(figsize=(10, 8))
        plt.barh(top["feature"], top["gain_or_importance"])
        plt.xlabel("Importance / Gain")
        plt.title("Top 20 Model Features")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "top20_feature_importance.png", dpi=160)
        plt.close()


# -----------------------------
# 18. BUSINESS/EDA REPORT TABLES
# -----------------------------
def save_basic_eda(master: pd.DataFrame):
    target = TARGET.lower()

    # Missingness
    miss = pd.DataFrame({
        "feature": master.columns,
        "missing_count": [master[c].isna().sum() for c in master.columns],
        "missing_pct": [master[c].isna().mean() * 100 for c in master.columns],
        "n_unique": [master[c].nunique(dropna=True) for c in master.columns],
    }).sort_values("missing_pct", ascending=False)
    miss.to_csv(OUTPUT_DIR / "master_missingness.csv", index=False)

    # Numeric univariate bad-rate correlation / summary
    numeric_cols = [
        c for c in master.select_dtypes(include=np.number).columns
        if c != target and master[c].nunique(dropna=True) > 1
    ]
    rows = []
    for c in numeric_cols:
        tmp = master[[c, target]].dropna()
        if len(tmp) < 20:
            continue
        try:
            corr = tmp[c].corr(tmp[target], method="spearman")
        except Exception:
            corr = np.nan
        rows.append({
            "feature": c,
            "spearman_with_bad_label": corr,
            "mean_good": tmp.loc[tmp[target] == 0, c].mean(),
            "mean_bad": tmp.loc[tmp[target] == 1, c].mean(),
            "median_good": tmp.loc[tmp[target] == 0, c].median(),
            "median_bad": tmp.loc[tmp[target] == 1, c].median(),
        })
    uni = pd.DataFrame(rows)
    if not uni.empty:
        uni["abs_spearman"] = uni["spearman_with_bad_label"].abs()
        uni = uni.sort_values("abs_spearman", ascending=False)
    uni.to_csv(OUTPUT_DIR / "numeric_univariate_summary.csv", index=False)


# -----------------------------
# 19. EXCEL SUMMARY
# -----------------------------
def save_excel_summary(
    validation_results: pd.DataFrame,
    final_metrics: Dict[str, float],
    deciles: pd.DataFrame,
    ks: pd.DataFrame,
    feature_imp: pd.DataFrame
):
    path = OUTPUT_DIR / "Bank_GoodCredit_Model_Report.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        validation_results.to_excel(writer, sheet_name="Model Comparison", index=False)
        pd.DataFrame([final_metrics]).to_excel(writer, sheet_name="Final Metrics", index=False)
        deciles.to_excel(writer, sheet_name="Decile Rank Order", index=False)
        ks.to_excel(writer, sheet_name="KS Table", index=False)
        feature_imp.to_excel(writer, sheet_name="Feature Matrix", index=False)
    print("Saved:", path)


# -----------------------------
# 20. MAIN WORKFLOW
# -----------------------------
def main():
    print("\nBANK GOODCREDIT - COMPLETE WORKFLOW")
    print("=" * 80)
    print_package_help()

    # A. Connect and load
    engine = make_engine()
    try:
        test_connection(engine)
        raw = load_all_tables(engine)
    except Exception as e:
        print("\nDATABASE CONNECTION/LOAD FAILED")
        print("Error:", repr(e))
        print("""
Check:
1) Internet connection
2) Host/port are reachable
3) MySQL server is running
4) Your public IP is allowed by the server/security group
5) Credentials/database name are correct
6) VPN/firewall is not blocking port 3306

The script stops here because modelling must use the real project tables.
""")
        raise

    # B. Normalize columns
    account = normalize_columns(raw["account"])
    enquiry = normalize_columns(raw["enquiry"])
    demo = normalize_columns(raw["demographics"])

    # normalize target spelling
    if "bad_label" not in demo.columns:
        possible = [c for c in demo.columns if c.replace("_", "") == "badlabel"]
        if possible:
            demo = demo.rename(columns={possible[0]: "bad_label"})

    # Normalize target immediately after loading so every downstream step uses
    # the same reliable 0/1 representation.
    if "bad_label" not in demo.columns:
        raise KeyError(
            f"Target 'Bad_label' was not found in Cust_Demographics. Columns: {list(demo.columns)}"
        )

    print_target_diagnostics(demo["bad_label"], "raw Cust_Demographics")
    raw_bad_label = demo["bad_label"].copy()
    demo["bad_label"] = normalize_bad_label(demo["bad_label"])
    print_target_diagnostics(demo["bad_label"], "normalized Cust_Demographics")

    unsupported = demo["bad_label"].isna() & raw_bad_label.notna()
    if unsupported.any():
        unsupported_values = (
            raw_bad_label.loc[unsupported]
            .astype(str)
            .value_counts(dropna=False)
            .head(20)
            .to_dict()
        )
        print(
            f"WARNING: {int(unsupported.sum()):,} target rows have unsupported values and "
            f"will not be used for modelling: {unsupported_values}"
        )

    normalized_counts = demo["bad_label"].value_counts(dropna=True).to_dict()
    if len(normalized_counts) < 2:
        raw_counts = raw_bad_label.astype(str).str.strip().value_counts(dropna=False).head(30).to_dict()
        raise ValueError(
            "Bad_label still has only one valid class after robust normalization. "
            f"Normalized counts={normalized_counts}; raw counts={raw_counts}. "
            "This means the loaded Cust_Demographics data itself does not currently "
            "contain both target classes, or the target is stored using an unexpected code. "
            "Do not fabricate the missing class; inspect the printed raw counts."
        )

    # C. Audit
    audit_dataframe(account, "Cust_Account")
    audit_dataframe(enquiry, "Cust_Enquiry")
    audit_dataframe(demo, "Cust_Demographics")
    save_target_summary(demo)

    # D. Dates
    demo, account, enquiry = prepare_snapshot_dates(demo, account, enquiry)

    # E. Feature engineering
    acct_feat = engineer_account_features(account, demo)
    enq_feat = engineer_enquiry_features(enquiry, demo)

    acct_feat.to_csv(OUTPUT_DIR / "account_customer_features.csv", index=False)
    enq_feat.to_csv(OUTPUT_DIR / "enquiry_customer_features.csv", index=False)

    # F. Master model matrix
    master = build_master_dataset(demo, acct_feat, enq_feat)
    master.to_csv(OUTPUT_DIR / "master_customer_dataset.csv", index=False)
    save_basic_eda(master)

    # G. Split
    X_train, X_valid, X_test, y_train, y_valid, y_test, split_type = make_splits(master)

    # H. Train and compare
    best_name, best_valid_model, validation_results = train_and_compare(
        X_train, X_valid, y_train, y_valid
    )

    # I. Final refit + test
    final_model, final_metrics, deciles, ks = refit_best_and_test(
        best_name,
        X_train, X_valid, X_test,
        y_train, y_valid, y_test
    )

    # J. Final predictions for plots
    test_prob = final_model.predict_proba(X_test)[:, 1]

    # K. Feature gain / importance
    feature_imp = extract_feature_importance(final_model, top_n=200)
    print("\nTop model features:")
    print(feature_imp.head(25).to_string(index=False))

    # L. Save plots and Excel
    save_plots(y_test, test_prob, deciles, feature_imp)
    save_excel_summary(validation_results, final_metrics, deciles, ks, feature_imp)

    # M. Rank ordering quality check
    risk_rates = deciles.sort_values("decile")["actual_bad_rate"].values
    monotonic_non_decreasing = np.all(np.diff(risk_rates) >= -1e-12)
    print("\nRank ordering monotonic from decile 1 -> 10:", monotonic_non_decreasing)

    print("\nTop-risk-to-low-risk decile table:")
    print(deciles.to_string(index=False))

    # N. Final project summary JSON
    summary = {
        "project": "Bank GoodCredit PM-PR-0015",
        "target": "Bad_label",
        "best_model": best_name,
        "split_type": split_type,
        "benchmark_gini": 0.379,
        "final_test_metrics": {k: float(v) for k, v in final_metrics.items()},
        "rank_order_monotonic": bool(monotonic_non_decreasing),
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "n_test": int(len(X_test)),
        "n_model_features_before_encoding": int(X_train.shape[1]),
    }
    with open(OUTPUT_DIR / "project_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETED.")
    print(f"All files are inside: {OUTPUT_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()

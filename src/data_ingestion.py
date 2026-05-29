import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ── Load credentials from .env ────────────────────────────────
load_dotenv()

DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

RAW_DATA_PATH = "data/raw"
BLS_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# ── Database connection ───────────────────────────────────────
def get_engine():
    """Create and return a SQLAlchemy engine connected to ai_observatory."""
    connection_string = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(connection_string)


# ── BLS Column Standardization ────────────────────────────────
# BLS column names vary slightly across years. This maps known
# variations to the standard names our schema expects.
BLS_COLUMN_MAP = {
    "area"       : "area_code",
    "area_code"  : "area_code",
    "area_title" : "area_title",
    "area_type"  : "area_type",
    "occ_code"   : "occ_code",
    "occ_title"  : "occ_title",
    "occ_group"  : "occ_group",
    "tot_emp"    : "tot_emp",
    "h_mean"     : "h_mean",
    "a_mean"     : "a_mean",
    "h_median"   : "h_median",
    "a_median"   : "a_median",
    "h_pct50"    : "h_median",  # some years use pct50 instead of median
    "a_pct50"    : "a_median",
}

KEEP_COLUMNS = [
    "area_code", "area_title", "area_type",
    "occ_code", "occ_title", "occ_group",
    "tot_emp", "h_mean", "a_mean", "h_median", "a_median"
]


def standardize_bls_columns(df):
    """
    Lowercase all column names, apply the column map,
    and keep only the columns we need.
    """
    # Lowercase and strip whitespace from all column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename using our map (only renames columns that exist)
    df = df.rename(columns=BLS_COLUMN_MAP)

    # Keep only columns that exist in both our target list and the DataFrame
    cols_to_keep = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[cols_to_keep]

    # Add any missing target columns as NaN so all years have the same shape
    for col in KEEP_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df[KEEP_COLUMNS]


# ── Load BLS OEWS Files ───────────────────────────────────────
def load_bls_files(engine):
    """
    Load all BLS OEWS Excel files (MSA + National) for all years
    into the bls_oews table. Appends year by year.
    """
    print("\n" + "="*60)
    print("LOADING BLS OEWS DATA")
    print("="*60)

    # Clear existing data from table before reloading
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE bls_oews RESTART IDENTITY;"))
        conn.commit()

    total_rows = 0

    for year in BLS_YEARS:
        for data_type in ["msa", "national"]:
            filename = f"bls_oews_{year}_{data_type}.xlsx"
            filepath = os.path.join(RAW_DATA_PATH, filename)

            if not os.path.exists(filepath):
                print(f"  SKIPPING (not found): {filename}")
                continue

            print(f"  Loading {filename}...", end=" ")

            try:
                df = pd.read_excel(filepath, dtype=str)
                df = standardize_bls_columns(df)
                df["year"]      = year
                df["data_type"] = data_type

                # Write to PostgreSQL — append mode since we loop
                df.to_sql(
                    "bls_oews",
                    engine,
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=1000
                )

                print(f"OK — {len(df):,} rows loaded")
                total_rows += len(df)

            except Exception as e:
                print(f"ERROR — {e}")

    print(f"\n  ✓ BLS loading complete. Total rows inserted: {total_rows:,}")


# ── Load Frey & Osborne Automation Scores ────────────────────
def load_automation_scores(engine):
    """
    Load the Frey & Osborne automation probability CSV
    into the automation_scores table.
    """
    print("\n" + "="*60)
    print("LOADING AUTOMATION SCORES")
    print("="*60)

    filepath = os.path.join(RAW_DATA_PATH, "frey_osborne_automation.csv")
    print("  Loading frey_osborne_automation.csv...", end=" ")

    df = pd.read_csv(filepath, dtype=str)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Find the SOC code column by looking for values shaped like "15-1252"
    # (7 characters with a dash at position 2)
    occ_code_col = None
    for col in df.columns:
        sample_vals = df[col].dropna().head(10).tolist()
        if any(isinstance(v, str) and len(v) == 7 and v[2] == '-'
               for v in sample_vals):
            occ_code_col = col
            break

    # Fallback: grab any column with 'code' in the name
    if occ_code_col is None:
        code_cols = [c for c in df.columns if 'code' in c]
        occ_code_col = code_cols[0] if code_cols else None

    print(f"\n  SOC code column identified as: '{occ_code_col}'")

    # Build a clean dataframe directly from the known column names.
    # Use 'probability' (the full-precision column) and ignore 'prob'
    # so we never end up with duplicate column names after mapping.
    result = pd.DataFrame({
        'occ_code'       : df[occ_code_col] if occ_code_col else np.nan,
        'occ_title'      : df['occupation'] if 'occupation' in df.columns
                           else np.nan,
        'automation_prob': pd.to_numeric(df['probability'], errors='coerce'),
        'median_wage'    : pd.to_numeric(df['median_ann_wage'], errors='coerce'),
        'education_req'  : df['education'] if 'education' in df.columns
                           else np.nan,
    })

    # Drop rows with no automation score
    result = result.dropna(subset=['automation_prob'])

    print(f"\n  Preview of what will be loaded:")
    print(result.head(3).to_string())

    # Clear existing data and reload
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE automation_scores RESTART IDENTITY;"))
        conn.commit()

    result.to_sql("automation_scores", engine,
                  if_exists="append", index=False, method="multi")

    print(f"\n  OK — {len(result):,} rows loaded")

# ── Load O*NET Work Activities ────────────────────────────────
def load_onet_work_activities(engine):
    """
    Load the O*NET Work Activities Excel file into
    the onet_work_activities table.
    """
    print("\n" + "="*60)
    print("LOADING O*NET WORK ACTIVITIES")
    print("="*60)

    filepath = os.path.join(RAW_DATA_PATH, "onet_work_activities.xlsx")

    print("  Loading onet_work_activities.xlsx...", end=" ")
    df = pd.read_excel(filepath, dtype=str)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_").replace("*", "")
                  for c in df.columns]
    print(f"\n  Raw columns: {list(df.columns)}")

    # Standard O*NET column names
    col_map = {
        "onet-soc_code"           : "onet_soc_code",
        "onet_soc_code"           : "onet_soc_code",
        "onetsoc_code"            : "onet_soc_code",
        "title"                   : "title",
        "element_id"              : "element_id",
        "element_name"            : "element_name",
        "scale_id"                : "scale_id",
        "data_value"              : "data_value",
        "recommend_suppress"      : "recommend_suppress",
    }
    df = df.rename(columns=col_map)

    schema_cols = ["onet_soc_code", "title", "element_id",
                   "element_name", "scale_id", "data_value",
                   "recommend_suppress"]
    df = df[[c for c in schema_cols if c in df.columns]]

    if "data_value" in df.columns:
        df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")

    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE onet_work_activities RESTART IDENTITY;"))
        conn.commit()

    df.to_sql("onet_work_activities", engine,
              if_exists="append", index=False, method="multi", chunksize=1000)

    print(f"  OK — {len(df):,} rows loaded")


# ── Load O*NET Occupation Reference ──────────────────────────
def load_onet_occupations(engine):
    """
    Load the O*NET Occupation Data reference file into
    the onet_occupations table.
    """
    print("\n" + "="*60)
    print("LOADING O*NET OCCUPATION REFERENCE")
    print("="*60)

    filepath = os.path.join(RAW_DATA_PATH, "onet_occupation_data.xlsx")

    print("  Loading onet_occupation_data.xlsx...", end=" ")
    df = pd.read_excel(filepath, dtype=str)

    df.columns = [c.strip().lower().replace(" ", "_").replace("*", "")
                  for c in df.columns]

    col_map = {
        "onet-soc_code" : "onet_soc_code",
        "onet_soc_code" : "onet_soc_code",
        "onetsoc_code"  : "onet_soc_code",
        "title"         : "title",
        "description"   : "description",
    }
    df = df.rename(columns=col_map)

    schema_cols = ["onet_soc_code", "title", "description"]
    df = df[[c for c in schema_cols if c in df.columns]]

    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE onet_occupations RESTART IDENTITY;"))
        conn.commit()

    df.to_sql("onet_occupations", engine,
              if_exists="append", index=False, method="multi")

    print(f"  OK — {len(df):,} rows loaded")


# ── Row Count Verification ────────────────────────────────────
def verify_row_counts(engine):
    """
    Print row counts for every table so we can confirm
    everything loaded correctly.
    """
    print("\n" + "="*60)
    print("VERIFICATION — ROW COUNTS")
    print("="*60)

    tables = ["bls_oews", "automation_scores",
              "onet_work_activities", "onet_occupations"]

    with engine.connect() as conn:
        for table in tables:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            print(f"  {table:<30} {count:>10,} rows")

    print("\n✓ Verification complete.")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Connecting to database...")
    engine = get_engine()

    # Test the connection before doing any work
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ Database connection successful\n")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("Check your .env credentials and make sure PostgreSQL is running.")
        exit(1)

    load_bls_files(engine)
    load_automation_scores(engine)
    load_onet_work_activities(engine)
    load_onet_occupations(engine)
    verify_row_counts(engine)
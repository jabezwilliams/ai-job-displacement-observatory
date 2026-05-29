import pandas as pd
import os

# ── Configuration ──────────────────────────────────────────────────────────────
RAW_DATA_PATH = "data/raw"

BLS_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

FILES_TO_CHECK = {
    "frey_osborne": "frey_osborne_automation.csv",
    "onet_work_activities": "onet_work_activities.xlsx",
    "onet_occupation_data": "onet_occupation_data.xlsx",
}

# ── Helper Functions ───────────────────────────────────────────────────────────

def load_bls_file(year, file_type):
    """
    Load a BLS OEWS Excel file for a given year and type (msa or national).
    Returns a DataFrame with a 'year' column added.
    """
    filename = f"bls_oews_{year}_{file_type}.xlsx"
    filepath = os.path.join(RAW_DATA_PATH, filename)

    print(f"  Loading {filename}...", end=" ")
    df = pd.read_excel(filepath, dtype=str)  # load everything as string first
    df["year"] = year                         # tag the row with its survey year
    print(f"OK — {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_csv_or_excel(key, filename):
    """
    Load a CSV or Excel file from data/raw by filename.
    Returns a DataFrame.
    """
    filepath = os.path.join(RAW_DATA_PATH, filename)
    print(f"  Loading {filename}...", end=" ")

    if filename.endswith(".csv"):
        df = pd.read_excel(filepath, dtype=str) if False else pd.read_csv(filepath, dtype=str)
    else:
        df = pd.read_excel(filepath, dtype=str)

    print(f"OK — {len(df):,} rows, {len(df.columns)} columns")
    return df


# ── Main Verification ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("DATA VERIFICATION — AI Job Displacement Observatory")
    print("=" * 60)

    # --- Check BLS OEWS files ---
    print("\n[1] BLS OEWS Files (MSA level):")
    bls_msa_frames = []
    for year in BLS_YEARS:
        try:
            df = load_bls_file(year, "msa")
            bls_msa_frames.append(df)
        except FileNotFoundError as e:
            print(f"  MISSING: {e}")

    print("\n[2] BLS OEWS Files (National level):")
    bls_nat_frames = []
    for year in BLS_YEARS:
        try:
            df = load_bls_file(year, "national")
            bls_nat_frames.append(df)
        except FileNotFoundError as e:
            print(f"  MISSING: {e}")

    # --- Check supplementary files ---
    print("\n[3] Supplementary Files:")
    supplementary = {}
    for key, filename in FILES_TO_CHECK.items():
        try:
            df = load_csv_or_excel(key, filename)
            supplementary[key] = df
        except FileNotFoundError as e:
            print(f"  MISSING: {e}")

    # --- Print column previews ---
    print("\n" + "=" * 60)
    print("COLUMN PREVIEWS")
    print("=" * 60)

    if bls_msa_frames:
        print("\nBLS MSA 2023 columns:")
        print(list(bls_msa_frames[-1].columns))

    if "frey_osborne" in supplementary:
        print("\nFrey-Osborne columns:")
        print(list(supplementary["frey_osborne"].columns))

    if "onet_work_activities" in supplementary:
        print("\nO*NET Work Activities columns:")
        print(list(supplementary["onet_work_activities"].columns))

    print("\n✓ Verification complete.")
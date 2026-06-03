"""
src/feature_builder.py
AI Job Displacement Observatory — Feature Engineering Pipeline

Builds the AI Displacement Risk Index from four features:
  1. Frey-Osborne automation probability      (weight: 40%)
  2. O*NET AI activity exposure score         (weight: 25%)
  3. Employment trend risk (CAGR 2018-2025)   (weight: 20%)
  4. Wage stagnation risk vs. CPI             (weight: 15%)

Outputs:
  data/processed/occupation_risk_index.csv  — one row per occupation
  data/processed/risk_index.csv             — one row per occupation × metro area
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()

# SQL filter constants (occ_group column is NULL in DB; derive from SOC code pattern)
DETAILED_FILTER = "RIGHT(occ_code, 1) != '0'"
MAJOR_FILTER    = "RIGHT(occ_code, 4) = '0000' AND occ_code != '00-0000'"

# The four O*NET activities most associated with AI-automatable work
HIGH_AI_ACTIVITIES = [
    "Processing Information",
    "Analyzing Data or Information",
    "Documenting/Recording Information",
    "Interacting With Computers",
]

# Index component weights — must sum to 1.0
DEFAULT_WEIGHTS = {
    'automation_prob':      0.40,
    'ai_exposure_score':    0.25,
    'emp_trend_risk':       0.20,
    'wage_stagnation_risk': 0.15,
}

# U.S. CPI annual growth rate 2018-2025 (approx. 3.5% based on BLS CPI data)
# Used to determine whether real wages kept pace with inflation
CPI_CAGR = 0.035

# Fill value for occupations missing one or more feature scores (neutral risk)
NEUTRAL_FILL = 0.5

PROCESSED_PATH = "data/processed"


# ── Database Connection ────────────────────────────────────────────────────────
def get_engine():
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )


# ── Step 1: Occupation Master Table ──────────────────────────────────────────
def build_occupation_master(engine):
    """
    Returns all unique detailed BLS occupations from 2025 national data.
    This becomes the spine every feature table joins onto.
    """
    df = pd.read_sql(f"""
        SELECT DISTINCT occ_code, occ_title
        FROM bls_oews
        WHERE year = 2025
          AND data_type = 'national'
          AND {DETAILED_FILTER}
    """, engine)
    df['major_code'] = df['occ_code'].str[:2]
    print(f"  Occupation master: {len(df):,} unique occupations")
    return df


# ── Step 2: Automation Feature ────────────────────────────────────────────────
def build_automation_feature(engine, df_master):
    """
    Frey-Osborne automation probability, with major-group median imputation
    for the ~15-20% of occupations that have no matching score.

    Imputation strategy: for each unmatched occupation, assign the median
    automation probability of all matched occupations in the same 2-digit
    major SOC group. If the entire group is unmatched, fall back to the
    dataset-wide median.
    """
    df_auto = pd.read_sql(
        "SELECT occ_code, automation_prob, education_req FROM automation_scores",
        engine
    )

    df = df_master.merge(df_auto, on='occ_code', how='left')
    df['score_source'] = np.where(df['automation_prob'].isna(), 'imputed', 'actual')

    # Compute median per major group from actual (non-imputed) scores
    major_medians   = (
        df[df['score_source'] == 'actual']
        .groupby('major_code')['automation_prob']
        .median()
        .to_dict()
    )
    overall_median = df['automation_prob'].median()

    def impute(row):
        if pd.isna(row['automation_prob']):
            return major_medians.get(row['major_code'], overall_median)
        return row['automation_prob']

    df['automation_prob'] = df.apply(impute, axis=1)

    actual_n  = (df['score_source'] == 'actual').sum()
    imputed_n = (df['score_source'] == 'imputed').sum()
    print(f"  Automation scores: {actual_n} actual, {imputed_n} imputed "
          f"({imputed_n / len(df) * 100:.1f}% imputed)")
    return df[['occ_code', 'automation_prob', 'score_source', 'education_req']]


# ── Step 3: Employment Trend Feature ─────────────────────────────────────────
def build_employment_trend_feature(engine):
    """
    Computes the Compound Annual Growth Rate (CAGR) in employment from 2018
    to 2025 for each occupation, using national-level BLS data. 2020 is
    excluded from the endpoint calculation to avoid COVID distortion.

    Converts CAGR to a 0-1 risk score via percentile rank (inverted):
      - Fastest-declining occupation → risk ≈ 1.0
      - Fastest-growing occupation  → risk ≈ 0.0
    """
    df_emp = pd.read_sql(f"""
        SELECT year, occ_code, tot_emp
        FROM bls_oews
        WHERE data_type = 'national'
          AND {DETAILED_FILTER}
          AND year IN (2018, 2025)
          AND tot_emp IS NOT NULL
    """, engine)
    df_emp['tot_emp'] = pd.to_numeric(df_emp['tot_emp'], errors='coerce')
    df_emp = df_emp.dropna(subset=['tot_emp'])

    df_2018 = (df_emp[df_emp['year'] == 2018][['occ_code', 'tot_emp']]
               .rename(columns={'tot_emp': 'emp_2018'}))
    df_2025 = (df_emp[df_emp['year'] == 2025][['occ_code', 'tot_emp']]
               .rename(columns={'tot_emp': 'emp_2025'}))

    df = df_2018.merge(df_2025, on='occ_code', how='inner')
    df = df[(df['emp_2018'] > 0) & (df['emp_2025'] >= 0)]

    # CAGR: annualized growth over 7 years
    df['emp_cagr'] = (df['emp_2025'] / df['emp_2018']) ** (1 / 7) - 1

    # Percentile rank → invert so declining job = high risk
    df['emp_trend_risk'] = 1 - df['emp_cagr'].rank(pct=True)

    print(f"  Employment trend: {len(df):,} occupations scored")
    print(f"    CAGR range: {df['emp_cagr'].min():.1%}  to  {df['emp_cagr'].max():.1%}")
    return df[['occ_code', 'emp_cagr', 'emp_trend_risk']]


# ── Step 4: AI Exposure Feature ───────────────────────────────────────────────
def build_ai_exposure_feature(engine):
    """
    For each occupation, averages the O*NET Importance (IM) score across the
    four work activities most associated with AI-automatable tasks. Normalizes
    the result from the 1-5 IM scale to a 0-1 score.

    Higher score = more data/information-processing work = higher AI exposure.
    """
    activity_sql = ", ".join([f"'{a}'" for a in HIGH_AI_ACTIVITIES])

    df_onet = pd.read_sql(f"""
        SELECT onet_soc_code, element_name, data_value
        FROM onet_work_activities
        WHERE scale_id    = 'IM'
          AND element_name IN ({activity_sql})
          AND data_value IS NOT NULL
    """, engine)

    df = (
        df_onet
        .groupby('onet_soc_code')['data_value']
        .mean()
        .reset_index()
        .rename(columns={'data_value': 'ai_exposure_raw'})
    )
    # Normalize: IM scale is 1-5, so (score - 1) / 4 maps to 0-1
    df['ai_exposure_score'] = (df['ai_exposure_raw'] - 1) / 4
    df['occ_code'] = df['onet_soc_code'].str[:7]   # strip ".00" suffix

    print(f"  AI exposure: {len(df):,} occupations scored")
    print(f"    Score range: {df['ai_exposure_score'].min():.3f} – "
          f"{df['ai_exposure_score'].max():.3f}")
    return df[['occ_code', 'ai_exposure_score', 'ai_exposure_raw']]


# ── Step 5: Wage Stagnation Feature ───────────────────────────────────────────
def build_wage_stagnation_feature(engine):
    """
    Computes real wage growth (wage CAGR minus CPI CAGR) for each occupation
    from 2018 to 2025. Converts to a 0-1 risk score via percentile rank:
      - Strongest real wage growth → risk ≈ 0.0 (wages kept pace or exceeded CPI)
      - Weakest / negative real growth → risk ≈ 1.0 (wages lost ground to inflation)

    CPI_CAGR = 3.5% (approx. annualized U.S. CPI 2018-2025, BLS data)
    """
    df_wages = pd.read_sql(f"""
        SELECT year, occ_code, a_mean
        FROM bls_oews
        WHERE data_type = 'national'
          AND {DETAILED_FILTER}
          AND year IN (2018, 2025)
          AND a_mean IS NOT NULL
    """, engine)
    df_wages['a_mean'] = pd.to_numeric(df_wages['a_mean'], errors='coerce')
    df_wages = df_wages.dropna(subset=['a_mean'])

    df_w18 = (df_wages[df_wages['year'] == 2018][['occ_code', 'a_mean']]
              .rename(columns={'a_mean': 'wage_2018'}))
    df_w25 = (df_wages[df_wages['year'] == 2025][['occ_code', 'a_mean']]
              .rename(columns={'a_mean': 'wage_2025'}))

    df = df_w18.merge(df_w25, on='occ_code', how='inner')
    df = df[df['wage_2018'] > 0]

    df['wage_cagr']           = (df['wage_2025'] / df['wage_2018']) ** (1 / 7) - 1
    df['real_wage_growth']    = df['wage_cagr'] - CPI_CAGR
    df['wage_stagnation_risk'] = 1 - df['real_wage_growth'].rank(pct=True)

    below_cpi = (df['real_wage_growth'] < 0).sum()
    print(f"  Wage stagnation: {len(df):,} occupations scored")
    print(f"    {below_cpi} ({below_cpi / len(df) * 100:.0f}%) have below-CPI wage growth")
    return df[['occ_code', 'wage_cagr', 'real_wage_growth', 'wage_stagnation_risk']]


# ── Step 6: Composite Risk Index ─────────────────────────────────────────────
def build_risk_index(engine, weights=None):
    """
    Merges all four features, fills missing values with NEUTRAL_FILL (0.5),
    applies the weight vector, and returns the occupation-level risk index.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    print("\n" + "=" * 60)
    print("BUILDING AI DISPLACEMENT RISK INDEX")
    print("=" * 60)

    df_master = build_occupation_master(engine)
    df_auto   = build_automation_feature(engine, df_master)
    df_trend  = build_employment_trend_feature(engine)
    df_ai     = build_ai_exposure_feature(engine)
    df_wage   = build_wage_stagnation_feature(engine)

    df = df_master.copy()
    for feat_df, key in [(df_auto, 'automation'), (df_trend, 'trend'),
                          (df_ai, 'ai'), (df_wage, 'wage')]:
        df = df.merge(feat_df, on='occ_code', how='left')

    # Fill missing features with neutral risk
    for col in ['emp_trend_risk', 'ai_exposure_score', 'wage_stagnation_risk']:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  Imputing {missing} missing {col} values → {NEUTRAL_FILL}")
        df[col] = df[col].fillna(NEUTRAL_FILL)

    # Composite index (weighted sum of normalized 0-1 features)
    df['risk_index'] = (
        df['automation_prob']      * weights['automation_prob']      +
        df['ai_exposure_score']    * weights['ai_exposure_score']    +
        df['emp_trend_risk']       * weights['emp_trend_risk']       +
        df['wage_stagnation_risk'] * weights['wage_stagnation_risk']
    )

    print(f"\n  Occupation-level index built: {len(df):,} occupations")
    print(f"  Risk index  min: {df['risk_index'].min():.3f}")
    print(f"  Risk index mean: {df['risk_index'].mean():.3f}")
    print(f"  Risk index  max: {df['risk_index'].max():.3f}")
    return df


# ── Step 7: Metro-Level Scoring ───────────────────────────────────────────────
def score_metro_level(engine, df_occ_index):
    """
    Joins the occupation-level risk index to MSA 2025 employment data.
    Output: one row per (occupation, metro area) with risk index + components.
    """
    df_msa = pd.read_sql(f"""
        SELECT occ_code, area_code, area_title, tot_emp
        FROM bls_oews
        WHERE year = 2025
          AND data_type = 'msa'
          AND {DETAILED_FILTER}
          AND tot_emp IS NOT NULL
    """, engine)
    df_msa['tot_emp'] = pd.to_numeric(df_msa['tot_emp'], errors='coerce')
    df_msa = df_msa.dropna(subset=['tot_emp'])

    export_cols = [
        'occ_code', 'occ_title', 'risk_index',
        'automation_prob', 'ai_exposure_score',
        'emp_trend_risk', 'wage_stagnation_risk',
        'score_source', 'education_req',
    ]
    df_output = df_msa.merge(df_occ_index[export_cols], on='occ_code', how='inner')

    print(f"\n  Metro-level output: {len(df_output):,} rows")
    print(f"  Metro areas covered: {df_output['area_code'].nunique():,}")
    return df_output


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run_pipeline():
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    engine = get_engine()

    df_occ   = build_risk_index(engine)
    df_metro = score_metro_level(engine, df_occ)

    occ_path   = os.path.join(PROCESSED_PATH, "occupation_risk_index.csv")
    metro_path = os.path.join(PROCESSED_PATH, "risk_index.csv")

    df_occ.to_csv(occ_path,   index=False)
    df_metro.to_csv(metro_path, index=False)

    print(f"\n✓ Saved: {occ_path}    ({len(df_occ):,} rows)")
    print(f"✓ Saved: {metro_path}  ({len(df_metro):,} rows)")
    print("\n✓ Feature engineering pipeline complete.")
    return df_occ, df_metro


if __name__ == "__main__":
    run_pipeline()
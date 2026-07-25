"""
src/forecasting.py
AI Job Displacement Observatory — ML & Forecasting Pipeline

Two models:
  1. Prophet: employment forecasts per occupation through 2027
  2. XGBoost: predict automation probability from O*NET activity scores

Outputs:
  data/processed/employment_forecasts.csv   — Prophet forecasts with confidence intervals
  data/processed/xgb_automation_predictions.csv — XGBoost predicted automation scores
"""

import os
import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()

DETAILED_FILTER   = "RIGHT(occ_code, 1) != '0'"
TRAIN_YEARS       = list(range(2018, 2024))     # 2018–2023 (inclusive)
TEST_YEARS        = [2024, 2025]
FORECAST_YEARS    = [2026, 2027]
TOP_N_OCCUPATIONS = 20
XGB_TEST_SIZE     = 0.20
RANDOM_STATE      = 42
PROCESSED_PATH    = "data/processed"


# ── Database Connection ────────────────────────────────────────────────────────
def get_engine():
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )


# ── Section 1: Prophet — Employment Forecasting ───────────────────────────────

def load_employment_timeseries(engine):
    """Load national employment for all years, detailed occupations."""
    df = pd.read_sql(f"""
        SELECT year, occ_code, occ_title, tot_emp
        FROM bls_oews
        WHERE data_type = 'national'
          AND {DETAILED_FILTER}
          AND tot_emp IS NOT NULL
        ORDER BY occ_code, year
    """, engine)
    df['tot_emp'] = pd.to_numeric(df['tot_emp'], errors='coerce')
    df = df.dropna(subset=['tot_emp'])
    return df


def get_top_occupations(df_emp, n=TOP_N_OCCUPATIONS):
    """Return the top N occupations by 2025 national employment."""
    df_2025 = (
        df_emp[df_emp['year'] == 2025]
        .nlargest(n, 'tot_emp')[['occ_code', 'occ_title', 'tot_emp']]
        .reset_index(drop=True)
    )
    print(f"  Top {n} occupations selected for forecasting")
    return df_2025


def prepare_prophet_df(df_emp, occ_code):
    """
    Format employment time series for a single occupation into Prophet's
    required (ds, y) format. Uses July 1 as the annual date anchor since
    BLS OEWS surveys are conducted in the spring/early summer.
    """
    df_occ = df_emp[df_emp['occ_code'] == occ_code].copy()
    df_occ['ds'] = pd.to_datetime(df_occ['year'].astype(str) + '-07-01')
    df_occ['y']  = df_occ['tot_emp']
    return df_occ[['ds', 'y']].sort_values('ds').reset_index(drop=True)


def fit_and_forecast(df_prophet, occ_code, occ_title):
    """
    Fit Prophet on training years (2018-2023), then forecast through 2027.
    2020 is flagged as an anomaly via Prophet's holidays mechanism so the
    COVID employment shock doesn't distort the trend estimate.

    Returns: (forecast_df, rmse, mape) where forecast covers all years
    including holdout (2024-2025) and future (2026-2027).
    """
    # COVID anomaly flag
    covid = pd.DataFrame({
        'holiday': 'covid_anomaly',
        'ds': pd.to_datetime(['2020-07-01']),
        'lower_window': 0,
        'upper_window': 0,
    })

    # Training set only: 2018-2023
    df_train = df_prophet[df_prophet['ds'].dt.year.isin(TRAIN_YEARS)]

    model = Prophet(
        holidays             = covid,
        yearly_seasonality   = False,   # annual data can't detect annual seasonality
        weekly_seasonality   = False,
        daily_seasonality    = False,
        changepoint_prior_scale = 0.1,  # conservative — prevents overfit on 6 points
        uncertainty_samples  = 500,
    )
    model.fit(df_train)

    # Build full date range: all known years + forecast years
    all_years = (
        list(df_prophet['ds'].dt.year.unique())
        + [y for y in FORECAST_YEARS if y not in df_prophet['ds'].dt.year.values]
    )
    future = pd.DataFrame({
        'ds': pd.to_datetime([f'{y}-07-01' for y in sorted(set(all_years))])
    })
    forecast = model.predict(future)

    # Evaluate on holdout (2024-2025)
    df_test   = df_prophet[df_prophet['ds'].dt.year.isin(TEST_YEARS)]
    df_pred   = forecast[forecast['ds'].dt.year.isin(TEST_YEARS)][['ds', 'yhat']]
    df_eval   = df_test.merge(df_pred, on='ds', how='inner')

    rmse, mape = np.nan, np.nan
    if len(df_eval) > 0:
        rmse = np.sqrt(((df_eval['y'] - df_eval['yhat']) ** 2).mean())
        mape = (np.abs(df_eval['y'] - df_eval['yhat']) / df_eval['y']).mean() * 100

    # Attach occupation info
    forecast['occ_code']  = occ_code
    forecast['occ_title'] = occ_title
    return forecast[['occ_code', 'occ_title', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']], rmse, mape


def run_prophet_pipeline(engine):
    """Forecast employment for top N occupations and return combined results."""
    print("\n" + "="*60)
    print("PROPHET — EMPLOYMENT FORECASTING")
    print("="*60)

    df_emp     = load_employment_timeseries(engine)
    top_occs   = get_top_occupations(df_emp)
    all_forecasts, evaluation_rows = [], []

    for _, row in top_occs.iterrows():
        occ_code, occ_title = row['occ_code'], row['occ_title']
        df_prophet = prepare_prophet_df(df_emp, occ_code)

        if len(df_prophet) < 4:
            print(f"  SKIP {occ_title[:40]} — insufficient data points")
            continue

        forecast, rmse, mape = fit_and_forecast(df_prophet, occ_code, occ_title)
        all_forecasts.append(forecast)
        evaluation_rows.append({'occ_code': occ_code, 'occ_title': occ_title,
                                  'rmse': rmse, 'mape_pct': mape})
        print(f"  ✓ {occ_title[:45]:<45}  RMSE: {rmse:>10,.0f}  MAPE: {mape:.1f}%")

    df_forecasts   = pd.concat(all_forecasts, ignore_index=True)
    df_evaluation  = pd.DataFrame(evaluation_rows)
    return df_forecasts, df_evaluation


# ── Section 2: XGBoost — Automation Score Prediction ─────────────────────────

def build_onet_feature_matrix(engine):
    """
    Pivot O*NET work activity importance scores into a wide feature matrix:
    rows = occupations, columns = work activities (~40 features).
    """
    df_onet = pd.read_sql("""
        SELECT onet_soc_code, element_name, data_value
        FROM onet_work_activities
        WHERE scale_id = 'IM' AND data_value IS NOT NULL
    """, engine)

    df_pivot = df_onet.pivot_table(
        index   = 'onet_soc_code',
        columns = 'element_name',
        values  = 'data_value',
        aggfunc = 'mean'
    ).reset_index()

    # Fill missing activity scores with column median
    activity_cols = [c for c in df_pivot.columns if c != 'onet_soc_code']
    for col in activity_cols:
        df_pivot[col] = df_pivot[col].fillna(df_pivot[col].median())

    df_pivot['occ_code'] = df_pivot['onet_soc_code'].str[:7]
    print(f"  O*NET feature matrix: {len(df_pivot):,} occupations × {len(activity_cols)} activities")
    return df_pivot, activity_cols


def train_xgboost(engine):
    """
    Train XGBoost to predict automation probability from O*NET activity scores.
    Returns the trained model, evaluation metrics, and feature importances.
    """
    print("\n" + "="*60)
    print("XGBOOST — AUTOMATION SCORE PREDICTION")
    print("="*60)

    df_pivot, activity_cols = build_onet_feature_matrix(engine)

    df_auto = pd.read_sql(
        "SELECT occ_code, automation_prob FROM automation_scores", engine
    )

    df_ml = df_pivot.merge(df_auto, on='occ_code', how='inner')
    print(f"  Training samples (matched occupations): {len(df_ml):,}")

    X = df_ml[activity_cols].values
    y = df_ml['automation_prob'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=XGB_TEST_SIZE, random_state=RANDOM_STATE
    )

    model = XGBRegressor(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        random_state      = RANDOM_STATE,
        verbosity         = 0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    print(f"  Test RMSE:  {rmse:.4f}")
    print(f"  Test R²:    {r2:.4f}")

    importances = pd.Series(model.feature_importances_, index=activity_cols)
    return model, activity_cols, df_ml, X_test, y_test, y_pred, importances, rmse, r2


def predict_unmatched(engine, model, df_pivot, activity_cols):
    """Apply trained XGBoost to occupations without Frey-Osborne scores."""
    df_all = pd.read_sql(f"""
        SELECT DISTINCT occ_code, occ_title
        FROM bls_oews
        WHERE year = 2025 AND data_type = 'national'
          AND {DETAILED_FILTER}
    """, engine)

    df_scored = pd.read_sql("SELECT occ_code FROM automation_scores", engine)
    df_unmatched = df_all[~df_all['occ_code'].isin(df_scored['occ_code'])]

    df_features = df_pivot[df_pivot['occ_code'].isin(df_unmatched['occ_code'])].copy()
    if len(df_features) == 0:
        print("  No unmatched occupations found in O*NET — nothing to predict.")
        return pd.DataFrame()

    X_new = df_features[activity_cols].fillna(0).values
    df_features['predicted_automation_prob'] = model.predict(X_new)
    df_features['prediction_source']         = 'xgboost'

    result = df_features[['occ_code', 'predicted_automation_prob', 'prediction_source']]
    print(f"  Predicted scores for {len(result):,} previously unmatched occupations")
    return result


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run_pipeline():
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    engine = get_engine()

    # Prophet
    df_forecasts, df_evaluation = run_prophet_pipeline(engine)
    forecast_path = os.path.join(PROCESSED_PATH, "employment_forecasts.csv")
    df_forecasts.to_csv(forecast_path, index=False)
    print(f"\n✓ Saved: {forecast_path}  ({len(df_forecasts):,} rows)")

    # XGBoost
    model, activity_cols, df_ml, X_test, y_test, y_pred, importances, rmse, r2 = (
        train_xgboost(engine)
    )
    df_pivot, _ = build_onet_feature_matrix(engine)
    df_unmatched_preds = predict_unmatched(engine, model, df_pivot, activity_cols)

    pred_path = os.path.join(PROCESSED_PATH, "xgb_automation_predictions.csv")
    df_unmatched_preds.to_csv(pred_path, index=False)
    print(f"✓ Saved: {pred_path}  ({len(df_unmatched_preds):,} rows)")

    print("\n✓ Part 6 pipeline complete.")
    return df_forecasts, df_evaluation, model, importances, rmse, r2


if __name__ == "__main__":
    run_pipeline()
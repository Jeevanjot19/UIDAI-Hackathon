"""
Generate Time-Series Forecasts for Dashboard
Creates ARIMA and Prophet forecasts for update volumes
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATING TIME-SERIES FORECASTS")
print("="*80)

# Load data
print("\n[1/6] Loading data...")
df = pd.read_csv('data/processed/aadhaar_with_indices.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
print(f"✅ Loaded {len(df):,} records")

# Aggregate to monthly national level
print("\n[2/6] Aggregating to monthly time series...")
ts_monthly = df.groupby('date').agg({
    'total_enrolments': 'sum',
    'total_updates': 'sum',
    'rolling_3m_updates': 'sum'
}).reset_index()
print(f"✅ {len(ts_monthly)} monthly data points")

# ARIMA Forecasting
print("\n[3/6] Fitting ARIMA model...")
# Use total_updates as target
endog = ts_monthly['total_updates'].values

# Fit ARIMA(2,1,2) - based on typical monthly patterns
model_arima = ARIMA(endog, order=(2, 1, 2))
fitted_arima = model_arima.fit()

# Forecast 6 months ahead
forecast_steps = 6
forecast_arima = fitted_arima.forecast(steps=forecast_steps)
forecast_se = fitted_arima.forecast(steps=forecast_steps, return_conf_int=False)

# Create forecast dates
last_date = ts_monthly['date'].max()
forecast_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')

arima_forecast_df = pd.DataFrame({
    'date': forecast_dates,
    'forecast': forecast_arima,
    'model': 'ARIMA'
})
print(f"✅ ARIMA forecast: {forecast_steps} months ahead")

# Prophet Forecasting
print("\n[4/6] Fitting Prophet model...")
# Prepare data for Prophet
prophet_df = ts_monthly[['date', 'total_updates']].copy()
prophet_df.columns = ['ds', 'y']

# Fit Prophet
model_prophet = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05
)
model_prophet.fit(prophet_df)

# Make future dataframe
future = model_prophet.make_future_dataframe(periods=6, freq='MS')
prophet_forecast = model_prophet.predict(future)

# Extract forecast for future dates only
prophet_forecast_df = prophet_forecast[prophet_forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
prophet_forecast_df.columns = ['date', 'forecast', 'lower', 'upper']
prophet_forecast_df['model'] = 'Prophet'
print(f"✅ Prophet forecast: {len(prophet_forecast_df)} months ahead")

# Combine historical and forecast
print("\n[5/6] Creating combined dataset...")
historical_df = ts_monthly[['date', 'total_updates']].copy()
historical_df['type'] = 'Historical'

combined_arima = pd.concat([
    historical_df.rename(columns={'total_updates': 'value'}),
    arima_forecast_df[['date', 'forecast']].rename(columns={'forecast': 'value'}).assign(type='ARIMA Forecast')
], ignore_index=True)

combined_prophet = pd.concat([
    historical_df.rename(columns={'total_updates': 'value'}),
    prophet_forecast_df[['date', 'forecast']].rename(columns={'forecast': 'value'}).assign(type='Prophet Forecast')
], ignore_index=True)

# Save results
print("\n[6/6] Saving forecasts...")
import os
os.makedirs('outputs/forecasts', exist_ok=True)

arima_forecast_df.to_csv('outputs/forecasts/arima_6m_forecast.csv', index=False)
prophet_forecast_df.to_csv('outputs/forecasts/prophet_6m_forecast.csv', index=False)
ts_monthly.to_csv('outputs/forecasts/historical_monthly.csv', index=False)
combined_arima.to_csv('outputs/forecasts/combined_arima.csv', index=False)
combined_prophet.to_csv('outputs/forecasts/combined_prophet.csv', index=False)

print("✅ Saved: outputs/forecasts/arima_6m_forecast.csv")
print("✅ Saved: outputs/forecasts/prophet_6m_forecast.csv")
print("✅ Saved: outputs/forecasts/historical_monthly.csv")
print("✅ Saved: outputs/forecasts/combined_arima.csv")
print("✅ Saved: outputs/forecasts/combined_prophet.csv")

# Summary
print("\n" + "="*80)
print("FORECAST SUMMARY")
print("="*80)
print(f"\nHistorical Period: {ts_monthly['date'].min().strftime('%Y-%m')} to {ts_monthly['date'].max().strftime('%Y-%m')}")
print(f"Forecast Horizon: 6 months ({forecast_dates[0].strftime('%Y-%m')} to {forecast_dates[-1].strftime('%Y-%m')})")
print(f"\nARIMA Forecast (6 months):")
print(arima_forecast_df[['date', 'forecast']].to_string(index=False))
print(f"\nProphet Forecast (6 months):")
print(prophet_forecast_df[['date', 'forecast']].to_string(index=False))
print(f"\nAverage Monthly Updates (Historical): {ts_monthly['total_updates'].mean():,.0f}")
print(f"Average Monthly Updates (ARIMA Forecast): {arima_forecast_df['forecast'].mean():,.0f}")
print(f"Average Monthly Updates (Prophet Forecast): {prophet_forecast_df['forecast'].mean():,.0f}")

print("\n✅ FORECASTS READY FOR DASHBOARD!")

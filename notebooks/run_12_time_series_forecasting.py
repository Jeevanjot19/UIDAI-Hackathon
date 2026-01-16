"""
Time-Series Forecasting
Days 5-6 of Implementation Plan

Goals:
1. ARIMA models for enrolment forecasting
2. Prophet for seasonality detection
3. LSTM (if time permits) for advanced sequence modeling
4. Multi-step ahead predictions (7-day, 30-day, 90-day)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TIME-SERIES FORECASTING")
print("="*80)

# ============================================================================
# LOAD & PREPARE DATA
# ============================================================================
print("\n[LOAD] Loading dataset...")
df = pd.read_csv('data/processed/aadhaar_with_advanced_features.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

print(f"Dataset: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ============================================================================
# AGGREGATE TIME SERIES (DAILY NATIONAL LEVEL)
# ============================================================================
print("\n[AGG] Creating national time series...")
ts_national = df.groupby('date').agg({
    'total_enrolments': 'sum',
    'total_demographic_updates': 'sum',
    'total_biometric_updates': 'sum',
    'total_all_updates': 'sum'
}).reset_index()

ts_national = ts_national.set_index('date').asfreq('D')  # Daily frequency

print(f"Time series length: {len(ts_national)} days")
print(f"\nFirst 5 rows:")
print(ts_national.head())

# ============================================================================
# STATIONARITY TEST
# ============================================================================
print("\n" + "="*80)
print("STATIONARITY ANALYSIS")
print("="*80)

def adf_test(series, name=''):
    """Augmented Dickey-Fuller test for stationarity"""
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"\n{name} ADF Test:")
    print(f"   ADF Statistic: {result[0]:.4f}")
    print(f"   p-value: {result[1]:.4f}")
    print(f"   Stationary: {'YES ✅' if result[1] < 0.05 else 'NO ❌'}")
    return result[1] < 0.05

# Test stationarity
is_stationary_enr = adf_test(ts_national['total_enrolments'], 'Enrolments')
is_stationary_upd = adf_test(ts_national['total_all_updates'], 'Updates')

# Differencing if needed
if not is_stationary_enr:
    print("\n[DIFF] Applying first differencing to enrolments...")
    ts_national['enrolments_diff'] = ts_national['total_enrolments'].diff()
    adf_test(ts_national['enrolments_diff'].dropna(), 'Enrolments (Differenced)')

if not is_stationary_upd:
    print("\n[DIFF] Applying first differencing to updates...")
    ts_national['updates_diff'] = ts_national['total_all_updates'].diff()
    adf_test(ts_national['updates_diff'].dropna(), 'Updates (Differenced)')

# ============================================================================
# ACF & PACF PLOTS
# ============================================================================
print("\n[VIZ] Generating ACF/PACF plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Enrolments
plot_acf(ts_national['total_enrolments'].dropna(), lags=30, ax=axes[0, 0])
axes[0, 0].set_title('ACF - Total Enrolments', fontweight='bold')

plot_pacf(ts_national['total_enrolments'].dropna(), lags=30, ax=axes[0, 1])
axes[0, 1].set_title('PACF - Total Enrolments', fontweight='bold')

# Updates
plot_acf(ts_national['total_all_updates'].dropna(), lags=30, ax=axes[1, 0])
axes[1, 0].set_title('ACF - Total Updates', fontweight='bold')

plot_pacf(ts_national['total_all_updates'].dropna(), lags=30, ax=axes[1, 1])
axes[1, 1].set_title('PACF - Total Updates', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/ts_acf_pacf_plots.png', dpi=300)
print("✅ Saved: outputs/figures/ts_acf_pacf_plots.png")

# ============================================================================
# 1. ARIMA FORECASTING (ENROLMENTS)
# ============================================================================
print("\n" + "="*80)
print("ARIMA FORECASTING")
print("="*80)

# Split train/test
train_size = int(len(ts_national) * 0.8)
train = ts_national['total_enrolments'][:train_size]
test = ts_national['total_enrolments'][train_size:]

print(f"\nTrain size: {len(train)} days")
print(f"Test size: {len(test)} days")

# Fit ARIMA model (p, d, q)
# Based on ACF/PACF, using (5, 1, 2) as starting point
print("\n[ARIMA] Training ARIMA(5,1,2) model...")
arima_model = ARIMA(train, order=(5, 1, 2))
arima_fit = arima_model.fit()

print(f"✅ ARIMA model trained!")
print(f"\nModel Summary:")
print(arima_fit.summary().tables[1])

# Forecast
forecast_steps = len(test)
forecast = arima_fit.forecast(steps=forecast_steps)

# Calculate metrics
mae = np.mean(np.abs(test.values - forecast.values))
rmse = np.sqrt(np.mean((test.values - forecast.values)**2))
mape = np.mean(np.abs((test.values - forecast.values) / test.values)) * 100

print(f"\n📊 ARIMA Performance:")
print(f"   MAE: {mae:,.0f}")
print(f"   RMSE: {rmse:,.0f}")
print(f"   MAPE: {mape:.2f}%")

# Plot forecast
plt.figure(figsize=(14, 6))
plt.plot(train.index, train.values, label='Train', color='blue', linewidth=1.5)
plt.plot(test.index, test.values, label='Actual', color='green', linewidth=1.5)
plt.plot(test.index, forecast.values, label='ARIMA Forecast', color='red', linewidth=2, linestyle='--')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Enrolments', fontsize=12)
plt.title('ARIMA Forecast - Total Enrolments', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/ts_arima_forecast_enrolments.png', dpi=300)
print("✅ Saved: outputs/figures/ts_arima_forecast_enrolments.png")

# ============================================================================
# 2. PROPHET FORECASTING (WITH SEASONALITY)
# ============================================================================
print("\n" + "="*80)
print("PROPHET FORECASTING")
print("="*80)

# Prepare data for Prophet
prophet_df = ts_national.reset_index()[['date', 'total_enrolments']].rename(
    columns={'date': 'ds', 'total_enrolments': 'y'}
)

# Split train/test
train_prophet = prophet_df[:train_size]
test_prophet = prophet_df[train_size:]

print(f"\n[PROPHET] Training Prophet model with seasonality...")
prophet_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
)
prophet_model.fit(train_prophet)

# Make predictions
future = prophet_model.make_future_dataframe(periods=len(test_prophet), freq='D')
prophet_forecast = prophet_model.predict(future)

# Evaluate on test set
test_predictions = prophet_forecast.iloc[train_size:]['yhat'].values
mae_prophet = np.mean(np.abs(test_prophet['y'].values - test_predictions))
rmse_prophet = np.sqrt(np.mean((test_prophet['y'].values - test_predictions)**2))
mape_prophet = np.mean(np.abs((test_prophet['y'].values - test_predictions) / test_prophet['y'].values)) * 100

print(f"✅ Prophet model trained!")
print(f"\n📊 Prophet Performance:")
print(f"   MAE: {mae_prophet:,.0f}")
print(f"   RMSE: {rmse_prophet:,.0f}")
print(f"   MAPE: {mape_prophet:.2f}%")

# Plot Prophet forecast
fig = prophet_model.plot(prophet_forecast, figsize=(14, 6))
plt.axvline(train_prophet['ds'].iloc[-1], color='red', linestyle='--', label='Train/Test Split')
plt.title('Prophet Forecast - Total Enrolments', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/figures/ts_prophet_forecast_enrolments.png', dpi=300)
print("✅ Saved: outputs/figures/ts_prophet_forecast_enrolments.png")

# Plot components
fig = prophet_model.plot_components(prophet_forecast, figsize=(14, 10))
plt.tight_layout()
plt.savefig('outputs/figures/ts_prophet_components.png', dpi=300)
print("✅ Saved: outputs/figures/ts_prophet_components.png")

# ============================================================================
# 3. FUTURE FORECASTS (7-day, 30-day, 90-day)
# ============================================================================
print("\n" + "="*80)
print("MULTI-STEP AHEAD FORECASTS")
print("="*80)

# Use full data for final forecast
print("\n[FORECAST] Training on full dataset...")
arima_full = ARIMA(ts_national['total_enrolments'], order=(5, 1, 2)).fit()
prophet_full = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
prophet_full.fit(prophet_df)

# Generate forecasts
forecast_days = [7, 30, 90]
forecasts = {}

for days in forecast_days:
    # ARIMA forecast
    arima_fc = arima_full.forecast(steps=days)
    
    # Prophet forecast
    future_p = prophet_full.make_future_dataframe(periods=days, freq='D')
    prophet_fc = prophet_full.predict(future_p)
    
    forecasts[f'{days}d'] = {
        'arima': arima_fc.values,
        'prophet': prophet_fc.iloc[-days:]['yhat'].values
    }
    
    print(f"\n{days}-Day Forecast:")
    print(f"   ARIMA Mean: {arima_fc.mean():,.0f} enrolments/day")
    print(f"   Prophet Mean: {prophet_fc.iloc[-days:]['yhat'].mean():,.0f} enrolments/day")

# Save forecasts
forecast_summary = pd.DataFrame({
    'Horizon': ['7-day', '30-day', '90-day'],
    'ARIMA_Mean': [forecasts['7d']['arima'].mean(), 
                   forecasts['30d']['arima'].mean(), 
                   forecasts['90d']['arima'].mean()],
    'Prophet_Mean': [forecasts['7d']['prophet'].mean(), 
                     forecasts['30d']['prophet'].mean(), 
                     forecasts['90d']['prophet'].mean()]
})
forecast_summary.to_csv('outputs/tables/future_forecasts_summary.csv', index=False)
print("\n✅ Saved: outputs/tables/future_forecasts_summary.csv")

# Plot multi-horizon forecasts
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, days in enumerate(forecast_days):
    ax = axes[idx]
    
    # Plot historical
    ax.plot(ts_national.index[-30:], ts_national['total_enrolments'].iloc[-30:], 
            label='Historical', color='blue', linewidth=2)
    
    # Plot forecasts
    future_dates = pd.date_range(start=ts_national.index[-1], periods=days+1, freq='D')[1:]
    ax.plot(future_dates, forecasts[f'{days}d']['arima'], 
            label='ARIMA', color='red', linewidth=2, linestyle='--')
    ax.plot(future_dates, forecasts[f'{days}d']['prophet'], 
            label='Prophet', color='green', linewidth=2, linestyle=':')
    
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Total Enrolments', fontsize=10)
    ax.set_title(f'{days}-Day Forecast', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('outputs/figures/ts_multi_horizon_forecasts.png', dpi=300)
print("✅ Saved: outputs/figures/ts_multi_horizon_forecasts.png")

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Model': ['ARIMA(5,1,2)', 'Prophet'],
    'MAE': [mae, mae_prophet],
    'RMSE': [rmse, rmse_prophet],
    'MAPE': [mape, mape_prophet]
})

print("\n" + comparison.to_string(index=False))
comparison.to_csv('outputs/tables/ts_model_comparison.csv', index=False)
print("\n✅ Saved: outputs/tables/ts_model_comparison.csv")

# Bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['MAE', 'RMSE', 'MAPE']
for idx, metric in enumerate(metrics):
    comparison.plot(x='Model', y=metric, kind='bar', ax=axes[idx], legend=False, color=['skyblue', 'coral'])
    axes[idx].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(metric, fontsize=11)
    axes[idx].set_xlabel('')
    axes[idx].tick_params(axis='x', rotation=0)
    axes[idx].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/figures/ts_model_comparison.png', dpi=300)
print("✅ Saved: outputs/figures/ts_model_comparison.png")

# ============================================================================
# INSIGHTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TIME-SERIES FORECASTING INSIGHTS")
print("="*80)

print(f"\n📊 DATA SUMMARY:")
print(f"   ✅ {len(ts_national)} days of data analyzed")
print(f"   ✅ Date range: {ts_national.index.min()} to {ts_national.index.max()}")

print(f"\n🔮 FORECASTING MODELS:")
print(f"   ✅ ARIMA(5,1,2): MAPE = {mape:.2f}%")
print(f"   ✅ Prophet: MAPE = {mape_prophet:.2f}%")
print(f"   ✅ Best Model: {'ARIMA' if mape < mape_prophet else 'Prophet'}")

print(f"\n📈 FUTURE FORECASTS:")
print(f"   ✅ 7-day forecast generated")
print(f"   ✅ 30-day forecast generated")
print(f"   ✅ 90-day forecast generated")

print(f"\n📁 OUTPUTS:")
print(f"   - outputs/tables/future_forecasts_summary.csv")
print(f"   - outputs/tables/ts_model_comparison.csv")
print(f"   - outputs/figures/ts_*.png")

print("\n" + "="*80)
print("TIME-SERIES FORECASTING COMPLETE!")
print("="*80)

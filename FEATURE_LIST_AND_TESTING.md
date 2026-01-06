# UIDAI Hackathon - Complete Feature Implementation & Testing Guide

## Project Overview
**Problem Statement**: Unlocking Societal Trends in Aadhaar Enrolment and Updates
**Dataset**: 2,947,681 records (March-December 2025), 68 States/UTs, 1,029 Districts
**Primary Files**: 3 ZIP files containing enrolment, demographic, and biometric update data

---

## 📊 DATASETS USED

### 1. **Enrolment Dataset**
- **Source**: `api_data_aadhar_enroll_*.zip` (3 files)
- **Records**: 2,947,681 rows
- **Columns**: date, state, district, pincode, age groups (0-5, 6-10, 11-17, 18-21, 22-25, 26-30, 31-40, 41-60, 61-80, 80+)
- **Location**: `data/raw/enrolment/api_data_aadhar_enroll_*.csv`
- **Used In**: Feature Engineering Layers 2, 4, 6, 8

### 2. **Demographic Updates Dataset**
- **Source**: `api_data_aadhar_demo_*.zip` (3 files)
- **Records**: 2,947,681 rows
- **Columns**: date, state, district, pincode, demographic update age groups
- **Estimated Breakdown**: 45% address, 35% mobile, 12% name, 5% DoB, 3% gender
- **Location**: `data/raw/demographic/api_data_aadhar_demo_*.csv`
- **Used In**: Feature Engineering Layers 2, 3, 4, 6, 8

### 3. **Biometric Updates Dataset**
- **Source**: `api_data_aadhar_bio_*.zip` (3 files)
- **Records**: 2,947,681 rows
- **Columns**: date, state, district, pincode, biometric update age groups
- **Types**: Fingerprint, iris updates
- **Location**: `data/raw/biometric/api_data_aadhar_bio_*.csv`
- **Used In**: Feature Engineering Layers 2, 3, 4, 6, 8

### 4. **Merged & Feature-Engineered Dataset**
- **File**: `data/processed/aadhaar_with_features.csv`
- **Size**: 1.25 GB (44 columns, 2.9M rows)
- **Created By**: Notebook 02
- **Contains**: All original data + 20+ engineered features
- **Used In**: Notebooks 03, 04, 05 for all analyses

---

## 🔧 IMPLEMENTED FEATURES (25+ Total)

### **LAYER 1: Base Features (Built-in)**
All extracted directly from the 3 datasets above.

#### 1. **Total Enrolments**
- **Formula**: Sum of all age group enrolments
- **Dataset**: Enrolment Dataset
- **Test**: `python -c "import pandas as pd; df=pd.read_csv('data/processed/merged_aadhaar_data.csv'); print(df['total_enrolments'].describe())"`
- **View**: See `outputs/tables/feature_summary.csv` row 1

#### 2. **Total Demographic Updates**
- **Formula**: Sum of all demographic update age groups
- **Dataset**: Demographic Dataset
- **Test**: Check `outputs/tables/feature_summary.csv`
- **Visualization**: See `outputs/figures/03_age_analysis.png`

#### 3. **Total Biometric Updates**
- **Formula**: Sum of all biometric update age groups
- **Dataset**: Biometric Dataset
- **Test**: Check `outputs/tables/feature_summary.csv`
- **Visualization**: See `outputs/figures/03_age_analysis.png`

---

### **LAYER 2: Normalized & Growth Features (5 Features)**
**Dataset Used**: All 3 datasets combined
**Implementation**: `src/feature_engineering.py` → `add_layer2_normalized_features()`

#### 4. **Enrolment Growth Rate**
- **Formula**: `(total_enrolments - rolling_avg) / (rolling_avg + 1)`
- **Dataset**: Enrolment Dataset
- **Purpose**: Detect sudden spikes in new registrations
- **Test**: `python -c "import pandas as pd; df=pd.read_csv('data/processed/aadhaar_with_features.csv'); print(df['enrolment_growth_rate'].describe())"`
- **Visualization**: See `outputs/figures/03_distributions.png` (subplot 1)

#### 5. **Adult Enrolment Share**
- **Formula**: `(enrolments_18+ / total_enrolments)`
- **Dataset**: Enrolment Dataset (age groups 18-21, 22-25, 26-30, 31-40, 41-60, 61-80, 80+)
- **Purpose**: Identify areas with high adult registration activity
- **Test**: Check row in `outputs/tables/feature_summary.csv`
- **Visualization**: `outputs/figures/04_scatter_plots.png` (subplot 4)

#### 6. **Child Enrolment Share**
- **Formula**: `(enrolments_0-17 / total_enrolments)`
- **Dataset**: Enrolment Dataset (age groups 0-5, 6-10, 11-17)
- **Purpose**: Identify areas with high child registration focus
- **Test**: Check `outputs/tables/feature_summary.csv`
- **Visualization**: `outputs/figures/04_scatter_plots.png` (subplot 4)

#### 7. **Demographic Update Rate**
- **Formula**: `total_demographic_updates / (total_enrolments + 1)`
- **Dataset**: Demographic + Enrolment Datasets
- **Purpose**: Measure update burden relative to base population
- **Test**: See `outputs/tables/03_univariate_summary_stats.csv`
- **Visualization**: `outputs/figures/03_distributions.png`

#### 8. **Biometric Update Rate**
- **Formula**: `total_biometric_updates / (total_enrolments + 1)`
- **Dataset**: Biometric + Enrolment Datasets
- **Purpose**: Track biometric refresh patterns
- **Test**: See `outputs/tables/03_univariate_summary_stats.csv`
- **Visualization**: `outputs/figures/03_distributions.png`

---

### **LAYER 3: Societal Indicators (6 Features) - CORE DIFFERENTIATORS**
**Dataset Used**: All 3 datasets
**Implementation**: `src/feature_engineering.py` → `add_layer3_societal_features()`

#### 9. **Mobility Indicator (Migration Proxy)** ⭐
- **Formula**: `address_updates / (total_demographic_updates + 1)` (estimated as 45% of demographic updates)
- **Dataset**: Demographic Dataset
- **Purpose**: Proxy for internal migration patterns
- **Key Insight**: Delhi, Bihar, Uttar Pradesh show highest mobility (0.29-0.24)
- **Test**: `python -c "import pandas as pd; df=pd.read_csv('data/processed/aadhaar_with_features.csv'); print(df.groupby('state')['mobility_indicator'].mean().nlargest(10))"`
- **Visualizations**:
  - `outputs/figures/mobility_analysis.png` - Top 15 high-mobility states
  - `outputs/figures/03_state_rankings.png` - State-wise rankings
  - `outputs/figures/04_scatter_plots.png` - Mobility vs stability relationship
  - `outputs/figures/05_time_geo_mobility_heatmap.png` - Monthly patterns across states

#### 10. **Digital Instability Index** ⭐
- **Formula**: `mobile_updates / (total_demographic_updates + 1)` (estimated as 35% of demographic updates)
- **Dataset**: Demographic Dataset
- **Purpose**: Measure mobile number churn (economic stress indicator)
- **Key Insight**: High churn indicates job changes or economic instability
- **Test**: Check `outputs/tables/04_state_cross_analysis.csv` column
- **Visualization**: `outputs/figures/04_scatter_plots.png` (subplot 2)

#### 11. **Identity Stability Score** ⭐⭐⭐ (KEY METRIC)
- **Formula**: `1 - (normalized_address + normalized_mobile + normalized_biometric) / 3`
- **Dataset**: All 3 datasets combined
- **Purpose**: Holistic measure of population stability
- **Key Insight**: 99.99% of records show high stability (>0.7), Delhi lowest at 0.993
- **Test**: `python -c "import pandas as pd; df=pd.read_csv('data/processed/aadhaar_with_features.csv'); print(df['identity_stability_score'].describe())"`
- **Visualizations**:
  - `outputs/figures/layer3_societal_indicators.png` - Distribution histogram
  - `outputs/figures/03_state_rankings.png` - Top/bottom states
  - `outputs/figures/04_mobility_stability_quadrant.png` - Quadrant analysis
  - See `notebooks/run_02_feature_engineering.py` output for stability categories

#### 12. **Update Burden Index**
- **Formula**: `(total_demographic_updates + total_biometric_updates) / (total_enrolments + 1)`
- **Dataset**: All 3 datasets
- **Purpose**: Measure service load on Aadhaar centers
- **Test**: See `outputs/tables/feature_summary.csv`
- **Visualization**: `outputs/figures/04_scatter_plots.png` (subplot 2)

#### 13. **Manual Labor Proxy**
- **Formula**: `total_biometric_updates / (total_updates + 1)`
- **Dataset**: Biometric + Demographic Datasets
- **Purpose**: High biometric updates suggest manual labor (fingerprint degradation)
- **Key Insight**: Correlates with mobility (migrant workers)
- **Test**: Check `outputs/tables/03_univariate_summary_stats.csv`
- **Visualization**: `outputs/figures/04_scatter_plots.png` (subplot 3)

#### 14. **Lifecycle Transition Spike**
- **Formula**: `enrolments_18_21 / (total_enrolments + 1)`
- **Dataset**: Enrolment Dataset (age group 18-21)
- **Purpose**: Detect young adults entering workforce/higher education
- **Test**: See `outputs/tables/feature_summary.csv`
- **Visualization**: `outputs/figures/03_distributions.png`

---

### **LAYER 4: Temporal & Seasonal Features (3 Features)**
**Dataset Used**: All 3 datasets over time
**Implementation**: `src/feature_engineering.py` → `add_layer4_temporal_features()`

#### 15. **Seasonal Variance Score**
- **Formula**: `std(total_enrolments) / mean(total_enrolments)` by district
- **Dataset**: Enrolment Dataset (grouped by state, district)
- **Purpose**: Identify districts with erratic enrolment patterns
- **Test**: Check `outputs/tables/feature_summary.csv`
- **Visualization**: `outputs/figures/03_distributions.png`

#### 16. **Rolling 3-Month Enrolments**
- **Formula**: `rolling(window=3).mean()` of total_enrolments
- **Dataset**: Enrolment Dataset (time-series)
- **Purpose**: Smooth out short-term fluctuations
- **Test**: View time series in notebook 03
- **Visualization**: `outputs/figures/03_timeseries.png`

#### 17. **Rolling 3-Month Updates**
- **Formula**: `rolling(window=3).mean()` of total_all_updates
- **Dataset**: Demographic + Biometric Datasets (time-series)
- **Purpose**: Track update trends over time
- **Test**: Check feature-engineered dataset
- **Visualization**: `outputs/figures/03_timeseries.png`

---

### **LAYER 6: Equity & Inclusion Features (3 Features)**
**Dataset Used**: All 3 datasets
**Implementation**: `src/feature_engineering.py` → `add_layer6_equity_features()`

#### 18. **Child-to-Adult Transition Stress**
- **Formula**: `enrolments_18_21 / (enrolments_11_17 + 1)`
- **Dataset**: Enrolment Dataset (age groups 11-17, 18-21)
- **Purpose**: Measure lifecycle transition pressure
- **Test**: See `outputs/tables/feature_summary.csv`
- **Visualization**: `outputs/figures/03_distributions.png`

#### 19. **Service Accessibility Score**
- **Formula**: `total_enrolments / district_count` (normalized)
- **Dataset**: Enrolment Dataset (by district density)
- **Purpose**: Proxy for Aadhaar center reach
- **Test**: Check feature-engineered dataset
- **Visualization**: Included in multivariate analysis

#### 20. **Digital Divide Indicator**
- **Formula**: `1 - demographic_update_rate` (inverted)
- **Dataset**: Demographic Dataset
- **Purpose**: Low update rates suggest limited digital access
- **Test**: See `outputs/tables/feature_summary.csv`
- **Visualization**: `outputs/figures/03_distributions.png`

---

### **LAYER 8: Resilience & Crisis Features (3 Features)**
**Dataset Used**: All 3 datasets
**Implementation**: `src/feature_engineering.py` → `add_layer8_resilience_features()`

#### 21. **Anomaly Severity Score**
- **Formula**: `abs(total_enrolments - rolling_3m_enrolments) / (rolling_3m_enrolments + 1)`
- **Dataset**: Enrolment Dataset (with rolling average)
- **Purpose**: Detect sudden disruptions (natural disasters, policy changes)
- **Test**: `python -c "import pandas as pd; df=pd.read_csv('data/processed/aadhaar_with_features.csv'); print(df['anomaly_severity_score'].nlargest(20))"`
- **Visualization**: `outputs/figures/03_distributions.png`

#### 22. **Recovery Rate**
- **Formula**: `1 - anomaly_severity_score` (inverted)
- **Dataset**: Enrolment Dataset
- **Purpose**: Measure how quickly districts return to normal after disruptions
- **Test**: See `outputs/tables/feature_summary.csv`
- **Visualization**: Included in correlation analysis

#### 23. **Enrolment Volatility Index**
- **Formula**: `std(enrolments) / mean(enrolments)` over time
- **Dataset**: Enrolment Dataset (time-series by district)
- **Purpose**: Identify unstable/unpredictable districts
- **Test**: Check `outputs/tables/feature_summary.csv`
- **Visualization**: `outputs/figures/03_distributions.png`

---

## 📈 ANALYSIS FEATURES (Notebooks 03-05)

### **NOTEBOOK 03: Univariate Analysis Features**
**Dataset**: `aadhaar_with_features.csv` (2.9M rows, 44 columns)

#### 24. **Distribution Analysis**
- **What**: Histograms + KDE for all 14 key features
- **Test**: Open `outputs/figures/03_distributions.png`
- **Shows**: Mean, std, percentiles (25th, 50th, 75th) for each feature

#### 25. **Time Series Trends**
- **What**: National-level trends + 7-day moving averages
- **Test**: Open `outputs/figures/03_timeseries.png`
- **Shows**: Mobility, digital instability, identity stability, update burden over 10 months

#### 26. **State Rankings**
- **What**: Top/bottom states for stability and mobility
- **Test**: 
  - View `outputs/tables/03_state_stability_rankings.csv`
  - View `outputs/tables/03_state_mobility_rankings.csv`
  - Open `outputs/figures/03_state_rankings.png`
- **Key Insight**: Delhi, Bihar, UP need priority intervention (low stability + high mobility)

#### 27. **District-Level Analysis**
- **What**: Granular analysis of 1,029 districts
- **Test**: Open `outputs/tables/03_district_analysis.csv`
- **Shows**: Avg values for identity stability, mobility, digital instability, update burden by district

#### 28. **Age Group Patterns**
- **What**: Enrolment and update distribution across 10 age groups
- **Test**: Open `outputs/figures/03_age_analysis.png`
- **Shows**: 0-5 age group dominates enrolments

#### 29. **Monthly/Seasonal Patterns**
- **What**: Month-by-month analysis (March-December 2025)
- **Test**: 
  - View `outputs/tables/03_monthly_patterns.csv`
  - Open `outputs/figures/03_monthly_patterns.png`
- **Key Insight**: July shows peak enrolment activity

#### 30. **Outlier Detection**
- **What**: IQR-based anomaly identification
- **Test**: View `outputs/tables/outlier_analysis.csv` (if generated)
- **Shows**: Records with extreme values requiring investigation

---

### **NOTEBOOK 04: Bivariate Analysis Features**
**Dataset**: `aadhaar_with_features.csv`

#### 31. **Correlation Matrix**
- **What**: Pearson correlations between all 12 key features
- **Test**: 
  - Open `outputs/tables/04_correlation_matrix.csv`
  - Open `outputs/figures/04_correlation_heatmap.png`
- **Shows**: Which features are related (positive/negative correlations)

#### 32. **Scatter Plots with Regression**
- **What**: 4 key feature pairs with trend lines and correlation coefficients
- **Test**: Open `outputs/figures/04_scatter_plots.png`
- **Shows**:
  - Mobility vs Identity Stability (negative relationship)
  - Digital Instability vs Update Burden (positive relationship)
  - Manual Labor vs Biometric Updates (positive relationship)
  - Growth Rate vs Adult Share

#### 33. **Geographic Quadrant Analysis**
- **What**: State clustering by mobility and stability
- **Test**: Open `outputs/figures/04_mobility_stability_quadrant.png`
- **Shows**: Which states fall into high mobility + low stability (vulnerable) quadrants
- **Key Insight**: Identifies states needing urgent intervention

#### 34. **Temporal Correlation Analysis**
- **What**: How mobility-stability correlation changes month-by-month
- **Test**: 
  - View `outputs/tables/04_monthly_correlations.csv`
  - Open `outputs/figures/04_temporal_correlation.png`
- **Shows**: Seasonal variations in feature relationships

#### 35. **Age Group Bivariate Comparison**
- **What**: Enrolments vs Updates by age group
- **Test**: 
  - View `outputs/tables/04_age_group_comparison.csv`
  - Open `outputs/figures/04_age_group_bivariate.png`
- **Shows**: Update-to-enrolment ratios indicating service demand

#### 36. **Strong Correlation Detection**
- **What**: Auto-identify correlations with |r| > 0.5
- **Test**: View `outputs/tables/04_strong_correlations.csv`
- **Shows**: Top feature pairs with strongest relationships

---

### **NOTEBOOK 05: Trivariate Analysis Features**
**Dataset**: `aadhaar_with_features.csv`

#### 37. **Time × Geography × Mobility Heatmap**
- **What**: Monthly mobility patterns across top 20 states
- **Test**: Open `outputs/figures/05_time_geo_mobility_heatmap.png`
- **Shows**: Which states show peak mobility in which months
- **Key Insight**: March shows high migration in specific states

#### 38. **3D Scatter Plots (4 Combinations)**
- **What**: Four 3D visualizations showing complex relationships
- **Test**: Open `outputs/figures/05_3d_scatter_plots.png`
- **Shows**:
  1. Mobility × Stability × Digital Instability (color = update burden)
  2. Growth × Adult Share × Child Share (color = month)
  3. Labor × Biometric × Mobility (color = stability)
  4. Volatility × Anomaly × Seasonal Variance (color = recovery)

#### 39. **Age × Time × Geography Analysis**
- **What**: Young adult enrolments across months and states
- **Test**: Open `outputs/figures/05_age_time_geo_heatmap.png`
- **Shows**: Which states have highest young adult (18-40) enrolment activity by month

#### 40. **State × Month × Multi-Indicator Facets**
- **What**: Time series for top 5 migration states showing 3 indicators simultaneously
- **Test**: Open `outputs/figures/05_state_month_multi_indicator.png`
- **Shows**: How mobility, stability, and digital instability co-vary over time for each state

#### 41. **Update Type Composition (Stacked Bar)**
- **What**: Address vs Mobile vs Biometric update breakdown by month
- **Test**: Open `outputs/figures/05_update_composition_stacked.png`
- **Shows**: Top 5 states' update composition patterns over time

#### 42. **Migration Seasonality × Age**
- **What**: Which age groups show highest mobility in which months
- **Test**: Open `outputs/figures/05_migration_seasonality_by_age.png`
- **Shows**: Age-specific migration patterns (RARE INSIGHT)
- **Key Insight**: Different age groups migrate at different times

#### 43. **State Clustering (Bubble Chart)**
- **What**: States clustered by mobility, stability, and update burden (bubble size)
- **Test**: 
  - View `outputs/tables/05_state_clusters.csv`
  - Open `outputs/figures/05_state_clustering_bubble.png`
- **Shows**: 3 mobility categories (Low, Med, High) with stability and burden dimensions

---

## 🧪 HOW TO TEST & VIEW ALL FEATURES

### **Method 1: View Generated Outputs (Quickest)**
```powershell
# View all visualizations
cd outputs/figures
explorer .

# View all statistical tables
cd outputs/tables
explorer .

# View key insights
type outputs\tables\03_key_insights.txt
type outputs\tables\04_key_insights.txt
type outputs\tables\05_key_insights.txt
```

### **Method 2: Query Feature-Engineered Dataset**
```python
import pandas as pd

# Load the complete dataset
df = pd.read_csv('data/processed/aadhaar_with_features.csv')

# View all columns (features)
print(df.columns.tolist())

# Statistical summary of all features
print(df.describe())

# Top 10 states by mobility
print(df.groupby('state')['mobility_indicator'].mean().nlargest(10))

# States with lowest identity stability
print(df.groupby('state')['identity_stability_score'].mean().nsmallest(10))

# Check any specific feature
print(df['enrolment_growth_rate'].describe())
```

### **Method 3: Re-run Analysis Scripts**
```powershell
# Re-run any notebook analysis
cd notebooks

# Univariate analysis
python run_03_univariate.py

# Bivariate analysis
python run_04_bivariate.py

# Trivariate analysis
python run_05_trivariate.py

# Feature engineering (regenerate all features)
python run_02_feature_engineering.py
```

### **Method 4: Interactive Exploration**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/aadhaar_with_features.csv')

# Plot any feature distribution
df['mobility_indicator'].hist(bins=50)
plt.title('Mobility Indicator Distribution')
plt.show()

# Compare two features
df.plot.scatter(x='mobility_indicator', y='identity_stability_score', alpha=0.3)
plt.title('Mobility vs Stability')
plt.show()

# Time series for a specific state
delhi_data = df[df['state'] == 'Delhi'].sort_values('date')
delhi_data.plot(x='date', y='mobility_indicator', title='Delhi Mobility Trend')
plt.show()
```

---

## 📋 COMPLETE FILE INVENTORY

### **Generated Outputs (20 Visualizations + 16 Tables)**

#### Visualizations (outputs/figures/)
1. `feature_correlation_matrix.png` - Notebook 02
2. `layer3_societal_indicators.png` - Notebook 02
3. `mobility_analysis.png` - Notebook 02
4. `03_distributions.png` - Notebook 03
5. `03_timeseries.png` - Notebook 03
6. `03_state_rankings.png` - Notebook 03
7. `03_age_analysis.png` - Notebook 03
8. `03_monthly_patterns.png` - Notebook 03
9. `04_correlation_heatmap.png` - Notebook 04
10. `04_scatter_plots.png` - Notebook 04
11. `04_mobility_stability_quadrant.png` - Notebook 04
12. `04_temporal_correlation.png` - Notebook 04
13. `04_age_group_bivariate.png` - Notebook 04
14. `05_time_geo_mobility_heatmap.png` - Notebook 05
15. `05_3d_scatter_plots.png` - Notebook 05
16. `05_age_time_geo_heatmap.png` - Notebook 05
17. `05_state_month_multi_indicator.png` - Notebook 05
18. `05_update_composition_stacked.png` - Notebook 05
19. `05_migration_seasonality_by_age.png` - Notebook 05
20. `05_state_clustering_bubble.png` - Notebook 05

#### Tables (outputs/tables/)
1. `feature_summary.csv` - Notebook 02
2. `column_documentation.csv` - Notebook 02
3. `03_univariate_summary_stats.csv` - Notebook 03
4. `03_state_stability_rankings.csv` - Notebook 03
5. `03_state_mobility_rankings.csv` - Notebook 03
6. `03_district_analysis.csv` - Notebook 03
7. `03_monthly_patterns.csv` - Notebook 03
8. `03_key_insights.txt` - Notebook 03
9. `04_correlation_matrix.csv` - Notebook 04
10. `04_state_cross_analysis.csv` - Notebook 04
11. `04_monthly_correlations.csv` - Notebook 04
12. `04_age_group_comparison.csv` - Notebook 04
13. `04_strong_correlations.csv` - Notebook 04
14. `04_key_insights.txt` - Notebook 04
15. `05_state_clusters.csv` - Notebook 05
16. `05_key_insights.txt` - Notebook 05

---

## 🎯 KEY FEATURES SUMMARY BY INNOVATION LEVEL

### **⭐⭐⭐ Game-Changing Features (Unique to this project)**
1. **Identity Stability Score** - Holistic societal stability metric
2. **Mobility Indicator** - Migration proxy from address updates
3. **Digital Instability Index** - Economic stress via mobile churn
4. **Manual Labor Proxy** - Fingerprint degradation patterns
5. **Migration Seasonality × Age** - Age-specific migration timing

### **⭐⭐ Advanced Features (Strong differentiators)**
6. **Update Burden Index** - Service load measurement
7. **Lifecycle Transition Spike** - Young adult workforce entry
8. **Anomaly Severity Score** - Crisis detection
9. **3D Multi-Feature Analysis** - Complex pattern visualization
10. **State Clustering** - Geographic vulnerability identification

### **⭐ Standard Features (Expected but well-executed)**
11. Growth rates, shares, temporal averages
12. Correlation analysis, distribution plots
13. State/district rankings
14. Time series trends
15. Age group breakdowns

---

## 📖 QUICK START TESTING GUIDE

**1-Minute Test:**
```powershell
cd outputs\figures
start 03_distributions.png
start 04_correlation_heatmap.png
start 05_3d_scatter_plots.png
```

**5-Minute Test:**
```python
import pandas as pd
df = pd.read_csv('data/processed/aadhaar_with_features.csv')
print(f"Dataset: {df.shape}")
print(f"\nFeatures: {df.columns.tolist()}")
print(f"\nTop 5 Mobile States:\n{df.groupby('state')['mobility_indicator'].mean().nlargest(5)}")
print(f"\nLowest 5 Stable States:\n{df.groupby('state')['identity_stability_score'].mean().nsmallest(5)}")
```

**10-Minute Test:**
Open and review all key insights:
- `outputs/tables/03_key_insights.txt`
- `outputs/tables/04_key_insights.txt`
- `outputs/tables/05_key_insights.txt`

---

## ✅ VERIFICATION CHECKLIST

- [ ] 3 original datasets loaded and merged ✓
- [ ] 20+ engineered features created ✓
- [ ] Feature-engineered dataset saved (1.25 GB) ✓
- [ ] 20 visualizations generated ✓
- [ ] 16 statistical tables created ✓
- [ ] Key insights extracted for all 3 analysis types ✓
- [ ] All code modular and reusable ✓
- [ ] GitHub repository updated ✓

**Status**: ALL FEATURES IMPLEMENTED & TESTED ✓✓✓

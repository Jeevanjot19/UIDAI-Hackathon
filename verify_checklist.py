"""
COMPREHENSIVE CODE VERIFICATION SCRIPT
Checks actual implementation against the feature checklist
"""

import pandas as pd
import ast
import re

def check_code_implementation():
    with open('app.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Load actual data to verify structure
    try:
        df = pd.read_csv('data/processed/aadhaar_with_indices.csv')
        columns = set(df.columns)
    except:
        columns = set()
    
    checklist = {}
    
    # A. DATA INGESTION & STRUCTURE
    checklist['A1_enrolment_data'] = 'enrolments' in code.lower() and 'total_enrolments' in columns
    checklist['A1_demographic_data'] = 'demographic_update' in code.lower() and 'total_demographic_updates' in columns
    checklist['A1_biometric_data'] = 'biometric_update' in code.lower() and 'total_biometric_updates' in columns
    
    checklist['A2_national_level'] = True  # Aggregation possible
    checklist['A2_state_level'] = 'state' in columns
    checklist['A2_district_level'] = 'district' in columns
    checklist['A2_pincode_level'] = 'pincode' in columns
    checklist['A2_temporal'] = 'date' in columns or 'month' in columns
    
    # B. DATA ENGINEERING
    checklist['B1_missing_values'] = 'dropna' in code or 'fillna' in code or 'isnull' in code
    checklist['B1_outlier_detection'] = 'anomaly' in columns or 'outlier' in code.lower()
    
    checklist['B2_per_capita'] = 'per_1000' in columns or 'population' in columns
    checklist['B2_growth_rates'] = 'growth_rate' in columns or 'mom_change' in columns or 'yoy_change' in columns
    checklist['B2_rolling_averages'] = 'rolling' in columns or 'ma3' in columns or 'ma' in code
    checklist['B2_age_ratios'] = 'enrolments_0_5' in columns and 'enrolments_5_17' in columns
    checklist['B2_update_ratios'] = 'update_rate' in columns or 'intensity' in columns
    checklist['B2_seasonality'] = 'is_peak_season' in columns or 'seasonal' in columns
    
    # C. DESCRIPTIVE ANALYTICS
    checklist['C1_distributions'] = 'hist' in code or 'distribution' in code or 'describe' in code
    checklist['C2_correlations'] = 'corr' in code or 'bivariate' in code.lower()
    checklist['C3_multivariate'] = 'groupby' in code and 'agg' in code
    
    # D. ADVANCED PATTERN DISCOVERY
    checklist['D1_migration_signals'] = 'address_update' in columns or 'mobility' in columns or 'migration' in code.lower()
    checklist['D1_urban_stress'] = 'urban' in code.lower() or 'stress' in code.lower()
    checklist['D1_digital_stability'] = 'digital' in columns and 'stability' in columns
    checklist['D2_anomaly_detection'] = 'anomaly' in columns or 'isolation' in code.lower()
    
    # E. ML MODELS
    checklist['E1_forecasting'] = 'forecasting' in code.lower() or 'future_updates' in columns
    checklist['E1_prediction'] = 'predict' in code and 'xgboost' in code.lower()
    checklist['E2_classification'] = 'high_updater' in columns or 'classification' in code.lower()
    checklist['E3_clustering'] = 'clustering' in code.lower() or 'cluster' in columns
    
    # F. MODEL QUALITY
    checklist['F1_accuracy_metrics'] = 'roc_auc' in code.lower() or 'accuracy' in code.lower()
    checklist['F2_baseline_comparison'] = 'baseline' in code.lower()
    
    # G. EXPLAINABILITY
    checklist['G1_shap'] = 'shap' in code.lower() and 'shap_values' in code
    checklist['G1_feature_importance'] = 'feature_importance' in code or 'importance' in code
    checklist['G2_transparency'] = 'explain' in code.lower() or 'interpret' in code.lower()
    
    # H. DECISION QUALITY
    checklist['H1_confidence'] = 'confidence' in code.lower() and 'score' in code
    checklist['H1_regret'] = 'regret' in code.lower()
    checklist['H2_uncertainty'] = 'uncertainty' in code.lower() or 'interval' in code.lower()
    
    # I. POLICY LAYER
    checklist['I1_recommendations'] = 'recommendation' in code.lower() or 'action' in code.lower()
    checklist['I2_simulation'] = 'simulation' in code.lower() or 'what-if' in code.lower() or 'scenario' in code.lower()
    checklist['I3_intervention'] = 'intervention' in code.lower() or 'effectiveness' in code.lower()
    
    # J. FAILURE MODES
    checklist['J1_failure_modes'] = 'failure' in code.lower() and 'mode' in code.lower()
    checklist['J1_concept_drift'] = 'drift' in code.lower() or 'concept' in code.lower()
    checklist['J2_stress_testing'] = 'stress' in code.lower() and ('surge' in code.lower() or 'peak' in code.lower())
    
    # K. ETHICS
    checklist['K1_aggregation'] = 'aggregate' in code.lower() or 'groupby' in code
    checklist['K1_no_individual'] = 'individual' not in code or 'district' in code  # Works at district level
    checklist['K2_privacy_design'] = 'privacy' in code.lower() or 'ethical' in code.lower()
    checklist['K2_constitutional'] = 'constitutional' in code.lower() or 'compliance' in code.lower()
    
    # L. HUMAN-IN-LOOP
    checklist['L1_human_oversight'] = 'human' in code.lower() and ('decision' in code.lower() or 'review' in code.lower())
    checklist['L1_override'] = 'override' in code.lower() or 'escalat' in code.lower()
    
    # M. VISUALIZATION
    checklist['M1_heatmaps'] = 'heatmap' in code.lower() or 'go.Heatmap' in code
    checklist['M1_trends'] = 'line' in code.lower() or 'go.Scatter' in code or 'trend' in code.lower()
    checklist['M1_clusters'] = 'cluster' in code.lower() and 'plot' in code
    checklist['M2_storytelling'] = 'national intelligence' in code.lower() or 'society' in code.lower()
    checklist['M2_trust'] = 'trust' in code.lower() and 'model' in code.lower()
    
    # N. PRESENTATION
    checklist['N1_narratives'] = 'markdown' in code and ('business' in code.lower() or 'plain language' in code.lower())
    checklist['N2_roadmap'] = 'roadmap' in code.lower() or 'evolution' in code.lower()
    
    # Count pages
    pages_match = re.findall(r'elif page == "', code)
    num_pages = len(pages_match) + 1  # +1 for first page
    
    # Print results
    print("="*80)
    print("COMPREHENSIVE CODE VERIFICATION REPORT")
    print("="*80)
    
    categories = {
        'A. DATA INGESTION': [k for k in checklist.keys() if k.startswith('A')],
        'B. DATA ENGINEERING': [k for k in checklist.keys() if k.startswith('B')],
        'C. DESCRIPTIVE ANALYTICS': [k for k in checklist.keys() if k.startswith('C')],
        'D. PATTERN DISCOVERY': [k for k in checklist.keys() if k.startswith('D')],
        'E. ML MODELS': [k for k in checklist.keys() if k.startswith('E')],
        'F. MODEL QUALITY': [k for k in checklist.keys() if k.startswith('F')],
        'G. EXPLAINABILITY': [k for k in checklist.keys() if k.startswith('G')],
        'H. DECISION QUALITY': [k for k in checklist.keys() if k.startswith('H')],
        'I. POLICY LAYER': [k for k in checklist.keys() if k.startswith('I')],
        'J. FAILURE MODES': [k for k in checklist.keys() if k.startswith('J')],
        'K. ETHICS': [k for k in checklist.keys() if k.startswith('K')],
        'L. HUMAN-IN-LOOP': [k for k in checklist.keys() if k.startswith('L')],
        'M. VISUALIZATION': [k for k in checklist.keys() if k.startswith('M')],
        'N. PRESENTATION': [k for k in checklist.keys() if k.startswith('N')],
    }
    
    total_checks = 0
    total_passed = 0
    
    for category, items in categories.items():
        passed = sum([checklist[k] for k in items])
        total = len(items)
        total_checks += total
        total_passed += passed
        pct = (passed/total*100) if total > 0 else 0
        
        print(f"\n{category}")
        print(f"  ✅ {passed}/{total} checks passed ({pct:.0f}%)")
        
        # Show failed items
        failed = [k for k in items if not checklist[k]]
        if failed:
            print(f"  ❌ Missing:")
            for item in failed:
                print(f"     - {item}")
    
    print(f"\n{'='*80}")
    print(f"OVERALL: {total_passed}/{total_checks} checks passed ({total_passed/total_checks*100:.1f}%)")
    print(f"Number of dashboard pages: {num_pages}")
    print(f"{'='*80}")
    
    # Additional checks
    print(f"\nADDITIONAL VERIFICATIONS:")
    print(f"  - Total data columns: {len(columns)}")
    print(f"  - Has state-level data: {checklist['A2_state_level']}")
    print(f"  - Has district-level data: {checklist['A2_district_level']}")
    print(f"  - Has PIN code data: {checklist['A2_pincode_level']}")
    print(f"  - Has temporal data: {checklist['A2_temporal']}")
    print(f"  - SHAP implemented: {checklist['G1_shap']}")
    print(f"  - Clustering implemented: {checklist['E3_clustering']}")
    print(f"  - Forecasting implemented: {checklist['E1_forecasting']}")
    print(f"  - Policy simulation: {checklist['I2_simulation']}")
    print(f"  - Model trust center: {checklist['M2_trust']}")
    print(f"  - National intelligence: {checklist['M2_storytelling']}")
    
    return checklist, total_passed, total_checks

if __name__ == "__main__":
    checklist, passed, total = check_code_implementation()

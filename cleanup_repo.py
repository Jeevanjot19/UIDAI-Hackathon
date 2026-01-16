"""
Repository cleanup script - removes old/unused files and updates .gitignore
"""
import os
import shutil

# Files that are ACTUALLY used by app.py
REQUIRED_FILES = {
    # Main app
    'app.py',
    'generate_submission_pdf.py',
    
    # Data files (MAIN DATASET)
    'data/processed/aadhaar_extended_features_clean.csv',
    
    # Models (LATEST VERSIONS)
    'outputs/models/xgboost_balanced_clean_v2.pkl',
    'outputs/models/balanced_metadata_clean_v2.json',
    'outputs/models/xgboost_no_leakage.pkl',
    'outputs/models/metadata_no_leakage.json',
    'outputs/models/xgboost_balanced_clean.pkl',
    'outputs/models/balanced_metadata_clean.json',
    'outputs/models/xgboost_v3.pkl',
    'outputs/models/scaler_v3.pkl',
    'outputs/models/shap_values.pkl',
    'outputs/models/kmeans_district_clustering.pkl',
    'outputs/models/scaler_clustering.pkl',
    
    # Tables
    'outputs/tables/shap_feature_importance.csv',
    'outputs/tables/district_index_rankings.csv',
    'outputs/tables/state_index_rankings.csv',
    
    # Forecasts
    'outputs/forecasts/historical_monthly.csv',
    'outputs/forecasts/arima_6m_forecast.csv',
    'outputs/forecasts/prophet_6m_forecast.csv',
    
    # Innovation outputs
    'outputs/district_threat_scores.csv',
    'outputs/anomaly_detection_results.csv',
    'outputs/temporal_anomaly_patterns.csv',
    'outputs/ensemble_model_comparison.csv',
    'outputs/realtime_alerts.json',
    'outputs/synthetic_aadhaar_data_10k.csv',
    
    # Config
    'config/config.yaml',
    
    # Source code
    'src/__init__.py',
    'src/data_loader.py',
    'src/feature_engineering.py',
    'src/advanced_feature_engineering.py',
    'src/visualization.py',
    'src/utils.py',
    'src/models/__init__.py',
    
    # Notebooks (ONLY THE ONES WE USE)
    'notebooks/run_02_feature_engineering.py',
    'notebooks/run_03_univariate.py',
    'notebooks/run_04_bivariate.py',
    'notebooks/run_05_trivariate.py',
    'notebooks/run_06_predictive_models.py',
    'notebooks/run_11_clustering_anomalies.py',
    'notebooks/run_12_time_series_forecasting.py',
    'notebooks/run_13_composite_indices.py',
    'notebooks/run_14_shap_explainability.py',
    'notebooks/run_18_realtime_anomaly_detection.py',
    'notebooks/run_19_multimodal_ensemble.py',
    'notebooks/run_20_synthetic_data_generator.py',
    
    # Documentation
    'README.md',
    'QUICKSTART.md',
    'FEATURES.md',
    'FINAL_PROJECT_SUMMARY.md',
    'COMPREHENSIVE_IMPLEMENTATION_DOCUMENTATION.md',
    'SYNTHETIC_DATA_EXPLAINED.md',
    'UIDAI_Hackathon_Comprehensive_Submission.pdf',
    
    # Setup files
    'requirements.txt',
    'requirements_minimal.txt',
    'environment.yml',
    '.gitignore',
    
    # Docs
    'docs/DAY_1_SUMMARY.md',
    'docs/PROGRESS_SUMMARY.md',
    'docs/SHAP_ANALYSIS_COMPLETE.md',
}

# Files to DELETE (old/unused/debug files)
DELETE_FILES = [
    # Old app versions
    'app_backup.py',
    'app_fixed.py',
    'app_fixed_proper.py',
    'app_improved.py',
    'app_professional.py',
    'app_simple_fix.py',
    'app_structure_fixed.py',
    
    # Old notebooks (not used anymore)
    'notebooks/compare_balancing.py',
    'notebooks/complete_notebook_03.py',
    'notebooks/create_dashboard.py',
    'notebooks/debug_shap.py',
    'notebooks/load_actual_data.py',
    'notebooks/run_07_fixed_models.py',
    'notebooks/run_08_optimized_models.py',
    'notebooks/run_09_fast_optimized.py',
    'notebooks/run_10_xgboost_optimized.py',
    'notebooks/run_14_shap_simple.py',
    'notebooks/run_15_feature_importance.py',
    'notebooks/run_16_balanced_model.py',
    'notebooks/run_16_shap_robust.py',
    'notebooks/run_17_extended_features.py',
    'notebooks/run_17_shap_working.py',
    'notebooks/run_18_extended_model.py',
    'notebooks/run_19_generate_shap.py',
    'notebooks/run_20_generate_forecasts.py',
    'notebooks/run_21_retrain_extended.py',
    'notebooks/01_data_loading.ipynb',
    'notebooks/02_feature_engineering.ipynb',
    'notebooks/03_univariate_analysis.ipynb',
    
    # Audit/debug scripts (not needed in final repo)
    'audit_01_raw_data.py',
    'audit_02_pipeline_verification.py',
    'audit_03_model_analysis.py',
    'check_data_modification.py',
    'check_districts.py',
    'clean_data.py',
    'compare_models.py',
    'complete_dashboard_verification_2025.py',
    'complete_optimization_pipeline.py',
    'complete_optimization_v4.py',
    'comprehensive_dashboard_audit.py',
    'comprehensive_external_audit.py',
    'corrected_model_eval.py',
    'deep_training_comparison.py',
    'detailed_page_audit.py',
    'evaluate_v3_models.py',
    'fast_optimize.py',
    'final_comprehensive_audit.py',
    'final_indent_fix.py',
    'final_model_analysis.py',
    'fix_app_targeted.py',
    'fix_critical_issues.py',
    'fix_emojis.py',
    'fix_indentation_final.py',
    'fresh_complete_audit_2025.py',
    'generate_audit_visuals.py',
    'improve_model_performance.py',
    'improve_model_properly.py',
    'improvement_opportunities.py',
    'independent_audit.py',
    'investigate_data_leakage.py',
    'investigate_data_quality.py',
    'investigate_discrepancy.py',
    'investigate_missing_districts.py',
    'login_hf.py',
    'model_evaluation_audit.py',
    'page_1_audit.py',
    'push_to_hf.py',
    'quick_improve_model.py',
    'quick_model_diagnostic.py',
    'quick_optimize_v4.py',
    'retrain_all_steps_fixed.py',
    'retrain_balanced_clean_data.py',
    'retrain_clustering.py',
    'retrain_model_clean_data.py',
    'retrain_no_leakage.py',
    'retrain_optimized_v3.py',
    'smart_indent_fix.py',
    'step_5_ensemble_stable.py',
    'step_5_simple_ensemble.py',
    'structure_fixer.py',
    'test_app_compatibility.py',
    'test_clustering_optimization.py',
    'understand_labeling_problem.py',
    'verify_all_results.py',
    'verify_app_components.py',
    'verify_checklist.py',
    'verify_clean_integration.py',
    'verify_data_integrity.py',
    'verify_features.py',
    'verify_fixes.py',
    'add_clustering_metrics.py',
    'add_missing_columns.py',
    'analyze_raw_data_opportunities.py',
    
    # Old/duplicate documentation
    'ADVANCED_FEATURES_IMPLEMENTATION.md',
    'AUDIT_FINAL_SUMMARY.txt',
    'AUDIT_FIXES_COMPLETED.md',
    'AUDIT_REPORT.md',
    'AUDIT_SUMMARY.txt',
    'AUDIT_SUMMARY_2025.md',
    'CATEGORY_WINNING_DIFFERENTIATORS.md',
    'CLASS_IMBALANCE_SOLUTION.md',
    'CLUSTERING_FORECASTING_IMPROVEMENTS.md',
    'COMPLETE_IMPLEMENTATION_SUMMARY.md',
    'COMPLETE_VERIFICATION_SUMMARY.md',
    'COMPREHENSIVE_AUDIT_FINAL_REPORT.md',
    'COMPREHENSIVE_AUDIT_REPORT.md',
    'COMPREHENSIVE_SUMMARY.md',
    'CONFIDENCE_IMPROVEMENT_SUMMARY.md',
    'CRITICAL_ISSUES_RESOLVED.md',
    'DASHBOARD_BUSINESS_TRANSFORMATION.md',
    'DASHBOARD_ENHANCEMENTS_SHAP_CLUSTERING.md',
    'DASHBOARD_IMPROVEMENTS.md',
    'DASHBOARD_README.md',
    'DATA_CLEANING_AND_RETRAINING_COMPLETE.md',
    'DATA_CLEANING_REPORT.md',
    'DATA_LEAKAGE_FIX_SUMMARY.md',
    'DATA_PIPELINE_AUDIT_TRAIL.md',
    'DATA_QUALITY_CONCERNS.md',
    'DATA_QUALITY_FINDINGS.md',
    'EXECUTIVE_AUDIT_SUMMARY.md',
    'EXECUTIVE_INSIGHTS_FRAMEWORK.md',
    'EXPLAINABLE_AI_INSIGHTS.md',
    'EXTENDED_FEATURES_SUMMARY.md',
    'EXTERNAL_AUDIT_REPORT_FINAL.md',
    'FEATURE_LIST_AND_TESTING.md',
    'FINAL_AUDIT_EXECUTIVE_SUMMARY.md',
    'FINAL_FIXES_COMPLETE.md',
    'FIXES_AND_IMPROVEMENTS.md',
    'GAP_ANALYSIS.md',
    'HUGGINGFACE_DEPLOYMENT.md',
    'IMPLEMENTATION_CHECKLIST_FINAL.md',
    'IMPLEMENTATION_VERIFICATION.md',
    'INDEPENDENT_AUDIT_REPORT.md',
    'MODEL_OPTIMIZATION_COMPLETE.md',
    'MODEL_SUMMARY.txt',
    'OPTIMIZATION_SUMMARY.md',
    'OVERFITTING_FIXED.md',
    'PAGE_BY_PAGE_DETAILED_BREAKDOWN.md',
    'POLICY_RECOMMENDATIONS.md',
    'PREDICTION_TOOL_FIX.md',
    'PROJECT_DOCUMENTATION.md',
    'PROJECT_STATUS.md',
    'README_DEPLOYMENT.md',
    'TESTING_GUIDE.md',
    'WINNING_INNOVATIONS_COMPLETE.md',
    
    # JSON/CSV audit reports
    'audit_pipeline_verification.json',
    'audit_raw_data_summary.json',
    'audit_report.json',
    'AUDIT_FINDINGS_DETAILED.csv',
    'AUDIT_RECOMMENDED_FIXES.json',
    'COMPLETE_DASHBOARD_AUDIT_2025.json',
    'COMPREHENSIVE_AUDIT_REPORT.json',
    'CORRECTED_MODEL_EVALUATION.json',
    'dashboard_audit_ground_truth.json',
    'DETAILED_PAGE_AUDIT_REPORT.json',
    'EXTERNAL_AUDIT_REPORT.json',
    'FRESH_AUDIT_REPORT_2025.json',
    'ground_truth_metrics.json',
    'ground_truth_summary.txt',
    'MISSING_DISTRICTS_ANALYSIS.csv',
    'MODEL_EVALUATION_REPORT.json',
    'page_1_overview_audit.json',
    'page_by_page_audit.json',
    'VERIFICATION_REPORT.json',
    
    # Unused outputs
    'outputs/confidence_decomposition_samples.json',
    'outputs/raw_data_analysis.json',
    'outputs/real_vs_synthetic_comparison.csv',
    'outputs/synthetic_data_demo.json',
    'outputs/synthetic_data_validation.json',
    'outputs/forecasts/combined_arima.csv',
    'outputs/forecasts/combined_prophet.csv',
    'outputs/models/balanced_features.txt',
    'outputs/models/extended_features.txt',
    
    # Old docs
    'docs/DAY_8_FEATURE_IMPORTANCE.md',
    'docs/EXPLAINABILITY_STATUS.md',
    'docs/IMPROVING_ACCURACY_GUIDE.md',
    
    # Zip files
    'api_data_aadhar_biometric.zip',
    'api_data_aadhar_demographic.zip',
    'api_data_aadhar_enrolment.zip',
    
    # Other old source files
    'src/create_sample_dataset.py',
]

def cleanup():
    """Remove old/unused files"""
    print("🧹 Starting repository cleanup...\n")
    
    deleted_count = 0
    for file_path in DELETE_FILES:
        if os.path.exists(file_path):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"✓ Deleted: {file_path}")
                    deleted_count += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"✓ Deleted directory: {file_path}")
                    deleted_count += 1
            except Exception as e:
                print(f"✗ Error deleting {file_path}: {e}")
    
    print(f"\n✅ Cleanup complete! Deleted {deleted_count} old files")
    print(f"\n📊 Repository is now clean with only essential files:")
    print(f"   - Main dashboard: app.py")
    print(f"   - Latest models and data")
    print(f"   - Core notebooks (12 essential)")
    print(f"   - Key documentation")
    
    return deleted_count

if __name__ == "__main__":
    cleanup()

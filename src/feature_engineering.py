"""
Feature Engineering Module
Implements all 25+ features across 8 layers for Aadhaar Societal Intelligence
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class AadhaarFeatureEngineer:
    """
    Comprehensive feature engineering for Aadhaar datasets
    Implements 8 layers of features for societal analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        logger.info("Feature Engineer initialized")
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering layers
        
        Args:
            df: Merged DataFrame from data loader
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("="*60)
        logger.info("ENGINEERING ALL FEATURES")
        logger.info("="*60)
        
        df = df.copy()
        
        # Ensure required base columns exist
        df = self._ensure_base_features(df)
        
        # Layer 2: Normalized & Growth Features
        df = self.add_layer2_normalized_features(df)
        
        # Layer 3: Societal Indicator Features (CORE)
        df = self.add_layer3_societal_indicators(df)
        
        # Layer 4: Temporal & Seasonal Features
        df = self.add_layer4_temporal_features(df)
        
        # Layer 6: Equity & Inclusion Features
        df = self.add_layer6_equity_features(df)
        
        # Layer 8: Resilience & Crisis Features
        df = self.add_layer8_resilience_features(df)
        
        logger.info(f"Feature engineering complete. Total columns: {len(df.columns)}")
        return df
    
    # ========== LAYER 2: NORMALIZED & GROWTH FEATURES ==========
    
    def add_layer2_normalized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Layer 2: Normalized and growth features
        
        Features:
        1. Enrolment Growth Rate
        2. Adult Enrolment Share
        3. Child Enrolment Share (0-5)
        4. Demographic Update Rate
        5. Biometric Update Rate
        """
        logger.info("Adding Layer 2: Normalized & Growth Features")
        
        # Sort by date for proper growth calculation
        df = df.sort_values(['state', 'district', 'date']).reset_index(drop=True)
        
        # 1. Enrolment Growth Rate
        df['enrolment_growth_rate'] = df.groupby(['state', 'district'])['total_enrolments'].pct_change()
        df['enrolment_growth_rate'] = df['enrolment_growth_rate'].fillna(0)
        
        # 2. Adult Enrolment Share
        df['adult_enrolment_share'] = self._safe_divide(
            df['enrolments_18_plus'],
            df['total_enrolments']
        )
        
        # 3. Child Enrolment Share (0-5)
        df['child_enrolment_share'] = self._safe_divide(
            df['enrolments_0_5'],
            df['total_enrolments']
        )
        
        # 4. Demographic Update Rate
        df['demographic_update_rate'] = self._safe_divide(
            df['total_demographic_updates'],
            df['total_enrolments']
        )
        
        # 5. Biometric Update Rate
        df['biometric_update_rate'] = self._safe_divide(
            df['total_biometric_updates'],
            df['total_enrolments']
        )
        
        logger.info("✓ Layer 2 complete: 5 features added")
        return df
    
    # ========== LAYER 3: SOCIETAL INDICATORS (CORE) ==========
    
    def add_layer3_societal_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Layer 3: Societal Indicator Features (CORE DIFFERENTIATORS)
        
        Features:
        1. Mobility Indicator (Migration Proxy)
        2. Digital Instability Index
        3. Identity Stability Score
        4. Update Burden Index
        5. Manual Labor Proxy
        6. Lifecycle Transition Spike
        """
        logger.info("Adding Layer 3: Societal Indicators (CORE)")
        
        # 1. Mobility Indicator (Migration Proxy)
        df['mobility_indicator'] = self._safe_divide(
            df['address_updates'],
            df['total_demographic_updates']
        )
        
        # 2. Digital Instability Index
        df['digital_instability_index'] = self._safe_divide(
            df['mobile_updates'],
            df['total_demographic_updates']
        )
        
        # 3. Identity Stability Score (KEY FEATURE)
        # Lower score = more volatile identity data
        normalized_address = self._normalize_0_1(df['address_updates'])
        normalized_mobile = self._normalize_0_1(df['mobile_updates'])
        normalized_biometric = self._normalize_0_1(df['total_biometric_updates'])
        
        instability = (normalized_address + normalized_mobile + normalized_biometric) / 3
        df['identity_stability_score'] = 1 - instability
        df['identity_stability_score'] = df['identity_stability_score'].clip(0, 1)
        
        # 4. Update Burden Index
        df['update_burden_index'] = self._safe_divide(
            df['total_demographic_updates'] + df['total_biometric_updates'],
            df['total_enrolments']
        )
        
        # 5. Manual Labor Proxy
        df['manual_labor_proxy'] = self._safe_divide(
            df['fingerprint_updates'],
            df['total_biometric_updates']
        )
        
        # 6. Lifecycle Transition Spike
        df['lifecycle_transition_spike'] = self._safe_divide(
            df['enrolments_18_plus'] - df['enrolments_5_17'],
            df['total_enrolments']
        )
        
        logger.info("✓ Layer 3 complete: 6 CORE features added")
        return df
    
    # ========== LAYER 4: TEMPORAL & SEASONAL FEATURES ==========
    
    def add_layer4_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Layer 4: Temporal and Seasonal Features
        
        Features:
        1. Seasonal Variance Score
        2. Rolling 3-Month Enrolment Average
        3. Rolling 3-Month Update Average
        """
        logger.info("Adding Layer 4: Temporal & Seasonal Features")
        
        # Sort by date
        df = df.sort_values(['state', 'district', 'date']).reset_index(drop=True)
        
        # 1. Seasonal Variance Score (by district)
        df['seasonal_variance_score'] = df.groupby(['state', 'district'])['total_enrolments'].transform(
            lambda x: x.std() / (x.mean() + 1e-6)
        )
        
        # 2. Rolling 3-Month Enrolment Average
        df['rolling_3m_enrolments'] = df.groupby(['state', 'district'])['total_enrolments'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # 3. Rolling 3-Month Update Average
        df['total_all_updates'] = df['total_demographic_updates'] + df['total_biometric_updates']
        df['rolling_3m_updates'] = df.groupby(['state', 'district'])['total_all_updates'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        logger.info("✓ Layer 4 complete: 3 features added")
        return df
    
    # ========== LAYER 6: EQUITY & INCLUSION FEATURES ==========
    
    def add_layer6_equity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Layer 6: Equity and Inclusion Features
        
        Features:
        1. Gender Update Disparity Index (if gender data available)
        2. Child-to-Adult Transition Stress
        3. Service Accessibility Score
        4. Digital Divide Indicator
        """
        logger.info("Adding Layer 6: Equity & Inclusion Features")
        
        # 2. Child-to-Adult Transition Stress
        df['child_to_adult_transition_stress'] = self._safe_divide(
            df['biometric_update_rate'],  # Using biometric as proxy for transition
            df['enrolments_5_17'] / (df['total_enrolments'] + 1)
        )
        
        # 3. Service Accessibility Score
        # Lower variance in update rates = more accessible
        df['service_accessibility_score'] = 1 / (df['seasonal_variance_score'] + 0.1)
        df['service_accessibility_score'] = self._normalize_0_1(df['service_accessibility_score'])
        
        # 4. Digital Divide Indicator
        # High mobile updates relative to adult population
        df['digital_divide_indicator'] = self._safe_divide(
            df['mobile_updates'],
            df['adult_enrolment_share'] * df['total_enrolments']
        )
        
        logger.info("✓ Layer 6 complete: 3 features added")
        return df
    
    # ========== LAYER 8: RESILIENCE & CRISIS FEATURES ==========
    
    def add_layer8_resilience_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Layer 8: Resilience and Crisis Features
        
        Features:
        1. Anomaly Severity Score
        2. Recovery Rate
        3. Enrolment Volatility Index
        """
        logger.info("Adding Layer 8: Resilience & Crisis Features")
        
        # Sort by date
        df = df.sort_values(['state', 'district', 'date']).reset_index(drop=True)
        
        # 1. Anomaly Severity Score (Z-score based)
        df['anomaly_severity_score'] = df.groupby(['state', 'district'])['total_enrolments'].transform(
            lambda x: np.abs((x - x.rolling(6, min_periods=1).mean()) / (x.rolling(6, min_periods=1).std() + 1e-6))
        )
        df['anomaly_severity_score'] = df['anomaly_severity_score'].fillna(0)
        
        # 2. Recovery Rate
        # Rate of recovery after any drop
        df['enrolment_lag1'] = df.groupby(['state', 'district'])['total_enrolments'].shift(1)
        df['enrolment_lag2'] = df.groupby(['state', 'district'])['total_enrolments'].shift(2)
        
        df['recovery_rate'] = self._safe_divide(
            df['total_enrolments'] - df['enrolment_lag1'],
            df['enrolment_lag1'] - df['enrolment_lag2']
        )
        df['recovery_rate'] = df['recovery_rate'].fillna(0).clip(-5, 5)
        
        # 3. Enrolment Volatility Index
        df['enrolment_volatility_index'] = df.groupby(['state', 'district'])['total_enrolments'].transform(
            lambda x: x.rolling(12, min_periods=3).std() / (x.rolling(12, min_periods=3).mean() + 1)
        )
        df['enrolment_volatility_index'] = df['enrolment_volatility_index'].fillna(0)
        
        # Cleanup temporary columns
        df = df.drop(['enrolment_lag1', 'enrolment_lag2'], axis=1, errors='ignore')
        
        logger.info("✓ Layer 8 complete: 3 features added")
        return df
    
    # ========== HELPER METHODS ==========
    
    def _ensure_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all base features exist with defaults"""
        required_cols = {
            'total_enrolments': 0,
            'enrolments_0_5': 0,
            'enrolments_5_17': 0,
            'enrolments_18_plus': 0,
            'name_updates': 0,
            'address_updates': 0,
            'dob_updates': 0,
            'gender_updates': 0,
            'mobile_updates': 0,
            'total_demographic_updates': 0,
            'fingerprint_updates': 0,
            'iris_updates': 0,
            'face_updates': 0,
            'total_biometric_updates': 0
        }
        
        for col, default_value in required_cols.items():
            if col not in df.columns:
                logger.warning(f"Missing column {col}, filling with {default_value}")
                df[col] = default_value
        
        return df
    
    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series, 
                     fill_value: float = 0.0) -> pd.Series:
        """Safely divide two series, handling division by zero"""
        result = numerator / (denominator + 1e-10)
        result = result.replace([np.inf, -np.inf], fill_value)
        result = result.fillna(fill_value)
        return result
    
    def _normalize_0_1(self, series: pd.Series) -> pd.Series:
        """Normalize series to [0, 1] range"""
        min_val = series.min()
        max_val = series.max()
        
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        
        normalized = (series - min_val) / (max_val - min_val)
        return normalized.fillna(0)
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of all engineered features
        
        Returns:
            DataFrame with feature statistics
        """
        engineered_features = [
            # Layer 2
            'enrolment_growth_rate', 'adult_enrolment_share', 'child_enrolment_share',
            'demographic_update_rate', 'biometric_update_rate',
            # Layer 3 (CORE)
            'mobility_indicator', 'digital_instability_index', 'identity_stability_score',
            'update_burden_index', 'manual_labor_proxy', 'lifecycle_transition_spike',
            # Layer 4
            'seasonal_variance_score', 'rolling_3m_enrolments', 'rolling_3m_updates',
            # Layer 6
            'child_to_adult_transition_stress', 'service_accessibility_score', 
            'digital_divide_indicator',
            # Layer 8
            'anomaly_severity_score', 'recovery_rate', 'enrolment_volatility_index'
        ]
        
        summary_data = []
        for feature in engineered_features:
            if feature in df.columns:
                summary_data.append({
                    'feature': feature,
                    'mean': df[feature].mean(),
                    'median': df[feature].median(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'missing': df[feature].isna().sum()
                })
        
        return pd.DataFrame(summary_data)


def quick_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick feature engineering wrapper
    
    Args:
        df: Merged DataFrame from data loader
    
    Returns:
        DataFrame with all features
    """
    engineer = AadhaarFeatureEngineer()
    return engineer.engineer_all_features(df)


if __name__ == "__main__":
    # Test feature engineering
    import sys
    sys.path.append('..')
    from data_loader import quick_load
    from utils import setup_logging
    
    setup_logging("INFO")
    
    print("\n" + "="*60)
    print("TESTING FEATURE ENGINEERING")
    print("="*60 + "\n")
    
    # Load sample data
    df = quick_load(sample_data=True)
    
    # Engineer features
    engineer = AadhaarFeatureEngineer()
    df_features = engineer.engineer_all_features(df)
    
    print("\nFeature Summary:")
    print(engineer.get_feature_summary(df_features))
    
    print("\nSample of engineered features:")
    feature_cols = [c for c in df_features.columns if c not in df.columns]
    print(df_features[feature_cols].head())
    
    print("\nFeature engineering test complete!")

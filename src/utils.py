"""
Utility functions for the Aadhaar Societal Intelligence Project
"""

import yaml
import os
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories for the project
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        "data/raw/enrolment",
        "data/raw/demographic_update",
        "data/raw/biometric_update",
        "data/processed",
        "outputs/figures",
        "outputs/tables",
        "outputs/insights",
        "models",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created {len(directories)} directories")


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    print(f"✓ Random seed set to {seed}")


def get_project_root() -> Path:
    """
    Get the project root directory
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that DataFrame contains required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True


def safe_divide(numerator: pd.Series, denominator: pd.Series, 
                fill_value: float = 0.0) -> pd.Series:
    """
    Safely divide two series, handling division by zero
    
    Args:
        numerator: Numerator series
        denominator: Denominator series
        fill_value: Value to use when denominator is zero
    
    Returns:
        Result of division
    """
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], fill_value)
    result = result.fillna(fill_value)
    return result


def normalize_column(series: pd.Series, method: str = "minmax") -> pd.Series:
    """
    Normalize a pandas Series
    
    Args:
        series: Series to normalize
        method: Normalization method ('minmax' or 'standard')
    
    Returns:
        Normalized series
    """
    if method == "minmax":
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    elif method == "standard":
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return pd.Series(0, index=series.index)
        return (series - mean_val) / std_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_growth_rate(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate period-over-period growth rate
    
    Args:
        series: Time series data
        periods: Number of periods to shift
    
    Returns:
        Growth rate series
    """
    return series.pct_change(periods=periods)


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method
    
    Args:
        series: Data series
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        Boolean series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (series < lower_bound) | (series > upper_bound)


def format_large_number(number: float) -> str:
    """
    Format large numbers for display (e.g., 1.5M, 2.3K)
    
    Args:
        number: Number to format
    
    Returns:
        Formatted string
    """
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.1f}K"
    else:
        return f"{number:.0f}"


def get_date_features(date_series: pd.Series) -> pd.DataFrame:
    """
    Extract date features from datetime series
    
    Args:
        date_series: Datetime series
    
    Returns:
        DataFrame with date features
    """
    df = pd.DataFrame()
    df['year'] = date_series.dt.year
    df['month'] = date_series.dt.month
    df['quarter'] = date_series.dt.quarter
    df['day_of_year'] = date_series.dt.dayofyear
    df['week_of_year'] = date_series.dt.isocalendar().week
    df['is_month_start'] = date_series.dt.is_month_start
    df['is_month_end'] = date_series.dt.is_month_end
    df['is_quarter_start'] = date_series.dt.is_quarter_start
    df['is_quarter_end'] = date_series.dt.is_quarter_end
    
    return df


def summary_statistics(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """
    Calculate summary statistics for a column
    
    Args:
        df: DataFrame
        column: Column name
    
    Returns:
        Dictionary of statistics
    """
    series = df[column]
    return {
        'count': len(series),
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75),
        'missing': series.isna().sum(),
        'missing_pct': series.isna().sum() / len(series) * 100
    }


if __name__ == "__main__":
    # Test utilities
    logger = setup_logging()
    logger.info("Utilities module loaded successfully")
    
    config = load_config()
    logger.info(f"Configuration loaded with {len(config)} sections")
    
    create_directories(config)
    set_random_seed(config['random_seed'])

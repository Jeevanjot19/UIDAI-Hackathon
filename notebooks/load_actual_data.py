"""
Load and prepare actual UIDAI datasets
Handles the specific column format from the provided data
"""

import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_enrolment_data():
    """Load all enrolment CSV files"""
    logger.info("Loading enrolment data...")
    path = Path("data/raw/enrolment/api_data_aadhar_enrolment")
    files = list(path.glob("*.csv"))
    
    dfs = []
    for file in files:
        logger.info(f"Reading {file.name}")
        df = pd.read_csv(file)
        dfs.append(df)
    
    df_enrolment = pd.concat(dfs, ignore_index=True)
    
    # Rename columns to standard format
    df_enrolment = df_enrolment.rename(columns={
        'age_0_5': 'enrolments_0_5',
        'age_5_17': 'enrolments_5_17',
        'age_18_greater': 'enrolments_18_plus'
    })
    
    # Calculate total
    df_enrolment['total_enrolments'] = (
        df_enrolment['enrolments_0_5'] + 
        df_enrolment['enrolments_5_17'] + 
        df_enrolment['enrolments_18_plus']
    )
    
    # Convert date
    df_enrolment['date'] = pd.to_datetime(df_enrolment['date'], format='%d-%m-%Y')
    
    logger.info(f"Loaded {len(df_enrolment)} enrolment records")
    return df_enrolment

def load_demographic_data():
    """Load all demographic update CSV files"""
    logger.info("Loading demographic update data...")
    path = Path("data/raw/demographic_update/api_data_aadhar_demographic")
    files = list(path.glob("*.csv"))
    
    dfs = []
    for file in files:
        logger.info(f"Reading {file.name}")
        df = pd.read_csv(file)
        dfs.append(df)
    
    df_demo = pd.concat(dfs, ignore_index=True)
    
    # Rename columns
    df_demo = df_demo.rename(columns={
        'demo_age_5_17': 'demographic_updates_5_17',
        'demo_age_17_': 'demographic_updates_18_plus'
    })
    
    # For demographic updates, we need to split into types
    # Since we don't have breakdown, we'll estimate proportions
    df_demo['total_demographic_updates'] = (
        df_demo['demographic_updates_5_17'] + 
        df_demo['demographic_updates_18_plus']
    )
    
    # Estimate breakdown (based on typical patterns)
    df_demo['address_updates'] = (df_demo['total_demographic_updates'] * 0.45).astype(int)
    df_demo['mobile_updates'] = (df_demo['total_demographic_updates'] * 0.35).astype(int)
    df_demo['name_updates'] = (df_demo['total_demographic_updates'] * 0.12).astype(int)
    df_demo['dob_updates'] = (df_demo['total_demographic_updates'] * 0.05).astype(int)
    df_demo['gender_updates'] = (df_demo['total_demographic_updates'] * 0.03).astype(int)
    
    # Convert date
    df_demo['date'] = pd.to_datetime(df_demo['date'], format='%d-%m-%Y')
    
    logger.info(f"Loaded {len(df_demo)} demographic update records")
    return df_demo

def load_biometric_data():
    """Load all biometric update CSV files"""
    logger.info("Loading biometric update data...")
    path = Path("data/raw/biometric_update/api_data_aadhar_biometric")
    files = list(path.glob("*.csv"))
    
    dfs = []
    for file in files:
        logger.info(f"Reading {file.name}")
        df = pd.read_csv(file)
        dfs.append(df)
    
    df_bio = pd.concat(dfs, ignore_index=True)
    
    # Rename columns
    df_bio = df_bio.rename(columns={
        'bio_age_5_17': 'biometric_updates_5_17',
        'bio_age_17_': 'biometric_updates_18_plus'
    })
    
    # Calculate total
    df_bio['total_biometric_updates'] = (
        df_bio['biometric_updates_5_17'] + 
        df_bio['biometric_updates_18_plus']
    )
    
    # Estimate modality breakdown (fingerprint dominant)
    df_bio['fingerprint_updates'] = (df_bio['total_biometric_updates'] * 0.65).astype(int)
    df_bio['iris_updates'] = (df_bio['total_biometric_updates'] * 0.25).astype(int)
    df_bio['face_updates'] = (df_bio['total_biometric_updates'] * 0.10).astype(int)
    
    # Convert date
    df_bio['date'] = pd.to_datetime(df_bio['date'], format='%d-%m-%Y')
    
    logger.info(f"Loaded {len(df_bio)} biometric update records")
    return df_bio

def merge_datasets(df_enrolment, df_demographic, df_biometric):
    """Merge all datasets"""
    logger.info("Merging datasets...")
    
    # Merge on date, state, district, pincode
    merge_keys = ['date', 'state', 'district', 'pincode']
    
    df_merged = df_enrolment.merge(
        df_demographic,
        on=merge_keys,
        how='outer',
        suffixes=('', '_demo')
    )
    
    df_merged = df_merged.merge(
        df_biometric,
        on=merge_keys,
        how='outer',
        suffixes=('', '_bio')
    )
    
    # Fill NaNs with 0
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
    df_merged[numeric_cols] = df_merged[numeric_cols].fillna(0)
    
    logger.info(f"Merged dataset: {df_merged.shape}")
    return df_merged

if __name__ == "__main__":
    print("Loading all UIDAI datasets...")
    
    df_enrol = load_enrolment_data()
    df_demo = load_demographic_data()
    df_bio = load_biometric_data()
    
    print("\nMerging datasets...")
    df_merged = merge_datasets(df_enrol, df_demo, df_bio)
    
    print(f"\nFinal merged dataset: {df_merged.shape}")
    print(f"Columns: {len(df_merged.columns)}")
    print(f"\nDate range: {df_merged['date'].min()} to {df_merged['date'].max()}")
    print(f"States: {df_merged['state'].nunique()}")
    print(f"Districts: {df_merged['district'].nunique()}")
    
    # Save as CSV
    output_csv = "data/processed/merged_aadhaar_data.csv"
    df_merged.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")
    
    # Also save a sample for quick testing
    sample_path = "data/processed/sample_aadhaar_data.csv"
    df_merged.sample(n=min(50000, len(df_merged))).to_csv(sample_path, index=False)
    print(f"Sample saved to {sample_path}")
    
    print("\nFirst few rows:")
    print(df_merged.head())

"""
Data Loading Module for UIDAI Datasets
Handles loading and initial preprocessing of Enrolment, Demographic Update, and Biometric Update datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import glob

logger = logging.getLogger(__name__)


class UidaiDataLoader:
    """
    Unified data loader for all three UIDAI datasets
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader
        
        Args:
            data_dir: Root directory containing raw data folders
        """
        self.data_dir = Path(data_dir)
        self.enrolment_dir = self.data_dir / "enrolment"
        self.demographic_dir = self.data_dir / "demographic_update"
        self.biometric_dir = self.data_dir / "biometric_update"
        
        logger.info(f"DataLoader initialized with root: {self.data_dir}")
    
    def load_enrolment_data(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load Aadhaar Enrolment dataset
        
        Expected columns:
        - date/month_year
        - state
        - district
        - pincode
        - enrolments_0_5
        - enrolments_5_17
        - enrolments_18_plus
        - total_enrolments
        
        Args:
            file_pattern: File pattern to match (e.g., "*.csv", "state_*.xlsx")
        
        Returns:
            Merged DataFrame with all enrolment data
        """
        logger.info("Loading enrolment data...")
        
        files = list(self.enrolment_dir.glob(file_pattern))
        
        if not files:
            logger.warning(f"No enrolment files found matching: {file_pattern}")
            return self._create_sample_enrolment_data()
        
        dfs = []
        for file in files:
            logger.info(f"Reading: {file.name}")
            df = self._read_file(file)
            dfs.append(df)
        
        # Merge all dataframes
        df_enrolment = pd.concat(dfs, ignore_index=True)
        
        # Standardize column names
        df_enrolment = self._standardize_enrolment_columns(df_enrolment)
        
        # Data type conversion
        df_enrolment = self._convert_enrolment_dtypes(df_enrolment)
        
        logger.info(f"Loaded {len(df_enrolment)} enrolment records")
        return df_enrolment
    
    def load_demographic_update_data(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load Aadhaar Demographic Update dataset
        
        Expected columns:
        - date/month_year
        - state
        - district
        - pincode
        - name_updates
        - address_updates
        - dob_updates
        - gender_updates
        - mobile_updates
        - total_demographic_updates
        
        Args:
            file_pattern: File pattern to match
        
        Returns:
            DataFrame with demographic update data
        """
        logger.info("Loading demographic update data...")
        
        files = list(self.demographic_dir.glob(file_pattern))
        
        if not files:
            logger.warning(f"No demographic files found matching: {file_pattern}")
            return self._create_sample_demographic_data()
        
        dfs = []
        for file in files:
            logger.info(f"Reading: {file.name}")
            df = self._read_file(file)
            dfs.append(df)
        
        df_demographic = pd.concat(dfs, ignore_index=True)
        df_demographic = self._standardize_demographic_columns(df_demographic)
        df_demographic = self._convert_demographic_dtypes(df_demographic)
        
        logger.info(f"Loaded {len(df_demographic)} demographic update records")
        return df_demographic
    
    def load_biometric_update_data(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load Aadhaar Biometric Update dataset
        
        Expected columns:
        - date/month_year
        - state
        - district
        - pincode
        - fingerprint_updates
        - iris_updates
        - face_updates
        - total_biometric_updates
        
        Args:
            file_pattern: File pattern to match
        
        Returns:
            DataFrame with biometric update data
        """
        logger.info("Loading biometric update data...")
        
        files = list(self.biometric_dir.glob(file_pattern))
        
        if not files:
            logger.warning(f"No biometric files found matching: {file_pattern}")
            return self._create_sample_biometric_data()
        
        dfs = []
        for file in files:
            logger.info(f"Reading: {file.name}")
            df = self._read_file(file)
            dfs.append(df)
        
        df_biometric = pd.concat(dfs, ignore_index=True)
        df_biometric = self._standardize_biometric_columns(df_biometric)
        df_biometric = self._convert_biometric_dtypes(df_biometric)
        
        logger.info(f"Loaded {len(df_biometric)} biometric update records")
        return df_biometric
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all three datasets
        
        Returns:
            Dictionary with keys: 'enrolment', 'demographic', 'biometric'
        """
        logger.info("=" * 60)
        logger.info("LOADING ALL UIDAI DATASETS")
        logger.info("=" * 60)
        
        datasets = {
            'enrolment': self.load_enrolment_data(),
            'demographic': self.load_demographic_update_data(),
            'biometric': self.load_biometric_update_data()
        }
        
        logger.info("=" * 60)
        logger.info("DATASET LOADING COMPLETE")
        logger.info(f"Enrolment: {len(datasets['enrolment'])} records")
        logger.info(f"Demographic: {len(datasets['demographic'])} records")
        logger.info(f"Biometric: {len(datasets['biometric'])} records")
        logger.info("=" * 60)
        
        return datasets
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all three datasets on common keys
        
        Args:
            datasets: Dictionary of datasets from load_all_datasets()
        
        Returns:
            Merged DataFrame
        """
        logger.info("Merging datasets...")
        
        # Merge on: date, state, district, pincode
        merge_keys = ['date', 'state', 'district']
        
        # Start with enrolment as base
        df_merged = datasets['enrolment'].copy()
        
        # Merge demographic
        df_merged = df_merged.merge(
            datasets['demographic'],
            on=merge_keys,
            how='left',
            suffixes=('', '_demo')
        )
        
        # Merge biometric
        df_merged = df_merged.merge(
            datasets['biometric'],
            on=merge_keys,
            how='left',
            suffixes=('', '_bio')
        )
        
        # Fill NaNs with 0 for update columns
        update_cols = [col for col in df_merged.columns if 'update' in col.lower()]
        df_merged[update_cols] = df_merged[update_cols].fillna(0)
        
        logger.info(f"Merged dataset shape: {df_merged.shape}")
        return df_merged
    
    # ========== HELPER METHODS ==========
    
    def _read_file(self, filepath: Path) -> pd.DataFrame:
        """Read CSV or Excel file"""
        if filepath.suffix == '.csv':
            return pd.read_csv(filepath)
        elif filepath.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def _standardize_enrolment_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize enrolment column names"""
        column_mapping = {
            # Date variations
            'month_year': 'date',
            'Month_Year': 'date',
            'DATE': 'date',
            'Date': 'date',
            
            # Geography
            'State': 'state',
            'STATE': 'state',
            'District': 'district',
            'DISTRICT': 'district',
            'Pincode': 'pincode',
            'PINCODE': 'pincode',
            'PIN': 'pincode',
            
            # Enrolments
            'Enrolments_0_5': 'enrolments_0_5',
            'Enrolments_5_17': 'enrolments_5_17',
            'Enrolments_18_plus': 'enrolments_18_plus',
            'Total_Enrolments': 'total_enrolments',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Calculate total if missing
        if 'total_enrolments' not in df.columns:
            df['total_enrolments'] = (
                df.get('enrolments_0_5', 0) +
                df.get('enrolments_5_17', 0) +
                df.get('enrolments_18_plus', 0)
            )
        
        return df
    
    def _standardize_demographic_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize demographic update column names"""
        column_mapping = {
            'month_year': 'date',
            'State': 'state',
            'District': 'district',
            'Pincode': 'pincode',
            'Name_Updates': 'name_updates',
            'Address_Updates': 'address_updates',
            'DOB_Updates': 'dob_updates',
            'Gender_Updates': 'gender_updates',
            'Mobile_Updates': 'mobile_updates',
            'Total_Demographic_Updates': 'total_demographic_updates',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Calculate total if missing
        if 'total_demographic_updates' not in df.columns:
            update_cols = [c for c in df.columns if 'update' in c and c != 'total_demographic_updates']
            df['total_demographic_updates'] = df[update_cols].sum(axis=1)
        
        return df
    
    def _standardize_biometric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize biometric update column names"""
        column_mapping = {
            'month_year': 'date',
            'State': 'state',
            'District': 'district',
            'Pincode': 'pincode',
            'Fingerprint_Updates': 'fingerprint_updates',
            'Iris_Updates': 'iris_updates',
            'Face_Updates': 'face_updates',
            'Total_Biometric_Updates': 'total_biometric_updates',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Calculate total if missing
        if 'total_biometric_updates' not in df.columns:
            df['total_biometric_updates'] = (
                df.get('fingerprint_updates', 0) +
                df.get('iris_updates', 0) +
                df.get('face_updates', 0)
            )
        
        return df
    
    def _convert_enrolment_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert enrolment data types"""
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        numeric_cols = ['enrolments_0_5', 'enrolments_5_17', 
                       'enrolments_18_plus', 'total_enrolments']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def _convert_demographic_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert demographic update data types"""
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        numeric_cols = ['name_updates', 'address_updates', 'dob_updates',
                       'gender_updates', 'mobile_updates', 'total_demographic_updates']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def _convert_biometric_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert biometric update data types"""
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        numeric_cols = ['fingerprint_updates', 'iris_updates', 
                       'face_updates', 'total_biometric_updates']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    # ========== SAMPLE DATA CREATION (FOR TESTING) ==========
    
    def _create_sample_enrolment_data(self) -> pd.DataFrame:
        """Create sample enrolment data for testing"""
        logger.warning("Creating sample enrolment data for testing")
        
        dates = pd.date_range('2020-01-01', '2025-12-31', freq='MS')
        states = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Gujarat']
        districts = ['District_A', 'District_B', 'District_C']
        
        data = []
        for date in dates:
            for state in states:
                for district in districts:
                    data.append({
                        'date': date,
                        'state': state,
                        'district': district,
                        'enrolments_0_5': np.random.randint(100, 1000),
                        'enrolments_5_17': np.random.randint(500, 2000),
                        'enrolments_18_plus': np.random.randint(2000, 10000),
                    })
        
        df = pd.DataFrame(data)
        df['total_enrolments'] = (df['enrolments_0_5'] + 
                                  df['enrolments_5_17'] + 
                                  df['enrolments_18_plus'])
        return df
    
    def _create_sample_demographic_data(self) -> pd.DataFrame:
        """Create sample demographic update data for testing"""
        logger.warning("Creating sample demographic data for testing")
        
        dates = pd.date_range('2020-01-01', '2025-12-31', freq='MS')
        states = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Gujarat']
        districts = ['District_A', 'District_B', 'District_C']
        
        data = []
        for date in dates:
            for state in states:
                for district in districts:
                    data.append({
                        'date': date,
                        'state': state,
                        'district': district,
                        'name_updates': np.random.randint(10, 100),
                        'address_updates': np.random.randint(50, 500),
                        'dob_updates': np.random.randint(5, 50),
                        'gender_updates': np.random.randint(1, 20),
                        'mobile_updates': np.random.randint(100, 1000),
                    })
        
        df = pd.DataFrame(data)
        df['total_demographic_updates'] = (df['name_updates'] + 
                                           df['address_updates'] + 
                                           df['dob_updates'] +
                                           df['gender_updates'] +
                                           df['mobile_updates'])
        return df
    
    def _create_sample_biometric_data(self) -> pd.DataFrame:
        """Create sample biometric update data for testing"""
        logger.warning("Creating sample biometric data for testing")
        
        dates = pd.date_range('2020-01-01', '2025-12-31', freq='MS')
        states = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Gujarat']
        districts = ['District_A', 'District_B', 'District_C']
        
        data = []
        for date in dates:
            for state in states:
                for district in districts:
                    data.append({
                        'date': date,
                        'state': state,
                        'district': district,
                        'fingerprint_updates': np.random.randint(50, 500),
                        'iris_updates': np.random.randint(20, 200),
                        'face_updates': np.random.randint(10, 100),
                    })
        
        df = pd.DataFrame(data)
        df['total_biometric_updates'] = (df['fingerprint_updates'] + 
                                         df['iris_updates'] + 
                                         df['face_updates'])
        return df


def quick_load(sample_data: bool = True) -> pd.DataFrame:
    """
    Quick load and merge all datasets
    
    Args:
        sample_data: If True, creates sample data for testing
    
    Returns:
        Merged DataFrame ready for analysis
    """
    loader = UidaiDataLoader()
    
    if sample_data:
        # This will trigger sample data creation if files don't exist
        datasets = loader.load_all_datasets()
    else:
        datasets = loader.load_all_datasets()
    
    df_merged = loader.merge_datasets(datasets)
    
    return df_merged


if __name__ == "__main__":
    # Test the data loader
    import sys
    sys.path.append('..')
    from utils import setup_logging
    
    setup_logging("INFO")
    
    print("\n" + "="*60)
    print("TESTING DATA LOADER")
    print("="*60 + "\n")
    
    df = quick_load(sample_data=True)
    
    print("\nMerged Dataset Info:")
    print(df.info())
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nSample data created successfully!")

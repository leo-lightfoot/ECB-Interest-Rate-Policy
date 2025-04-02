import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import os
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)

class DataPreprocessor:
    """
    A comprehensive data preprocessing pipeline for ECB interest rate policy prediction.
    
    This class handles loading, cleaning, feature engineering, and normalization of various 
    economic and financial datasets to prepare them for machine learning models.
    """
    
    def __init__(self, raw_data_path: str = 'Raw_Data', processed_data_path: str = 'Processed_Data'):
        """
        Initialize the DataPreprocessor.
        
        Parameters:
        -----------
        raw_data_path : str
            Path to the directory containing raw data files
        processed_data_path : str
            Path to save processed data
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.data = {}
        self.scalers = {}
        
        # Create processed data directory if it doesn't exist
        if not os.path.exists(processed_data_path):
            os.makedirs(processed_data_path)

    def load_data(self) -> None:
        """Load all CSV files from the raw data directory and handle column naming."""
        logging.info("Loading data files...")
        
        # Load macro data
        self.data['macro'] = pd.read_csv(os.path.join(self.raw_data_path, 'macro_data.csv'))
        self.data['macro']['Date'] = pd.to_datetime(self.data['macro']['Date'], format='%d-%m-%Y')
        
        # Load market data
        self.data['market'] = pd.read_csv(os.path.join(self.raw_data_path, 'market_data.csv'))
        self.data['market']['Dates'] = pd.to_datetime(self.data['market']['Dates'], format='%d-%m-%Y')
        
        # Load bond yields
        self.data['bonds'] = pd.read_csv(os.path.join(self.raw_data_path, 'German_bondyield_data.csv'))
        self.data['bonds']['Dates'] = pd.to_datetime(self.data['bonds']['Dates'])
        
        # Rename bond yield columns for consistency
        self.data['bonds'] = self.data['bonds'].rename(columns={
            '10 Y Bond Yield': '10Y',
            '2 Y Bond Yield': '2Y',
            '5 Y Bond Yield': '5Y',
            '1 year Bond Yield': '1Y'
        })
        
        # Load oil prices
        self.data['oil'] = pd.read_csv(os.path.join(self.raw_data_path, 'Brent_Oil_Prices_data.csv'))
        self.data['oil']['Dates'] = pd.to_datetime(self.data['oil']['Dates'])
        
        # Rename oil price column for consistency
        self.data['oil'] = self.data['oil'].rename(columns={
            'Oil Prices Brent': 'Brent Oil Price'
        })
        
        # Load Fed Funds Rate
        self.data['fed'] = pd.read_csv(os.path.join(self.raw_data_path, 'Fed_Fund_Rate_data.csv'))
        self.data['fed']['Dates'] = pd.to_datetime(self.data['fed']['Dates'])
        
        # Rename Fed Funds Rate column for consistency
        self.data['fed'] = self.data['fed'].rename(columns={
            'Fed Funds Rate': 'Fed_Fund_Rate'
        })
        
        # Load GPR data
        self.data['gpr'] = pd.read_csv(os.path.join(self.raw_data_path, 'GPR_data.csv'))
        # Convert month column to datetime
        self.data['gpr']['Date'] = pd.to_datetime(self.data['gpr']['month'], format='%d-%m-%Y')
        
        logging.info("All data files loaded successfully")

    def validate_data(self) -> None:
        """Validate data formats and types before processing."""
        logging.info("Validating data...")
        
        for dataset_name, df in self.data.items():
            logging.info(f"Validating {dataset_name} dataset with columns: {df.columns.tolist()}")
            
            # Check for required columns based on dataset
            if dataset_name == 'macro':
                required_cols = ['Date', 'Eurozone GDP Growth', 'CPI_Inflation_YoY', 'Core_CPI']
            elif dataset_name == 'market':
                required_cols = ['Dates', 'ECBRateFutures', 'EURUSD Exchange rate', 'EuroStoxx 50 (Equity Markets)']
            elif dataset_name == 'bonds':
                required_cols = ['Dates', '10Y', '2Y', '5Y', '1Y']
            elif dataset_name == 'oil':
                required_cols = ['Dates', 'Brent Oil Price']
            elif dataset_name == 'fed':
                required_cols = ['Dates', 'Fed_Fund_Rate']
            elif dataset_name == 'gpr':
                required_cols = ['Date', 'GPR', 'GPRC_DEU', 'GPRHC_USA']
            
            logging.info(f"Required columns for {dataset_name}: {required_cols}")
            
            # Check if all required columns exist
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in {dataset_name}: {missing_cols}")
            
            # Check for non-numeric values in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            logging.info(f"Numeric columns in {dataset_name}: {numeric_cols.tolist()}")
            
            for col in numeric_cols:
                if df[col].dtype not in [np.float64, np.int64]:
                    raise ValueError(f"Column {col} in {dataset_name} should be numeric but is {df[col].dtype}")
            
            logging.info(f"Data validation passed for {dataset_name}")

    def handle_missing_data(self) -> None:
        """Handle missing values in all datasets with appropriate interpolation methods."""
        logging.info("Handling missing data...")
        
        for dataset_name, df in self.data.items():
            # Identify datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            
            # Forward fill first (including datetime columns)
            df.fillna(method='ffill', inplace=True)
            
            # Then interpolate remaining missing values (excluding datetime columns)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if dataset_name in ['macro', 'fed', 'gpr']:
                # Use cubic interpolation for monthly data
                df[numeric_cols] = df[numeric_cols].interpolate(method='cubic')
            else:
                # Use linear interpolation for daily data
                df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            
            # Backward fill any remaining NaNs (including datetime columns)
            df.fillna(method='bfill', inplace=True)
            
            self.data[dataset_name] = df
            logging.info(f"Missing data handled for {dataset_name}")

    def create_yield_curve_features(self) -> None:
        """Create yield curve features from bond data including slopes, curvature and volatility."""
        logging.info("Creating yield curve features...")
        
        bonds_df = self.data['bonds']
        
        # Calculate yield curve slopes
        bonds_df['slope_10y_2y'] = bonds_df['10Y'] - bonds_df['2Y']
        bonds_df['slope_5y_1y'] = bonds_df['5Y'] - bonds_df['1Y']
        
        # Calculate yield curve curvature
        bonds_df['curvature'] = 2 * bonds_df['5Y'] - (bonds_df['2Y'] + bonds_df['10Y'])
        
        # Calculate yield volatility (30-day rolling std)
        for maturity in ['1Y', '2Y', '5Y', '10Y']:
            bonds_df[f'{maturity}_volatility'] = bonds_df[maturity].rolling(window=30).std()
        
        self.data['bonds'] = bonds_df
        logging.info("Yield curve features created successfully")

    def create_economic_indicators(self) -> None:
        """Create economic indicators from macro data including YoY changes and momentum indicators."""
        logging.info("Creating economic indicators...")
        
        macro_df = self.data['macro']
        
        # Calculate YoY changes
        for col in ['Eurozone GDP Growth', 'CPI_Inflation_YoY', 'Core_CPI', 
                   'Industrial_Production', 'Retail_Sales', 'M3_Money_Supply']:
            macro_df[f'{col}_YoY'] = macro_df[col].pct_change(periods=12) * 100
        
        # Calculate momentum indicators
        for col in ['Eurozone GDP Growth', 'CPI_Inflation_YoY', 'Core_CPI']:
            macro_df[f'{col}_3m_change'] = macro_df[col].pct_change(periods=3) * 100
            macro_df[f'{col}_6m_change'] = macro_df[col].pct_change(periods=6) * 100
        
        self.data['macro'] = macro_df
        logging.info("Economic indicators created successfully")

    def create_market_indicators(self) -> None:
        """Create market indicators from market data including returns and volatility metrics."""
        logging.info("Creating market indicators...")
        
        market_df = self.data['market']
        oil_df = self.data['oil']
        
        # Calculate returns and volatility for EuroStoxx 50
        market_df['EuroStoxx_returns'] = market_df['EuroStoxx 50 (Equity Markets)'].pct_change()
        market_df['EuroStoxx_volatility'] = market_df['EuroStoxx_returns'].rolling(window=30).std()
        
        # Calculate EUR/USD volatility
        market_df['EURUSD_volatility'] = market_df['EURUSD Exchange rate'].rolling(window=30).std()
        
        # Calculate oil price changes and volatility
        oil_df['Oil_returns'] = oil_df['Brent Oil Price'].pct_change()
        oil_df['Oil_volatility'] = oil_df['Oil_returns'].rolling(window=30).std()
        
        self.data['market'] = market_df
        self.data['oil'] = oil_df
        logging.info("Market indicators created successfully")

    def create_policy_indicators(self) -> None:
        """Create policy-related indicators such as rate spreads and rate changes."""
        logging.info("Creating policy indicators...")
        
        # Calculate Fed Funds Rate spread vs ECB rate
        self.data['fed']['Fed_ECB_spread'] = self.data['fed']['Fed_Fund_Rate'] - self.data['market']['ECBRateFutures']
        
        # Calculate policy rate changes
        self.data['market']['ECB_rate_change'] = self.data['market']['ECBRateFutures'].diff()
        self.data['market']['ECB_rate_acceleration'] = self.data['market']['ECB_rate_change'].diff()
        
        logging.info("Policy indicators created successfully")

    def create_lag_features(self) -> None:
        """Create lag features for relevant variables to capture temporal dynamics."""
        logging.info("Creating lag features...")
        
        # Market data lags (1-5 days)
        market_df = self.data['market']
        for col in ['ECBRateFutures', 'EURUSD Exchange rate', 'EuroStoxx 50 (Equity Markets)']:
            for lag in range(1, 6):
                market_df[f'{col}_lag_{lag}'] = market_df[col].shift(lag)
        
        # Macro data lags (1-3 months)
        macro_df = self.data['macro']
        for col in ['Eurozone GDP Growth', 'CPI_Inflation_YoY', 'Core_CPI']:
            for lag in range(1, 4):
                macro_df[f'{col}_lag_{lag}'] = macro_df[col].shift(lag)
        
        self.data['market'] = market_df
        self.data['macro'] = macro_df
        logging.info("Lag features created successfully")

    def clean_data(self) -> None:
        """Clean data by handling infinite values, outliers, and scaling large values."""
        logging.info("Cleaning data...")
        
        for dataset_name, df in self.data.items():
            # Replace infinite values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Handle percentage values (if they're too large)
            percentage_cols = [col for col in df.columns if any(x in col.lower() for x in ['growth', 'inflation', 'change', 'returns', 'yoy'])]
            for col in percentage_cols:
                # Cap percentage values at 1000% (10.0) and -1000% (-10.0)
                df[col] = df[col].clip(-10.0, 10.0)
            
            # Scale down very large monetary values
            if dataset_name == 'macro' and 'M3_Money_Supply' in df.columns:
                # Scale M3 Money Supply to millions
                df['M3_Money_Supply'] = df['M3_Money_Supply'] / 1000000
                logging.info("Scaled M3_Money_Supply to millions for easier handling")
            
            # Forward fill NaN values
            df.fillna(method='ffill', inplace=True)
            # Backward fill any remaining NaN values
            df.fillna(method='bfill', inplace=True)
            
            self.data[dataset_name] = df
            logging.info(f"Data cleaned for {dataset_name}")

    def validate_clean_data(self) -> None:
        """Validate that data is clean and ready for normalization."""
        logging.info("Validating cleaned data...")
        
        for dataset_name, df in self.data.items():
            # Check for infinite values
            inf_cols = df.columns[df.isin([np.inf, -np.inf]).any()]
            if len(inf_cols) > 0:
                raise ValueError(f"Infinite values found in {dataset_name} columns: {inf_cols.tolist()}")
            
            # Check for NaN values
            nan_cols = df.columns[df.isna().any()]
            if len(nan_cols) > 0:
                raise ValueError(f"NaN values found in {dataset_name} columns: {nan_cols.tolist()}")
            
            # Check for extremely large values (only for numeric columns)
            # Skip the M3_Money_Supply check since we've scaled it down already
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            exclude_cols = []
            if dataset_name == 'macro':
                # We've already scaled M3_Money_Supply so exclude it from the check
                exclude_cols = ['M3_Money_Supply']
            
            check_cols = [col for col in numeric_cols if col not in exclude_cols]
            large_vals = df[check_cols].abs().gt(1e6).any()
            large_cols = [col for i, col in enumerate(check_cols) if large_vals.iloc[i] if i < len(large_vals)]
            
            if len(large_cols) > 0:
                raise ValueError(f"Extremely large values found in {dataset_name} columns: {large_cols}")
            
            logging.info(f"Clean data validation passed for {dataset_name}")

    def normalize_data(self) -> None:
        """Normalize features using appropriate scalers for different data types."""
        logging.info("Normalizing data...")
        
        # Use StandardScaler for most features
        standard_scaler = StandardScaler()
        
        # Use MinMaxScaler for interest rates and yields
        minmax_scaler = MinMaxScaler()
        
        # Normalize market data
        market_cols = ['ECBRateFutures', 'EURUSD Exchange rate', 'EuroStoxx 50 (Equity Markets)']
        self.data['market'][market_cols] = minmax_scaler.fit_transform(self.data['market'][market_cols])
        
        # Normalize bond yields
        bond_cols = ['1Y', '2Y', '5Y', '10Y']
        self.data['bonds'][bond_cols] = minmax_scaler.fit_transform(self.data['bonds'][bond_cols])
        
        # Normalize macro data - exclude percentage columns
        macro_df = self.data['macro']
        percentage_cols = [col for col in macro_df.columns if any(x in col.lower() for x in ['growth', 'inflation', 'change', 'yoy'])]
        numeric_cols = macro_df.select_dtypes(include=[np.number]).columns
        non_percentage_cols = [col for col in numeric_cols if col not in percentage_cols]
        
        # Normalize non-percentage columns
        if non_percentage_cols:
            macro_df[non_percentage_cols] = standard_scaler.fit_transform(macro_df[non_percentage_cols])
        
        # Normalize percentage columns separately
        if percentage_cols:
            macro_df[percentage_cols] = macro_df[percentage_cols].clip(-10.0, 10.0) / 10.0
        
        self.data['macro'] = macro_df
        logging.info("Data normalization completed")

    def combine_datasets(self) -> pd.DataFrame:
        """Combine all processed datasets into a single DataFrame using time-based alignment."""
        logging.info("Combining datasets...")
        
        # Start with market data as base
        final_df = self.data['market'].copy()
        
        # Merge with macro data (monthly)
        final_df = pd.merge_asof(
            final_df,
            self.data['macro'],
            left_on='Dates',
            right_on='Date',
            direction='backward'
        )
        
        # Merge with bond data
        final_df = pd.merge_asof(
            final_df,
            self.data['bonds'],
            left_on='Dates',
            right_on='Dates',
            direction='backward'
        )
        
        # Merge with oil data
        final_df = pd.merge_asof(
            final_df,
            self.data['oil'],
            left_on='Dates',
            right_on='Dates',
            direction='backward'
        )
        
        # Merge with Fed Funds Rate
        final_df = pd.merge_asof(
            final_df,
            self.data['fed'],
            left_on='Dates',
            right_on='Dates',
            direction='backward'
        )
        
        # Merge with GPR data
        final_df = pd.merge_asof(
            final_df,
            self.data['gpr'],
            left_on='Dates',
            right_on='Date',
            direction='backward'
        )
        
        # Create target variable (ECB rate changes)
        final_df['ECB_rate_change'] = final_df['ECBRateFutures'].diff()
        
        # Drop rows with missing values
        final_df.dropna(inplace=True)
        
        logging.info("Datasets combined successfully")
        return final_df

    def save_processed_data(self, df: pd.DataFrame) -> None:
        """Save the processed dataset to CSV file."""
        output_path = os.path.join(self.processed_data_path, 'processed_data.csv')
        df.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}")

    def run_pipeline(self) -> None:
        """Run the complete preprocessing pipeline in sequence."""
        try:
            self.load_data()
            self.validate_data()
            self.handle_missing_data()
            self.create_yield_curve_features()
            self.create_economic_indicators()
            self.create_market_indicators()
            self.create_policy_indicators()
            self.create_lag_features()
            self.clean_data()
            self.validate_clean_data()
            self.normalize_data()
            final_df = self.combine_datasets()
            self.save_processed_data(final_df)
            logging.info("Preprocessing pipeline completed successfully")
        except Exception as e:
            logging.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_pipeline() 
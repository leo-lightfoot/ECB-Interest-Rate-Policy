import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import os
import logging
from typing import List, Tuple
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecb_preprocessing.log'),
        logging.StreamHandler()
    ]
)

class ECBMeetingPreprocessor:
    """
    A data preprocessing pipeline for ECB interest rate policy prediction,
    organized around the 6-week ECB meeting schedule.
    
    This class handles loading, cleaning, and organizing data to align with
    ECB's 6-week meeting intervals to better reflect the actual policy-making process.
    """
    
    def __init__(self, raw_data_path: str = 'Raw_Data', processed_data_path: str = 'Processed_Data'):
        """
        Initialize the ECBMeetingPreprocessor.
        
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
        self.ecb_meetings = []
        self.meeting_data = pd.DataFrame()
        
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
        self.data['bonds'] = self.data['bonds'].rename(columns={
            '10 Y Bond Yield': '10Y',
            '2 Y Bond Yield': '2Y',
            '5 Y Bond Yield': '5Y',
            '1 year Bond Yield': '1Y'
        })
        
        # Load oil prices
        self.data['oil'] = pd.read_csv(os.path.join(self.raw_data_path, 'Brent_Oil_Prices_data.csv'))
        self.data['oil']['Dates'] = pd.to_datetime(self.data['oil']['Dates'])
        self.data['oil'] = self.data['oil'].rename(columns={
            'Oil Prices Brent': 'Brent Oil Price'
        })
        
        # Load Fed Funds Rate
        self.data['fed'] = pd.read_csv(os.path.join(self.raw_data_path, 'Fed_Fund_Rate_data.csv'))
        self.data['fed']['Dates'] = pd.to_datetime(self.data['fed']['Dates'])
        self.data['fed'] = self.data['fed'].rename(columns={
            'Fed Funds Rate': 'Fed_Fund_Rate'
        })
        
        # Load GPR data
        self.data['gpr'] = pd.read_csv(os.path.join(self.raw_data_path, 'GPR_data.csv'))
        self.data['gpr']['Date'] = pd.to_datetime(self.data['gpr']['month'], format='%d-%m-%Y')
        
        # Load ECB Interest Rate data
        self.load_ecb_rate_data()
        
        logging.info("All data files loaded successfully")
        
    def load_ecb_rate_data(self) -> None:
        """
        Load ECB interest rate data and create consistent 6-week meeting windows.
        If there is any rate change within a 6-week window, it's marked as a change period.
        Otherwise, it's considered a hold period.
        """
        ecb_rates_path = os.path.join(self.raw_data_path, 'ECBDFR.csv')
        
        if os.path.exists(ecb_rates_path):
            logging.info("Loading ECB interest rate data...")
            
            ecb_rates_df = pd.read_csv(ecb_rates_path)
            
            if 'observation_date' in ecb_rates_df.columns and 'ECBDFR' in ecb_rates_df.columns:
                ecb_rates_df['observation_date'] = pd.to_datetime(ecb_rates_df['observation_date'])
                
                ecb_rates_df = ecb_rates_df.rename(columns={
                    'observation_date': 'Dates',
                    'ECBDFR': 'ECB_rate_actual'
                })
                
                ecb_rates_df = ecb_rates_df.sort_values('Dates')
                
                ecb_rates_df['rate_change'] = ecb_rates_df['ECB_rate_actual'].diff()
                rate_change_dates = ecb_rates_df[ecb_rates_df['rate_change'] != 0]['Dates'].tolist()
                
                if rate_change_dates:
                    first_meeting = rate_change_dates[0]
                    logging.info(f"First rate change detected on: {first_meeting}")
                    
                    self.create_consistent_meeting_schedule(first_meeting)
                    self.identify_rate_change_meetings(ecb_rates_df)
                else:
                    first_meeting = pd.to_datetime('2000-02-04')
                    logging.warning("No rate changes found. Using default first meeting date.")
                    self.create_consistent_meeting_schedule(first_meeting)
                
                self.data['ecb_rates'] = ecb_rates_df
                
                logging.info(f"Loaded ECB rates data with {len(ecb_rates_df)} rows")
                logging.info(f"Created {len(self.ecb_meetings)} consistent 6-week meeting periods")
            else:
                logging.warning("ECB interest rate data file missing required columns. Using default schedule.")
                self.create_consistent_meeting_schedule(pd.to_datetime('2000-02-04'))
        else:
            logging.warning(f"ECB interest rate data file not found at {ecb_rates_path}. Using default schedule.")
            self.create_consistent_meeting_schedule(pd.to_datetime('2000-02-04'))

    def create_consistent_meeting_schedule(self, start_date) -> None:
        """
        Create a consistent 6-week meeting schedule starting from the given date.
        """
        logging.info(f"Creating consistent 6-week ECB meeting schedule from {start_date}")
        
        # Find the latest date in our data to set the end of the schedule
        end_dates = []
        for dataset_name, df in self.data.items():
            date_col = 'Date' if 'Date' in df.columns else 'Dates'
            if date_col in df.columns:
                end_dates.append(df[date_col].max())
        
        last_date = max(end_dates) if end_dates else pd.to_datetime('2025-03-12')
        
        # Generate meetings every 6 weeks
        meetings = [start_date]
        current_date = start_date
        
        while current_date < last_date:
            # Add 6 weeks (42 days) for the next meeting
            current_date = current_date + timedelta(days=42)
            meetings.append(current_date)
        
        self.ecb_meetings = meetings
        logging.info(f"Created consistent schedule with {len(meetings)} meetings from {start_date} to {meetings[-1]}")
        
    def identify_rate_change_meetings(self, ecb_rates_df) -> None:
        """
        Identify which of the 6-week meeting periods had a rate change.
        Stores this information in the meeting_changes data.
        """
        meeting_rate_changes = []
        
        # For each meeting period, check if there was a rate change
        for i in range(len(self.ecb_meetings) - 1):
            meeting_date = self.ecb_meetings[i]
            next_meeting_date = self.ecb_meetings[i + 1]
            
            # Get the rate at this meeting
            current_rate = ecb_rates_df[ecb_rates_df['Dates'] <= meeting_date]['ECB_rate_actual'].iloc[-1] if not ecb_rates_df[ecb_rates_df['Dates'] <= meeting_date].empty else 0
            
            # Check for any rate changes in this period
            period_data = ecb_rates_df[(ecb_rates_df['Dates'] > meeting_date) & 
                                       (ecb_rates_df['Dates'] <= next_meeting_date)]
            
            rate_changed = False
            next_rate_change = 0.0
            
            if not period_data.empty:
                # Get the rate at the end of the period
                end_period_rate = period_data['ECB_rate_actual'].iloc[-1]
                next_rate_change = end_period_rate - current_rate
                rate_changed = abs(next_rate_change) > 0
            
            meeting_rate_changes.append({
                'Meeting_Date': meeting_date,
                'ECB_rate': current_rate,
                'Next_Change': next_rate_change,
                'Rate_Changed': rate_changed
            })
        
        # Add the last meeting (with no next change info)
        if self.ecb_meetings:
            last_meeting = self.ecb_meetings[-1]
            last_rate = ecb_rates_df[ecb_rates_df['Dates'] <= last_meeting]['ECB_rate_actual'].iloc[-1] if not ecb_rates_df[ecb_rates_df['Dates'] <= last_meeting].empty else 0
            
            meeting_rate_changes.append({
                'Meeting_Date': last_meeting,
                'ECB_rate': last_rate,
                'Next_Change': 0.0,
                'Rate_Changed': False
            })
        
        self.data['meeting_changes'] = pd.DataFrame(meeting_rate_changes)
        logging.info(f"Identified rate changes for {len(meeting_rate_changes)} consistent meeting periods")

    def generate_meeting_intervals(self) -> List[Tuple[datetime, datetime]]:
        """
        Generate time intervals between ECB meetings.
        
        Returns:
        --------
        List[Tuple[datetime, datetime]]
            List of tuples with (start_date, end_date) for each interval
        """
        if not self.ecb_meetings:
            raise ValueError("No ECB meeting dates available. Run load_ecb_rate_data first.")
        
        # Sort meeting dates to ensure chronological order
        sorted_meetings = sorted(self.ecb_meetings)
        
        # Create intervals between meetings
        meeting_intervals = []
        for i in range(len(sorted_meetings) - 1):
            start_date = sorted_meetings[i]
            end_date = sorted_meetings[i+1] - timedelta(days=1)  # Exclude the next meeting date
            meeting_intervals.append((start_date, end_date))
        
        return meeting_intervals
        
    def handle_missing_data(self) -> None:
        """Handle missing values in all datasets with appropriate interpolation methods."""
        logging.info("Handling missing data...")
        
        for dataset_name, df in self.data.items():
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
            if col in macro_df.columns:
                macro_df[f'{col}_YoY'] = macro_df[col].pct_change(periods=12) * 100
        
        # Calculate momentum indicators
        for col in ['Eurozone GDP Growth', 'CPI_Inflation_YoY', 'Core_CPI']:
            if col in macro_df.columns:
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
        if 'EuroStoxx 50 (Equity Markets)' in market_df.columns:
            market_df['EuroStoxx_returns'] = market_df['EuroStoxx 50 (Equity Markets)'].pct_change()
            market_df['EuroStoxx_volatility'] = market_df['EuroStoxx_returns'].rolling(window=30).std()
        
        # Calculate EUR/USD volatility
        if 'EURUSD Exchange rate' in market_df.columns:
            market_df['EURUSD_volatility'] = market_df['EURUSD Exchange rate'].rolling(window=30).std()
        
        # Calculate oil price changes and volatility
        if 'Brent Oil Price' in oil_df.columns:
            oil_df['Oil_returns'] = oil_df['Brent Oil Price'].pct_change()
            oil_df['Oil_volatility'] = oil_df['Oil_returns'].rolling(window=30).std()
        
        self.data['market'] = market_df
        self.data['oil'] = oil_df
        logging.info("Market indicators created successfully")
        
    def create_policy_indicators(self) -> None:
        """Create policy-related indicators tailored to the ECB 6-week meeting schedule."""
        logging.info("Creating policy indicators...")
        
        # Calculate Fed Funds Rate spread vs ECB rate
        if 'ecb_rates' in self.data and 'fed' in self.data:
            # Get the Fed Fund Rate on ECB meeting days or closest day
            fed_df = self.data['fed']
            ecb_df = self.data['ecb_rates']
            
            # For each ECB meeting, find the corresponding Fed rate
            meeting_fed_rates = []
            for meeting_date in self.ecb_meetings:
                # Find closest Fed rate date before or on meeting date
                closest_date = fed_df[fed_df['Dates'] <= meeting_date]['Dates'].max()
                if pd.notna(closest_date):
                    fed_rate = fed_df[fed_df['Dates'] == closest_date]['Fed_Fund_Rate'].values[0]
                    ecb_rate = ecb_df[ecb_df['Dates'] <= meeting_date]['ECB_rate_actual'].iloc[-1]
                    meeting_fed_rates.append({
                        'Meeting_Date': meeting_date,
                        'Fed_Fund_Rate': fed_rate,
                        'ECB_rate_actual': ecb_rate,
                        'Fed_ECB_spread': fed_rate - ecb_rate
                    })
            
            # Create a DataFrame with Fed-ECB spreads for each meeting
            self.data['policy_spreads'] = pd.DataFrame(meeting_fed_rates)
            logging.info(f"Created Fed-ECB rate spread data for {len(meeting_fed_rates)} meetings")
        
        # Use the meeting_changes data created during load_ecb_rate_data
        # No need to recalculate rate changes here
        
        logging.info("Policy indicators created successfully")
        
    def aggregate_by_meeting_interval(self) -> None:
        """
        Aggregate data by ECB meeting intervals (6-week periods).
        This creates a dataset where each row represents one 6-week interval between meetings.
        """
        logging.info("Aggregating data by ECB meeting intervals...")
        
        # Generate meeting intervals
        meeting_intervals = self.generate_meeting_intervals()
        
        # Create combined base dataframe with all data
        combined_df = self.combine_datasets()
        
        # List to store aggregated data for each meeting interval
        meeting_aggregated_data = []
        
        for interval_idx, (start_date, end_date) in enumerate(meeting_intervals):
            # Filter data for this interval
            interval_data = combined_df[(combined_df['Dates'] >= start_date) & 
                                        (combined_df['Dates'] <= end_date)]
            
            if interval_data.empty:
                logging.warning(f"No data for interval {start_date} to {end_date}. Skipping.")
                continue
            
            # Get the meeting date (start of the interval)
            meeting_date = start_date
            
            # Determine target (next meeting's decision)
            next_meeting_idx = interval_idx + 1
            if next_meeting_idx < len(self.ecb_meetings):
                next_meeting_date = self.ecb_meetings[next_meeting_idx]
                
                # Get the rate change at the next meeting
                if 'meeting_changes' in self.data:
                    next_change = self.data['meeting_changes'][
                        self.data['meeting_changes']['Meeting_Date'] == next_meeting_date
                    ]['Next_Change'].values
                    
                    rate_change = next_change[0] if len(next_change) > 0 else 0.0
                else:
                    # Fallback if meeting changes data unavailable
                    if 'ecb_rates' in self.data:
                        ecb_df = self.data['ecb_rates']
                        current_rate = ecb_df[ecb_df['Dates'] <= meeting_date]['ECB_rate_actual'].iloc[-1]
                        next_rate = ecb_df[ecb_df['Dates'] <= next_meeting_date]['ECB_rate_actual'].iloc[-1]
                        rate_change = next_rate - current_rate
                    else:
                        rate_change = 0.0
            else:
                # No next meeting (last interval)
                rate_change = 0.0
            
            # Aggregate features for this interval
            agg_data = {
                'Meeting_Date': meeting_date,
                'Interval_Start': start_date,
                'Interval_End': end_date,
                'Next_Rate_Change': rate_change,
            }
            
            # Aggregate numerical features by mean, min, max, std
            numeric_cols = interval_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'Dates':  # Skip date columns
                    agg_data[f'{col}_mean'] = interval_data[col].mean()
                    agg_data[f'{col}_min'] = interval_data[col].min()
                    agg_data[f'{col}_max'] = interval_data[col].max()
                    agg_data[f'{col}_std'] = interval_data[col].std()
                    
                    # For rate-related features, also capture the end of period value
                    if any(term in col.lower() for term in ['rate', 'yield', 'inflation', 'gdp']):
                        end_value = interval_data.loc[interval_data['Dates'].idxmax(), col]
                        agg_data[f'{col}_end'] = end_value
            
            meeting_aggregated_data.append(agg_data)
        
        # Convert to DataFrame
        self.meeting_data = pd.DataFrame(meeting_aggregated_data)
        
        logging.info(f"Created aggregated dataset with {len(self.meeting_data)} meeting intervals")
        
    def clean_meeting_data(self) -> None:
        """Clean meeting data by handling infinite values, outliers, and other issues."""
        logging.info("Cleaning meeting data...")
        
        if self.meeting_data.empty:
            raise ValueError("Meeting data is empty. Run aggregate_by_meeting_interval first.")
        
        # Identify date columns
        date_cols = ['Meeting_Date', 'Interval_Start', 'Interval_End']
        
        # Get all numeric columns
        numeric_cols = self.meeting_data.select_dtypes(include=[np.number]).columns
        
        # Replace infinite values with NaN
        self.meeting_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # For each numeric column, handle outliers and cap extreme values
        for col in numeric_cols:
            # Ignore date columns and columns with all NaN
            if col in date_cols or self.meeting_data[col].isna().all():
                continue
                
            # Calculate reasonable bounds for the data (3 std from mean)
            col_mean = self.meeting_data[col].mean()
            col_std = self.meeting_data[col].std()
            
            if pd.notna(col_mean) and pd.notna(col_std) and col_std > 0:
                lower_bound = col_mean - 5 * col_std
                upper_bound = col_mean + 5 * col_std
                
                # Cap values outside these bounds
                self.meeting_data[col] = self.meeting_data[col].clip(lower_bound, upper_bound)
            
            # Fill remaining NaN with median (more robust than mean)
            if self.meeting_data[col].isna().any():
                self.meeting_data[col].fillna(self.meeting_data[col].median(), inplace=True)
        
        # Log columns with remaining issues
        problematic_cols = []
        for col in numeric_cols:
            if self.meeting_data[col].isna().any() or np.isinf(self.meeting_data[col]).any():
                problematic_cols.append(col)
                
        if problematic_cols:
            logging.warning(f"Columns with remaining NA or infinite values: {problematic_cols}")
            # Drop problematic columns if they can't be fixed
            self.meeting_data.drop(columns=problematic_cols, inplace=True)
            logging.warning(f"Dropped {len(problematic_cols)} problematic columns")
        
        logging.info("Meeting data cleaned successfully")
    
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
        
        # Add ECB interest rate data if available
        if 'ecb_rates' in self.data:
            final_df = pd.merge_asof(
                final_df,
                self.data['ecb_rates'],
                left_on='Dates',
                right_on='Dates',
                direction='backward'
            )
        
        # Drop rows with missing values
        final_df.dropna(inplace=True)
        
        logging.info(f"Datasets combined successfully with {len(final_df)} rows")
        return final_df
    
    def normalize_meeting_data(self) -> None:
        """Normalize the meeting aggregated data for machine learning models."""
        logging.info("Normalizing meeting data...")
        
        if self.meeting_data.empty:
            raise ValueError("Meeting data is empty. Run aggregate_by_meeting_interval first.")
        
        # Identify columns to normalize (exclude date columns and target)
        date_cols = ['Meeting_Date', 'Interval_Start', 'Interval_End']
        target_col = 'Next_Rate_Change'
        
        # Get all numeric columns except target
        numeric_cols = self.meeting_data.select_dtypes(include=[np.number]).columns
        normalize_cols = [col for col in numeric_cols if col != target_col]
        
        # Use StandardScaler for most features
        standard_scaler = StandardScaler()
        
        # Normalize the data
        if normalize_cols:
            self.meeting_data[normalize_cols] = standard_scaler.fit_transform(self.meeting_data[normalize_cols])
            self.scalers['meeting_data'] = standard_scaler
        
        logging.info("Meeting data normalization completed")
    
    def save_processed_data(self) -> None:
        """Save only the processed meeting data CSV file."""
        logging.info("Saving processed data...")
        
        if self.meeting_data is None or self.meeting_data.empty:
            raise ValueError("No meeting data to save. Run the pipeline first.")
        
        # Create directory if it doesn't exist
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # Save only the meeting data
        try:
            meeting_data_path = os.path.join(self.processed_data_path, 'ecb_meeting_data.csv')
            self.meeting_data.to_csv(meeting_data_path, index=False)
            logging.info(f"Meeting data saved to {meeting_data_path}")
        except Exception as e:
            logging.error(f"Error saving processed data: {str(e)}")
            
            # Try to save to a backup file
            try:
                backup_filename = f"ecb_meeting_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                emergency_path = os.path.join(
                    self.processed_data_path, 
                    backup_filename
                )
                self.meeting_data.to_csv(emergency_path, index=False)
                logging.info(f"Processed data saved to emergency file: {emergency_path}")
            except Exception as e2:
                logging.error(f"Failed to save even to emergency file: {str(e2)}")
                raise
    
    def run_pipeline(self) -> None:
        """Run the complete preprocessing pipeline in sequence."""
        try:
            logging.info("Starting ECB meeting-based preprocessing pipeline...")
            
            # Load and clean the data
            self.load_data()
            self.handle_missing_data()
            
            # Create features
            self.create_yield_curve_features()
            self.create_economic_indicators()
            self.create_market_indicators()
            self.create_policy_indicators()
            
            # Aggregate by meeting interval
            self.aggregate_by_meeting_interval()
            
            # Clean and normalize the meeting data
            self.clean_meeting_data()
            self.normalize_meeting_data()
            
            # Save the processed data
            self.save_processed_data()
            
            logging.info("ECB meeting-based preprocessing pipeline completed successfully")
        except Exception as e:
            logging.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    preprocessor = ECBMeetingPreprocessor()
    preprocessor.run_pipeline() 
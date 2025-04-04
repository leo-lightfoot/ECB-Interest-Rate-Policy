# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import os
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')

class ECBMeetingPreprocessor:
    
    def __init__(self, raw_data_path: str = 'Raw_Data', processed_data_path: str = 'Processed_Data'):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.data = {}
        self.scalers = {}
        self.ecb_meetings = []
        self.meeting_data = pd.DataFrame()
        
        if not os.path.exists(processed_data_path):
            os.makedirs(processed_data_path)
            
    def load_data(self) -> None:
        """
        Loads all necessary datasets from raw CSV files and handles column naming.
        This function is foundational for the entire pipeline, gathering all required data sources.
        """
        self.data['macro'] = pd.read_csv(os.path.join(self.raw_data_path, 'macro_data.csv'))
        self.data['macro']['Date'] = pd.to_datetime(self.data['macro']['Date'], format='%d-%m-%Y')
        
        self.data['market'] = pd.read_csv(os.path.join(self.raw_data_path, 'market_data.csv'))
        self.data['market']['Dates'] = pd.to_datetime(self.data['market']['Dates'], format='%d-%m-%Y')
        
        self.data['bonds'] = pd.read_csv(os.path.join(self.raw_data_path, 'German_bondyield_data.csv'))
        self.data['bonds']['Dates'] = pd.to_datetime(self.data['bonds']['Dates'])
        self.data['bonds'] = self.data['bonds'].rename(columns={
            '10 Y Bond Yield': '10Y',
            '2 Y Bond Yield': '2Y',
            '5 Y Bond Yield': '5Y',
            '1 year Bond Yield': '1Y'
        })
        
        self.data['oil'] = pd.read_csv(os.path.join(self.raw_data_path, 'Brent_Oil_Prices_data.csv'))
        self.data['oil']['Dates'] = pd.to_datetime(self.data['oil']['Dates'])
        self.data['oil'] = self.data['oil'].rename(columns={
            'Oil Prices Brent': 'Brent Oil Price'
        })
        
        self.data['fed'] = pd.read_csv(os.path.join(self.raw_data_path, 'Fed_Fund_Rate_data.csv'))
        self.data['fed']['Dates'] = pd.to_datetime(self.data['fed']['Dates'])
        self.data['fed'] = self.data['fed'].rename(columns={
            'Fed Funds Rate': 'Fed_Fund_Rate'
        })
        
        self.data['gpr'] = pd.read_csv(os.path.join(self.raw_data_path, 'GPR_data.csv'))
        self.data['gpr']['Date'] = pd.to_datetime(self.data['gpr']['month'], format='%d-%m-%Y')
        
        self.load_ecb_rate_data()
        
    def load_ecb_rate_data(self) -> None:
        """
        Core function that loads ECB interest rate data and establishes the 6-week meeting schedule.
        This is critical as it creates the fundamental temporal structure used throughout the analysis.
        """
        ecb_rates_path = os.path.join(self.raw_data_path, 'ECBDFR.csv')
        
        if os.path.exists(ecb_rates_path):
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
                    self.create_consistent_meeting_schedule(first_meeting)
                    self.identify_rate_change_meetings(ecb_rates_df)
                else:
                    first_meeting = pd.to_datetime('2000-02-04')
                    self.create_consistent_meeting_schedule(first_meeting)
                
                self.data['ecb_rates'] = ecb_rates_df
            else:
                self.create_consistent_meeting_schedule(pd.to_datetime('2000-02-04'))
        else:
            self.create_consistent_meeting_schedule(pd.to_datetime('2000-02-04'))

    def create_consistent_meeting_schedule(self, start_date) -> None:
        end_dates = []
        for dataset_name, df in self.data.items():
            date_col = 'Date' if 'Date' in df.columns else 'Dates'
            if date_col in df.columns:
                end_dates.append(df[date_col].max())
        
        last_date = max(end_dates) if end_dates else pd.to_datetime('2025-03-12')
        
        meetings = [start_date]
        current_date = start_date
        
        while current_date < last_date:
            current_date = current_date + timedelta(days=42)
            meetings.append(current_date)
        
        self.ecb_meetings = meetings
        
    def identify_rate_change_meetings(self, ecb_rates_df) -> None:
        meeting_rate_changes = []
        
        for i in range(len(self.ecb_meetings) - 1):
            meeting_date = self.ecb_meetings[i]
            next_meeting_date = self.ecb_meetings[i + 1]
            
            current_rate = ecb_rates_df[ecb_rates_df['Dates'] <= meeting_date]['ECB_rate_actual'].iloc[-1] if not ecb_rates_df[ecb_rates_df['Dates'] <= meeting_date].empty else 0
            
            period_data = ecb_rates_df[(ecb_rates_df['Dates'] > meeting_date) & 
                                       (ecb_rates_df['Dates'] <= next_meeting_date)]
            
            rate_changed = False
            next_rate_change = 0.0
            
            if not period_data.empty:
                end_period_rate = period_data['ECB_rate_actual'].iloc[-1]
                next_rate_change = end_period_rate - current_rate
                rate_changed = abs(next_rate_change) > 0
            
            meeting_rate_changes.append({
                'Meeting_Date': meeting_date,
                'ECB_rate': current_rate,
                'Next_Change': next_rate_change,
                'Rate_Changed': rate_changed
            })
        
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

    def generate_meeting_intervals(self) -> List[Tuple[datetime, datetime]]:
        if not self.ecb_meetings:
            raise ValueError("No ECB meeting dates available. Run load_ecb_rate_data first.")
        
        sorted_meetings = sorted(self.ecb_meetings)
        
        meeting_intervals = []
        for i in range(len(sorted_meetings) - 1):
            start_date = sorted_meetings[i]
            end_date = sorted_meetings[i+1] - timedelta(days=1)
            meeting_intervals.append((start_date, end_date))
        
        return meeting_intervals
        
    def handle_missing_data(self) -> None:
        """
        Handles missing values using appropriate interpolation methods for different data types.
        Critical for ensuring complete datasets without gaps before feature engineering.
        """
        for dataset_name, df in self.data.items():
            df.fillna(method='ffill', inplace=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if dataset_name in ['macro', 'fed', 'gpr']:
                df[numeric_cols] = df[numeric_cols].interpolate(method='cubic')
            else:
                df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            
            df.fillna(method='bfill', inplace=True)
            
            self.data[dataset_name] = df
            
    def create_yield_curve_features(self) -> None:
        bonds_df = self.data['bonds']
        
        bonds_df['slope_10y_2y'] = bonds_df['10Y'] - bonds_df['2Y']
        bonds_df['slope_5y_1y'] = bonds_df['5Y'] - bonds_df['1Y']
        
        bonds_df['curvature'] = 2 * bonds_df['5Y'] - (bonds_df['2Y'] + bonds_df['10Y'])
        
        for maturity in ['1Y', '2Y', '5Y', '10Y']:
            bonds_df[f'{maturity}_volatility'] = bonds_df[maturity].rolling(window=30).std()
        
        self.data['bonds'] = bonds_df
        
    def create_economic_indicators(self) -> None:
        macro_df = self.data['macro']
        
        for col in ['Eurozone GDP Growth', 'CPI_Inflation_YoY', 'Core_CPI', 
                   'Industrial_Production', 'Retail_Sales', 'M3_Money_Supply']:
            if col in macro_df.columns:
                macro_df[f'{col}_YoY'] = macro_df[col].pct_change(periods=12) * 100
        
        for col in ['Eurozone GDP Growth', 'CPI_Inflation_YoY', 'Core_CPI']:
            if col in macro_df.columns:
                macro_df[f'{col}_3m_change'] = macro_df[col].pct_change(periods=3) * 100
                macro_df[f'{col}_6m_change'] = macro_df[col].pct_change(periods=6) * 100
        
        self.data['macro'] = macro_df
        
    def create_market_indicators(self) -> None:
        market_df = self.data['market']
        oil_df = self.data['oil']
        
        if 'EuroStoxx 50 (Equity Markets)' in market_df.columns:
            market_df['EuroStoxx_returns'] = market_df['EuroStoxx 50 (Equity Markets)'].pct_change()
            market_df['EuroStoxx_volatility'] = market_df['EuroStoxx_returns'].rolling(window=30).std()
        
        if 'EURUSD Exchange rate' in market_df.columns:
            market_df['EURUSD_volatility'] = market_df['EURUSD Exchange rate'].rolling(window=30).std()
        
        if 'Brent Oil Price' in oil_df.columns:
            oil_df['Oil_returns'] = oil_df['Brent Oil Price'].pct_change()
            oil_df['Oil_volatility'] = oil_df['Oil_returns'].rolling(window=30).std()
        
        self.data['market'] = market_df
        self.data['oil'] = oil_df
        
    def create_policy_indicators(self) -> None:
        if 'ecb_rates' in self.data and 'fed' in self.data:
            fed_df = self.data['fed']
            ecb_df = self.data['ecb_rates']
            
            meeting_fed_rates = []
            for meeting_date in self.ecb_meetings:
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
            
            self.data['policy_spreads'] = pd.DataFrame(meeting_fed_rates)
        
    def aggregate_by_meeting_interval(self) -> None:
        """
        Key function that aligns all datasets by ECB's 6-week meeting intervals.
        This transforms the data from time-series to meeting-based format which directly maps to ECB's decision-making cycle.
        """
        meeting_intervals = self.generate_meeting_intervals()
        
        combined_df = self.combine_datasets()
        
        meeting_aggregated_data = []
        
        for interval_idx, (start_date, end_date) in enumerate(meeting_intervals):
            interval_data = combined_df[(combined_df['Dates'] >= start_date) & 
                                        (combined_df['Dates'] <= end_date)]
            
            if interval_data.empty:
                continue
            
            meeting_date = start_date
            
            next_meeting_idx = interval_idx + 1
            if next_meeting_idx < len(self.ecb_meetings):
                next_meeting_date = self.ecb_meetings[next_meeting_idx]
                
                if 'meeting_changes' in self.data:
                    next_change = self.data['meeting_changes'][
                        self.data['meeting_changes']['Meeting_Date'] == next_meeting_date
                    ]['Next_Change'].values
                    
                    rate_change = next_change[0] if len(next_change) > 0 else 0.0
                else:
                    if 'ecb_rates' in self.data:
                        ecb_df = self.data['ecb_rates']
                        current_rate = ecb_df[ecb_df['Dates'] <= meeting_date]['ECB_rate_actual'].iloc[-1]
                        next_rate = ecb_df[ecb_df['Dates'] <= next_meeting_date]['ECB_rate_actual'].iloc[-1]
                        rate_change = next_rate - current_rate
                    else:
                        rate_change = 0.0
            else:
                rate_change = 0.0
            
            agg_data = {
                'Meeting_Date': meeting_date,
                'Interval_Start': start_date,
                'Interval_End': end_date,
                'Next_Rate_Change': rate_change,
            }
            
            numeric_cols = interval_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'Dates':
                    agg_data[f'{col}_mean'] = interval_data[col].mean()
                    agg_data[f'{col}_min'] = interval_data[col].min()
                    agg_data[f'{col}_max'] = interval_data[col].max()
                    agg_data[f'{col}_std'] = interval_data[col].std()
                    
                    if any(term in col.lower() for term in ['rate', 'yield', 'inflation', 'gdp']):
                        end_value = interval_data.loc[interval_data['Dates'].idxmax(), col]
                        agg_data[f'{col}_end'] = end_value
            
            meeting_aggregated_data.append(agg_data)
        
        self.meeting_data = pd.DataFrame(meeting_aggregated_data)
        
    def clean_meeting_data(self) -> None:
        """
        Addresses infinite values, outliers, and missing data in the aggregated meeting dataset.
        Essential for preparing clean data for the model training process.
        """
        if self.meeting_data.empty:
            raise ValueError("Meeting data is empty. Run aggregate_by_meeting_interval first.")
        
        date_cols = ['Meeting_Date', 'Interval_Start', 'Interval_End']
        
        numeric_cols = self.meeting_data.select_dtypes(include=[np.number]).columns
        
        self.meeting_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        for col in numeric_cols:
            if col in date_cols or self.meeting_data[col].isna().all():
                continue
                
            col_mean = self.meeting_data[col].mean()
            col_std = self.meeting_data[col].std()
            
            if pd.notna(col_mean) and pd.notna(col_std) and col_std > 0:
                lower_bound = col_mean - 5 * col_std
                upper_bound = col_mean + 5 * col_std
                
                self.meeting_data[col] = self.meeting_data[col].clip(lower_bound, upper_bound)
            
            if self.meeting_data[col].isna().any():
                self.meeting_data[col].fillna(self.meeting_data[col].median(), inplace=True)
        
        problematic_cols = []
        for col in numeric_cols:
            if self.meeting_data[col].isna().any() or np.isinf(self.meeting_data[col]).any():
                problematic_cols.append(col)
                
        if problematic_cols:
            self.meeting_data.drop(columns=problematic_cols, inplace=True)
        
    def combine_datasets(self) -> pd.DataFrame:
        """
        Combines all processed datasets into a unified time-aligned DataFrame.
        This integration is vital as it brings together diverse data sources needed for comprehensive analysis.
        """
        final_df = self.data['market'].copy()
        
        final_df = pd.merge_asof(
            final_df,
            self.data['macro'],
            left_on='Dates',
            right_on='Date',
            direction='backward'
        )
        
        final_df = pd.merge_asof(
            final_df,
            self.data['bonds'],
            left_on='Dates',
            right_on='Dates',
            direction='backward'
        )
        
        final_df = pd.merge_asof(
            final_df,
            self.data['oil'],
            left_on='Dates',
            right_on='Dates',
            direction='backward'
        )
        
        final_df = pd.merge_asof(
            final_df,
            self.data['fed'],
            left_on='Dates',
            right_on='Dates',
            direction='backward'
        )
        
        final_df = pd.merge_asof(
            final_df,
            self.data['gpr'],
            left_on='Dates',
            right_on='Date',
            direction='backward'
        )
        
        if 'ecb_rates' in self.data:
            final_df = pd.merge_asof(
                final_df,
                self.data['ecb_rates'],
                left_on='Dates',
                right_on='Dates',
                direction='backward'
            )
        
        final_df.dropna(inplace=True)
        
        return final_df
    
    def normalize_meeting_data(self) -> None:
        if self.meeting_data.empty:
            raise ValueError("Meeting data is empty. Run aggregate_by_meeting_interval first.")
        
        target_col = 'Next_Rate_Change'
        
        numeric_cols = self.meeting_data.select_dtypes(include=[np.number]).columns
        normalize_cols = [col for col in numeric_cols if col != target_col]
        
        standard_scaler = StandardScaler()
        
        if normalize_cols:
            self.meeting_data[normalize_cols] = standard_scaler.fit_transform(self.meeting_data[normalize_cols])
            self.scalers['meeting_data'] = standard_scaler
    
    def save_processed_data(self) -> None:
        if self.meeting_data is None or self.meeting_data.empty:
            raise ValueError("No meeting data to save. Run the pipeline first.")
        
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        try:
            meeting_data_path = os.path.join(self.processed_data_path, 'ecb_meeting_data.csv')
            self.meeting_data.to_csv(meeting_data_path, index=False)
        except Exception:
            try:
                backup_filename = f"ecb_meeting_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                emergency_path = os.path.join(
                    self.processed_data_path, 
                    backup_filename
                )
                self.meeting_data.to_csv(emergency_path, index=False)
            except Exception:
                pass
    
    def run_pipeline(self) -> None:
        """
        Orchestrates the complete data preprocessing workflow from raw data to ML-ready dataset.
        This is the main entry point that executes all processing steps in the proper sequence.
        """
        try:
            self.load_data()
            self.handle_missing_data()
            
            self.create_yield_curve_features()
            self.create_economic_indicators()
            self.create_market_indicators()
            self.create_policy_indicators()
            
            self.aggregate_by_meeting_interval()
            
            self.clean_meeting_data()
            self.normalize_meeting_data()
            
            self.save_processed_data()
            
        except Exception:
            raise

if __name__ == "__main__":
    preprocessor = ECBMeetingPreprocessor()
    preprocessor.run_pipeline() 
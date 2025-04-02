# ECB Interest Rate Policy Prediction

This project aims to predict ECB interest rate policy using machine learning models. The data has been collected from Bloomberg and other sources.

## Project Overview

The European Central Bank (ECB) interest rate policy is influenced by various economic and financial factors. This project aims to build a predictive model that can forecast ECB rate changes based on:

- Macroeconomic indicators (GDP, inflation, unemployment)
- Market data (exchange rates, stock indices)
- Bond yield curves
- Oil prices
- Fed policy rates
- Geopolitical risk factors

## Data Sources
- Macro Data (Monthly): GDP, Inflation, Unemployment, Industrial Production, Retail Sales, Money Supply
- Market Data (Daily): EUR/USD, ECB Rate Futures, EuroStoxx 50
- Bond Yields (Daily): German government bonds at various maturities (1Y, 2Y, 5Y, 10Y)
- Oil Prices (Daily): Brent Oil Prices
- Fed Funds Rate (Monthly): US Federal Reserve interest rates
- GPR Data (Monthly): Geopolitical Risk indicators

## Data Preprocessing Pipeline

The preprocessing pipeline (`Data_Preprocessing.py`) implements several key steps:

### 1. Data Loading and Initial Processing
- Load all data files from their respective sources
- Handle different date formats consistently
- Standardize column names across datasets
- Transform data into pandas DataFrames

### 2. Data Validation
- Verify presence of required columns in each dataset
- Check data types for correctness
- Validate that numerical columns contain proper numerical values
- Log validation results for each dataset

### 3. Missing Data Treatment
- Apply different interpolation strategies based on data frequency:
  - Linear interpolation for short gaps in daily data
  - Cubic interpolation for monthly economic indicators
- Forward fill for values that should persist (like policy rates)
- Backward fill for any remaining missing values
- Handle each dataset with appropriate methods based on its characteristics

### 4. Feature Engineering

#### a. Yield Curve Features
- Calculate yield curve slopes to capture market expectations:
  - 10Y - 2Y spread (long-term expectations)
  - 5Y - 1Y spread (medium-term expectations)
- Calculate yield curve curvature: 2*(5Y) - (2Y + 10Y)
- Generate rolling yield volatility metrics (30-day window)

#### b. Economic Indicators
- Calculate year-over-year (YoY) changes for:
  - GDP Growth
  - CPI Inflation
  - Core CPI
  - Industrial Production
  - Retail Sales
  - M3 Money Supply
- Create momentum indicators (3-month and 6-month changes)
- These derived features capture the rate of change and acceleration in economic conditions

#### c. Market Indicators
- Calculate returns and volatility metrics for equity markets
- Compute exchange rate volatility
- Generate oil price returns and volatility
- These indicators capture market sentiment and risk conditions

#### d. Policy Indicators
- Calculate Fed-ECB rate spreads to capture policy divergence
- Compute policy rate changes (first differences)
- Calculate rate accelerations (second differences)
- These features help model the influence of current policy on future decisions

### 5. Data Cleaning
- Replace infinite values with NaN and then fill appropriately
- Handle percentage values by clipping to reasonable ranges (-1000% to 1000%)
- Scale large monetary values (M3_Money_Supply scaled to millions)
- Remove outliers that might distort the model
- Add validation steps to ensure data quality after cleaning

### 6. Lag Features
- Create multiple lag periods to account for delayed policy responses:
  - Short-term lags (1-5 days) for market data
  - Medium-term lags (1-3 months) for economic indicators
- ECB policy tends to respond with a delay to economic changes, so lags are essential

### 7. Data Normalization
- Apply different normalization strategies based on data characteristics:
  - StandardScaler for general economic indicators
  - MinMaxScaler for interest rates and bond yields
  - Special handling for percentage-based features
- This ensures all features contribute appropriately to the model

### 8. Dataset Combination
- Merge all datasets based on dates using asof merges
- Handle different frequencies (daily vs. monthly)
- Create the final feature set with all engineered features
- Generate target variables (ECB rate changes)
- Remove any remaining rows with missing values

## Implementation Challenges and Solutions

### 1. Handling Mixed Frequency Data
- **Challenge**: Combining daily market data with monthly economic indicators
- **Solution**: Used pandas' merge_asof to properly align data points by date, with backward filling

### 2. Column Naming Inconsistencies
- **Challenge**: Different datasets had inconsistent column naming
- **Solution**: Standardized column names during loading for consistent processing

### 3. Large Value Scaling
- **Challenge**: M3_Money_Supply had values in millions, causing numerical issues
- **Solution**: Scaled down by dividing by 1,000,000 and excluded from large value checks

### 4. Infinite Values in Percentage Calculations
- **Challenge**: Some percentage changes resulted in infinite values
- **Solution**: Implemented replacement of infinity with NaN and clipping of extreme values

### 5. DateTime Column Handling
- **Challenge**: Different date formats across datasets
- **Solution**: Standardized date parsing and excluded datetime columns from numerical operations

## Usage
1. Place raw data files in the `Raw_Data` directory
2. Run the preprocessing pipeline:
   ```
   python Data_Preprocessing.py
   ```
3. Check the processed data in the `Processed_Data` directory
4. Use the processed data for model training

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Future Improvements
- Add more feature engineering techniques specific to monetary policy prediction
- Implement automated feature selection
- Add support for real-time data updates
- Explore alternative normalization techniques for economic data 
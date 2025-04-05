# ECB Interest Rate Policy Prediction - Comprehensive Documentation

This document provides comprehensive details about the ECB Interest Rate Policy Prediction project, including its architecture, implementation, deployment, and future development plans.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Sources](#data-sources)
3. [Project Structure](#project-structure)
4. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
5. [Machine Learning Models](#machine-learning-models)
6. [Model Evaluation](#model-evaluation)
7. [Addressing Class Imbalance](#addressing-class-imbalance)
8. [Installation and Usage](#installation-and-usage)
9. [Streamlit Dashboard Deployment](#streamlit-dashboard-deployment)
10. [Future Development Roadmap](#future-development-roadmap)

## Project Overview

The European Central Bank (ECB) interest rate policy has significant impacts on the European economy and financial markets. This project aims to build machine learning models to predict ECB rate decisions (hike, hold, or cut) based on various economic and financial indicators.

Key components:
- Data collection and preprocessing from multiple sources
- Feature engineering to create relevant predictors
- Multiple modeling approaches to address class imbalance nad period imbalances
- Specialized analysis for crisis periods
- Comprehensive model evaluation and visualization
- Interactive Streamlit dashboard for predictions

## Data Sources

| Data Type | Frequency | Source File | Description |
|-----------|-----------|-------------|-------------|
| ECB Interest Rates | Daily | `ECBDFR.csv` | Historical ECB interest rates (2000-present) |
| Macro Data | Monthly | `macro_data.csv` | GDP, Inflation, Unemployment, Industrial Production, Retail Sales, Money Supply |
| Market Data | Daily | `market_data.csv` | EUR/USD, ECB Rate Futures, EuroStoxx 50 |
| Bond Yields | Daily | `German_bondyield_data.csv` | German government bonds (1Y, 2Y, 5Y, 10Y) |
| Oil Prices | Daily | `Brent_Oil_Prices_data.csv` | Brent Oil Prices |
| Fed Funds Rate | Monthly | `Fed_Fund_Rate_data.csv` | US Federal Reserve interest rates |
| Geopolitical Risk | Monthly | `GPR_data.csv` | Geopolitical Risk indicators |

## Project Structure

```
ECB-Interest-Rate-Policy/
├── Documentation/           # Project documentation
├── Raw_Data/                # Raw input data files
├── Processed_Data/          # Processed data ready for modeling
├── models/                  # Saved trained models for each approach
│   ├── standard/            # Standard ML models
│   ├── crisis_aware/        # Models incorporating crisis indicators
│   ├── smote/               # Models trained with SMOTE oversampling
│   ├── weighted/            # Class-weighted models
│   └── two_stage/           # Hierarchical two-stage models
├── plots/                   # Visualization outputs
├── results/                 # Performance metrics and analysis
├── crisis_analysis/         # Analysis of model performance during crisis periods
├── shap_analysis/           # SHAP feature importance analysis
├── Data_Preprocessing.py    # Data loading and preprocessing pipeline
├── model_training.py        # Model training and evaluation script
├── app.py                   # Streamlit dashboard application
├── Deploy/                  # Deployment utilities
└── requirements.txt         # Project dependencies
```

## Data Preprocessing Pipeline

The preprocessing pipeline (`Data_Preprocessing.py`) implements several key steps:

### Data Loading and Initial Processing
- Loading from multiple data sources with different formats
- Standardizing date formats and column names
- Creating a consistent ECB meeting schedule (typically every 6 weeks)

### Missing Data Treatment
- Linear interpolation for short gaps in daily data
- Cubic interpolation for monthly economic indicators
- Forward/backward filling for appropriate time series

### Feature Engineering

#### Yield Curve Features
- Slopes: 10Y-2Y spread, 5Y-1Y spread
- Curvature: 2*(5Y) - (2Y + 10Y)
- Yield volatility metrics

#### Economic Indicators
- Year-over-year changes (GDP Growth, Inflation, etc.)
- Momentum indicators (3-month and 6-month changes)

#### Market Indicators
- Returns and volatility for equity markets
- Exchange rate volatility
- Oil price indicators

#### Policy Indicators
- Fed-ECB rate spreads
- Policy rate momentum
- Historical ECB rate patterns

### Meeting-Based Aggregation
- Creating a consistent meeting schedule based on ECB's 6-week cycle
- Aggregating daily/monthly data to the meeting intervals
- Feature statistics (mean, std, min, max) for each meeting period

### Data Normalization
- StandardScaler for economic indicators
- MinMaxScaler for interest rates and bond yields
- Handling of percentage-based features

### ECB Meeting Schedule Construction
The script constructs a consistent 6-week meeting schedule. If ECB rate data is available, it identifies the first actual rate change date as the starting point. Otherwise, it uses February 4, 2000, as a default starting point. It then generates meetings at exact 6-week (42-day) intervals.

### Rate Change Identification
For each meeting interval, the script determines if a rate change occurred and calculates its magnitude. This becomes the target variable for prediction models.

### Dataset Combination
The script combines all datasets using pandas' merge_asof function, which performs time-based alignment. This ensures that only data that would have been available at each point in time is used (avoiding look-ahead bias).

### Feature Aggregation Strategy
For each 6-week meeting interval, the script calculates:
- Statistical summaries (mean, min, max, standard deviation) for all numeric features
- For rate-related features (interest rates, yields, inflation, GDP), it captures the end-of-period value, which might be more relevant for policy decisions

## Machine Learning Models

Multiple modeling approaches implemented to address class imbalance in rate decisions:

### 1. Standard Model
- Random Forest Classifier with default class weights
- Used as the baseline for comparison

### 2. Crisis-Aware Model
- Incorporates a binary 'is_crisis' feature
- Trained to recognize different patterns during crisis periods

### 3. SMOTE Oversampling
- Synthetic Minority Over-sampling Technique
- Generates synthetic examples of minority classes (rate hikes/cuts)

### 4. Class-Weighted Model
- Assigns higher importance to minority classes during training
- Forces the model to pay more attention to rare events

### 5. Two-Stage Hierarchical Model
- First predicts whether a change will occur
- Then predicts direction (cut or hike) if a change is predicted

### Model Training Process

The model training script:
1. Divides data into train(80%) and test(20%) set using Stratified Kfold techniques to prevent class imbalance.
2. Trains each model with optimized parameters
3. Performs hyperparameter tuning using RandomizedSearchCV
4. Evaluates performance on test data
5. Saves the trained model and performance metrics

## Model Evaluation

Comprehensive evaluation metrics:
- Overall accuracy
- Class-specific F1 scores for Rate Cut, Hold, and Rate Hike decisions
- Confusion matrices
- ROC curves for each class
- SHAP feature importance analysis

### Evaluation Methods

The script uses several evaluation techniques:
1. Classification metrics:
   - Accuracy
   - Precision, recall, and F1-score for each class
   - Confusion matrices

2. Visualizations:
   - Confusion matrix heatmaps
   - ROC curves with AUC values
   - Feature importance plots
   - Model comparison bar charts

3. SHAP (SHapley Additive exPlanations) analysis:
   - Global feature importance
   - Class-specific feature importance
   - Feature dependence plots for top features

### Results Analysis

After training all models, the script:
1. Creates a summary DataFrame of all model performances
2. Identifies the best-performing models overall and for specific rate decisions
3. Analyzes feature importance across models
4. Provides recommendations for which models to use in different scenarios

## Addressing Class Imbalance

ECB rate decisions show significant class imbalance with 'Hold' decisions being much more common than 'Hike' or 'Cut'. Approaches to address this:

## Installation and Usage

### Requirements
- Python 3.8+
- Required packages listed in requirements.txt

### Setup

```bash
# Clone the repository
git clone https://github.com/leo-lightfoot/ECB-Interest-Rate-Policy.git
cd ECB-Interest-Rate-Policy

# Install dependencies
pip install -r requirements.txt

# Run data preprocessing
python Data_Preprocessing.py

# Train models and evaluate
python model_training.py
```

### Key Output Files
- `Processed_Data/ecb_meeting_data.csv`: Processed dataset ready for modeling
- `models/*/`: Trained models for each approach
- `results/model_performance_summary.csv`: Comparison of all model approaches
- `plots/`: Visualizations of model performance and feature importance

## Streamlit Dashboard

You can access the dashboard at https://github.com/leo-lightfoot/ECB-Interest-Rate-Policy/blob/main/README.md

## Future Development Roadmap

### 1. GitHub Repository Enhancements

- [ ] Add detailed README.md with screenshots of the Streamlit app
- [ ] Set up branch protection rules for the main branch
- [ ] Configure the GitHub repository for proper presentation
- [ ] Add badges for build status, code quality, etc.
- [ ] Add LICENSE file with appropriate open-source license

### 2. Data Pipeline Improvements

- [ ] Set up automated data retrieval from official sources
- [ ] Implement regular data updates (weekly/monthly)
- [ ] Add data validation tests to ensure quality
- [ ] Create a historical data archive system
- [ ] Document data sources and update frequencies

### 3. Model Enhancements

- [ ] Experiment with more advanced models (XGBoost, LGBM, deep learning)
- [ ] Implement model versioning and tracking
- [ ] Add uncertainty quantification
- [ ] Conduct thorough feature selection analysis
- [ ] Develop ensemble methods combining multiple approaches

### 4. Streamlit Dashboard Enhancements

- [ ] Add time series visualization of economic indicators
- [ ] Create a forecast calendar for upcoming ECB meetings
- [ ] Implement user session management for tracking predictions
- [ ] Add downloadable reports of predictions and analyses
- [ ] Enhance UI/UX with more interactive elements

### 5. Monitoring and Maintenance

- [ ] Implement automated testing for the application
- [ ] Set up monitoring for app performance
- [ ] Create maintenance schedule for model retraining
- [ ] Add error logging and alerts for issues
- [ ] Document maintenance procedures

### 6. Documentation

- [ ] Create comprehensive API documentation
- [ ] Add a detailed methodology guide
- [ ] Create a user guide for the Streamlit dashboard
- [ ] Document model performance metrics and limitations
- [ ] Add tutorials for common use cases

### 7. Community and Collaboration

- [ ] Set up issue templates for feature requests and bug reports
- [ ] Create contribution guidelines for open-source collaboration
- [ ] Add a code of conduct for community interaction
- [ ] Consider establishing a changelog for tracking updates
- [ ] Set up a discussion forum or communication channel

### 8. Advanced Features

- [ ] Implement scenario analysis capabilities
- [ ] Add confidence intervals for predictions
- [ ] Create comparative views of different models
- [ ] Develop API endpoints for programmatic access
- [ ] Integrate with other financial analysis tools

### 9. Performance Optimization

- [ ] Optimize model inference for faster predictions
- [ ] Implement caching strategies for Streamlit
- [ ] Refactor code for improved maintainability
- [ ] Conduct code review for optimization opportunities
- [ ] Improve load times and resource usage

### 10. Publicity and Outreach

- [ ] Prepare a blog post about the project
- [ ] Create a short demonstration video
- [ ] Share with relevant communities
- [ ] Gather feedback from domain experts
- [ ] Consider submitting to contests or journals

### Priority Tasks

These tasks should be addressed first:

1. Complete the Streamlit Cloud deployment
2. Add screenshots to README.md
3. Implement automated testing
4. Set up model versioning
5. Create comprehensive user documentation 
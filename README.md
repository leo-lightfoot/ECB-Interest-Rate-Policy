# ECB Interest Rate Policy Prediction

A machine learning system that predicts European Central Bank (ECB) interest rate decisions using economic and financial indicators.

## Project Overview

This project uses machine learning to predict ECB rate decisions (hike, hold, or cut) based on economic and market data. The models account for economic regimes (normal/crisis periods) and address class imbalance challenges.

## Key Features

- Data pipeline integrating multiple economic and financial indicators
- Random Forest classifier models with specialized approaches for class imbalance
- Interactive Streamlit dashboard for making predictions
- Comprehensive documentation and deployment guides

## Quick Start

```bash
# Clone the repository
git clone https://github.com/leo-lightfoot/ECB-Interest-Rate-Policy.git
cd ECB-Interest-Rate-Policy

# Install dependencies
pip install -r requirements.txt

# Run setup script to check environment
python Deploy/setup.py

# Run the Streamlit app locally
streamlit run app.py
```

## Documentation

- [Comprehensive Documentation](docs/DOCUMENTATION.md) - Detailed project information
- [Model Analysis](docs/model_analysis.txt) - Model performance and feature importance
- [Preprocessing Pipeline](docs/ECB_Preprocessing_Documentation.txt) - Data preparation workflow
- [Deployment Guide](docs/DEPLOYMENT.md) - Deployment instructions

## Project Structure

```
ECB-Interest-Rate-Policy/
├── app.py                   # Streamlit application
├── Data_Preprocessing.py    # Data preprocessing pipeline
├── model_training.py        # Model training script
├── models/                  # Saved trained models
├── Processed_Data/          # Processed data ready for modeling
├── Raw_Data/                # Original source data
├── shap_analysis/           # SHAP value analysis results
├── plots/                   # Generated plots and visualizations
├── results/                 # Model results and evaluation metrics
├── crisis_analysis/         # Analysis of crisis periods
├── .streamlit/              # Streamlit configuration
├── docs/                    # Project documentation
├── Deploy/                  # Deployment utilities
│   └── setup.py             # Setup script for environment verification
└── requirements.txt         # Project dependencies
```

## Future Enhancements

- Enhanced feature selection techniques
- Deep learning approaches (LSTM, Transformer models)
- Integration of text data from ECB statements
- Real-time data updates and predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
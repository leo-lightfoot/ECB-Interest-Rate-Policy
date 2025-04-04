import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO
import shap

# Page configuration
st.set_page_config(
    page_title="ECB Interest Rate Policy Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed on mobile
)

# Custom CSS for better appearance and mobile responsiveness
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #1a5276;
        font-size: calc(1.5rem + 1vw);
        margin-bottom: 1rem;
        text-align: center;
    }
    h2, h3 {
        color: #1a5276;
        font-size: calc(1.2rem + 0.5vw);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .cut {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    .hold {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
    }
    .hike {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        white-space: normal;
    }
    /* Mobile optimization */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
        }
        .stButton > button {
            width: 100%;
            margin-top: 10px;
        }
        .metric-card {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
    }
    /* Metric card styling */
    .metric-card {
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    /* Button styling */
    .stButton > button {
        background-color: #1a5276;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Function to load model
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'smote', 'smote_rf.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # Load feature names from a JSON file
        feature_path = os.path.join('models', 'feature_names.json')
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                feature_names = json.load(f)
        else:
            # Default feature names based on model
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_.tolist()
            else:
                feature_names = []
        return model, feature_names
    else:
        st.error(f"Model file not found at {model_path}")
        return None, []

# Load sample data for demo
@st.cache_data
def load_sample_data():
    data_path = os.path.join('Processed_Data', 'ecb_meeting_data.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        st.warning(f"Sample data file not found at {data_path}")
        return None

# Function to create feature importance plot
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create a DataFrame for easier handling
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(15)
        
        # Create plotly figure
        fig = px.bar(
            importance_df, 
            x='importance', 
            y='feature', 
            orientation='h',
            title='Top 15 Feature Importances',
            labels={'importance': 'Importance', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        # Update layout for better mobile display
        fig.update_layout(
            autosize=True,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Importance",
            yaxis_title="Feature",
            yaxis={'categoryorder':'total ascending'},
            title_x=0.5,  # Center the title
            font=dict(size=12)
        )
        
        return fig
    else:
        return None

# Function to plot prediction probabilities
def plot_probabilities(probabilities):
    categories = ['Rate Cut', 'Hold', 'Rate Hike']
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=probabilities,
            marker_color=['#ff6b6b', '#4dabf7', '#69db7c'],
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Decision Probabilities',
        title_x=0.5,  # Center the title
        xaxis_title='ECB Rate Decision',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        autosize=True,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(size=12)
    )
    
    return fig

# Display metric in a styled card
def display_metric(label, value, unit=""):
    if value != 'N/A':
        formatted_value = f"{value:.2f}{unit}"
    else:
        formatted_value = "N/A"
    
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin-top:0; font-size:1rem; color:#666;">{label}</h3>
        <p style="font-size:1.5rem; font-weight:bold; margin:0;">{formatted_value}</p>
    </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    st.title("ECB Interest Rate Policy Predictor")
    
    # Load model and feature names
    model, feature_names = load_model()
    
    # Sidebar with app description
    with st.sidebar:
        st.title("About")
        st.info(
            "This application predicts European Central Bank (ECB) interest rate policy decisions "
            "based on economic and financial indicators. It was trained on historical data "
            "from 2000 to present using a Random Forest model with SMOTE oversampling."
        )
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Importance", "ℹ️ How It Works"])
    
    with tab1:
        st.header("Predict ECB Rate Decision")
        
        # Option to use sample data or manual input
        input_method = st.radio(
            "Select input method",
            ["Use sample data from past meetings", "Manual input"]
        )
        
        input_data = {}
        
        if input_method == "Use sample data from past meetings":
            # Load sample data
            sample_data = load_sample_data()
            
            if sample_data is not None:
                # Convert date columns to datetime for sorting
                for date_col in ['Meeting_Date', 'Interval_Start', 'Interval_End']:
                    if date_col in sample_data.columns:
                        sample_data[date_col] = pd.to_datetime(sample_data[date_col])
                
                # Sort by meeting date
                sample_data = sample_data.sort_values('Meeting_Date', ascending=False)
                
                # Create a selection of past meetings
                meeting_options = sample_data['Meeting_Date'].dt.strftime('%Y-%m-%d').tolist()
                selected_meeting = st.selectbox("Select a past ECB meeting", meeting_options)
                
                # Get data for the selected meeting
                meeting_data = sample_data[sample_data['Meeting_Date'].dt.strftime('%Y-%m-%d') == selected_meeting].iloc[0]
                
                # Show the actual decision
                actual_decision = {-1: "Rate Cut", 0: "Hold", 1: "Rate Hike"}
                actual_change = meeting_data.get('Next_Rate_Change', 0)
                actual_policy = np.sign(actual_change)
                
                st.info(f"Actual decision for this meeting: **{actual_decision.get(actual_policy, 'Unknown')}** (Change: {actual_change}%)")
                
                # Use this data as input
                for feature in feature_names:
                    if feature in meeting_data:
                        input_data[feature] = meeting_data[feature]
                
                # Show some key indicators for the selected meeting
                st.subheader("Key Indicators for this Meeting")
                
                # Responsive grid layout for mobile
                indicators_col1, indicators_col2 = st.columns([1, 1])
                
                with indicators_col1:
                    display_metric("Inflation (HICP)", meeting_data.get('CPI_Inflation_YoY_mean', 'N/A'), "%")
                    display_metric("GDP Growth", meeting_data.get('Eurozone GDP Growth_mean', 'N/A'), "%")
                    display_metric("Unemployment", meeting_data.get('Eurozone_Unemployment_mean', 'N/A'), "%")
                
                with indicators_col2:
                    display_metric("10Y-2Y Yield Spread", meeting_data.get('slope_10y_2y_mean', 'N/A'), " pp")
                    display_metric("EUR/USD", meeting_data.get('EURUSD Exchange rate_mean', 'N/A'), "")
                    display_metric("Fed Funds Rate", meeting_data.get('Fed_Fund_Rate_mean', 'N/A'), "%")
                
        else:
            # Manual input for key features with mobile-friendly layout
            st.subheader("Enter Key Economic Indicators")
            
            # Use expanders to save vertical space on mobile
            with st.expander("Economic Indicators", expanded=True):
                input_data['CPI_Inflation_YoY_mean'] = st.slider("Inflation (HICP) %", min_value=-1.0, max_value=10.0, value=2.0, step=0.1)
                input_data['Eurozone GDP Growth_mean'] = st.slider("GDP Growth %", min_value=-8.0, max_value=8.0, value=1.5, step=0.1)
                input_data['Eurozone_Unemployment_mean'] = st.slider("Unemployment %", min_value=1.0, max_value=20.0, value=7.0, step=0.1)
                input_data['Industrial_Production_mean'] = st.slider("Industrial Production (YoY) %", min_value=-20.0, max_value=20.0, value=1.0, step=0.1)
            
            with st.expander("Market Indicators", expanded=True):
                input_data['slope_10y_2y_mean'] = st.slider("10Y-2Y Yield Spread (pp)", min_value=-2.0, max_value=3.0, value=0.5, step=0.01)
                input_data['EURUSD Exchange rate_mean'] = st.slider("EUR/USD Exchange Rate", min_value=0.8, max_value=1.6, value=1.1, step=0.01)
                input_data['Fed_Fund_Rate_mean'] = st.slider("Fed Funds Rate %", min_value=0.0, max_value=8.0, value=4.0, step=0.25)
                input_data['Brent Oil Price_mean'] = st.slider("Brent Oil Price (USD)", min_value=20.0, max_value=150.0, value=80.0, step=1.0)
        
        # Make prediction button - full width for mobile
        col_button = st.columns([1])[0]
        with col_button:
            predict_button = st.button("Predict ECB Decision", use_container_width=True)
        
        if predict_button:
            if model is not None and feature_names:
                # Create a DataFrame with all required features
                prediction_data = {}
                missing_features = []
                
                for feature in feature_names:
                    if feature in input_data:
                        prediction_data[feature] = [float(input_data[feature])]
                    else:
                        prediction_data[feature] = [0.0]
                        missing_features.append(feature)
                
                # Warn about missing features
                if missing_features and input_method == "Manual input":
                    st.warning(f"Missing {len(missing_features)} features. Using default values for them.")
                
                # Create DataFrame
                input_df = pd.DataFrame(prediction_data)
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0].tolist()
                
                # Convert prediction to human-readable format
                policy_mapping = {-1: "Rate Cut", 0: "Hold", 1: "Rate Hike"}
                policy_prediction = policy_mapping[prediction]
                
                # Display prediction
                prediction_class = "cut" if prediction == -1 else ("hold" if prediction == 0 else "hike")
                
                st.markdown(f"""
                <div class="prediction-box {prediction_class}">
                    <h3 style="text-align:center; margin-top:0;">Prediction: {policy_prediction}</h3>
                    <p style="text-align:center;">The model predicts that the ECB will {policy_prediction.lower()} interest rates at this meeting.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display probability plot
                st.plotly_chart(plot_probabilities(probabilities), use_container_width=True)
                
                # Add explanation about confidence
                max_prob = max(probabilities)
                if max_prob > 0.7:
                    confidence = "high confidence"
                elif max_prob > 0.5:
                    confidence = "moderate confidence"
                else:
                    confidence = "low confidence"
                
                st.info(f"The model predicts this outcome with **{confidence}** (probability: {max_prob:.2f}).")
            else:
                st.error("Model could not be loaded. Please check if the model file exists.")
    
    with tab2:
        st.header("Feature Importance Analysis")
        
        if model is not None and feature_names:
            st.markdown("""
            This section shows which economic and financial indicators most strongly 
            influence the ECB's interest rate decisions according to our model.
            """)
            
            # Feature importance plot
            importance_fig = plot_feature_importance(model, feature_names)
            if importance_fig:
                st.plotly_chart(importance_fig, use_container_width=True)
            
            # Load feature importance from saved results if available
            results_path = os.path.join('results', 'feature_importance.csv')
            if os.path.exists(results_path):
                feature_importance_df = pd.read_csv(results_path)
                
                st.subheader("Detailed Feature Importance Across Models")
                st.dataframe(feature_importance_df, use_container_width=True)
        else:
            st.error("Model could not be loaded. Please check if the model file exists.")
    
    with tab3:
        st.header("How the Model Works")
        
        with st.expander("Data Sources", expanded=True):
            st.markdown("""
            The model is trained on multiple data sources:
            - Macroeconomic indicators (GDP, inflation, unemployment)
            - Market data (exchange rates, stock indices)
            - Bond yield curves
            - Oil prices
            - Fed policy rates
            - Geopolitical risk factors
            """)
        
        with st.expander("Model Approach", expanded=True):
            st.markdown("""
            This application uses a Random Forest Classifier with SMOTE (Synthetic Minority Over-sampling Technique) 
            to address the class imbalance in ECB rate decisions, where 'Hold' decisions are much more common than 
            'Rate Hike' or 'Rate Cut' decisions.
            """)
        
        with st.expander("Performance", expanded=True):
            st.markdown("""
            Our best performing model (SMOTE) achieves:
            - Overall accuracy: 66.4%
            - F1 score for Rate Cut predictions: 52.6%
            - F1 score for Rate Hike predictions: 44.7%
            """)
        
        with st.expander("Limitations", expanded=True):
            st.markdown("""
            While the model performs significantly better than random guessing, it still has limitations:
            - ECB decisions can be influenced by factors not captured in our data
            - Rare events like financial crises are difficult to predict
            - Past patterns may not always predict future decisions
            """)

if __name__ == "__main__":
    main() 
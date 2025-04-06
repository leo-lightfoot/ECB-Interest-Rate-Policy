import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.express as px
import plotly.graph_objects as go
import base64
from PIL import Image

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
        overflow-x: hidden;  /* Hide horizontal scrollbar */
    }
    /* Hide scrollbar for Chrome, Safari and Opera */
    .stApp::-webkit-scrollbar {
        display: none;
    }
    /* Hide scrollbar for IE, Edge and Firefox */
    .stApp {
        -ms-overflow-style: none;  /* IE and Edge */
        scrollbar-width: none;  /* Firefox */
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
        width: 100%;
        overflow-wrap: break-word;
        word-wrap: break-word;
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
    /* Custom tab styling with more spacing */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        flex-wrap: wrap;
        justify-content: center;
        margin-bottom: 25px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 15px 20px;
        white-space: nowrap;
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 500;
        color: #1a5276;
        border: 1px solid #dee2e6;
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1a5276;
        color: white;
    }
    /* Mobile optimization */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
            padding: 8px 12px;
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
        /* Improved mobile styling for prediction box */
        .prediction-box {
            padding: 15px;
            margin: 10px 0 20px 0;
            width: 100% !important;
            box-sizing: border-box;
            overflow: hidden;
        }
        .prediction-box h3 {
            font-size: 1.3rem;
            margin-top: 0;
            margin-bottom: 10px;
        }
        .prediction-box p {
            font-size: 1rem;
            margin: 0;
        }
        /* Fix chart display on mobile */
        .stPlotlyChart {
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
        }
        /* Fix info box display on mobile */
        .stAlert {
            width: 100% !important;
            box-sizing: border-box;
            padding: 10px !important;
            margin: 10px 0 !important;
        }
        /* Make sure content doesn't overflow */
        .element-container {
            width: 100% !important;
            overflow-x: hidden !important;
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
    /* Model performance section styling */
    .performance-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    /* Model comparison table styling */
    .styled-table {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        border-radius: 10px;
        overflow: hidden;
        width: 100%;
    }
    .styled-table thead tr {
        background-color: #1a5276;
        color: #ffffff;
        text-align: left;
    }
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #1a5276;
    }
    /* Selector styling */
    .selector-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid #dee2e6;
    }
    /* SHAP Plot container */
    .shap-container {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    /* Additional mobile optimization for charts and images */
    @media (max-width: 768px) {
        .js-plotly-plot .main-svg {
            max-width: 100% !important;
        }
        .stImage img {
            width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load model
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

# Calculate default values based on recent data (last year)
@st.cache_data
def calculate_default_values():
    sample_data = load_sample_data()
    default_values = {}
    
    if sample_data is not None:
        # Convert date to datetime for filtering
        if 'Meeting_Date' in sample_data.columns:
            sample_data['Meeting_Date'] = pd.to_datetime(sample_data['Meeting_Date'])
            
            # Get data from last year
            last_year = sample_data['Meeting_Date'].max() - pd.DateOffset(years=1)
            recent_data = sample_data[sample_data['Meeting_Date'] >= last_year]
            
            # Calculate mean values for all numeric columns
            if not recent_data.empty:
                for col in recent_data.columns:
                    if col != 'Meeting_Date' and pd.api.types.is_numeric_dtype(recent_data[col]):
                        default_values[col] = recent_data[col].mean()
            
    return default_values

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

# Function to load confusion matrix image for a specific model
def load_confusion_matrix(model_name):
    """Load confusion matrix image for the specified model"""
    cm_path = os.path.join('plots', f"confusion_matrix_{model_name}.png")
    if os.path.exists(cm_path):
        img = Image.open(cm_path)
        return img
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
            textposition='auto',
            textfont=dict(size=12)  # Slightly smaller text
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
        font=dict(size=12),
        height=350  # Fixed height to prevent resizing issues
    )
    
    # Make more mobile-friendly
    fig.update_layout(
        xaxis=dict(
            tickangle=0,  # Horizontal labels
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            tickfont=dict(size=11)
        ),
        title_font=dict(size=14)
    )
    
    return fig

# Function to plot model performance comparison
def plot_model_comparison():
    # Load model performance data
    performance_path = os.path.join('results', 'final_model_comparison.csv')
    if os.path.exists(performance_path):
        df = pd.read_csv(performance_path)
        
        # Create a mobile-friendly visualization with better spacing
        fig = go.Figure()
        
        # Add accuracy bars
        fig.add_trace(go.Bar(
            y=df['model_type'],
            x=df['accuracy'],
            name='Accuracy',
            orientation='h',
            marker=dict(color='#1a5276'),
            text=[f"{x:.1%}" for x in df['accuracy']],
            textposition='auto',
            hoverinfo='text',
            hovertext=[f"Accuracy: {x:.1%}" for x in df['accuracy']]
        ))
        
        # Add F1 scores for each class
        fig.add_trace(go.Bar(
            y=df['model_type'],
            x=df['f1_rate_cut'],
            name='F1 Rate Cut',
            orientation='h',
            marker=dict(color='#ff6b6b'),
            text=[f"{x:.1%}" for x in df['f1_rate_cut']],
            textposition='auto',
            hoverinfo='text',
            hovertext=[f"F1 Rate Cut: {x:.1%}" for x in df['f1_rate_cut']]
        ))
        
        fig.add_trace(go.Bar(
            y=df['model_type'],
            x=df['f1_hold'],
            name='F1 Hold',
            orientation='h',
            marker=dict(color='#4dabf7'),
            text=[f"{x:.1%}" for x in df['f1_hold']],
            textposition='auto',
            hoverinfo='text',
            hovertext=[f"F1 Hold: {x:.1%}" for x in df['f1_hold']]
        ))
        
        fig.add_trace(go.Bar(
            y=df['model_type'],
            x=df['f1_rate_hike'],
            name='F1 Rate Hike',
            orientation='h',
            marker=dict(color='#69db7c'),
            text=[f"{x:.1%}" for x in df['f1_rate_hike']],
            textposition='auto',
            hoverinfo='text',
            hovertext=[f"F1 Rate Hike: {x:.1%}" for x in df['f1_rate_hike']]
        ))
        
        # Update layout for better mobile display
        fig.update_layout(
            barmode='group',
            xaxis=dict(
                title='Score',
                tickformat='.0%',
                range=[0, 1.0]  # Fix axis range to prevent overflow
            ),
            yaxis=dict(
                title='Model Type',
                categoryorder='total ascending'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.15,  # Increase vertical space for legend
                xanchor="center",
                x=0.5,
                font=dict(size=10),
                traceorder="normal"
            ),
            margin=dict(l=10, r=10, t=100, b=10),  # Increased top margin
            height=550,  # Increase height to ensure enough space for legend
            autosize=True,
            font=dict(size=10),  # Smaller font for mobile
        )
        
        return fig, df
    else:
        return None, None

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
    # Initialize session state variables
    if 'input_method' not in st.session_state:
        st.session_state['input_method'] = "Use sample data from past meetings"
        
    st.title("ECB Interest Rate Policy Predictor")
    
    # Load model and feature names
    model, feature_names = load_model()
    
    # Create tabs with enough spacing
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Prediction", 
        "📈 Feature Importance", 
        "📉 Model Performance", 
        "ℹ️ Documentation"
    ])
    
    with tab1:
        st.header("Predict ECB Rate Decision")
        
        # Option to use sample data or manual input
        input_method = st.radio(
            "Select input method",
            ["Use sample data from past meetings", "Manual input"],
            index=0,
            key="input_method"
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
                    display_metric("Inflation (CPI)", meeting_data.get('CPI_Inflation_YoY_mean', 'N/A'), "%")
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
                input_data['CPI_Inflation_YoY_mean'] = st.slider("Inflation (CPI) %", min_value=-1.0, max_value=10.0, value=2.0, step=0.1)
                input_data['Eurozone GDP Growth_mean'] = st.slider("GDP Growth %", min_value=-8.0, max_value=8.0, value=1.5, step=0.1)
                input_data['Eurozone_Unemployment_mean'] = st.slider("Unemployment %", min_value=1.0, max_value=20.0, value=7.0, step=0.1)
                input_data['Industrial_Production_mean'] = st.slider("Industrial Production (YoY) %", min_value=-20.0, max_value=20.0, value=1.0, step=0.1)
                input_data['Core_CPI_mean'] = st.slider("Core Inflation %", min_value=-1.0, max_value=10.0, value=1.8, step=0.1)
                input_data['M3_Money_Supply_mean'] = st.slider("M3 Money Supply Growth %", min_value=-2.0, max_value=15.0, value=4.5, step=0.1)
            
            with st.expander("Market Indicators", expanded=True):
                input_data['slope_10y_2y_mean'] = st.slider("10Y-2Y Yield Spread (pp)", min_value=-2.0, max_value=3.0, value=0.5, step=0.01)
                input_data['EURUSD Exchange rate_mean'] = st.slider("EUR/USD Exchange Rate", min_value=0.8, max_value=1.6, value=1.1, step=0.01)
                input_data['Fed_Fund_Rate_mean'] = st.slider("Fed Funds Rate %", min_value=0.0, max_value=8.0, value=4.0, step=0.25)
                input_data['Brent Oil Price_mean'] = st.slider("Brent Oil Price (USD)", min_value=20.0, max_value=150.0, value=80.0, step=1.0)
                input_data['10Y_mean'] = st.slider("10Y German Bond Yield %", min_value=-1.0, max_value=8.0, value=2.0, step=0.1)
                input_data['2Y_mean'] = st.slider("2Y German Bond Yield %", min_value=-2.0, max_value=7.0, value=1.5, step=0.1)
            
            with st.expander("Policy & Risk Indicators", expanded=True):
                input_data['ECB_rate_actual_mean'] = st.slider("Current ECB Rate %", min_value=-1.0, max_value=5.0, value=2.5, step=0.1)
                input_data['Previous_Rate_Decision'] = st.select_slider(
                    "Previous Rate Decision", 
                    options=[-0.5, -0.25, 0.0, 0.25, 0.5],
                    value=0.0
                )
                input_data['GPR_mean'] = st.slider("Geopolitical Risk Index", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
        
        # Make prediction button - full width for mobile
        col_button = st.columns([1])[0]
        with col_button:
            predict_button = st.button("Predict ECB Decision", use_container_width=True)
        
        if predict_button:
            if model is not None and feature_names:
                # Create a DataFrame with all required features
                prediction_data = {}
                missing_features = []
                
                # Calculate derived features from input data
                if input_method == "Manual input":
                    # Load default values for missing features (from last year's data)
                    default_values = calculate_default_values()
                    
                    # Derive yield curve features exactly as in Data_Preprocessing.py
                    input_data['5Y_mean'] = (input_data['10Y_mean'] + input_data['2Y_mean']) / 2  # Rough approximation
                    input_data['1Y_mean'] = input_data['2Y_mean'] - 0.2  # Rough approximation
                    input_data['slope_10y_2y_mean'] = input_data['10Y_mean'] - input_data['2Y_mean']
                    input_data['slope_5y_1y_mean'] = input_data['5Y_mean'] - input_data['1Y_mean']
                    input_data['curvature_mean'] = 2 * input_data['5Y_mean'] - (input_data['2Y_mean'] + input_data['10Y_mean'])
                    
                    # Create volatility metrics (approx 30-day standard deviation)
                    volatility_factor = 0.05  # This approximates a rolling 30-day std
                    for yield_type in ['1Y', '2Y', '5Y', '10Y']:
                        input_data[f'{yield_type}_volatility_mean'] = input_data[f'{yield_type}_mean'] * volatility_factor
                    
                    # Create market volatility indicators
                    input_data['EURUSD_volatility_mean'] = input_data['EURUSD Exchange rate_mean'] * volatility_factor
                    input_data['Oil_volatility_mean'] = input_data['Brent Oil Price_mean'] * volatility_factor
                    
                    # Create returns (approx daily returns)
                    input_data['Oil_returns_mean'] = 0.001  # Default daily return of 0.1%
                    
                    # Create Fed-ECB spread
                    input_data['Fed_ECB_spread_mean'] = input_data['Fed_Fund_Rate_mean'] - input_data['ECB_rate_actual_mean']
                    
                    # Derive year-over-year percentage changes
                    for col in ['Eurozone GDP Growth', 'CPI_Inflation_YoY', 'Core_CPI', 
                               'Industrial_Production', 'M3_Money_Supply']:
                        base_col = f'{col}_mean'
                        if base_col in input_data:
                            input_data[f'{col}_YoY_mean'] = input_data[base_col] * 0.9  # Slight decrease from current
                    
                    # Create 3-month and 6-month changes for key indicators
                    for col in ['Eurozone GDP Growth', 'CPI_Inflation_YoY', 'Core_CPI']:
                        base_col = f'{col}_mean'
                        if base_col in input_data:
                            input_data[f'{col}_3m_change_mean'] = input_data[base_col] * 0.1  # 10% quarterly change
                            input_data[f'{col}_6m_change_mean'] = input_data[base_col] * 0.2  # 20% 6-month change
                    
                    # Derive Yield_Curve_Status using same logic as in model_training.py
                    if input_data['slope_10y_2y_mean'] < -0.5:
                        input_data['Yield_Curve_Status'] = -1  # Inverted (recession signal)
                    elif input_data['slope_10y_2y_mean'] > 1.0:
                        input_data['Yield_Curve_Status'] = 1   # Steep (growth signal)
                    else:
                        input_data['Yield_Curve_Status'] = 0   # Neutral
                    
                    # Create market volatility average
                    volatility_cols = ['EURUSD_volatility_mean', 'Oil_volatility_mean', 
                                      '1Y_volatility_mean', '2Y_volatility_mean', 
                                      '5Y_volatility_mean', '10Y_volatility_mean']
                    
                    volatility_values = [input_data.get(col, 0.02) for col in volatility_cols]
                    input_data['Market_Volatility_Avg'] = sum(volatility_values) / len(volatility_values)
                    
                    # Create inflation indicators average
                    inflation_cols = ['CPI_Inflation_YoY_mean', 'Core_CPI_mean']
                    inflation_values = [input_data.get(col, 2.0) for col in inflation_cols]
                    input_data['Inflation_Indicators_Avg'] = sum(inflation_values) / len(inflation_values)
                    
                    # Create political risk average
                    input_data['Political_Risk_Avg'] = input_data['GPR_mean']
                    
                    # Add interval length (typical value of 42 days for ECB meetings)
                    input_data['Interval_Length'] = 42.0
                    
                    # Calculate min/max/std values more accurately
                    for base_feature in ['CPI_Inflation_YoY', 'Eurozone GDP Growth', 'Eurozone_Unemployment',
                                         'EURUSD Exchange rate', 'Fed_Fund_Rate', 'Brent Oil Price',
                                         'ECB_rate_actual', '10Y', '2Y', '5Y', '1Y']:
                        if f'{base_feature}_mean' in input_data:
                            mean_value = input_data[f'{base_feature}_mean']
                            
                            # Standard deviation generally higher for more volatile metrics
                            if 'GDP' in base_feature or 'Production' in base_feature:
                                std_factor = 0.08  # Higher variance for growth metrics
                            elif 'Inflation' in base_feature or 'CPI' in base_feature:
                                std_factor = 0.06  # Medium variance for inflation
                            elif 'Oil' in base_feature:
                                std_factor = 0.10  # High variance for oil prices
                            else:
                                std_factor = 0.04  # Lower variance for other metrics
                            
                            input_data[f'{base_feature}_std'] = abs(mean_value * std_factor)
                            
                            # Min/max roughly spans ±2 standard deviations for typical data
                            std_val = input_data[f'{base_feature}_std']
                            input_data[f'{base_feature}_min'] = mean_value - 2 * std_val
                            input_data[f'{base_feature}_max'] = mean_value + 2 * std_val
                            
                            # End values typically near but not exactly at mean (removed randomness for consistency)
                            input_data[f'{base_feature}_end'] = mean_value * 1.01
                    
                    # Set previous decisions
                    input_data['Two_Meetings_Ago_Decision'] = input_data['Previous_Rate_Decision'] / 2
                    
                    # Create crisis indicator (default to no crisis)
                    input_data['is_crisis'] = 0
                    
                # Create prediction data with derived features and defaults from last year's data
                for feature in feature_names:
                    if feature in input_data:
                        prediction_data[feature] = [float(input_data[feature])]
                    else:
                        # Use historical data for default values when available
                        default_values = calculate_default_values()
                        if feature in default_values:
                            prediction_data[feature] = [float(default_values[feature])]
                        else:
                            # Fallback to intelligent defaults if historical data not available
                            if 'volatility' in feature.lower():
                                prediction_data[feature] = [0.02]  # Default volatility
                            elif 'std' in feature.lower():
                                prediction_data[feature] = [0.5]  # Default std
                            elif 'min' in feature.lower():
                                # For min features, use slightly lower values
                                base_feature = feature.replace('_min', '_mean')
                                if base_feature in input_data:
                                    prediction_data[feature] = [input_data[base_feature] * 0.9]
                                else:
                                    prediction_data[feature] = [0.0]
                            elif 'max' in feature.lower():
                                # For max features, use slightly higher values
                                base_feature = feature.replace('_max', '_mean')
                                if base_feature in input_data:
                                    prediction_data[feature] = [input_data[base_feature] * 1.1]
                                else:
                                    prediction_data[feature] = [0.0]
                            elif 'end' in feature.lower():
                                # For end features, use values close to mean
                                base_feature = feature.replace('_end', '_mean')
                                if base_feature in input_data:
                                    prediction_data[feature] = [input_data[base_feature] * 1.02]
                                else:
                                    prediction_data[feature] = [0.0]
                            elif 'YoY' in feature:
                                # For year-over-year features, set reasonable defaults
                                prediction_data[feature] = [1.5]  # Default YoY change
                            elif '3m_change' in feature or '6m_change' in feature:
                                # For short-term changes, set small default values
                                prediction_data[feature] = [0.3]  # Default short-term change
                            elif any(term in feature.lower() for term in ['slope', 'curvature']):
                                prediction_data[feature] = [0.2]  # Default curve measure
                            elif 'return' in feature.lower():
                                prediction_data[feature] = [0.001]  # Default return
                            else:
                                prediction_data[feature] = [0.0]
                        missing_features.append(feature)
                
                # Remove warnings about missing features
                if missing_features and input_method == "Manual input":
                    # No warnings or messages about missing features
                    pass
                
                # Create DataFrame ensuring features are in the correct order
                input_df = pd.DataFrame(prediction_data)
                
                # Make sure the DataFrame has all the required features in the correct order
                for feature in feature_names:
                    if feature not in input_df.columns:
                        input_df[feature] = 0.0
                
                # Reorder columns to match the model's expected order
                input_df = input_df[feature_names]
                
                # Make prediction
                try:
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0].tolist()
                    
                    # Convert prediction to human-readable format
                    policy_mapping = {-1: "Rate Cut", 0: "Hold", 1: "Rate Hike"}
                    policy_prediction = policy_mapping[prediction]
                    
                    # Add back the missing prediction_class variable
                    prediction_class = "cut" if prediction == -1 else ("hold" if prediction == 0 else "hike")
                    
                    # Display prediction with better mobile formatting
                    st.markdown(f"""
                    <div class="prediction-box {prediction_class}" style="max-width:100%; box-sizing:border-box;">
                        <h3 style="text-align:center; margin-top:0; font-size:calc(1.2rem + 0.5vw);">Prediction: {policy_prediction}</h3>
                        <p style="text-align:center; font-size:calc(0.9rem + 0.2vw);">The model predicts that the ECB will {policy_prediction.lower()} interest rates at this meeting.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Make probability plot more mobile friendly
                    prob_fig = plot_probabilities(probabilities)
                    # Ensure chart is properly sized on mobile
                    prob_fig.update_layout(
                        autosize=True,
                        margin=dict(l=5, r=5, t=30, b=5),
                        height=300,  # Smaller height for mobile
                        font=dict(size=10)  # Smaller font for mobile
                    )
                    st.plotly_chart(prob_fig, use_container_width=True, config={'responsive': True, 'displayModeBar': False})
                    
                    # Add explanation about confidence with better mobile formatting
                    max_prob = max(probabilities)
                    if max_prob > 0.7:
                        confidence = "high confidence"
                    elif max_prob > 0.5:
                        confidence = "moderate confidence"
                    else:
                        confidence = "low confidence"
                    
                    st.markdown(f"""
                    <div style="background-color:#f0f7fb; border-left:5px solid #4dabf7; padding:10px; margin:10px 0; border-radius:5px; width:100%; box-sizing:border-box;">
                        <p style="margin:0; font-size:calc(0.9rem + 0.1vw);">The model predicts this outcome with <strong>{confidence}</strong> (probability: {max_prob:.2f}).</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add debug and feature information AFTER the prediction for those who want details
                    with st.expander("Technical Details and Feature Information", expanded=False):
                        st.write("### Key Feature Values Used for Prediction:")
                        debug_features = [
                            'CPI_Inflation_YoY_mean', 'Core_CPI_mean', 'Eurozone GDP Growth_mean',
                            'Eurozone_Unemployment_mean', '10Y_mean', '2Y_mean', 
                            'slope_10y_2y_mean', 'Fed_Fund_Rate_mean', 'ECB_rate_actual_mean',
                            'Inflation_Indicators_Avg', 'Yield_Curve_Status', 'Previous_Rate_Decision'
                        ]
                        
                        debug_data = {}
                        for feature in debug_features:
                            if feature in input_df.columns:
                                debug_data[feature] = float(input_df[feature].iloc[0])
                        
                        st.json(debug_data)
                        
                        st.write("### Top Features by Importance:")
                        if hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                            indices = np.argsort(importances)[::-1][:10]
                            
                            importance_data = {}
                            for i in indices:
                                if i < len(feature_names):
                                    feature = feature_names[i]
                                    importance_data[feature] = float(importances[i])
                            
                            st.json(importance_data)
                        
                        st.write("### Derived Features Information:")
                        st.markdown("""
                        Additional features have been calculated based on your inputs:
                        - Yield curve metrics (slopes and curvature)
                        - Volatility estimates
                        - Min/max/std values for each indicator
                        - Rate change indicators
                        
                        These derived features help the model make more accurate predictions.
                        """)
                
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.write("Error details:", str(e))
                    
            else:
                st.error("Model could not be loaded. Please check if the model file exists.")
    
    with tab2:
        st.header("Feature Importance Analysis")
        
        if model is not None and feature_names:
            # Feature importance plot
            importance_fig = plot_feature_importance(model, feature_names)
            if importance_fig:
                st.plotly_chart(importance_fig, use_container_width=True)
            
            # Load feature importance from saved results if available
            results_path = os.path.join('results', 'feature_importance.csv')
            if os.path.exists(results_path):
                with st.expander("Detailed Feature Importance Across Models", expanded=False):
                    feature_importance_df = pd.read_csv(results_path)
                    st.dataframe(feature_importance_df, use_container_width=True)
        else:
            st.error("Model could not be loaded. Please check if the model file exists.")
    
    with tab3:
        st.header("Model Performance Evaluation")
        
        # Model selection
        st.markdown("""
        <div class="selector-container">
            <h3 style="margin-top:0; font-size:1.2rem; color:#1a5276;">Select a model to view detailed performance metrics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        model_options = ["SMOTE", "Two_Stage", "Weighted", "Crisis-Aware", "Standard"]
        selected_model = st.selectbox("Model", model_options, label_visibility="collapsed")
        
        # Display performance metrics for selected model in a clean container
        st.markdown(f"""
        <div class="performance-container">
            <h3 style="margin-top:0;">Performance Metrics: {selected_model} Model</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create one column for the confusion matrix only
        col1 = st.columns([1])[0]
        
        with col1:
            # Load confusion matrix for selected model
            cm_img = load_confusion_matrix(selected_model)
            if cm_img:
                st.image(cm_img, caption=f"Confusion Matrix - {selected_model} Model", use_container_width=True)
            else:
                st.warning(f"Confusion matrix for {selected_model} model not found.")
        
        # Model comparison section
        st.subheader("Model Comparison")
        
        # Plot model comparison
        model_comp_fig, perf_df = plot_model_comparison()
        if model_comp_fig:
            # Add custom CSS to ensure mobile responsiveness
            st.markdown("""
            <style>
            .js-plotly-plot, .plot-container {
                max-width: 100%;
            }
            @media (max-width: 768px) {
                .js-plotly-plot .main-svg {
                    max-width: 100% !important;
                }
                .stPlotlyChart {
                    overflow-x: scroll !important;
                    margin-top: 20px !important;  /* Add more space above the chart */
                }
                /* Fix for legend overlapping with title */
                .gtitle {
                    transform: translate(0, -30px) !important;
                }
                .legend {
                    transform: translate(0, 30px) !important;
                }
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Add explicit title using Streamlit (more control than Plotly title)
            st.markdown("<h3 style='text-align: center; margin-bottom: 30px;'>Model Performance Comparison</h3>", unsafe_allow_html=True)
            
            # Use container width but make responsive, with increased top margin
            st.plotly_chart(model_comp_fig, use_container_width=True, config={'responsive': True})
        
        # Confusion matrices for all models in a grid with improved mobile display
        st.subheader("Confusion Matrices for All Models")
        
        # Create rows with one model per row for mobile (instead of two models)
        for model_name in model_options:
            cm_img = load_confusion_matrix(model_name)
            if cm_img:
                st.image(cm_img, caption=f"{model_name} Model", use_container_width=True)
        
        # Show detailed performance metrics in a table
        with st.expander("Detailed Model Performance Metrics Table", expanded=False):
            # Load detailed performance metrics
            detailed_perf_path = os.path.join('results', 'model_performance_summary.csv')
            if os.path.exists(detailed_perf_path):
                detailed_df = pd.read_csv(detailed_perf_path)
                
                # Display as a styled table
                st.dataframe(detailed_df, use_container_width=True)
                
                # Add download button for the performance data
                csv = detailed_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv" class="btn">Download Performance Data</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("Detailed performance metrics file not found.")
    
    with tab4:
        st.header("Documentation")
        st.subheader("Project Repository")
        st.markdown("The entire data, model and project can be found at [GitHub](https://github.com/leo-lightfoot/ECB-Interest-Rate-Policy")
        
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
            
            We also implemented several other approaches:
            
            1. **Standard** - Basic Random Forest without special handling for imbalance
            2. **Crisis-Aware** - Model trained with awareness of financial crisis periods
            3. **Weighted** - Class-weighted Random Forest
            4. **SMOTE** - Random Forest with SMOTE oversampling
            5. **Two-Stage** - Decision tree first classifies between 'Hold' vs 'Change', then a second model determines 'Hike' vs 'Cut'
            """)
        
        with st.expander("Performance", expanded=True):
            st.markdown("""
            Our best performing model (Two-Stage) achieves:
            - Overall accuracy: 83.3%
            - F1 score for Rate Cut predictions: 44.4%
            - F1 score for Rate Hike predictions: 75.0%
            
            The SMOTE model (used for predictions in this app) achieves:
            - Overall accuracy: 76.2%
            - F1 score for Rate Cut predictions: 36.4%
            - F1 score for Rate Hike predictions: 57.1%
            """)
        
        with st.expander("Limitations", expanded=True):
            st.markdown("""
            While the model performs significantly better than random guessing, it still has limitations:
            - ECB decisions can be influenced by factors not captured in our data
            - Rare events like financial crises are difficult to predict
            - Past patterns may not always predict future decisions
            - Structural changes in the economy may affect the relationship between indicators and rate decisions
            - Forward guidance by the ECB has become more important but is challenging to quantify
            """)
        
        with st.expander("How to Use This App", expanded=True):
            st.markdown("""
            1. **Prediction Tab**: Enter economic data or select a past meeting to see what the model would predict
            2. **Feature Importance Tab**: Explore which factors most influence ECB rate decisions
            3. **Model Performance Tab**: Compare different models and examine their performance metrics
            4. **Documentation Tab**: Learn about the methodology and limitations
            
            The app is optimized for both desktop and mobile use.
            """)
        
        # Add links to data preprocessing and model analysis
        with st.expander("Code Repository & Documentation", expanded=True):
            st.markdown("""
            ### Access the Project's Code and Documentation
            
            #### Data Preprocessing
            The data preprocessing pipeline handles the collection, cleaning, and transformation of economic indicators and ECB meeting data.
            
            [View Data Preprocessing Code](https://github.com/leo-lightfoot/ECB-Interest-Rate-Policy/blob/main/Data_Preprocessing.py)
            
            ```python
            # Key steps in data preprocessing:
            # 1. Collection of macroeconomic indicators
            # 2. Cleaning and imputation of missing values
            # 3. Feature engineering and time window calculations
            # 4. Merging with ECB decision data
            # 5. Creating training/validation/test splits
            ```
            
            #### Model Training & Analysis
            The model training script contains the implementation of all modeling approaches and performance evaluation.
            
            [View Model Training Code](https://github.com/leo-lightfoot/ECB-Interest-Rate-Policy/blob/main/model_training.py)
            
            ```python
            # Key components of model training:
            # 1. Implementation of different modeling approaches (Standard, SMOTE, etc.)
            # 2. Cross-validation and hyperparameter tuning
            # 3. Performance metrics calculation and visualization
            # 4. Model serialization and export
            ```
            
            #### Results and Documentation
            For detailed analysis and documentation of the models, including performance metrics and feature importance analysis:
            
            [View Results Directory](https://github.com/leo-lightfoot/ECB-Interest-Rate-Policy/tree/main/results)
            
            [Project README](https://github.com/leo-lightfoot/ECB-Interest-Rate-Policy/blob/main/README.md)
            """)

if __name__ == "__main__":
    main() 
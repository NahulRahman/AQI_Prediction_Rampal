import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
from statsmodels.tsa.seasonal import STL
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="RampalAir Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3.5em;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2em;
        color: #555;
        margin-bottom: 30px;
    }
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .aqi-value {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .date-info {
        font-size: 0.9em;
        margin: 3px 0;
    }
    .health-concern {
        margin-top: 10px;
        padding: 8px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .good { background-color: #00e400; color: black; }
    .moderate { background-color: #ffff00; color: black; }
    .usg { background-color: #ff7e00; color: white; }
    .unhealthy { background-color: #ff0000; color: white; }
    .very-unhealthy { background-color: #8f3f97; color: white; }
    .hazardous { background-color: #7e0023; color: white; }
    
    .info-section {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        height: 100%;
    }
    .info-item {
        margin: 15px 0;
        line-height: 1.6;
    }
    .info-title {
        font-weight: bold;
        color: #1f77b4;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# Google Sheets URL
SHEET_URL = "https://docs.google.com/spreadsheets/d/1J4pNPY9QGUmfxn80rwNAM7sMmRQrkQ108EpHJcIGjPU/edit?usp=sharing"

# Station information
STATION_INFO = {
    "name": "Bagerhat C-CAMS-20 Rampal, Bagerhat",
    "location": "Maytree Super Thermal Power Project",
    "lat": 22.595556,
    "lon": 89.554028,
    "type": "Rural/Industrial"
}


@st.cache_data(ttl=3600)
def load_data_from_gsheets():
    """Load data from Google Sheets"""
    try:
        # Convert Google Sheets URL to CSV export URL
        sheet_id = SHEET_URL.split('/d/')[1].split('/')[0]
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        df = pd.read_csv(csv_url)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def format_date(date_str):
    """Format date to MM/DD/YYYY"""
    try:
        parts = str(date_str).split('/')
        month = parts[0].zfill(2)
        day = parts[1].zfill(2)
        year = parts[2]
        return f"{month}/{day}/{year}"
    except:
        return date_str

def preprocess_data(df):
    """Preprocess the data with STL imputation"""
    if df is None:
        return None
    
    # Select only Date and AQI columns
    df = df[['Date', 'AQI']].copy()
    
    # Apply date formatting
    df['Date'] = df['Date'].apply(format_date)
    
    # Convert to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['Date'])
    
    # Sort by date
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # Replace 'DNA' with NaN
    df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
    
    # Apply STL imputation
    df = stl_imputation(df)
    
    return df

def stl_imputation(df):
    """Method: Seasonal Decomposition (STL) Imputation"""
    df_copy = df.copy()
    
    # First fill NaNs temporarily for STL
    df_copy['AQI_temp'] = df_copy['AQI'].fillna(method='ffill').fillna(method='bfill')
    
    try:
        # Apply STL decomposition (seasonal period of 7 for weekly pattern)
        if len(df_copy) >= 14:  # Need at least 2 periods
            stl = STL(df_copy['AQI_temp'], seasonal=7, robust=True)
            result = stl.fit()
            # Use trend + seasonal components
            df_copy['AQI_filled'] = result.trend + result.seasonal
            # Only replace original NaN values
            mask = df_copy['AQI'].isna()
            df_copy.loc[mask, 'AQI'] = df_copy.loc[mask, 'AQI_filled']
            df_copy = df_copy[['Date', 'AQI']]
        else:
            # Not enough data for STL, use interpolation
            df_copy['AQI'] = df_copy['AQI'].interpolate(method='linear')
            df_copy['AQI'] = df_copy['AQI'].fillna(method='ffill').fillna(method='bfill')
            df_copy = df_copy[['Date', 'AQI']]
    except Exception as e:
        print(f"STL failed, using linear interpolation: {e}")
        df_copy['AQI'] = df_copy['AQI'].interpolate(method='linear')
        df_copy['AQI'] = df_copy['AQI'].fillna(method='ffill').fillna(method='bfill')
        df_copy = df_copy[['Date', 'AQI']]
    
    return df_copy

def aqi_to_category(aqi):
    """Convert AQI value to health concern category"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_health_concern_class(health_concern):
    """Get CSS class for health concern"""
    mapping = {
        "Good": "good",
        "Moderate": "moderate",
        "Unhealthy for Sensitive Groups": "usg",
        "Unhealthy": "unhealthy",
        "Very Unhealthy": "very-unhealthy",
        "Hazardous": "hazardous"
    }
    return mapping.get(health_concern, "")

def create_features(df):
    """Create time series features for XGBoost"""
    df = df.copy()
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_month'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['day_of_year'] = df['Date'].dt.dayofyear
    
    # Lag features
    for i in range(1, 8):
        df[f'lag_{i}'] = df['AQI'].shift(i)
    
    # Rolling statistics
    df['rolling_mean_7'] = df['AQI'].rolling(window=7, min_periods=1).mean()
    df['rolling_std_7'] = df['AQI'].rolling(window=7, min_periods=1).std()
    
    return df

def predict_aqi_xgboost(df, days=7):
    """Predict AQI using XGBoost"""
    # Create features
    df_features = create_features(df)
    
    # Drop rows with NaN in lag features (first few rows)
    df_features = df_features.dropna()
    
    if len(df_features) < 10:
        return [None] * days
    
    # Prepare training data
    feature_cols = ['day_of_week', 'day_of_month', 'month', 'day_of_year',
                    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7',
                    'rolling_mean_7', 'rolling_std_7']
    
    X = df_features[feature_cols]
    y = df_features['AQI']
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(X, y)
    
    # Make predictions
    predictions = []
    last_date = df['Date'].max()
    
    # Use the last rows for initial lag values
    prediction_df = df_features.copy()
    
    for i in range(days):
        # Create features for next day
        next_date = last_date + timedelta(days=i+1)
        
        next_features = {
            'day_of_week': next_date.dayofweek,
            'day_of_month': next_date.day,
            'month': next_date.month,
            'day_of_year': next_date.dayofyear,
        }
        
        # Get lag features from recent data
        recent_aqi = list(prediction_df['AQI'].tail(7).values)
        
        for lag in range(1, 8):
            if lag <= len(recent_aqi):
                next_features[f'lag_{lag}'] = recent_aqi[-lag]
            else:
                next_features[f'lag_{lag}'] = recent_aqi[0]
        
        next_features['rolling_mean_7'] = np.mean(recent_aqi)
        next_features['rolling_std_7'] = np.std(recent_aqi)
        
        # Predict
        X_next = pd.DataFrame([next_features])
        pred = model.predict(X_next)[0]
        predictions.append(int(round(pred)))
        
        # Add prediction to dataframe for next iteration
        new_row = pd.DataFrame({
            'Date': [next_date],
            'AQI': [pred]
        })
        prediction_df = pd.concat([prediction_df, new_row], ignore_index=True)
    
    return predictions

def create_dynamic_map():
    """Create a dynamic Folium map for Rampal Power Plant"""
    lat, lon = STATION_INFO["lat"], STATION_INFO["lon"]
    
    m = folium.Map(location=[lat, lon], zoom_start=12)
    
    folium.Marker(
        location=[lat, lon],
        popup=f"{STATION_INFO['name']}<br>{STATION_INFO['location']}",
        icon=folium.Icon(color="red", icon="industry", prefix='fa'),
    ).add_to(m)
    
    return m

# Main App
def main():
    # Title and subtitle
    st.markdown('<h1 class="main-title">RampalAir</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">A Dashboard Web Application Showing the Real-time AQI monitoring with Interactive visualizations, 7-day forecasting with XGBoost alongside Historical trend analysis of Rampal Powerplant, Bagerhat</p>', unsafe_allow_html=True)
    
    # Load and preprocess data
    with st.spinner('Loading data from Google Sheets...'):
        raw_data = load_data_from_gsheets()
        preprocessed_df = preprocess_data(raw_data)
    
    if preprocessed_df is None or len(preprocessed_df) == 0:
        st.error("Unable to load or process data. Please check the Google Sheets URL.")
        return
    
    # Add AQI Category
    preprocessed_df['AQI Category'] = preprocessed_df['AQI'].apply(aqi_to_category)
    
    # Interactive Scatter Plot
    #st.markdown("### 📈 Interactive AQI Trend Visualization")
    
    fig = px.scatter(
        preprocessed_df, 
        x='Date', 
        y='AQI', 
        color='AQI Category',
        size='AQI',
        title='Interactive AQI Trend Visualization - AQI Trends Over Time - Rampal Power Plant',
        color_discrete_map={
            "Good": "#00e400",
            "Moderate": "#ffff00",
            "Unhealthy for Sensitive Groups": "#ff7e00",
            "Unhealthy": "#ff0000",
            "Very Unhealthy": "#8f3f97",
            "Hazardous": "#7e0023"
        },
        height= 470
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="AQI Value",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 7-Day Predictions
    st.markdown("### 🔮 Upcoming 7 Days AQI Value Forecast")
    
    if len(preprocessed_df) >= 10:
        predictions = predict_aqi_xgboost(preprocessed_df, days=7)
        
        cols = st.columns(7)
        current_date = datetime.now()
        
        for i, col in enumerate(cols):
            with col:
                pred_date = current_date + timedelta(days=i+1)
                day_name = pred_date.strftime("%A")
                date_str = pred_date.strftime("%m/%d")
                
                if i < len(predictions) and predictions[i] is not None:
                    aqi_value = predictions[i]
                    health_concern = aqi_to_category(aqi_value)
                    health_class = get_health_concern_class(health_concern)
                    
                    st.markdown(f"""
                    <div class="prediction-container">
                        <div class="aqi-value">{aqi_value}</div>
                        <div class="date-info">{date_str}   {day_name}</div>
                        <div class="date-info"></div>
                        <div class="health-concern {health_class}">{health_concern}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-container">
                        <div class="aqi-value">--</div>
                        <div class="date-info">{date_str}  {day_name}</div>
                        <div class="date-info"></div>
                        <div class="health-concern">No Data</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Insufficient data to generate reliable predictions. Need at least 10 data points.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Three Column Layout
    col1, col2, col3 = st.columns(3)
    
    # Column 1: AQI Health Concern Table
    with col1:
        st.markdown("### 📊 AQI Health Index")
        
        # Create colored table using HTML
        table_html = """
        <table style="width:100%; border-collapse: collapse;">
            <thead>
                <tr style="background-color: #1f77b4; color: white;">
                    <th style="padding: 10px; border: 1px solid #ddd;">AQI Range</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Health Concern</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background-color: #00e400;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">0-50</td>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Good</td>
                </tr>
                <tr style="background-color: #ffff00;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">51-100</td>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Moderate</td>
                </tr>
                <tr style="background-color: #ff7e00; color: white;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">101-150</td>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Unhealthy for Sensitive Groups</td>
                </tr>
                <tr style="background-color: #ff0000; color: white;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">151-200</td>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Unhealthy</td>
                </tr>
                <tr style="background-color: #8f3f97; color: white;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">201-300</td>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Very Unhealthy</td>
                </tr>
                <tr style="background-color: #7e0023; color: white;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">301+</td>
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Hazardous</td>
                </tr>
            </tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)
    
    # Column 2: Station Information
    with col2:
        st.markdown("### 🏭 Station Information")
        st.markdown(f"""
        <div class="info-section">
            <div class="info-item">
                <div class="info-title">📍 Location: {STATION_INFO['name']}, {STATION_INFO['location']}</div>
            </div>
            <div class="info-item">
                <div class="info-title">🌍 Coordinates: Lat: {STATION_INFO['lat']}°N, Lon: {STATION_INFO['lon']}°E</div>
            </div>
            <div class="info-item">
                <div class="info-title">🏷️ Station Type: {STATION_INFO['type']}</div>
            </div>
            <div class="info-item">
                <div class="info-title">📊 About AQI: Air Quality Index (AQI) measures air pollution levels.<br>Higher values indicate greater health concerns.</div>
            </div>
            <div class="info-item">
                <div class="info-title">🔍 Dashboard Features</div>
                • Real-time AQI monitoring with Interactive visualizations<br>
                • 7-day forecasting with XGBoost alongside Historical trend analysis<br></div>
             
        </div>
        """, unsafe_allow_html=True)
    
    # Column 3: Dynamic Map
    with col3:
        st.markdown("### 🗺️ Station Location Map")
        folium_map = create_dynamic_map()
        folium_static(folium_map, width=950, height=350)
    
    # Footer
    #st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>RampalAQI Dashboard | Data Source: https://doe.gov.bd/ | Dashboard designed & Developed by Md. Nahul Rahman</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
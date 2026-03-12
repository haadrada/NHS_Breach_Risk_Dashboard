import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(
    page_title = "NHS_Diagnostic Breach Risk",
    layout = "wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv('predictions_dec2025.csv')
    return df

@st.cache_resource
def load_model():
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

df = load_data()
model, scaler = load_model()

st.title("NHS Diagnostic Breach Risk Dashboard")
st.markdown("Predicted breach risk for March 2026 based on December 2025 diagnostic data")



## getting addys


## testing map 

test_data = pd.DataFrame({
    'provider_org_code': ['RKL', 'RJ8', 'RWH', 'NDJ', 'RHU'],
    'lat': [54.2, 51.5, 51.7, 53.8, 50.8],
    'lon': [-2.7, -0.1, -0.3, -1.5, -1.1],
    'breach_risk_prob': [0.98, 0.97, 0.95, 0.88, 0.75]
})

fig = px.scatter_mapbox(
    test_data,
    lat='lat',
    lon='lon',
    color='breach_risk_prob',
    size='breach_risk_prob',
    hover_name='provider_org_code',
    color_continuous_scale=['green', 'yellow', 'red'],
    zoom=5,
    center={"lat": 52.5, "lon": -1.5},
    mapbox_style='carto-positron',
    title='NHS Provider Breach Risk'
)

fig.update_layout(
    height=700,  
    margin={"r":0,"t":0,"l":0,"b":0}  
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Providers", len(df))
st.divider()
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import requests

st.set_page_config(
    page_title = "NHS_Diagnostic Breach Risk",
    layout = "wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv('predictions_dec2025.csv')
    return df

@st.cache_data
def load_geojson():
    url = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Integrated_Care_Boards_April_2023_EN_BGC/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson"
    response = requests.get(url)
    return response.json()

@st.cache_resource
def load_model():
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

df = load_data()
model, scaler = load_model()
geojson = load_geojson()

st.title("NHS Diagnostic Breach Risk Dashboard")
st.markdown("Predicted breach risk for March 2026 based on December 2025 diagnostic data")



## choropleth map:

## Mapping chloropeth
## get ONS code matched with the name, add to icb_risk 

icb_risk = df.groupby('provider_parent_name').agg(
    avg_breach_prob = ('breach_risk_prob', 'mean')
).reset_index()

icb_risk['name_upper'] = icb_risk['provider_parent_name'].str.strip().str.upper()

geojson_name_to_code = {
    f['properties']['ICB23NM'].upper().strip(): f['properties']['ICB23CD']
    for f in geojson['features']
}

icb_risk['icb_ons_code'] = icb_risk['name_upper'].map(geojson_name_to_code)

fig = px.choropleth(
    icb_risk,
    geojson=geojson,
    locations = 'icb_ons_code',
    featureidkey = 'properties.ICB23CD',
    color = 'avg_breach_prob',
    color_continuous_scale = ['green', 'yellow', 'red'],
    hover_name = 'provider_parent_name',
    hover_data= {
        'avg_breach_prob': 'Avg Breach Probability',
        'icb_ons_code': False,
        'name_upper': False
    },
    labels={
        'avg_breach_prob': 'Avg Breach Probability'
    },
    title = 'NHS ICB Diagnostic Breach Risk - March 2026'
)

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(height=700, margin={"r":0,"t":30,"l":0,"b":0})
fig.show()




## testing map 

test_data = pd.DataFrame({
    'provider_org_code': ['RKL', 'RJ8', 'RWH', 'NDJ', 'RHU'],
    'lat': [54.2, 51.5, 51.7, 53.8, 50.8],
    'lon': [-2.7, -0.1, -0.3, -1.5, -1.1],
    'breach_risk_prob': [0.98, 0.97, 0.95, 0.88, 0.75]
})

fig2 = px.scatter_mapbox(
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

fig2.update_layout(
    height=700,  
    margin={"r":0,"t":0,"l":0,"b":0}  
)

fig2.show()

st.plotly_chart(fig, use_container_width=True)

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Providers", len(df))
st.divider()
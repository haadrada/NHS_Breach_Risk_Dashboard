import streamlit as st
import pandas as pd
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

df = load_data()
geojson = load_geojson()

st.markdown("<h3 style='text-align: center;'> NHS Diagnostic Breach Risk Dashboard </h3>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'> Predictions for March 2026 based on December 2025 Diagnostic Data </h5>", unsafe_allow_html=True)
st.divider()


st.markdown("<h5> Information about this Dashboard: </h5>", unsafe_allow_html=True)

with st.expander("What is Breach Risk?", expanded=False):
    st.markdown("""
    **Breach Risk** refers to the predicted probability that a diagnostic provider will 
    exceed **15.7%** of their waiting list in the 13+ week category within the next **3 months**.
    
    This threshold represents the **90th percentile** of provider performance during the 
    training period (pre-2023), identifying providers in the top 10% of observed backlog pressure.
    """)

with st.expander("Methodology", expanded=False):
    st.markdown("""
    - **Data Source:** NHS England Diagnostic Waiting Times dataset (2018–2025)
    - **Model and Performance:** XGBoost classifier trained on provider-level diagnostic activity: ROC-AUC 0.9428, Recall 0.81, Precision 0.52
    - **Prediction horizon:** 3 months ahead
    - **Training period:** 2018-2023data
    - **Test period:** 2023 onwards
    
    The breach label was defined using the empirical P90 of the training period (15.7%), 
    identifying providers whose 13-week waiting percentage places them in the top 10% 
    of observed values.
    """)

with st.expander("Limitations", expanded=False):
    st.markdown("""
    - 6 ICBs are absent from the regional map as they appear only as commissioners 
      in the source dataset, not providers.
    - Predictions for December 2025 extrapolate slightly beyond the training period.
    - Provider coordinates are derived from ODS postcode data. UPRN was not available for all provider codes, and as such precise building 
      locations may differ slightly.
    """)

st.divider()

## choropleth map:

## Mapping chloropeth
## get ONS code matched with the name, add to icb_risk 

icb_risk = df.groupby('provider_parent_name').agg(
    avg_breach_prob = ('breach_risk_prob', 'mean'),
    high_risk_count = ('breach_risk_pred', 'sum'),
    total_providers = ('provider_org_code', 'count')
).reset_index()

icb_risk['name_upper'] = icb_risk['provider_parent_name'].str.strip().str.upper()

geojson_name_to_code = {
    f['properties']['ICB23NM'].upper().strip(): f['properties']['ICB23CD']
    for f in geojson['features']
}

icb_risk['icb_ons_code'] = icb_risk['name_upper'].map(geojson_name_to_code)

# missing ICB data
missing_icbs = [
    { 
        'provider_parent_name': f['properties']['ICB23NM'],
        'avg_breach_prob': None,
        'high_risk_count': 0,
        'total_providers': 0,
        'icb_ons_code': f['properties']['ICB23CD'],
        'name_upper': f['properties']['ICB23NM'].upper()
    }
    for f in geojson['features']
    if f['properties']['ICB23CD'] not in icb_risk['icb_ons_code'].values 
]
missing_df = pd.DataFrame(missing_icbs)
icb_risk_full = pd.concat([icb_risk, missing_df], ignore_index = True)
icb_risk_full['avg_breach_prob'] = icb_risk_full['avg_breach_prob'].fillna(-1)

fig = px.choropleth(
    icb_risk_full,
    geojson=geojson,
    locations = 'icb_ons_code',
    featureidkey = 'properties.ICB23CD',
    color = 'avg_breach_prob',
    range_color = [-1, 0.75],
    color_continuous_scale = ['lightgrey', 'lightgrey', 'green', 'yellow', 'red'],
    hover_name = 'provider_parent_name',
    hover_data= {
        'avg_breach_prob': 'Avg Regional ICB Breach Probability',
        'high_risk_count': 'Breach Count',
        'total_providers': 'Total Providers',
        'icb_ons_code': False,
        'name_upper': False
    },
    custom_data = ['avg_breach_prob', 'high_risk_count', 'total_providers'],
    labels={
        'avg_breach_prob': 'Avg Breach Probability'
    },
    title = 'Average Breach Risk Probability mapped by Integrated Care Boards'
)


fig.update_traces(
    hovertemplate = "<b>%{hovertext}</b><br>" +
                  "Avg Breach Probability: %{customdata[0]:.3f}<br>" +
                  "High Risk Providers: %{customdata[1]:.0f}<br>" +
                  "Total Providers: %{customdata[2]:.0f}<br>"
)
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(height=700, margin={"r":0,"t":30,"l":0,"b":0}, title_x= 0.25)


## lay out of map 

st.markdown("<h5> Interactive Maps: </h5>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 2])

with col1: 
    map_view = st.radio(
        "Select View",
        ["Breach Risk by Provider Sites", "Regional Overview (ICB)"],
        horizontal=True
    )

map_df = df.copy()

if map_view == "Breach Risk by Provider Sites":
    with col2:
        st.markdown("<p style='font-size: 14px;'>Filter by risk level</p>", unsafe_allow_html=True)
        show_high_risk = st.checkbox("Show High Risk Only", value = False)
    with col3:
        top_n = st.slider("Top N Provides by Risk", min_value = 10, max_value = len(df), value = 208)


    if show_high_risk: 
        map_df = map_df[map_df['breach_risk_pred'] == 1]
        color_range = [map_df['breach_risk_prob'].min(), map_df['breach_risk_prob'].max()]
        color_scale = ['yellow', 'orange', 'red'] 

    else:
        color_range = [0,1]
        color_scale = ['green', 'yellow', 'red']

    map_df = map_df.nlargest(top_n, 'breach_risk_prob')
    center_lat = map_df['lat'].mean()
    center_lon = map_df['lon'].mean()

    fig2 = px.scatter_mapbox(
        map_df,
        lat='lat',
        lon='lon',
        color='breach_risk_prob',
        size='breach_risk_prob',
        hover_name='name',
        hover_data = {
            'provider_org_code': True,
            'breach_risk_prob': ':.3f',
            'lat': False,
            'lon': False
        },
        range_color = color_range, 
        color_continuous_scale = color_scale,
        zoom=5,
        center={"lat": center_lat, "lon": center_lon},
        mapbox_style='carto-positron',
        title='NHS Provider Breach Risk - March 2026'
    )

    fig2.update_layout(
        height=700,  
        margin={"r":0,"t":0,"l":0,"b":0}  
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Size and Colour of Dots indicate breach risk probability")

elif map_view == "Regional Overview (ICB)":
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.markdown("<h5> 2025 December Data Snapshot </h5>", unsafe_allow_html=True)





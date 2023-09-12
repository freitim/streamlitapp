import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from urllib.request import urlopen
import json
from copy import deepcopy
from cmcrameri import cm

st.set_page_config(layout = "wide")

def mpl_to_plotly(cmap, pl_entries=11, rdigits=2, reverse=False):
    # cmap - colormap 
    # pl_entries - int = number of Plotly colorscale entries
    # rdigits - int -=number of digits for rounding scale values
    scale = np.linspace(0, 1, pl_entries)
    colors = (cmap(scale)[:, :3]*255).astype(np.uint8)
    if reverse:
        colors = colors[::-1]
    pl_colorscale = [[round(s, rdigits), f'rgb{tuple(color)}'] for s, color in zip(scale, colors)]
    return pl_colorscale

## data handling
data_url = "https://data.stadt-zuerich.ch/dataset/sid_stapo_hundebestand_od1001/download/KUL100OD1001.csv"

geo_kreise_url = "https://www.ogd.stadt-zuerich.ch/wfs/geoportal/Stadtkreise?service=WFS&version=1.1.0&request=GetFeature&outputFormat=GeoJSON&typename=adm_stadtkreise_a"

geo_quar_url = "https://www.ogd.stadt-zuerich.ch/wfs/geoportal/Statistische_Quartiere?service=WFS&version=1.1.0&request=GetFeature&outputFormat=GeoJSON&typename=adm_statistische_quartiere_map"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

@st.cache_data
def load_geo(url):
    with urlopen(url) as response:
        gd = json.load(response)
    return gd

raw_df = load_data(data_url)
raw_geo_kreis = load_geo(geo_kreise_url)
raw_geo_quar = load_geo(geo_quar_url)

df = raw_df.copy(deep = True)
geo_kreis = deepcopy(raw_geo_kreis)
geo_quar = deepcopy(raw_geo_quar)

# drop unknown kreise
plt_theme = "ggplot2"

## streamlit
st.title("Dogs of Zürich")
st.header("Header")
st.subheader("Subheader")

if any(any(df[col].isna()) for col in df.columns):
    st.warning("There are missing values in your data.", icon="⚠️")

left_column, right_column = st.columns([4, 1])

options = df["StichtagDatJahr"].unique()
year = left_column.selectbox("Year", options)

region = right_column.radio("Regions", options = ["Kreis", "Quartier"])

if region == "Kreis":
    region_feat = "KreisLang"
    feat_id = "properties.bezeichnung"
    geo_dat = geo_kreis
else:
    region_feat = "QuarLang"
    feat_id = "properties.qname"
    geo_dat = geo_quar

cmap_cont = mpl_to_plotly(cm.roma, reverse = True)
cmap_cat = mpl_to_plotly(cm.roma, pl_entries = len(df[region_feat].unique()), reverse = True)

df = df[~df[region_feat].str.match(r"Unbekannt")]

if region == "Kreis":
    kreise = pd.CategoricalDtype([f"Kreis {i}" for i in range(1, 13)], ordered = True)
    df["KreisLang"] = df["KreisLang"].astype(kreise)

count_df = df[(df["StichtagDatJahr"] == year) ].groupby(region_feat)["AnzHunde"].sum().reset_index().sort_values(region_feat)

## figure 1
w = 900
h = 650
left_column, right_column = st.columns([1, 1])

fig = go.Figure(
    go.Choroplethmapbox(
        geojson = geo_dat,
        locations = count_df[region_feat],
        featureidkey = feat_id,
        z = count_df["AnzHunde"],
        colorscale = cmap_cont,
        marker_opacity = 0.5,
        hovertemplate = "%{location}<br>Number of Dogs: %{z}<extra></extra>",
        colorbar_title = "Total Number<br>of Dogs"
    )
)

fig.update_layout(
    mapbox_style = "carto-positron",
    mapbox_zoom = 10.6, 
    mapbox_center = {"lat": 47.37793765556796, "lon": 8.535132625542666}
)

fig.update_layout(
    title = {"text": f"Dogs per {region} in {year}", "font": {"size": 28}},
    template = plt_theme,
    width = h,
    height = h,
    margin={"r":0,"l":0,"b":100}
)

left_column.plotly_chart(fig)

## second figure
grouped = df.groupby(["StichtagDatJahr", region_feat])["AnzHunde"].sum().reset_index()

traces = [go.Scatter(
    name = name, 
    x = group["StichtagDatJahr"], 
    y = group["AnzHunde"], 
    marker = {"color": f"{cmap_cat[i][1]}"},
) for i, (name, group) in enumerate(grouped.groupby(region_feat))]

fig = go.Figure(
    data = traces
)

for fig_data in fig.data:
    fig_data["customdata"] = [fig_data["name"]] * len(fig_data["x"])

fig.update_traces(
    hovertemplate = "<b>%{customdata}</b><br>NoD: %{y}<extra></extra>"
)

fig.update_layout(
    title = {"text": f"Dogs per {region} over Time", "font": {"size": 28}},
    xaxis = {"title": {"text": "Year", "font": {"size": 24}}},
    yaxis = {"title": {"text": "Total Number of Dogs", "font": {"size": 24}}},
    hovermode = "x unified",
    template = plt_theme,
    width = w,
    height = h
)

right_column.plotly_chart(fig)

st.dataframe(df)

## third
feature_opts = ["Age Group", "Owner's Sex"]
selection = st.selectbox("Feature", options = feature_opts)

if selection == "Age Group":
    feature = "AlterV10Lang"
elif selection == "Owner's Sex":
    feature = "SexLang"

feat_nod = df.groupby(["StichtagDatJahr", feature])["AnzHunde"].sum().reset_index()
if selection == "Age Group":
    feat_nod = feat_nod[feat_nod["AlterV10Lang"] != "Unbekannt"]

feat_nod_piv = feat_nod.pivot(index = "StichtagDatJahr", columns = feature, values = "AnzHunde")

fig = go.Figure()

fig.add_heatmap(
    x = feat_nod_piv.columns,
    y = feat_nod_piv.index,
    z = feat_nod_piv,
    colorscale = cmap_cont,
    colorbar_title = "Total Number<br>of Dogs",
    hovertemplate = "Age Group: %{x}<br>Year: %{y}<br>NoD: %{z}<extra></extra>"
)

fig.update_layout(
    title = {"text": f"Dogs per {selection} over Time", "font": {"size": 28}},
    xaxis = {"title": {"text": selection, "font": {"size": 24}}},
    yaxis = {"title": {"text": "Year", "font": {"size": 24}}},
    width = 900,
    height = 600
)

st.plotly_chart(fig)

# checked = st.sidebar.checkbox("Show stuff")

# if checked:
#     st.subheader("That stuff from the sidebar")

# years = ["All"] + sorted(df["year"].unique)
# year = st.selectbox("Choose a Year", years)

# mpl_fig

# st.pyplot(mpl_fig)

# options = ["Option 1", "Option 2"]
# something = left_column.selectbox("Make a choice", options)

# somthing_else = right_column.selectbox("Make another choice", options)

# if something == "Option 1":
#     st.write("You chose Option 1")
# else:
#     st.write("You chose Option 2")

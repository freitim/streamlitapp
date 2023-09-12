import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.request import urlopen
import json
from copy import deepcopy
from cmcrameri import cm

st.set_page_config(layout="wide")


def mpl_to_plotly(cmap, pl_entries=11, rdigits=2, reverse=False):
    # cmap - colormap
    # pl_entries - int = number of Plotly colorscale entries
    # rdigits - int -=number of digits for rounding scale values
    scale = np.linspace(0, 1, pl_entries)
    colors = (cmap(scale)[:, :3] * 255).astype(np.uint8)
    if reverse:
        colors = colors[::-1]
    pl_colorscale = [
        [round(s, rdigits), f"rgb{tuple(color)}"] for s, color in zip(scale, colors)
    ]
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

df = raw_df.copy(deep=True)
geo_kreis = deepcopy(raw_geo_kreis)
geo_quar = deepcopy(raw_geo_quar)

# drop unknown kreise
plt_theme = "ggplot2"

## streamlit
st.title("Dogs of Zürich")

if any(any(df[col].isna()) for col in df.columns):
    st.warning("There are missing values in your data.", icon="⚠️")

# left_column, right_column, _ = st.columns([1, 1, 8])

st.sidebar.header("Settings")

year_opts = df["StichtagDatJahr"].unique()
year = st.sidebar.selectbox("Year", year_opts)

region = st.sidebar.radio("Regions", options=["Kreis", "Quartier"])

if region == "Kreis":
    region_feat = "KreisLang"
    feat_id = "properties.bezeichnung"
    geo_dat = geo_kreis
else:
    region_feat = "QuarLang"
    feat_id = "properties.qname"
    geo_dat = geo_quar

cmap_cont = mpl_to_plotly(cm.roma, reverse=True)
cmap_cat = mpl_to_plotly(
    cm.roma, pl_entries=len(df[region_feat].unique()), reverse=True
)

df = df[~df[region_feat].str.match(r"Unbekannt")]

if region == "Kreis":
    kreise = pd.CategoricalDtype([f"Kreis {i}" for i in range(1, 13)], ordered=True)
    df["KreisLang"] = df["KreisLang"].astype(kreise)

count_df = (
    df[(df["StichtagDatJahr"] == year)]
    .groupby(region_feat)["AnzHunde"]
    .sum()
    .reset_index()
    .sort_values(region_feat)
)

## 1st figure ##################################################################
### left ###
h = 600
w = 4 / 3 * h

left_column, right_column = st.columns([2, 3])

fig = go.Figure(
    go.Choroplethmapbox(
        geojson=geo_dat,
        locations=count_df[region_feat],
        featureidkey=feat_id,
        z=count_df["AnzHunde"],
        colorscale=cmap_cont,
        marker_opacity=0.5,
        hovertemplate="%{location}<br>Number of Dogs: %{z}<extra></extra>",
        colorbar_title="Number<br>of Dogs",
    )
)

fig.update_layout(
    # mapbox_style="carto-darkmatter",
    mapbox_style="stamen-toner",
    mapbox_zoom=10.6,
    mapbox_center={"lat": 47.37793765556796, "lon": 8.535132625542666},
)

fig.update_layout(
    title={"text": f"Dogs per {region} in {year}", "font": {"size": 28}},
    template=plt_theme,
    width=7 / 8 * h,
    height=h,
    margin={"r": 0, "l": 0},
)

left_column.plotly_chart(fig)

### right ###
grouped = df.groupby(["StichtagDatJahr", region_feat])["AnzHunde"].sum().reset_index()

traces = [
    go.Scatter(
        name=name,
        x=group["StichtagDatJahr"],
        y=group["AnzHunde"],
        marker={"color": f"{cmap_cat[i][1]}"},
    )
    for i, (name, group) in enumerate(grouped.groupby(region_feat))
]

fig = go.Figure(data=traces)

if year == max(year_opts):
    position = "top left"
    just = "right"
else:
    position = "top right"
    just = "left"

fig.add_vline(
    year,
    line_width=3,
    line_dash="dash",
    line_color="lightgray",
    annotation={
        "text": "Year<br>Shown",
        "font": {"size": 24},
        "align": just,
    },
    annotation_position=position,
)

for fig_data in fig.data:
    fig_data["customdata"] = [fig_data["name"]] * len(fig_data["x"])

fig.update_traces(hovertemplate="<b>%{customdata}</b><br>Dogs: %{y}<extra></extra>")

fig.update_layout(
    title={"text": f"Dogs per {region} over Time", "font": {"size": 28}},
    xaxis={"title": {"text": "Year", "font": {"size": 24}}},
    yaxis={"title": {"text": "Total Number of Dogs", "font": {"size": 24}}},
    hovermode="x unified",
    template=plt_theme,
    width=w,
    height=h,
)

right_column.plotly_chart(fig)


## 2nd figure ##################################################################
l_col, r_col = st.columns([1, 5])

feature_opts = {"Age Group": "AlterV10Lang", "Owner's Sex": "SexLang"}
selection_x = l_col.selectbox("X-Axis Feature", options=feature_opts)
selection_y = l_col.selectbox(
    "Y-Axis Feature",
    options={key: value for key, value in feature_opts.items() if key != selection_x},
)

feature_x = feature_opts[selection_x]
feature_y = feature_opts[selection_y]

feat_feat = (
    df.groupby(["StichtagDatJahr", feature_x, feature_y])["AnzHunde"]
    .sum()
    .reset_index()
)

if selection_x == "Age Group" or selection_y == "Age Group":
    feat_feat = feat_feat[feat_feat["AlterV10Lang"] != "Unbekannt"]


feat_feat_piv = feat_feat.pivot(
    index=["StichtagDatJahr", feature_y], columns=feature_x, values="AnzHunde"
).reset_index(feature_y)

grouped = feat_feat_piv.groupby("StichtagDatJahr")

fig = make_subplots(
    rows=3,
    cols=3,
    subplot_titles=list(grouped.groups.keys()),
    shared_xaxes=True,
    shared_yaxes=True,
)

for i, (name, group) in enumerate(grouped):
    row = i % 3 + 1
    col = i // 3 + 1
    fig.add_heatmap(
        name=name,
        x=group.columns[1:],
        y=group[feature_y],
        z=group[group.columns[1:]],
        coloraxis="coloraxis",
        row=row,
        col=col,
    )

fig.update_layout(width=900, height=600, coloraxis={"colorscale": cmap_cont})

st.plotly_chart(fig)

st.dataframe(df)

# feature_opts = ["Age Group", "Owner's Sex"]
# selection = st.selectbox("Feature", options=feature_opts)

# if selection == "Age Group":
#     feature = "AlterV10Lang"
# elif selection == "Owner's Sex":
#     feature = "SexLang"

# feat_nod = df.groupby(["StichtagDatJahr", feature])["AnzHunde"].sum().reset_index()
# if selection == "Age Group":
#     feat_nod = feat_nod[feat_nod["AlterV10Lang"] != "Unbekannt"]

# feat_nod_piv = feat_nod.pivot(
#     index="StichtagDatJahr", columns=feature, values="AnzHunde"
# )

# fig = go.Figure()

# fig.add_heatmap(
#     x=feat_nod_piv.columns,
#     y=feat_nod_piv.index,
#     z=feat_nod_piv,
#     colorscale=cmap_cont,
#     colorbar_title="Total Number<br>of Dogs",
#     hovertemplate="Age Group: %{x}<br>Year: %{y}<br>NoD: %{z}<extra></extra>",
# )

# fig.update_layout(
#     title={"text": f"Dogs per {selection} over Time", "font": {"size": 28}},
#     xaxis={"title": {"text": selection, "font": {"size": 24}}},
#     yaxis={"title": {"text": "Year", "font": {"size": 24}}},
#     width=900,
#     height=600,
# )
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

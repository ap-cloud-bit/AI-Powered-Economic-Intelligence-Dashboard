import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Economic Intelligence Dashboard", layout="wide")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    unemployment = pd.read_csv("data/unemployment.csv")
    gdp = pd.read_csv("data/richest_countries.csv")
    cost = pd.read_csv("data/cost_of_living.csv")
    corruption = pd.read_csv("data/corruption.csv")
    tourism = pd.read_csv("data/tourism.csv")

    unemployment.columns = ["country", "unemployment_rate"]
    gdp.columns = ["country", "gdp_per_capita"]

    df = gdp.merge(unemployment, on="country", how="left")
    df = df.merge(cost, on="country", how="left")
    df = df.merge(corruption[["country", "corruption_index"]], on="country", how="left")
    df = df.merge(
        tourism[["country", "receipts_in_billions", "percentage_of_gdp"]],
        on="country", how="left"
    )

    df = df.fillna(df.median(numeric_only=True))
    return df

df = load_data()

# -----------------------------
# Feature Engineering
# -----------------------------
df["affordability_index"] = df["gdp_per_capita"] / df["cost_index"]
df["labor_stress"] = df["unemployment_rate"] * df["cost_index"]
df["tourism_dependency"] = df["percentage_of_gdp"]

# -----------------------------
# Opportunity Score
# -----------------------------
features = [
    "gdp_per_capita",
    "purchasing_power_index",
    "receipts_in_billions",
    "unemployment_rate",
    "corruption_index"
]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
scaled_df = pd.DataFrame(scaled, columns=features)

df["opportunity_score"] = (
    scaled_df["gdp_per_capita"] * 0.30 +
    scaled_df["purchasing_power_index"] * 0.25 +
    scaled_df["receipts_in_billions"] * 0.15 -
    scaled_df["unemployment_rate"] * 0.15 -
    scaled_df["corruption_index"] * 0.15
) * 100

# -----------------------------
# Clustering
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
df["market_cluster"] = kmeans.fit_predict(
    df[["opportunity_score", "tourism_dependency", "corruption_index"]]
)

# --------------------------------------------------
# Sidebar – Market Filters
# --------------------------------------------------
st.sidebar.title("Market Filters")

selected_country = st.sidebar.selectbox(
    "Primary country",
    sorted(df["country"].unique())
)

st.sidebar.markdown("---")

compare_countries = st.sidebar.multiselect(
    "Compare with other countries",
    options=sorted(df["country"].unique()),
    default=[]
)

# -----------------------------
# Main Dashboard
# -----------------------------
st.title("🌍 AI‑Powered Economic Intelligence Dashboard")

country_data = df[df["country"] == selected_country].iloc[0]

comparison_df = None

if compare_countries:
    comparison_df = df[
        df["country"].isin([selected_country] + compare_countries)
    ][
        [
            "country",
            "opportunity_score",
            "unemployment_rate",
            "corruption_index",
            "gdp_per_capita",
            "purchasing_power_index"
        ]
    ].set_index("country")

col1, col2, col3 = st.columns(3)

col1.metric("Opportunity Score", f"{selected.opportunity_score:.1f}")
col2.metric("Unemployment Rate", f"{selected.unemployment_rate}%")
col3.metric("Corruption Index", f"{int(selected.corruption_index)}")

st.subheader("Country Profile")
st.write(selected[
    ["gdp_per_capita", "cost_index", "purchasing_power_index", "tourism_dependency"]
])

# -----------------------------
# Visualization
# -----------------------------
st.subheader("Market Landscape")

fig, ax = plt.subplots()
sns.scatterplot(
    data=df,
    x="corruption_index",
    y="opportunity_score",
    hue="market_cluster",
    ax=ax
)
st.pyplot(fig)

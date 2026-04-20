import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# ==================================================
# Page Config
# ==================================================
st.set_page_config(
    page_title="AI‑Powered Economic Intelligence Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ==================================================
# Data Loading
# ==================================================
@st.cache_data
def load_data():
    unemployment = pd.read_csv("data/unemployment.csv")
    gdp = pd.read_csv("data/richest_countries.csv")
    cost = pd.read_csv("data/cost_of_living.csv")
    corruption = pd.read_csv("data/corruption.csv")
    tourism = pd.read_csv("data/tourism.csv")

    unemployment.columns = ["country", "unemployment_rate"]
    gdp.columns = ["country", "gdp_per_capita"]

    df = (
        gdp.merge(unemployment, on="country", how="left")
           .merge(cost, on="country", how="left")
           .merge(corruption[["country", "corruption_index"]], on="country", how="left")
           .merge(
               tourism[["country", "receipts_in_billions", "percentage_of_gdp"]],
               on="country", how="left"
           )
    )

    df = df.fillna(df.median(numeric_only=True))
    return df


df = load_data()

# ==================================================
# Feature Engineering
# ==================================================
df["affordability_index"] = df["gdp_per_capita"] / df["cost_index"]
df["labor_stress"] = df["unemployment_rate"] * df["cost_index"]
df["tourism_dependency"] = df["percentage_of_gdp"]

# ==================================================
# Opportunity Score
# ==================================================
score_features = [
    "gdp_per_capita",
    "purchasing_power_index",
    "receipts_in_billions",
    "unemployment_rate",
    "corruption_index",
]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[score_features])
scaled_df = pd.DataFrame(scaled, columns=score_features)

df["opportunity_score"] = (
    scaled_df["gdp_per_capita"] * 0.30
    + scaled_df["purchasing_power_index"] * 0.25
    + scaled_df["receipts_in_billions"] * 0.15
    - scaled_df["unemployment_rate"] * 0.15
    - scaled_df["corruption_index"] * 0.15
) * 100

# ==================================================
# Clustering
# ==================================================
kmeans = KMeans(n_clusters=4, random_state=42)
df["market_cluster"] = kmeans.fit_predict(
    df[["opportunity_score", "tourism_dependency", "corruption_index"]]
)

# ==================================================
# Insight Function (✅ MUST COME BEFORE USE)
# ==================================================
def generate_insight(row):
    if row.opportunity_score > 70 and row.corruption_index < 40:
        return "High‑opportunity, low‑risk market suitable for expansion."
    elif row.opportunity_score > 60:
        return "Attractive market with moderate risk — due diligence recommended."
    elif row.tourism_dependency > 5:
        return "Tourism‑dependent economy — vulnerable to external shocks."
    else:
        return "Balanced opportunity with mixed economic signals."

# ==================================================
# Sidebar – Filters & Screenshot Mode
# ==================================================
st.sidebar.title("Market Filters")

selected_country = st.sidebar.selectbox(
    "Primary country",
    sorted(df["country"].unique())
)

compare_countries = st.sidebar.multiselect(
    "Compare with other countries",
    options=sorted(df["country"].unique()),
)

st.sidebar.markdown("---")
screenshot_mode = st.sidebar.checkbox("📸 Screenshot Mode", value=False)

# ==================================================
# Header
# ==================================================
st.title("🌍 AI‑Powered Economic Intelligence Dashboard")
st.caption(
    "Decision‑support dashboard for market entry, "
    "investment screening, and global business strategy."
)

# ==================================================
# Primary Country Data
# ==================================================
country_data = df[df["country"] == selected_country].iloc[0]

# ==================================================
# KPIs
# ==================================================
st.subheader("Key Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("Opportunity Score (0–100)", f"{country_data.opportunity_score:.1f}")
col2.metric("Unemployment Rate", f"{country_data.unemployment_rate:.1f}%")
col3.metric("Corruption Index", int(country_data.corruption_index))

st.markdown("---")

# ==================================================
# Country Profile
# ==================================================
st.subheader("Country Economic Profile")

profile_df = country_data[
    [
        "gdp_per_capita",
        "cost_index",
        "purchasing_power_index",
        "tourism_dependency",
    ]
].to_frame(name="Value")

st.dataframe(profile_df)

st.markdown("---")

# ==================================================
# Country Comparison
# ==================================================
if compare_countries:
    st.subheader("Country Comparison")

    comparison_df = df[
        df["country"].isin([selected_country] + compare_countries)
    ][
        [
            "country",
            "opportunity_score",
            "unemployment_rate",
            "corruption_index",
            "gdp_per_capita",
            "purchasing_power_index",
        ]
    ].set_index("country")

    st.dataframe(
        comparison_df.style.format(
            {
                "opportunity_score": "{:.1f}",
                "unemployment_rate": "{:.1f}",
                "corruption_index": "{:.0f}",
                "gdp_per_capita": "{:,.0f}",
                "purchasing_power_index": "{:.1f}",
            }
        )
    )

    if not screenshot_mode:
        fig, ax = plt.subplots()
        comparison_df["opportunity_score"].plot(kind="bar", ax=ax)
        ax.set_ylabel("Opportunity Score")
        ax.set_xlabel("")
        st.pyplot(fig)

    st.markdown("---")

# ==================================================
# Market Landscape (Hidden in Screenshot Mode)
# ==================================================
if not screenshot_mode:
    st.subheader("Global Market Landscape")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="corruption_index",
        y="opportunity_score",
        hue="market_cluster",
        palette="Set2",
        ax=ax,
    )
    ax.set_xlabel("Corruption Index (Lower is Better)")
    ax.set_ylabel("Opportunity Score")
    st.pyplot(fig)

# ==================================================
# Automated Business Insight (✅ FIXED & ENHANCED)
# ==================================================
st.subheader("Automated Business Insight")

col_a, col_b = st.columns([1, 3])

with col_a:
    st.metric(
        "Opportunity Level",
        "High" if country_data.opportunity_score >= 70
        else "Medium" if country_data.opportunity_score >= 50
        else "Low"
    )

with col_b:
    st.info(generate_insight(country_data))

st.markdown("**Key Drivers:**")

drivers = []

if country_data.gdp_per_capita > df["gdp_per_capita"].median():
    drivers.append("Above‑average GDP per capita")

if country_data.purchasing_power_index > df["purchasing_power_index"].median():
    drivers.append("Strong purchasing power")

if country_data.corruption_index > df["corruption_index"].median():
    drivers.append("Elevated governance risk")

if country_data.unemployment_rate > df["unemployment_rate"].median():
    drivers.append("Labor market stress")

if drivers:
    for d in drivers:
        st.write(f"• {d}")
else:
    st.write("• Balanced economic indicators with no extreme signals")

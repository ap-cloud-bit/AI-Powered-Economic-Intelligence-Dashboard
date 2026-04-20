import streamlit as st
import pandas as pd
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

    return df.fillna(df.median(numeric_only=True))


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
features = [
    "gdp_per_capita",
    "purchasing_power_index",
    "receipts_in_billions",
    "unemployment_rate",
    "corruption_index",
]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
scaled_df = pd.DataFrame(scaled, columns=features)

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
# Insight Functions
# ==================================================
def generate_single_insight(row):
    if row.opportunity_score > 70 and row.corruption_index < 40:
        return "High‑opportunity, low‑risk market suitable for expansion."
    elif row.opportunity_score > 60:
        return "Attractive market with moderate risk — due diligence recommended."
    elif row.tourism_dependency > 5:
        return "Tourism‑dependent economy — vulnerable to external shocks."
    else:
        return "Balanced opportunity with mixed economic signals."


def generate_comparative_insight(cdf):
    ranked = cdf.sort_values("opportunity_score", ascending=False)
    top = ranked.index[0]
    bottom = ranked.index[-1]

    insight = (
        f"Among the selected countries, **{top}** demonstrates the highest overall "
        f"opportunity score, reflecting stronger economic attractiveness. "
        f"In contrast, **{bottom}** ranks lower, indicating comparatively higher "
        f"economic or structural risk. "
        "Overall, the selected countries present clear risk‑reward trade‑offs "
        "rather than a single dominant choice."
    )
    return insight


def generate_comparative_report(cdf):
    report = "Comparative Economic Opportunity Report\n\n"
    ranked = cdf.sort_values("opportunity_score", ascending=False)

    for i, (country, row) in enumerate(ranked.iterrows(), 1):
        report += (
            f"{i}. {country}\n"
            f"   Opportunity Score: {row.opportunity_score:.1f}\n"
            f"   GDP per Capita: {row.gdp_per_capita:,.0f}\n"
            f"   Unemployment Rate: {row.unemployment_rate:.1f}%\n"
            f"   Corruption Index: {row.corruption_index:.0f}\n\n"
        )

    report += "Insight Summary:\n"
    report += generate_comparative_insight(cdf)
    return report

# ==================================================
# Sidebar
# ==================================================
st.sidebar.title("Market Filters")

selected_country = st.sidebar.selectbox(
    "Primary country",
    sorted(df["country"].unique())
)

compare_countries = st.sidebar.multiselect(
    "Compare with other countries",
    options=sorted(df["country"].unique())
)

st.sidebar.markdown("---")
screenshot_mode = st.sidebar.checkbox("📸 Screenshot Mode", value=False)

# ==================================================
# Header
# ==================================================
st.title("🌍 AI‑Powered Economic Intelligence Dashboard")
st.caption(
    "Data‑driven intelligence for market entry, investment screening, "
    "and global business decisions."
)

# ==================================================
# Primary Country
# ==================================================
country_data = df[df["country"] == selected_country].iloc[0]

# ==================================================
# KPIs
# ==================================================
st.subheader("Key Indicators")

c1, c2, c3 = st.columns(3)
c1.metric("Opportunity Score (0–100)", f"{country_data.opportunity_score:.1f}")
c2.metric("Unemployment Rate", f"{country_data.unemployment_rate:.1f}%")
c3.metric("Corruption Index", int(country_data.corruption_index))

st.markdown("---")

# ==================================================
# Country Profile
# ==================================================
st.subheader("Country Economic Profile")
profile = country_data[
    ["gdp_per_capita", "cost_index", "purchasing_power_index", "tourism_dependency"]
].to_frame("Value")
st.dataframe(profile)

st.markdown("---")

# ==================================================
# Top‑10 Opportunity Ranking ✅
# ==================================================
st.subheader("Top 10 Countries by Opportunity Score")

top10 = df.sort_values("opportunity_score", ascending=False).head(10)[
    ["country", "opportunity_score", "unemployment_rate", "corruption_index"]
]
st.dataframe(top10.set_index("country").style.format("{:.1f}"))

st.markdown("---")

# ==================================================
# Country Comparison
# ==================================================
if compare_countries:
    st.subheader("Country Comparison")

    comparison_df = df[
        df["country"].isin([selected_country] + compare_countries)
    ].set_index("country")

    st.dataframe(
        comparison_df[
            [
                "opportunity_score",
                "unemployment_rate",
                "corruption_index",
                "gdp_per_capita",
                "purchasing_power_index",
            ]
        ].style.format("{:.1f}")
    )

    if not screenshot_mode:
        fig, ax = plt.subplots()
        comparison_df["opportunity_score"].plot(kind="bar", ax=ax)
        ax.set_ylabel("Opportunity Score")
        st.pyplot(fig)

    # ✅ Downloadable comparative report
    st.download_button(
        "📄 Download Comparative Report",
        generate_comparative_report(comparison_df),
        file_name="comparative_economic_report.txt",
        mime="text/plain"
    )

    st.markdown("---")

# ==================================================
# Market Landscape
# ==================================================
if not screenshot_mode:
    st.subheader("Global Market Landscape")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="corruption_index",
        y="opportunity_score",
        hue="market_cluster",
        ax=ax,
        palette="Set2"
    )
    ax.set_xlabel("Corruption Index (Lower is Better)")
    ax.set_ylabel("Opportunity Score")
    st.pyplot(fig)

# ==================================================
# Insights
# ==================================================
st.subheader(
    "Comparative Business Insight" if compare_countries else "Automated Business Insight"
)

if compare_countries:
    st.info(generate_comparative_insight(comparison_df))
else:
    st.info(generate_single_insight(country_data))

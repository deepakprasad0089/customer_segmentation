import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# ---------- Page Setup ----------
st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide")
st.title("📊 Customer Segmentation Dashboard")
st.markdown("RFM-based customer segments from the Online Retail dataset")
st.divider()

# ---------- Absolute Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODELS_DIR = os.path.join(BASE_DIR, "..", "Models")
PICTURES_DIR = os.path.join(BASE_DIR, "..", "Pictures")

CLUSTER_NAMES = {
    0: "Regular Customers",
    1: "VIP / Champions",
    2: "At Risk / Lost",
    3: "Loyal Customers",
}
PALETTE = {
    "VIP / Champions": "#C0392B",
    "Loyal Customers": "#27AE60",
    "Regular Customers": "#2980B9",
    "At Risk / Lost": "#95A5A6",
}

# ---------- Load Data & Models ----------
@st.cache_data
def load_data():
    rfm = pd.read_excel(os.path.join(DATA_DIR, "rfm_with_clusters.xlsx"), index_col="CustomerID")
    rfm["Segment"] = rfm["Cluster"].map(CLUSTER_NAMES)
    return rfm

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))

rfm = load_data()
rf_model = load_model()

# ---------- Sidebar ----------
st.sidebar.header("Filters")
segments = st.sidebar.multiselect(
    "Select segments to view:",
    options=rfm["Segment"].unique(),
    default=rfm["Segment"].unique(),
)
filtered = rfm[rfm["Segment"].isin(segments)]

st.sidebar.divider()
st.sidebar.caption("Customer Segmentation Dashboard")
st.sidebar.caption("RFM Analysis + K-Means Clustering + Market Basket Analysis")

# ---------- KPI Cards ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(filtered):,}")
col2.metric("Avg. Recency (days)", f"{filtered['Recency'].mean():.0f}")
col3.metric("Avg. Frequency", f"{filtered['Frequency'].mean():.1f}")
col4.metric("Avg. Monetary (£)", f"£{filtered['Monetary'].mean():,.0f}")

st.divider()

# ---------- Segment Distribution Chart ----------
st.subheader("Segment Distribution")
segment_counts = filtered["Segment"].value_counts()

left, right = st.columns([2, 1])
with left:
    fig, ax = plt.subplots(figsize=(5.5, 3))
    bars = ax.bar(segment_counts.index, segment_counts.values,
                   color=[PALETTE[s] for s in segment_counts.index])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:,.0f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_ylabel("Number of Customers", fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

with right:
    st.dataframe(segment_counts.rename("Customers"))

st.divider()

# ---------- Saved Charts (fixed alignment: all images same constrained width) ----------
st.subheader("Saved Analysis Charts")
tab1, tab2, tab3, tab4 = st.tabs(["Cluster Scatter", "Cluster Profile", "Monthly Trend", "Market Basket"])

with tab1:
    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        st.image(os.path.join(PICTURES_DIR, "cluster_scatter_professional.png"), use_container_width=True)

with tab2:
    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        st.image(os.path.join(PICTURES_DIR, "cluster_profile_bars.png"), use_container_width=True)

with tab3:
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        st.image(os.path.join(PICTURES_DIR, "monthly_trend.png"), use_container_width=True)

with tab4:
    st.subheader("Product Association Rules")
    c1, c2 = st.columns(2)
    with c1:
        st.image(os.path.join(PICTURES_DIR, "basket_rules_scatter.png"), use_container_width=True)
    with c2:
        st.image(os.path.join(PICTURES_DIR, "basket_network_graph.png"), use_container_width=True)

    rules_path = os.path.join(DATA_DIR, "association_rules.xlsx")
    if os.path.exists(rules_path):
        rules_df = pd.read_excel(rules_path)
        st.dataframe(
            rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            .sort_values('lift', ascending=False)
        )
    else:
        st.warning(f"association_rules.xlsx not found in: {DATA_DIR}")

st.divider()

# ---------- Data Table ----------
st.subheader("Customer Data")
st.dataframe(filtered)

st.divider()

# ---------- Live Prediction Tool ----------
st.subheader("🔮 Predict a New Customer's Segment")
c1, c2, c3 = st.columns(3)
recency = c1.number_input("Recency (days since last order)", min_value=0, value=30)
frequency = c2.number_input("Frequency (number of orders)", min_value=1, value=3)
monetary = c3.number_input("Monetary (total spend £)", min_value=0.0, value=500.0)

if st.button("Predict Segment", type="primary"):
    new_customer = pd.DataFrame({"Recency": [recency], "Frequency": [frequency], "Monetary": [monetary]})
    pred_cluster = rf_model.predict(new_customer)[0]
    pred_segment = CLUSTER_NAMES[pred_cluster]

    st.success(f"Predicted Segment: **{pred_segment}**")

    segment_avg = rfm[rfm["Segment"] == pred_segment][["Recency", "Frequency", "Monetary"]].mean()
    comparison_df = pd.DataFrame({
        "This Customer": [recency, frequency, monetary],
        f"{pred_segment} Average": [segment_avg["Recency"], segment_avg["Frequency"], segment_avg["Monetary"]],
    }, index=["Recency", "Frequency", "Monetary"])

    st.markdown(f"#### How this customer compares to a typical **{pred_segment}**")

    bar_left, bar_right = st.columns([3, 1])
    with bar_left:
        fig2, axes2 = plt.subplots(1, 3, figsize=(8, 2.3))
        metrics = ["Recency", "Frequency", "Monetary"]
        for ax, metric in zip(axes2, metrics):
            values = comparison_df.loc[metric]
            bars = ax.bar(values.index, values.values,
                           color=[PALETTE[pred_segment], "#7f8c8d"], width=0.6)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:,.0f}',
                        ha='center', va='bottom', fontsize=6, fontweight='bold')
            ax.set_title(metric, fontsize=8, fontweight='bold')
            ax.tick_params(axis='both', labelsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("#### Where this customer sits among all segments")

    scatter_left, scatter_right = st.columns([2, 1])
    with scatter_left:
        fig3, ax3 = plt.subplots(figsize=(4.5, 3))
        for segment, color in PALETTE.items():
            subset = rfm[rfm["Segment"] == segment]
            ax3.scatter(subset["Recency"], subset["Monetary"], label=segment,
                        color=color, alpha=0.4, s=10)

        ax3.scatter(recency, monetary, color="black", marker="*", s=200,
                    edgecolor="yellow", linewidth=1.0, label="This Customer", zorder=5)

        ax3.set_xlabel("Recency (days)", fontsize=8)
        ax3.set_ylabel("Monetary (£)", fontsize=8)
        ax3.set_yscale("log")
        ax3.tick_params(axis='both', labelsize=7)
        ax3.legend(loc="upper right", fontsize=6)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)

# ---------- Footer ----------
st.divider()
st.markdown(
    "<p style='text-align: center; color: grey; font-size: 13px;'>"
    "Built by <b>Usman Haider</b> — Customer Segmentation & Market Basket Analysis Project"
    "</p>",
    unsafe_allow_html=True
)
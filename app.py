import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="MPI Performance Dashboard",
    layout="wide"
)

# =====================================
# Load Data
# =====================================

df = pd.read_csv("data_median_obs.csv")

# Rename algorithms for readability
algorithm_map = {
    "TTPL_BT": "HieAta",
    "2PhaseRBruck": "ParAta",
    "MPI_Alltoallv": "MPI_Alltoallv"
}

df["algorithm_name"] = df["algorithm"].map(algorithm_map)

# =====================================
# Title
# =====================================

st.title("MPI Communication Performance Dashboard")

st.write(
    """
    Interactive dashboard for exploring MPI communication benchmark performance
    across different algorithms, process counts, and system configurations.
    """
)

# =====================================
# Tabs
# =====================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Algorithm Comparison",
    "Scaling Analysis",
    "Parameter Optimization"
])

# =====================================
# TAB 1: OVERVIEW
# =====================================

with tab1:

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Experiments", f"{len(df):,}")
    col2.metric("Algorithms", df["algorithm_name"].nunique())
    col3.metric("Process Counts", df["num_process"].nunique())
    col4.metric("PPN Values", df["ppn"].nunique())

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20))

    st.subheader("Available Algorithms")

    st.write(df["algorithm_name"].unique())

# =====================================
# TAB 2: ALGORITHM COMPARISON
# =====================================

with tab2:

    st.header("Algorithm Comparison")

    selected_ppn = st.selectbox(
        "Processes Per Node (PPN)",
        sorted(df["ppn"].unique()),
        key="alg_ppn"
    )

    filtered_df = df[df["ppn"] == selected_ppn]

    avg_time = (
        filtered_df
        .groupby("algorithm_name")["time"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        avg_time,
        x="algorithm_name",
        y="time",
        title=f"Average Communication Time (PPN = {selected_ppn})",
        labels={
            "algorithm_name": "Algorithm",
            "time": "Average Latency"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance Summary")

    summary = (
        filtered_df
        .groupby("algorithm_name")["time"]
        .agg(["mean", "median", "min", "max"])
        .round(4)
        .reset_index()
    )

    summary.columns = [
        "Algorithm",
        "Average Time",
        "Median Time",
        "Minimum Time",
        "Maximum Time"
    ]

    st.dataframe(summary)

# =====================================
# TAB 3: SCALING ANALYSIS
# =====================================

with tab3:

    st.header("Scaling Analysis")

    scaling_data = (
        df.groupby(
            ["num_process", "algorithm_name"]
        )["time"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        scaling_data,
        x="num_process",
        y="time",
        color="algorithm_name",
        markers=True,
        title="Communication Time vs Number of Processes",
        labels={
            "num_process": "Number of Processes",
            "time": "Average Latency"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write(
        """
        This visualization shows how communication performance changes
        as the total number of processes increases.
        """
    )

# =====================================
# TAB 4: PARAMETER OPTIMIZATION
# =====================================

with tab4:

    st.header("Parameter Optimization")

    selected_algorithm = st.selectbox(
        "Algorithm",
        sorted(df["algorithm_name"].dropna().unique()),
        key="param_alg"
    )

    param_df = df[df["algorithm_name"] == selected_algorithm]

    st.subheader("Configuration Preview")

    st.dataframe(
        param_df[
            [
                "algorithm_name",
                "num_process",
                "ppn",
                "radix",
                "blocksize",
                "time"
            ]
        ].head(20)
    )

    # Heatmap only for HieAta

    if selected_algorithm == "HieAta":

        st.subheader("Radix vs Blocksize Performance Heatmap")

        heatmap_data = (
            param_df
            .groupby(["blocksize", "radix"])["time"]
            .mean()
            .reset_index()
        )

        heatmap_pivot = heatmap_data.pivot(
            index="blocksize",
            columns="radix",
            values="time"
        )

        fig = px.imshow(
            heatmap_pivot,
            aspect="auto",
            labels=dict(
                x="Radix",
                y="Blocksize",
                color="Average Time"
            ),
            title="Average Communication Time"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        st.subheader("Radix Analysis")

        radix_data = (
            param_df
            .groupby("radix")["time"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            radix_data,
            x="radix",
            y="time",
            markers=True,
            title="Average Time vs Radix"
        )

        st.plotly_chart(fig, use_container_width=True)
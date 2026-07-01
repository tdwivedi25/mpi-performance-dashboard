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
# Sidebar Filters
# =====================================

st.sidebar.header("Experiment Filters")

# Processes Per Node
selected_ppn = st.sidebar.selectbox(
    "Processes Per Node (PPN)",
    sorted(df["ppn"].unique())
)

# Topology
selected_topology = st.sidebar.selectbox(
    "Topology",
    sorted(df["topology"].unique())
)

# Message Size Range
min_msg = float(df["E[send_count]"].min())
max_msg = float(df["E[send_count]"].max())

selected_message_range = st.sidebar.slider(
    "Average Message Size",
    min_value=min_msg,
    max_value=max_msg,
    value=(min_msg, max_msg)
)

# Mode toggle
scaling_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Fixed Process Count (Comparison)", "Scaling Analysis"]
)

# =====================================
# BASE FILTER (always applied)
# =====================================

base_df = df[
    (df["ppn"] == selected_ppn) &
    (df["topology"] == selected_topology) &
    (df["E[send_count]"] >= selected_message_range[0]) &
    (df["E[send_count]"] <= selected_message_range[1])
]

# =====================================
# MODE-SPECIFIC FILTER
# =====================================

if scaling_mode == "Fixed Process Count (Comparison)":

    selected_process = st.sidebar.selectbox(
        "Number of Processes",
        sorted(df["num_process"].unique())
    )

    filtered_df = base_df[base_df["num_process"] == selected_process]

else:
    filtered_df = base_df

# Debug
st.write("Filtered rows:", len(filtered_df))

# =====================================
# Title
# =====================================

st.title("MPI Communication Performance Dashboard")

st.write(
    """
    Interactive dashboard for exploring MPI communication benchmark performance
    across different algorithms and system configurations.
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

# =====================================
# TAB 2: ALGORITHM COMPARISON
# =====================================

with tab2:

    st.header("Algorithm Comparison")

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
    else:

        # Create summary statistics for each algorithm
        summary = (
            filtered_df
            .groupby("algorithm_name")["time"]
            .agg(
                Average="mean",
                Median="median",
                Minimum="min",
                Maximum="max",
                Runs="count"
            )
            .round(4)
            .reset_index()
        )

        # Sort from fastest to slowest
        summary = summary.sort_values("Average")

        # Highlight the fastest algorithm
        fastest = summary.iloc[0]

        st.success(
            f"🏆 Fastest Algorithm: **{fastest['algorithm_name']}** "
            f"(Average Time = {fastest['Average']:.4f})"
        )

        # Bar chart
        fig = px.bar(
            summary,
            x="algorithm_name",
            y="Average",
            color="algorithm_name",
            title="Average Communication Time by Algorithm",
            labels={
                "algorithm_name": "Algorithm",
                "Average": "Average Communication Time"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistics table
        st.subheader("Performance Summary")

        st.dataframe(
            summary.rename(
                columns={
                    "algorithm_name": "Algorithm"
                }
            ),
            use_container_width=True
        )
# =====================================
# TAB 3: SCALING ANALYSIS
# =====================================

with tab3:

    st.header("Scaling Analysis")

    st.write(
        """
        This analysis shows how communication time changes as the number of
        MPI processes increases while keeping the selected PPN, topology,
        and message size range fixed.
        """
    )

    # Scaling only works when process count is NOT fixed
    if scaling_mode == "Fixed Process Count (Comparison)":

        st.info(
            """
            Scaling Analysis requires the number of processes to vary.

            Please switch the sidebar **Analysis Mode** to
            **Scaling Analysis** to view scaling behavior across
            different process counts.
            """
        )

    else:

        if filtered_df.empty:
            st.warning("No scaling data available for the selected filters.")

        else:

            scaling_data = (
                filtered_df
                .groupby(["num_process", "algorithm_name"])
                .agg(
                    Average_Time=("time", "mean"),
                    Runs=("time", "count")
                )
                .reset_index()
                .sort_values("num_process")
            )

            fig = px.line(
                scaling_data,
                x="num_process",
                y="Average_Time",
                color="algorithm_name",
                markers=True,
                hover_data=["Runs"],
                labels={
                    "num_process": "Number of Processes",
                    "Average_Time": "Average Communication Time",
                    "algorithm_name": "Algorithm"
                },
                title="Scaling Performance Across Process Counts"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Scaling Data")

            st.dataframe(
                scaling_data.rename(
                    columns={
                        "num_process": "Processes",
                        "algorithm_name": "Algorithm",
                        "Average_Time": "Average Time"
                    }
                ),
                use_container_width=True
            )

            st.info(
                f"Experiments included in this analysis: {len(filtered_df):,}"
            )

# =====================================
# TAB 4: PARAMETER OPTIMIZATION
# =====================================

with tab4:

    st.header("Parameter Optimization")

    st.write(
        """
        Explore how different parameter settings affect communication
        performance under the currently selected experimental conditions.
        """
    )

    selected_algorithm = st.selectbox(
        "Algorithm",
        sorted(df["algorithm_name"].dropna().unique()),
        key="param_alg"
    )

    # Apply global filters + algorithm selection
    param_df = filtered_df[
        filtered_df["algorithm_name"] == selected_algorithm
    ]

    if param_df.empty:

        st.warning(
            "No parameter data is available for the selected filters."
        )

    else:

        st.subheader("Configuration Preview")

        st.dataframe(
            param_df[
                [
                    "algorithm_name",
                    "num_process",
                    "ppn",
                    "topology",
                    "radix",
                    "blocksize",
                    "time"
                ]
            ].head(20),
            use_container_width=True
        )

        # =====================================
        # HieAta Heatmap
        # =====================================

        if selected_algorithm == "HieAta":

            st.subheader("Radix vs Blocksize Heatmap")

            st.write(
                """
                This heatmap shows the average communication time for
                different radix and blocksize combinations under the
                selected experimental conditions.
                Lower values indicate better performance.
                """
            )

            if (
                param_df["radix"].nunique() < 2
                or param_df["blocksize"].nunique() < 2
            ):

                st.info(
                    "Not enough parameter combinations are available "
                    "to generate a heatmap."
                )

            else:

                heatmap_data = (
                    param_df
                    .groupby(["blocksize", "radix"])
                    .agg(
                        Average_Time=("time", "mean")
                    )
                    .reset_index()
                )

                heatmap_pivot = heatmap_data.pivot(
                    index="blocksize",
                    columns="radix",
                    values="Average_Time"
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

        # =====================================
        # Other Algorithms
        # =====================================

        else:

            st.subheader("Radix Performance")

            radix_data = (
                param_df
                .groupby("radix")
                .agg(
                    Average_Time=("time", "mean")
                )
                .reset_index()
                .sort_values("radix")
            )

            fig = px.line(
                radix_data,
                x="radix",
                y="Average_Time",
                markers=True,
                labels={
                    "radix": "Radix",
                    "Average_Time": "Average Communication Time"
                },
                title="Communication Time vs Radix"
            )

            st.plotly_chart(fig, use_container_width=True)
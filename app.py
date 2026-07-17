import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="MPI Performance Dashboard",
    layout="wide"
)

st.caption(
    "Developed by Tanvi Dwivedi | Interactive visualization for MPI communication benchmark analysis"
)

# =====================================
# Load Data (cached so it isn't re-read on every widget interaction)
# =====================================

algorithm_map = {
    "TTPL_BT": "HieAta",
    "2PhaseRBruck": "ParAta",
    "MPI_Alltoallv": "MPI_Alltoallv"
}

@st.cache_data
def load_data():
    df = pd.read_csv("data_median_obs.csv")
    df["algorithm_name"] = df["algorithm"].map(algorithm_map)
    return df

# =====================================
# Helper Functions
# =====================================

def format_time(seconds):
    """Convert seconds to a readable unit."""

    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} microseconds"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} milliseconds"
    else:
        return f"{seconds:.3f} seconds"
    
df = load_data()

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

    # Options come from base_df so the list only shows process counts that
    # actually exist under the filters already chosen. Falls back to the
    # full dataset if the current filters return nothing, so the widget
    # never renders with an empty option list.
    process_options = (
        sorted(base_df["num_process"].unique())
        if not base_df.empty
        else sorted(df["num_process"].unique())
    )

    selected_process = st.sidebar.selectbox(
        "Number of Processes",
        process_options
    )

    filtered_df = base_df[base_df["num_process"] == selected_process]

else:
    filtered_df = base_df

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

with tab2:

    st.header("Algorithm Comparison")

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
    else:

        comparison_basis = st.radio(
            "Compare algorithms using:",
            ["Best parameter setting", "Average across all parameters"],
            horizontal=True,
            help=(
                "Best parameter setting compares each algorithm at its "
                "fastest radix/blocksize combination under the current "
                "filters. Average across all parameters compares algorithms "
                "using their mean time across every parameter combination "
                "tested (the original behavior)."
            )
        )

        if comparison_basis == "Best parameter setting":

            # Mean time for every (algorithm, radix, blocksize) combo
            # under the current filters ...
            param_means = (
                filtered_df
                .groupby(["algorithm_name", "radix", "blocksize"])["time"]
                .mean()
                .reset_index()
            )

            # ... then pick the fastest combo per algorithm.
            best_idx = param_means.groupby("algorithm_name")["time"].idxmin()
            best_params = param_means.loc[best_idx].reset_index(drop=True)

            summary_rows = []
            for _, row in best_params.iterrows():
                subset = filtered_df[
                    (filtered_df["algorithm_name"] == row["algorithm_name"]) &
                    (filtered_df["radix"] == row["radix"]) &
                    (filtered_df["blocksize"] == row["blocksize"])
                ]
                summary_rows.append({
                    "algorithm_name": row["algorithm_name"],
                    "Average": subset["time"].mean(),
                    "Median": subset["time"].median(),
                    "Minimum": subset["time"].min(),
                    "Maximum": subset["time"].max(),
                    "Runs": subset["time"].count(),
                    "Best Radix": row["radix"],
                    "Best Blocksize": row["blocksize"],
                })

            summary = pd.DataFrame(summary_rows)

        else:

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
                
                .reset_index()
            )

        # Sort from fastest to slowest
        summary = summary.sort_values("Average")

        # Highlight the fastest algorithm
        fastest = summary.iloc[0]

        st.success(
            f"🏆 Fastest Algorithm: **{fastest['algorithm_name']}** "
            f"(Average Time = {format_time(fastest['Average'])}, "
            f"based on {int(fastest['Runs'])} runs)"
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
        display_summary = summary.copy()

        for col in ["Average", "Median", "Minimum", "Maximum"]:
            display_summary[col] = (display_summary[col] * 1000).round(4)

        display_summary = display_summary.rename(columns={
    "algorithm_name": "Algorithm",
    "Average": "Average (milliseconds)",
    "Median": "Median (milliseconds)",
    "Minimum": "Minimum (milliseconds)",
    "Maximum": "Maximum (milliseconds)"
})

        st.dataframe(display_summary, use_container_width=True)

        # =====================================
        # Anomaly Detection (Outlier Analysis)
        # =====================================

        st.subheader("Anomaly Detection")

        st.caption(
            "Uses a simple machine learning model (Isolation Forest) to automatically "
            "flag runs that behave very differently from the rest under the current "
            "filters — useful for catching measurement noise or genuinely unusual "
            "performance behavior that's hard to spot by scanning the raw data."
        )

        if len(filtered_df) < 10:
            st.info(
                "Not enough data to run anomaly detection meaningfully. "
                "At least 10 runs are recommended."
            )
        else:
            # Prepare features for anomaly detection
            # Select "time" column, and "Var[send_count]" if available
            feature_cols = ["time"]
            if "Var[send_count]" in filtered_df.columns:
                feature_cols.append("Var[send_count]")
            
            # Build feature matrix with explicit column names and drop NaN rows
            X = filtered_df[feature_cols].dropna()
            
            # Track which rows are kept after NaN removal (by index)
            valid_indices = X.index
            
            # Run IsolationForest for anomaly detection
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            outlier_labels = iso_forest.fit_predict(X)
            
            # Initialize all rows as non-outliers
            analysis_df = filtered_df.copy()
            analysis_df["is_outlier"] = False
            
            # Mark only the valid (non-NaN) rows with outlier predictions
            analysis_df.loc[valid_indices, "is_outlier"] = (outlier_labels == -1)
            
            num_outliers = (analysis_df["is_outlier"]).sum()
            total_runs = len(analysis_df)
            
            # Show summary
            st.info(
                f"⚠️ **{num_outliers}** of **{total_runs}** runs flagged as unusual (outliers)."
            )
            
            if num_outliers > 0:
                # Show outlier table in a collapsible expander
                outlier_rows = analysis_df[analysis_df["is_outlier"]][
                    ["algorithm_name", "radix", "blocksize", "num_process", "ppn", "topology", "time"]
                ].copy()
                
                # Format time to milliseconds for display
                outlier_rows["time"] = (outlier_rows["time"] * 1000).round(4)
                
                outlier_rows = outlier_rows.rename(columns={
                    "algorithm_name": "Algorithm",
                    "num_process": "Processes",
                    "ppn": "PPN",
                    "topology": "Topology",
                    "time": "Time (ms)",
                    "radix": "Radix",
                    "blocksize": "Blocksize"
                })
                
                with st.expander("See flagged runs"):
                    st.dataframe(outlier_rows, use_container_width=True)

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

    # Parameter Optimization only works with fixed process count
    if scaling_mode == "Scaling Analysis":

        st.info(
            """
            Parameter Optimization requires a fixed number of processes.

            Please switch the sidebar **Analysis Mode** to
            **Fixed Process Count (Comparison)** to tune parameters.
            """
        )

    else:

        # Options come from filtered_df so the dropdown only shows algorithms
        # that actually have data under the current filters. Falls back to the
        # full dataset if the current filters return nothing. Exclude MPI_Alltoallv
        # since it has no tunable parameters and belongs only in the distribution view.
        all_algorithms = (
            filtered_df["algorithm_name"].dropna().unique()
            if not filtered_df.empty
            else df["algorithm_name"].dropna().unique()
        )
        algorithm_options = sorted([
            alg for alg in all_algorithms if alg != "MPI_Alltoallv"
        ])

        selected_algorithm = st.selectbox(
            "Algorithm",
            algorithm_options,
            key="param_alg"
        )

        # Apply global filters + algorithm selection
        param_df = filtered_df[
            filtered_df["algorithm_name"] == selected_algorithm
        ]

        # =====================================
        # Summary Line (updates with selections)
        # =====================================

        if param_df.empty:

            st.warning(
                "No parameter data is available for the selected filters and algorithm."
            )

        else:

            matching_runs = len(param_df)
            summary_text = (
                f"Showing **{selected_algorithm}** under PPN={selected_ppn}, "
                f"{selected_topology}, {selected_process} processes "
                f"({matching_runs} matching run{'s' if matching_runs != 1 else ''})"
            )
            st.info(summary_text)

            # =====================================
            # Tunable vs Non-Tunable Algorithm Handling
            # =====================================

            if selected_algorithm in ["HieAta", "ParAta"]:

                st.subheader("Radix vs Blocksize Heatmap")

                st.write(
                    """
                    This heatmap shows the average communication time for different
                    radix and blocksize combinations. Lower values indicate better
                    performance. Identify the optimal parameter setting for this
                    configuration.
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

                    # Generate heatmap data
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
                    # Best Parameters Callout
                    # =====================================

                    # Reuse the same best-parameter-finding logic from Tab 2
                    param_means = (
                        param_df
                        .groupby(["radix", "blocksize"])["time"]
                        .mean()
                        .reset_index()
                    )

                    best_idx = param_means["time"].idxmin()
                    best_row = param_means.loc[best_idx]

                    best_time = best_row["time"]
                    best_radix = best_row["radix"]
                    best_blocksize = best_row["blocksize"]

                    st.success(
                        f"🎯 **Best Setting:** radix={best_radix}, "
                        f"blocksize={best_blocksize} "
                        f"({format_time(best_time)} average)"
                    )
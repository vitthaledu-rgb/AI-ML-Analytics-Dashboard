```python
import os
import tempfile
import warnings

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ============================================================
# PAGE / THEME
# ============================================================

TITLE = "🧠 AI & ML Analytics Dashboard"
SUBTITLE = "End-to-End Data Science • Machine Learning • Prediction • Explainability"


# ============================================================
# GLOBAL DATA STATE
# ============================================================

CURRENT_DF = None
ORIGINAL_DF = None


# ============================================================
# DATA LOADING
# ============================================================

def load_dataframe(file_path):
    """
    Load CSV, XLSX or XLS file.
    """

    if file_path is None:
        raise ValueError("Please upload a CSV or Excel file first.")

    if not os.path.exists(file_path):
        raise ValueError("The uploaded file could not be found.")

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".csv":
        df = pd.read_csv(file_path)

    elif extension in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    else:
        raise ValueError(
            "Unsupported file type. Please upload CSV, XLSX or XLS."
        )

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    return df


# ============================================================
# BASIC INFORMATION
# ============================================================

def get_column_info(df):
    info = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data Type": df.dtypes.astype(str).values,
        "Non-Missing": df.notna().sum().values,
        "Missing": df.isna().sum().values,
        "Missing %": (
            df.isna().mean().values * 100
        ).round(2),
        "Unique Values": df.nunique(dropna=True).values
    })

    return info


def get_summary(df):
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        return pd.DataFrame(
            {"Message": ["No numeric columns available for statistics."]}
        )

    summary = numeric_df.describe().T.reset_index()
    summary = summary.rename(columns={"index": "Column"})

    return summary.round(3)


# ============================================================
# OVERVIEW
# ============================================================

def overview_text(df):
    rows = len(df)
    columns = len(df.columns)
    duplicates = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())

    if rows > 0:
        missing_percentage = round(
            missing_cells / (rows * columns) * 100,
            2
        )
    else:
        missing_percentage = 0

    numeric_columns = len(
        df.select_dtypes(include=np.number).columns
    )

    categorical_columns = len(
        df.select_dtypes(exclude=np.number).columns
    )

    return f"""
### Dataset Overview

| Metric | Value |
|---|---:|
| Total Rows | **{rows:,}** |
| Total Columns | **{columns:,}** |
| Duplicate Rows | **{duplicates:,}** |
| Missing Cells | **{missing_cells:,}** |
| Missing Percentage | **{missing_percentage}%** |
| Numeric Columns | **{numeric_columns:,}** |
| Categorical Columns | **{categorical_columns:,}** |

Your dataset has been successfully loaded.
"""


# ============================================================
# LOAD DATASET
# ============================================================

def process_uploaded_file(file_path):
    global CURRENT_DF
    global ORIGINAL_DF

    try:
        df = load_dataframe(file_path)

        ORIGINAL_DF = df.copy()
        CURRENT_DF = df.copy()

        overview = overview_text(df)
        preview = df.head(20)
        column_info = get_column_info(df)
        summary = get_summary(df)

        status = (
            f"✅ Dataset loaded successfully!\n\n"
            f"Rows: {len(df):,}\n"
            f"Columns: {len(df.columns):,}\n"
            f"File type: {os.path.splitext(file_path)[1].upper()}"
        )

        return (
            status,
            overview,
            preview,
            column_info,
            summary
        )

    except Exception as e:
        return (
            f"❌ Error loading dataset:\n{str(e)}",
            "",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

def missing_analysis():
    global CURRENT_DF

    if CURRENT_DF is None:
        return (
            "Please load a dataset first.",
            pd.DataFrame(),
            None
        )

    df = CURRENT_DF

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    result = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing.values,
        "Missing %": missing_pct.values
    })

    result = result[result["Missing Values"] > 0]

    if result.empty:
        message = "✅ No missing values found."
        return message, result, None

    fig, ax = plt.subplots(figsize=(10, 5))

    result_plot = result.sort_values(
        "Missing Values",
        ascending=False
    )

    sns.barplot(
        data=result_plot,
        x="Missing Values",
        y="Column",
        ax=ax
    )

    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Number of Missing Values")
    ax.set_ylabel("Column")
    plt.tight_layout()

    return (
        f"⚠️ {len(result)} column(s) contain missing values.",
        result,
        fig
    )


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def duplicate_analysis():
    global CURRENT_DF

    if CURRENT_DF is None:
        return "Please load a dataset first.", pd.DataFrame()

    df = CURRENT_DF

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        return "✅ No duplicate rows found.", pd.DataFrame()

    duplicates = df[df.duplicated(keep=False)].copy()

    return (
        f"⚠️ {duplicate_count:,} duplicate row(s) found.",
        duplicates.head(100)
    )


# ============================================================
# HISTOGRAM
# ============================================================

def create_histogram(column):
    global CURRENT_DF

    if CURRENT_DF is None:
        return None

    if not column:
        return None

    if column not in CURRENT_DF.columns:
        return None

    if not pd.api.types.is_numeric_dtype(CURRENT_DF[column]):
        return None

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        CURRENT_DF[column].dropna(),
        kde=True,
        ax=ax
    )

    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    return fig


# ============================================================
# BOX PLOT
# ============================================================

def create_boxplot(column):
    global CURRENT_DF

    if CURRENT_DF is None:
        return None

    if not column:
        return None

    if column not in CURRENT_DF.columns:
        return None

    if not pd.api.types.is_numeric_dtype(CURRENT_DF[column]):
        return None

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.boxplot(
        x=CURRENT_DF[column].dropna(),
        ax=ax
    )

    ax.set_title(f"Box Plot of {column}")
    ax.set_xlabel(column)

    plt.tight_layout()

    return fig


# ============================================================
# CORRELATION HEATMAP
# ============================================================

def create_correlation():
    global CURRENT_DF

    if CURRENT_DF is None:
        return None

    numeric_df = CURRENT_DF.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    fig, ax = plt.subplots(
        figsize=(
            max(8, len(corr.columns) * 0.8),
            max(6, len(corr.columns) * 0.6)
        )
    )

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    plt.tight_layout()

    return fig


# ============================================================
# OUTLIER DETECTION
# ============================================================

def detect_outliers():
    global CURRENT_DF

    if CURRENT_DF is None:
        return (
            "Please load a dataset first.",
            pd.DataFrame()
        )

    numeric_columns = CURRENT_DF.select_dtypes(
        include=np.number
    ).columns

    results = []

    for column in numeric_columns:

        series = CURRENT_DF[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = int(
            ((series < lower) | (series > upper)).sum()
        )

        percentage = round(
            outlier_count / len(series) * 100,
            2
        )

        results.append({
            "Column": column,
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "Lower Bound": round(lower, 3),
            "Upper Bound": round(upper, 3),
            "Outliers": outlier_count,
            "Outlier %": percentage
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return (
            "No numeric columns available.",
            result_df
        )

    total_outliers = int(result_df["Outliers"].sum())

    return (
        f"🔎 Total detected outliers: **{total_outliers:,}**",
        result_df
    )


# ============================================================
# IMPUTATION
# ============================================================

def impute_missing(method, constant_value):
    global CURRENT_DF

    if CURRENT_DF is None:
        return (
            "Please load a dataset first.",
            pd.DataFrame()
        )

    df = CURRENT_DF.copy()

    missing_before = int(df.isna().sum().sum())

    if missing_before == 0:
        return (
            "✅ No missing values to impute.",
            get_column_info(df)
        )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns

    try:

        if method == "Mean":
            for column in numeric_columns:
                df[column] = df[column].fillna(
                    df[column].mean()
                )

        elif method == "Median":
            for column in numeric_columns:
                df[column] = df[column].fillna(
                    df[column].median()
                )

        elif method == "Mode":

            for column in df.columns:

                mode = df[column].mode()

                if not mode.empty:
                    df[column] = df[column].fillna(
                        mode.iloc[0]
                    )

        elif method == "Constant":

            for column in numeric_columns:
                df[column] = df[column].fillna(0)

            for column in categorical_columns:
                df[column] = df[column].fillna(
                    constant_value
                )

        elif method == "Forward Fill":
            df = df.ffill()

        elif method == "Backward Fill":
            df = df.bfill()

        missing_after = int(df.isna().sum().sum())

        CURRENT_DF = df

        return (
            f"""
### Imputation Completed

Missing values before: **{missing_before:,}**

Missing values after: **{missing_after:,}**

Method: **{method}**
""",
            get_column_info(df)
        )

    except Exception as e:

        return (
            f"❌ Imputation failed: {str(e)}",
            get_column_info(df)
        )


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates():
    global CURRENT_DF

    if CURRENT_DF is None:
        return (
            "Please load a dataset first.",
            pd.DataFrame()
        )

    before = len(CURRENT_DF)

    CURRENT_DF = CURRENT_DF.drop_duplicates().reset_index(
        drop=True
    )

    after = len(CURRENT_DF)

    removed = before - after

    return (
        f"✅ Removed **{removed:,}** duplicate row(s).\n\n"
        f"Rows remaining: **{after:,}**",
        CURRENT_DF.head(20)
    )


# ============================================================
# REMOVE IQR OUTLIERS
# ============================================================

def remove_outliers():
    global CURRENT_DF

    if CURRENT_DF is None:
        return (
            "Please load a dataset first.",
            pd.DataFrame()
        )

    df = CURRENT_DF.copy()

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    before = len(df)

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df = df[
            (df[column].isna()) |
            (
                (df[column] >= lower) &
                (df[column] <= upper)
            )
        ]

    df = df.reset_index(drop=True)

    CURRENT_DF = df

    after = len(df)

    removed = before - after

    return (
        f"""
### Outlier Removal Completed

Rows before: **{before:,}**

Rows after: **{after:,}**

Rows removed: **{removed:,}**
""",
        df.head(20)
    )


# ============================================================
# RESET DATASET
# ============================================================

def reset_dataset():
    global CURRENT_DF
    global ORIGINAL_DF

    if ORIGINAL_DF is None:
        return (
            "No original dataset available.",
            pd.DataFrame()
        )

    CURRENT_DF = ORIGINAL_DF.copy()

    return (
        "🔄 Dataset restored to the original uploaded version.",
        CURRENT_DF.head(20)
    )


# ============================================================
# DOWNLOAD CLEANED DATA
# ============================================================

def save_cleaned_csv():
    global CURRENT_DF

    if CURRENT_DF is None:
        return None

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv"
    )

    temp_file.close()

    CURRENT_DF.to_csv(
        temp_file.name,
        index=False
    )

    return temp_file.name


def save_cleaned_excel():
    global CURRENT_DF

    if CURRENT_DF is None:
        return None

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".xlsx"
    )

    temp_file.close()

    CURRENT_DF.to_excel(
        temp_file.name,
        index=False
    )

    return temp_file.name


# ============================================================
# CLEANING REPORT
# ============================================================

def cleaning_report():
    global CURRENT_DF
    global ORIGINAL_DF

    if CURRENT_DF is None or ORIGINAL_DF is None:
        return "Please load a dataset first."

    original_rows = len(ORIGINAL_DF)
    current_rows = len(CURRENT_DF)

    original_missing = int(
        ORIGINAL_DF.isna().sum().sum()
    )

    current_missing = int(
        CURRENT_DF.isna().sum().sum()
    )

    original_duplicates = int(
        ORIGINAL_DF.duplicated().sum()
    )

    current_duplicates = int(
        CURRENT_DF.duplicated().sum()
    )

    return f"""
### Cleaning Audit Report

| Metric | Original | Current |
|---|---:|---:|
| Rows | {original_rows:,} | {current_rows:,} |
| Columns | {len(ORIGINAL_DF.columns):,} | {len(CURRENT_DF.columns):,} |
| Missing Cells | {original_missing:,} | {current_missing:,} |
| Duplicate Rows | {original_duplicates:,} | {current_duplicates:,} |

**Rows removed:** {original_rows - current_rows:,}

**Missing cells removed/imputed:** {original_missing - current_missing:,}
"""


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="AI & ML Analytics Dashboard",
    theme=gr.themes.Soft()
) as dashboard:

    gr.Markdown(
        f"""
# {TITLE}

### {SUBTITLE}

Upload a CSV or Excel dataset and explore, clean and analyze it.
"""
    )

    # --------------------------------------------------------
    # UPLOAD SECTION
    # --------------------------------------------------------

    with gr.Row():

        file_input = gr.File(
            label="📁 Upload Dataset",
            file_types=[
                ".csv",
                ".xlsx",
                ".xls"
            ],
            type="filepath"
        )

        load_button = gr.Button(
            "🚀 Load Dataset",
            variant="primary"
        )

    status_output = gr.Markdown()

    # --------------------------------------------------------
    # MAIN TABS
    # --------------------------------------------------------

    with gr.Tabs():

        # ====================================================
        # OVERVIEW
        # ====================================================

        with gr.Tab("📊 Overview"):

            overview_output = gr.Markdown()

            with gr.Row():

                preview_output = gr.Dataframe(
                    label="Data Preview",
                    interactive=False
                )

                column_info_output = gr.Dataframe(
                    label="Column Information",
                    interactive=False
                )

            summary_output = gr.Dataframe(
                label="Statistical Summary",
                interactive=False
            )

        # ====================================================
        # DATA QUALITY
        # ====================================================

        with gr.Tab("🔍 Data Quality"):

            gr.Markdown(
                "## Missing Value Analysis"
            )

            missing_button = gr.Button(
                "Analyze Missing Values"
            )

            missing_status = gr.Markdown()

            missing_table = gr.Dataframe(
                interactive=False
            )

            missing_plot = gr.Plot()

            gr.Markdown(
                "## Duplicate Analysis"
            )

            duplicate_button = gr.Button(
                "Analyze Duplicates"
            )

            duplicate_status = gr.Markdown()

            duplicate_table = gr.Dataframe(
                interactive=False
            )

        # ====================================================
        # VISUALIZATION
        # ====================================================

        with gr.Tab("📈 Visualizations"):

            numeric_column_dropdown = gr.Dropdown(
                label="Select Numeric Column",
                choices=[]
            )

            with gr.Row():

                histogram_button = gr.Button(
                    "📊 Histogram"
                )

                boxplot_button = gr.Button(
                    "📦 Box Plot"
                )

            histogram_plot = gr.Plot()

            boxplot_plot = gr.Plot()

            gr.Markdown(
                "## Correlation Analysis"
            )

            correlation_button = gr.Button(
                "🔥 Generate Correlation Heatmap"
            )

            correlation_plot = gr.Plot()

        # ====================================================
        # OUTLIERS
        # ====================================================

        with gr.Tab("🎯 Outliers"):

            outlier_button = gr.Button(
                "🔎 Detect IQR Outliers"
            )

            outlier_status = gr.Markdown()

            outlier_table = gr.Dataframe(
                interactive=False
            )

        # ====================================================
        # CLEANING
        # ====================================================

        with gr.Tab("🧹 Data Cleaning"):

            gr.Markdown(
                """
## Missing Value Imputation

Choose a method and apply it to the current dataset.
"""
            )

            with gr.Row():

                imputation_method = gr.Dropdown(
                    choices=[
                        "Mean",
                        "Median",
                        "Mode",
                        "Constant",
                        "Forward Fill",
                        "Backward Fill"
                    ],
                    value="Median",
                    label="Imputation Method"
                )

                constant_value = gr.Textbox(
                    value="Unknown",
                    label="Constant Value"
                )

            impute_button = gr.Button(
                "Apply Imputation",
                variant="primary"
            )

            imputation_status = gr.Markdown()

            imputation_info = gr.Dataframe(
                interactive=False
            )

            gr.Markdown(
                """
## Duplicate Removal
"""
            )

            remove_duplicate_button = gr.Button(
                "Remove Duplicate Rows"
            )

            duplicate_clean_status = gr.Markdown()

            duplicate_clean_preview = gr.Dataframe(
                interactive=False
            )

            gr.Markdown(
                """
## IQR Outlier Removal
"""
            )

            remove_outlier_button = gr.Button(
                "Remove IQR Outliers"
            )

            outlier_clean_status = gr.Markdown()

            outlier_clean_preview = gr.Dataframe(
                interactive=False
            )

            gr.Markdown(
                """
## Reset Dataset
"""
            )

            reset_button = gr.Button(
                "🔄 Reset to Original Dataset"
            )

            reset_status = gr.Markdown()

            reset_preview = gr.Dataframe(
                interactive=False
            )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        with gr.Tab("💾 Download"):

            gr.Markdown(
                """
## Cleaning Report
"""
            )

            report_button = gr.Button(
                "Generate Cleaning Report"
            )

            report_output = gr.Markdown()

            gr.Markdown(
                """
## Download Cleaned Dataset
"""
            )

            with gr.Row():

                csv_button = gr.Button(
                    "⬇️ Prepare CSV"
                )

                excel_button = gr.Button(
                    "⬇️ Prepare Excel"
                )

            csv_download = gr.File(
                label="Cleaned CSV"
            )

            excel_download = gr.File(
                label="Cleaned Excel"
            )

    # ========================================================
    # EVENTS
    # ========================================================

    # LOAD DATASET
    load_button.click(
        fn=process_uploaded_file,
        inputs=file_input,
        outputs=[
            status_output,
            overview_output,
            preview_output,
            column_info_output,
            summary_output
        ]
    )

    # MISSING ANALYSIS
    missing_button.click(
        fn=missing_analysis,
        inputs=[],
        outputs=[
            missing_status,
            missing_table,
            missing_plot
        ]
    )

    # DUPLICATES
    duplicate_button.click(
        fn=duplicate_analysis,
        inputs=[],
        outputs=[
            duplicate_status,
            duplicate_table
        ]
    )

    # HISTOGRAM
    histogram_button.click(
        fn=create_histogram,
        inputs=numeric_column_dropdown,
        outputs=histogram_plot
    )

    # BOX PLOT
    boxplot_button.click(
        fn=create_boxplot,
        inputs=numeric_column_dropdown,
        outputs=boxplot_plot
    )

    # CORRELATION
    correlation_button.click(
        fn=create_correlation,
        inputs=[],
        outputs=correlation_plot
    )

    # OUTLIERS
    outlier_button.click(
        fn=detect_outliers,
        inputs=[],
        outputs=[
            outlier_status,
            outlier_table
        ]
    )

    # IMPUTATION
    impute_button.click(
        fn=impute_missing,
        inputs=[
            imputation_method,
            constant_value
        ],
        outputs=[
            imputation_status,
            imputation_info
        ]
    )

    # REMOVE DUPLICATES
    remove_duplicate_button.click(
        fn=remove_duplicates,
        inputs=[],
        outputs=[
            duplicate_clean_status,
            duplicate_clean_preview
        ]
    )

    # REMOVE OUTLIERS
    remove_outlier_button.click(
        fn=remove_outliers,
        inputs=[],
        outputs=[
            outlier_clean_status,
            outlier_clean_preview
        ]
    )

    # RESET
    reset_button.click(
        fn=reset_dataset,
        inputs=[],
        outputs=[
            reset_status,
            reset_preview
        ]
    )

    # REPORT
    report_button.click(
        fn=cleaning_report,
        inputs=[],
        outputs=report_output
    )

    # DOWNLOAD CSV
    csv_button.click(
        fn=save_cleaned_csv,
        inputs=[],
        outputs=csv_download
    )

    # DOWNLOAD EXCEL
    excel_button.click(
        fn=save_cleaned_excel,
        inputs=[],
        outputs=excel_download
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    dashboard.launch()
```
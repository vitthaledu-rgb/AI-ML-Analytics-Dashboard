gr.Markdown(
    """
    # 🧠 AI & ML Analytics Dashboard

    ### Automated EDA • Data Quality • Data Cleaning •
    Outlier Detection • Machine Learning
    """
)

gr.Markdown(
    """
    Upload a CSV or Excel dataset and automatically explore,
    clean, visualize and analyze your data.
    """
)

# ========================================================
# UPLOAD SECTION
# ========================================================

with gr.Tab("📂 Upload Dataset"):

    file_input = gr.File(
        label="Upload CSV / XLSX / XLS",
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

    load_status = gr.Markdown()

    dataset_summary = gr.Markdown()

    preview_output = gr.HTML(
        label="Data Preview"
    )

    dataset_state = gr.Dataframe(
        visible=False
    )

    load_button.click(
        fn=load_file,
        inputs=file_input,
        outputs=[
            load_status,
            dataset_summary,
            preview_output,
            dataset_state
        ]
    )

# ========================================================
# OVERVIEW
# ========================================================

with gr.Tab("🏠 Overview"):

    overview_button = gr.Button(
        "🔄 Refresh Overview"
    )

    overview_output = gr.Markdown()

    overview_button.click(
        fn=lambda: (
            create_summary(CURRENT_DF)
            if CURRENT_DF is not None
            else "❌ Please upload a dataset first."
        ),
        outputs=overview_output
    )

# ========================================================
# DATA INFORMATION
# ========================================================

with gr.Tab("🔍 Data Quality"):

    with gr.Row():

        column_info_button = gr.Button(
            "📋 Column Information"
        )

        missing_button = gr.Button(
            "❓ Missing Values"
        )

        duplicate_button = gr.Button(
            "🔁 Duplicate Rows"
        )

    column_info_output = gr.HTML()

    missing_output = gr.HTML()

    duplicate_output = gr.HTML()

    column_info_button.click(
        fn=column_information,
        outputs=column_info_output
    )

    missing_button.click(
        fn=missing_report,
        outputs=missing_output
    )

    duplicate_button.click(
        fn=duplicate_report,
        outputs=duplicate_output
    )

    gr.Markdown("### 📈 Statistical Summary")

    stats_button = gr.Button(
        "Calculate Statistics"
    )

    stats_output = gr.HTML()

    stats_button.click(
        fn=statistical_summary,
        outputs=stats_output
    )

# ========================================================
# EDA
# ========================================================

with gr.Tab("📊 EDA & Visualization"):

    chart_column = gr.Dropdown(
        label="Select Column",
        choices=[],
        allow_custom_value=True
    )

    refresh_columns_button = gr.Button(
        "🔄 Refresh Columns"
    )

    histogram_button = gr.Button(
        "📊 Histogram"
    )

    box_button = gr.Button(
        "📦 Box Plot"
    )

    bar_button = gr.Button(
        "📊 Bar Chart"
    )

    heatmap_button = gr.Button(
        "🔥 Correlation Heatmap"
    )

    chart_output = gr.Plot()

    refresh_columns_button.click(
        fn=lambda: (
            list(CURRENT_DF.columns)
            if CURRENT_DF is not None
            else []
        ),
        outputs=chart_column
    )

    histogram_button.click(
        fn=create_histogram,
        inputs=chart_column,
        outputs=chart_output
    )

    box_button.click(
        fn=create_boxplot,
        inputs=chart_column,
        outputs=chart_output
    )

    bar_button.click(
        fn=create_bar_chart,
        inputs=chart_column,
        outputs=chart_output
    )

    heatmap_button.click(
        fn=create_heatmap,
        outputs=chart_output
    )

    gr.Markdown("### 🔵 Scatter Plot")

    with gr.Row():

        scatter_x = gr.Dropdown(
            label="X Column",
            choices=[]
        )

        scatter_y = gr.Dropdown(
            label="Y Column",
            choices=[]
        )

    refresh_scatter_button = gr.Button(
        "🔄 Refresh Numeric Columns"
    )

    scatter_button = gr.Button(
        "🔵 Create Scatter Plot"
    )

    scatter_output = gr.Plot()

    refresh_scatter_button.click(
        fn=lambda: (
            list(
                CURRENT_DF.select_dtypes(
                    include=np.number
                ).columns
            )
            if CURRENT_DF is not None
            else []
        ),
        outputs=[
            scatter_x,
            scatter_y
        ]
    )

    scatter_button.click(
        fn=create_scatter,
        inputs=[
            scatter_x,
            scatter_y
        ],
        outputs=scatter_output
    )

# ========================================================
# OUTLIERS
# ========================================================

with gr.Tab("🚨 Outliers"):

    outlier_button = gr.Button(
        "🔎 Detect IQR Outliers"
    )

    outlier_output = gr.HTML()

    outlier_button.click(
        fn=outlier_report,
        outputs=outlier_output
    )

    gr.Markdown("### 📦 Visual Outlier Detection")

    outlier_column = gr.Dropdown(
        label="Select Numeric Column",
        choices=[]
    )

    refresh_outlier_columns = gr.Button(
        "🔄 Refresh Numeric Columns"
    )

    outlier_plot = gr.Plot()

    refresh_outlier_columns.click(
        fn=lambda: (
            list(
                CURRENT_DF.select_dtypes(
                    include=np.number
                ).columns
            )
            if CURRENT_DF is not None
            else []
        ),
        outputs=outlier_column
    )

    outlier_column.change(
        fn=create_boxplot,
        inputs=outlier_column,
        outputs=outlier_plot
    )

# ========================================================
# CLEANING
# ========================================================

with gr.Tab("🧹 Data Cleaning"):

    gr.Markdown(
        """
        ## Duplicate Cleaning
        """
    )

    duplicate_clean_button = gr.Button(
        "🗑️ Remove Duplicate Rows",
        variant="primary"
    )

    duplicate_clean_output = gr.Markdown()

    duplicate_clean_button.click(
        fn=remove_duplicates,
        outputs=duplicate_clean_output
    )

    gr.Markdown(
        """
        ## Missing Value Imputation
        """
    )

    imputation_column = gr.Dropdown(
        label="Select Column",
        choices=[]
    )

    refresh_imputation_columns = gr.Button(
        "🔄 Refresh Columns"
    )

    imputation_method = gr.Dropdown(
        label="Imputation Method",
        choices=[
            "Mean",
            "Median",
            "Mode",
            "Constant",
            "Forward Fill",
            "Backward Fill"
        ],
        value="Median"
    )

    constant_value = gr.Textbox(
        label="Constant Value",
        value="Unknown"
    )

    imputation_button = gr.Button(
        "🧹 Apply Imputation",
        variant="primary"
    )

    imputation_output = gr.Markdown()

    refresh_imputation_columns.click(
        fn=lambda: (
            list(CURRENT_DF.columns)
            if CURRENT_DF is not None
            else []
        ),
        outputs=imputation_column
    )

    imputation_button.click(
        fn=impute_missing,
        inputs=[
            imputation_column,
            imputation_method,
            constant_value
        ],
        outputs=imputation_output
    )

    gr.Markdown(
        """
        ## IQR Outlier Removal
        """
    )

    cleaning_outlier_column = gr.Dropdown(
        label="Select Numeric Column",
        choices=[]
    )

    refresh_cleaning_outlier = gr.Button(
        "🔄 Refresh Numeric Columns"
    )

    remove_outlier_button = gr.Button(
        "🚨 Remove Outliers"
    )

    remove_outlier_output = gr.Markdown()

    refresh_cleaning_outlier.click(
        fn=lambda: (
            list(
                CURRENT_DF.select_dtypes(
                    include=np.number
                ).columns
            )
            if CURRENT_DF is not None
            else []
        ),
        outputs=cleaning_outlier_column
    )

    remove_outlier_button.click(
        fn=remove_outliers,
        inputs=cleaning_outlier_column,
        outputs=remove_outlier_output
    )

    gr.Markdown(
        """
        ## Reset
        """
    )

    reset_button = gr.Button(
        "🔄 Reset to Original Dataset"
    )

    reset_output = gr.Markdown()

    reset_button.click(
        fn=reset_dataset,
        outputs=reset_output
    )

    gr.Markdown(
        """
        ## Current Data Preview
        """
    )

    preview_clean_button = gr.Button(
        "👀 Show Current Dataset"
    )

    cleaned_preview_output = gr.HTML()

    preview_clean_button.click(
        fn=cleaned_preview,
        outputs=cleaned_preview_output
    )

# ========================================================
# MACHINE LEARNING
# ========================================================

with gr.Tab("🤖 Machine Learning"):

    gr.Markdown(
        """
        ## Machine Learning

        Select a target column and one or more feature
        columns to train a basic machine-learning model.
        """
    )

    ml_target = gr.Dropdown(
        label="Target Column",
        choices=[]
    )

    ml_features = gr.Dropdown(
        label="Feature Columns",
        choices=[],
        multiselect=True
    )

    refresh_ml_columns = gr.Button(
        "🔄 Refresh ML Columns"
    )

    problem_type = gr.Dropdown(
        label="Problem Type",
        choices=[
            "Classification",
            "Regression"
        ],
        value="Classification"
    )

    model_type = gr.Dropdown(
        label="Model",
        choices=[
            "Random Forest",
            "Logistic Regression",
            "Linear Regression"
        ],
        value="Random Forest"
    )

    train_button = gr.Button(
        "🚀 Train Machine Learning Model",
        variant="primary"
    )

    ml_result = gr.Markdown()

    ml_predictions = gr.HTML()

    refresh_ml_columns.click(
        fn=lambda: (
            list(CURRENT_DF.columns)
            if CURRENT_DF is not None
            else []
        ),
        outputs=[
            ml_target,
            ml_features
        ]
    )

    train_button.click(
        fn=train_machine_learning,
        inputs=[
            ml_target,
            ml_features,
            problem_type,
            model_type
        ],
        outputs=[
            ml_result,
            ml_predictions
        ]
    )

# ========================================================
# DOWNLOAD
# ========================================================

with gr.Tab("⬇️ Download"):

    gr.Markdown(
        """
        ## Download Your Cleaned Dataset
        """
    )

    download_csv_button = gr.Button(
        "⬇️ Download CSV"
    )

    download_excel_button = gr.Button(
        "⬇️ Download Excel"
    )

    csv_file = gr.File(
        label="CSV File"
    )

    excel_file = gr.File(
        label="Excel File"
    )

    download_csv_button.click(
        fn=download_csv,
        outputs=csv_file
    )

    download_excel_button.click(
        fn=download_excel,
        outputs=excel_file
    )
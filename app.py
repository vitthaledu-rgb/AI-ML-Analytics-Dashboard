# ============================================================
# 🧠 AI & ML END-TO-END ANALYTICS DASHBOARD
# Google Colab - COMPLETE SINGLE-CELL VERSION
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# Gradio compatibility shim for legacy helper functions.
class _GrCompat:
    @staticmethod
    def update(**kwargs):
        return kwargs

gr = _GrCompat()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve
)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

df_original = None
df_working = None

model_global = None
pipeline_global = None

target_global = None
task_global = None

X_test_global = None
y_test_global = None
y_pred_global = None

feature_columns_global = []

comparison_results = []


# ============================================================
# CSS
# ============================================================

css = """

.gradio-container {
    max-width: 1550px !important;
    margin: auto !important;
    padding: 10px 22px !important;
}

#main_header {
    text-align: center;
    padding: 8px 0 12px 0;
}

#main_header h1 {
    font-size: 30px;
    margin: 0;
    font-weight: 800;
}

#main_header p {
    margin: 4px 0;
    font-size: 14px;
    color: #667085;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
    margin: 8px 0;
}

.kpi-card {
    background: white;
    border: 1px solid #e4e7ec;
    border-radius: 12px;
    padding: 9px 5px;
    text-align: center;
    min-height: 68px;
}

.kpi-title {
    color: #667085;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .5px;
}

.kpi-value {
    font-size: 23px;
    font-weight: 800;
    margin-top: 4px;
}

.compact-box {
    border-radius: 12px;
}

.result-card {
    padding: 12px;
    border-radius: 12px;
    background: #f7f8ff;
    border: 1px solid #dfe3ff;
}

button {
    border-radius: 9px !important;
}

"""

# ============================================================
# HELPER
# ============================================================

def kpi_card(title, value):
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """


def empty_fig(message="No result available"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=14
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


# ============================================================
# DATASET LOAD
# ============================================================

def load_dataset(file):

    global df_original
    global df_working

    if file is None:

        return (
            "❌ Please upload a CSV or Excel file.",
            "",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(choices=["None"], value="None")
        )

    try:

        path = file

        if isinstance(file, dict):
            path = file.get("path")

        extension = os.path.splitext(path)[1].lower()

        if extension == ".csv":

            df = pd.read_csv(path)

        elif extension in [".xlsx", ".xls"]:

            df = pd.read_excel(path)

        else:

            return (
                "❌ Unsupported file format.",
                "",
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                gr.update(choices=[]),
                gr.update(choices=[]),
                gr.update(choices=[]),
                gr.update(choices=["None"], value="None")
            )

        df_original = df.copy()
        df_working = df.copy()

        numeric_count = len(
            df.select_dtypes(
                include=np.number
            ).columns
        )

        categorical_count = len(
            df.select_dtypes(
                exclude=np.number
            ).columns
        )

        missing_count = int(
            df.isnull().sum().sum()
        )

        duplicate_count = int(
            df.duplicated().sum()
        )

        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        kpis = f"""
        <div class="kpi-grid">

            {kpi_card("Rows", f"{len(df):,}")}

            {kpi_card("Columns", f"{len(df.columns):,}")}

            {kpi_card("Numeric", f"{numeric_count:,}")}

            {kpi_card("Categorical", f"{categorical_count:,}")}

            {kpi_card("Missing", f"{missing_count:,}")}

            {kpi_card("Duplicates", f"{duplicate_count:,}")}

        </div>
        """

        # ----------------------------------------------------
        # DATA TYPE TABLE
        # ----------------------------------------------------

        dtype_table = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing": df.isnull().sum().values,
            "Missing %": (
                df.isnull().mean() * 100
            ).round(2).values,
            "Unique": df.nunique().values
        })

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        statistics = df.describe(
            include="all"
        ).T.reset_index()

        statistics.rename(
            columns={"index": "Column"},
            inplace=True
        )

        # ----------------------------------------------------
        # MISSING
        # ----------------------------------------------------

        missing_table = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": df.isnull().sum().values,
            "Missing %": (
                df.isnull().mean() * 100
            ).round(2).values
        })

        # ----------------------------------------------------
        # DUPLICATES
        # ----------------------------------------------------

        duplicate_table = pd.DataFrame({
            "Duplicate Rows": [duplicate_count],
            "Total Rows": [len(df)],
            "Duplicate %": [
                round(
                    duplicate_count /
                    max(len(df), 1) * 100,
                    2
                )
            ]
        })

        status = f"""
## ✅ Dataset Loaded

**File:** `{os.path.basename(path)}`

**Rows:** {len(df):,} &nbsp; | &nbsp;
**Columns:** {len(df.columns)} &nbsp; | &nbsp;
**Numeric:** {numeric_count} &nbsp; | &nbsp;
**Categorical:** {categorical_count}
"""

        columns = list(df.columns)

        return (
            status,
            kpis,
            df.head(10),
            dtype_table,
            statistics,
            missing_table,
            gr.update(
                choices=columns,
                value=columns[0] if columns else None
            ),
            gr.update(
                choices=["None"] + columns,
                value="None"
            ),
            gr.update(
                choices=columns,
                value=columns[0] if columns else None
            ),
            gr.update(
                choices=["None"] + columns,
                value="None"
            )
        )

    except Exception as e:

        return (
            f"❌ Error: `{str(e)}`",
            "",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(choices=["None"])
        )


# ============================================================
# CLEANING
# ============================================================

def clean_dataset(
    missing_method,
    duplicate_method,
    outlier_method,
    numeric_method,
    categorical_method,
    invalid_method
):

    global df_working

    if df_working is None:

        return (
            "❌ Load dataset first.",
            pd.DataFrame(),
            pd.DataFrame()
        )

    df = df_working.copy()

    original_rows = len(df)
    original_missing = int(
        df.isnull().sum().sum()
    )
    original_duplicates = int(
        df.duplicated().sum()
    )

    # ========================================================
    # 1. DUPLICATES
    # ========================================================

    if duplicate_method == "Remove duplicates":

        df = df.drop_duplicates()

    # ========================================================
    # 2. MISSING VALUES
    # ========================================================

    if missing_method == "Drop rows":

        df = df.dropna()

    elif missing_method == "Drop columns >50% missing":

        threshold = len(df) * 0.50

        df = df.dropna(
            axis=1,
            thresh=int(threshold)
        )

    elif missing_method == "Mean imputation":

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        for col in numeric_cols:

            df[col] = df[col].fillna(
                df[col].mean()
            )

    elif missing_method == "Median imputation":

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        for col in numeric_cols:

            df[col] = df[col].fillna(
                df[col].median()
            )

    elif missing_method == "Mode imputation":

        for col in df.columns:

            mode = df[col].mode()

            if len(mode) > 0:

                df[col] = df[col].fillna(
                    mode.iloc[0]
                )

    elif missing_method == "Forward fill":

        df = df.ffill()

    elif missing_method == "Backward fill":

        df = df.bfill()

    elif missing_method == "Fill numeric 0":

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        df[numeric_cols] = df[
            numeric_cols
        ].fillna(0)

    # ========================================================
    # 3. NUMERIC CLEANING
    # ========================================================

    if numeric_method == "Convert numeric-looking text":

        for col in df.columns:

            if df[col].dtype == "object":

                converted = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

                valid_ratio = (
                    converted.notna().mean()
                )

                if valid_ratio >= 0.80:

                    df[col] = converted

    elif numeric_method == "Round numeric values":

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        df[numeric_cols] = df[
            numeric_cols
        ].round(2)

    # ========================================================
    # 4. CATEGORICAL CLEANING
    # ========================================================

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    if categorical_method == "Remove extra spaces":

        for col in categorical_cols:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
            )

    elif categorical_method == "Lowercase":

        for col in categorical_cols:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    elif categorical_method == "Uppercase":

        for col in categorical_cols:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
            )

    # ========================================================
    # 5. INVALID VALUES
    # ========================================================

    if invalid_method == "Replace negative numeric values":

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        for col in numeric_cols:

            df.loc[
                df[col] < 0,
                col
            ] = np.nan

    # ========================================================
    # 6. OUTLIERS
    # ========================================================

    if outlier_method != "Do Nothing":

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        for col in numeric_cols:

            if df[col].dropna().empty:
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            if outlier_method == "Remove IQR outliers":

                df = df[
                    (df[col] >= lower) &
                    (df[col] <= upper)
                ]

            elif outlier_method == "Cap IQR outliers":

                df[col] = df[col].clip(
                    lower,
                    upper
                )

            elif outlier_method == "Remove Z-score >3":

                z = np.abs(
                    stats.zscore(
                        df[col].fillna(
                            df[col].median()
                        )
                    )
                )

                df = df[
                    z < 3
                ]

    df_working = df.copy()

    final_missing = int(
        df.isnull().sum().sum()
    )

    final_duplicates = int(
        df.duplicated().sum()
    )

    result = f"""
## 🧹 Cleaning Completed

| Quality Measure | Before | After |
|---|---:|---:|
| Rows | {original_rows:,} | {len(df):,} |
| Missing Values | {original_missing:,} | {final_missing:,} |
| Duplicate Rows | {original_duplicates:,} | {final_duplicates:,} |
| Columns | {len(df_working.columns)} | {len(df.columns)} |

### Changes

Rows removed: **{original_rows - len(df):,}**

Missing values remaining: **{final_missing:,}**

Duplicate rows remaining: **{final_duplicates:,}**
"""

    return (
        result,
        df.head(15),
        df
    )


# ============================================================
# RESET CLEANING
# ============================================================

def reset_cleaning():

    global df_working

    if df_original is None:

        return (
            "❌ No dataset loaded.",
            pd.DataFrame(),
            pd.DataFrame()
        )

    df_working = df_original.copy()

    return (
        "🔄 Dataset restored to original uploaded version.",
        df_working.head(15),
        df_working
    )


# ============================================================
# GRAPH CREATOR
# ============================================================

def create_graph(
    graph_type,
    x_column,
    y_column,
    hue_column
):

    if df_working is None:

        return empty_fig(
            "Load a dataset first"
        )

    df = df_working.copy()

    try:

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        hue = (
            None
            if hue_column in [None, "None"]
            else hue_column
        )

        # ====================================================
        # HISTOGRAM
        # ====================================================

        if graph_type == "Histogram":

            sns.histplot(
                data=df,
                x=x_column,
                hue=hue,
                kde=True,
                ax=ax
            )

        # ====================================================
        # BOX
        # ====================================================

        elif graph_type == "Box Plot":

            if y_column in [None, "None"]:

                sns.boxplot(
                    data=df,
                    y=x_column,
                    ax=ax
                )

            else:

                sns.boxplot(
                    data=df,
                    x=x_column,
                    y=y_column,
                    hue=hue,
                    ax=ax
                )

        # ====================================================
        # VIOLIN
        # ====================================================

        elif graph_type == "Violin Plot":

            if y_column in [None, "None"]:

                sns.violinplot(
                    data=df,
                    y=x_column,
                    ax=ax
                )

            else:

                sns.violinplot(
                    data=df,
                    x=x_column,
                    y=y_column,
                    hue=hue,
                    ax=ax
                )

        # ====================================================
        # COUNT
        # ====================================================

        elif graph_type == "Count Plot":

            sns.countplot(
                data=df,
                x=x_column,
                hue=hue,
                ax=ax
            )

            ax.tick_params(
                axis="x",
                rotation=45
            )

        # ====================================================
        # BAR
        # ====================================================

        elif graph_type == "Bar Chart":

            if y_column in [None, "None"]:

                counts = (
                    df[x_column]
                    .value_counts()
                    .head(15)
                )

                counts.plot(
                    kind="bar",
                    ax=ax
                )

            else:

                temp = (
                    df.groupby(x_column)[y_column]
                    .mean()
                    .sort_values(
                        ascending=False
                    )
                    .head(20)
                )

                temp.plot(
                    kind="bar",
                    ax=ax
                )

            ax.tick_params(
                axis="x",
                rotation=45
            )

        # ====================================================
        # SCATTER
        # ====================================================

        elif graph_type == "Scatter Plot":

            sns.scatterplot(
                data=df,
                x=x_column,
                y=y_column,
                hue=hue,
                ax=ax
            )

        # ====================================================
        # LINE
        # ====================================================

        elif graph_type == "Line Chart":

            sns.lineplot(
                data=df,
                x=x_column,
                y=y_column,
                hue=hue,
                ax=ax
            )

        # ====================================================
        # AREA
        # ====================================================

        elif graph_type == "Area Chart":

            temp = df[
                [x_column, y_column]
            ].dropna()

            temp = temp.head(100)

            if pd.api.types.is_numeric_dtype(
                temp[x_column]
            ):

                temp = temp.sort_values(
                    x_column
                )

                ax.fill_between(
                    temp[x_column],
                    temp[y_column],
                    alpha=0.4
                )

                ax.plot(
                    temp[x_column],
                    temp[y_column]
                )

            else:

                ax.fill_between(
                    range(len(temp)),
                    temp[y_column],
                    alpha=0.4
                )

                ax.plot(
                    range(len(temp)),
                    temp[y_column]
                )

        # ====================================================
        # KDE
        # ====================================================

        elif graph_type == "KDE":

            sns.kdeplot(
                data=df,
                x=x_column,
                hue=hue,
                fill=True,
                ax=ax
            )

        # ====================================================
        # PIE
        # ====================================================

        elif graph_type == "Pie Chart":

            values = (
                df[x_column]
                .value_counts()
                .head(10)
            )

            ax.pie(
                values.values,
                labels=values.index.astype(str),
                autopct="%1.1f%%",
                startangle=90
            )

        # ====================================================
        # CORRELATION
        # ====================================================

        elif graph_type == "Correlation Heatmap":

            numeric_df = df.select_dtypes(
                include=np.number
            )

            if numeric_df.shape[1] < 2:

                return empty_fig(
                    "At least 2 numeric columns required"
                )

            corr = numeric_df.corr()

            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                ax=ax
            )

        # ====================================================
        # MISSING VALUES
        # ====================================================

        elif graph_type == "Missing Values":

            missing = (
                df.isnull()
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            missing = missing[
                missing > 0
            ]

            if len(missing) == 0:

                return empty_fig(
                    "✅ No missing values"
                )

            missing.plot(
                kind="bar",
                ax=ax
            )

            ax.tick_params(
                axis="x",
                rotation=45
            )

        # ====================================================
        # QQ PLOT
        # ====================================================

        elif graph_type == "Q-Q Plot":

            values = pd.to_numeric(
                df[x_column],
                errors="coerce"
            ).dropna()

            stats.probplot(
                values,
                dist="norm",
                plot=ax
            )

        # ====================================================
        # COVARIANCE
        # ====================================================

        elif graph_type == "Covariance Heatmap":

            numeric_df = df.select_dtypes(
                include=np.number
            )

            sns.heatmap(
                numeric_df.cov(),
                annot=True,
                fmt=".2f",
                cmap="viridis",
                ax=ax
            )

        ax.set_title(
            graph_type,
            fontsize=16,
            fontweight="bold"
        )

        plt.tight_layout()

        return fig

    except Exception as e:

        return empty_fig(
            f"Graph Error\n\n{str(e)}"
        )


# ============================================================
# TARGET ANALYSIS
# ============================================================

def target_analysis(target):

    if df_working is None:

        return (
            pd.DataFrame(),
            empty_fig("Load dataset first")
        )

    if target not in df_working.columns:

        return (
            pd.DataFrame(),
            empty_fig("Select target")
        )

    df = df_working.copy()

    values = df[target].value_counts(
        dropna=False
    )

    table = pd.DataFrame({
        "Category": values.index.astype(str),
        "Count": values.values,
        "Percentage": (
            values.values /
            len(df) *
            100
        ).round(2)
    })

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    if pd.api.types.is_numeric_dtype(
        df[target]
    ):

        sns.histplot(
            df[target].dropna(),
            kde=True,
            ax=ax
        )

    else:

        values.head(15).plot(
            kind="bar",
            ax=ax
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

    ax.set_title(
        f"Target Analysis: {target}",
        fontweight="bold"
    )

    plt.tight_layout()

    return table, fig


# ============================================================
# MODEL DICTIONARY
# ============================================================

def get_models(task):

    if task == "Classification":

        return {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=2000
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                ),

            "KNN":
                KNeighborsClassifier(),

            "SVM":
                SVC(
                    probability=True
                ),

            "Gradient Boosting":
                GradientBoostingClassifier(
                    random_state=42
                )
        }

    else:

        return {

            "Linear Regression":
                LinearRegression(),

            "Ridge":
                Ridge(),

            "Lasso":
                Lasso(),

            "Decision Tree":
                DecisionTreeRegressor(
                    random_state=42
                ),

            "Random Forest":
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                ),

            "SVR":
                SVR(),

            "Gradient Boosting":
                GradientBoostingRegressor(
                    random_state=42
                )
        }


# ============================================================
# TASK CHANGE
# ============================================================

def change_task(task):

    models = list(
        get_models(task).keys()
    )

    default = models[0]

    return gr.update(
        choices=models,
        value=default
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    target,
    task,
    model_name,
    test_size,
    scaling_method
):

    global model_global
    global pipeline_global
    global target_global
    global task_global
    global X_test_global
    global y_test_global
    global y_pred_global
    global feature_columns_global

    if df_working is None:

        return (
            "❌ Load dataset first.",
            pd.DataFrame(),
            empty_fig(),
            empty_fig(),
            pd.DataFrame(
                columns=["Feature", "Value"]
            ),
            pd.DataFrame()
        )

    if target is None or target == "":

        return (
            "❌ Select target column.",
            pd.DataFrame(),
            empty_fig(),
            empty_fig(),
            pd.DataFrame(
                columns=["Feature", "Value"]
            ),
            pd.DataFrame()
        )

    if target not in df_working.columns:

        return (
            "❌ Invalid target.",
            pd.DataFrame(),
            empty_fig(),
            empty_fig(),
            pd.DataFrame(
                columns=["Feature", "Value"]
            ),
            pd.DataFrame()
        )

    df = df_working.copy()

    # --------------------------------------------------------
    # X / Y
    # --------------------------------------------------------

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # Remove completely empty columns
    X = X.dropna(
        axis=1,
        how="all"
    )

    feature_columns_global = list(
        X.columns
    )

    target_global = target
    task_global = task

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    if task == "Classification":

        y = y.astype(str)

    else:

        y = pd.to_numeric(
            y,
            errors="coerce"
        )

        valid = y.notna()

        X = X.loc[valid]

        y = y.loc[valid]

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    # --------------------------------------------------------
    # SCALER
    # --------------------------------------------------------

    if scaling_method == "StandardScaler":

        scaler = StandardScaler()

    elif scaling_method == "MinMaxScaler":

        scaler = MinMaxScaler()

    else:

        scaler = "passthrough"

    # --------------------------------------------------------
    # NUMERIC PIPE
    # --------------------------------------------------------

    if scaler == "passthrough":

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ])

    else:

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                scaler
            )
        ])

    # --------------------------------------------------------
    # CATEGORICAL PIPE
    # --------------------------------------------------------

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    if task == "Classification":

        try:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )

        except:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42
            )

    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    models = get_models(task)

    if model_name not in models:

        model_name = list(
            models.keys()
        )[0]

    model = models[
        model_name
    ]

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    model_global = model
    pipeline_global = pipeline

    X_test_global = X_test
    y_test_global = y_test
    y_pred_global = y_pred

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if task == "Classification":

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        metrics = pd.DataFrame({

            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],

            "Score": [
                round(accuracy, 4),
                round(precision, 4),
                round(recall, 4),
                round(f1, 4)
            ]
        })

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        fig_cm, ax_cm = plt.subplots(
            figsize=(7, 5)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax_cm
        )

        ax_cm.set_title(
            "Confusion Matrix",
            fontweight="bold"
        )

        ax_cm.set_xlabel(
            "Predicted"
        )

        ax_cm.set_ylabel(
            "Actual"
        )

        plt.tight_layout()

        # ----------------------------------------------------
        # ROC
        # ----------------------------------------------------

        fig_roc, ax_roc = plt.subplots(
            figsize=(7, 5)
        )

        try:

            probabilities = (
                pipeline.predict_proba(
                    X_test
                )
            )

            if probabilities.shape[1] == 2:

                positive_probability = (
                    probabilities[:, 1]
                )

                y_binary = (
                    pd.Series(y_test)
                    .astype("category")
                    .cat.codes
                )

                fpr, tpr, _ = roc_curve(
                    y_binary,
                    positive_probability
                )

                auc_score = roc_auc_score(
                    y_binary,
                    positive_probability
                )

                ax_roc.plot(
                    fpr,
                    tpr,
                    label=f"AUC = {auc_score:.3f}"
                )

                ax_roc.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--"
                )

                ax_roc.legend()

                ax_roc.set_xlabel(
                    "False Positive Rate"
                )

                ax_roc.set_ylabel(
                    "True Positive Rate"
                )

                ax_roc.set_title(
                    "ROC Curve",
                    fontweight="bold"
                )

            else:

                ax_roc.text(
                    0.5,
                    0.5,
                    "Multiclass ROC available\nwith One-vs-Rest extension",
                    ha="center",
                    va="center"
                )

                ax_roc.axis("off")

        except:

            ax_roc.text(
                0.5,
                0.5,
                "ROC curve unavailable",
                ha="center",
                va="center"
            )

            ax_roc.axis("off")

        plt.tight_layout()

    # ========================================================
    # REGRESSION
    # ========================================================

    else:

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        metrics = pd.DataFrame({

            "Metric": [
                "MAE",
                "RMSE",
                "R²"
            ],

            "Score": [
                round(mae, 4),
                round(rmse, 4),
                round(r2, 4)
            ]
        })

        # ----------------------------------------------------
        # ACTUAL VS PREDICTED
        # ----------------------------------------------------

        fig_cm, ax_cm = plt.subplots(
            figsize=(7, 5)
        )

        ax_cm.scatter(
            y_test,
            y_pred,
            alpha=0.65
        )

        minimum = min(
            y_test.min(),
            y_pred.min()
        )

        maximum = max(
            y_test.max(),
            y_pred.max()
        )

        ax_cm.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--"
        )

        ax_cm.set_xlabel(
            "Actual"
        )

        ax_cm.set_ylabel(
            "Predicted"
        )

        ax_cm.set_title(
            "Actual vs Predicted",
            fontweight="bold"
        )

        plt.tight_layout()

        # ----------------------------------------------------
        # RESIDUAL
        # ----------------------------------------------------

        residuals = (
            y_test - y_pred
        )

        fig_roc, ax_roc = plt.subplots(
            figsize=(7, 5)
        )

        ax_roc.scatter(
            y_pred,
            residuals,
            alpha=0.65
        )

        ax_roc.axhline(
            0,
            linestyle="--"
        )

        ax_roc.set_xlabel(
            "Predicted"
        )

        ax_roc.set_ylabel(
            "Residual"
        )

        ax_roc.set_title(
            "Residual Plot",
            fontweight="bold"
        )

        plt.tight_layout()

    # ========================================================
    # PREDICTION INPUT
    # ========================================================

    prediction_template = pd.DataFrame({

        "Feature":
            feature_columns_global,

        "Value":
            [
                ""
                for _ in feature_columns_global
            ]
    })

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance_df = get_feature_importance()

    if importance_df is None:

        importance_df = pd.DataFrame(
            columns=[
                "Feature",
                "Importance"
            ]
        )

    result = f"""
## 🤖 Model Training Completed

**Model:** `{model_name}`

**Task:** `{task}`

**Target:** `{target}`

**Training Rows:** `{len(X_train):,}`

**Testing Rows:** `{len(X_test):,}`

**Features:** `{len(feature_columns_global)}`

### Performance

{metrics.to_markdown(index=False)}
"""

    return (
        result,
        metrics,
        fig_cm,
        fig_roc,
        prediction_template,
        importance_df
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance():

    if pipeline_global is None:

        return None

    try:

        preprocessor = (
            pipeline_global
            .named_steps[
                "preprocessor"
            ]
        )

        model = (
            pipeline_global
            .named_steps[
                "model"
            ]
        )

        names = (
            preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance = (
                model.feature_importances_
            )

        elif hasattr(
            model,
            "coef_"
        ):

            coef = model.coef_

            if coef.ndim > 1:

                importance = np.mean(
                    np.abs(coef),
                    axis=0
                )

            else:

                importance = np.abs(
                    coef
                )

        else:

            return None

        result = pd.DataFrame({

            "Feature":
                names,

            "Importance":
                importance
        })

        result = (
            result
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(25)
        )

        return result

    except:

        return None


# ============================================================
# PREDICTION
# ============================================================

def make_prediction(input_data):

    if pipeline_global is None:

        return """
## ❌ No Model Available

Train a model first.
"""

    if input_data is None:

        return """
## ❌ Enter prediction values.
"""

    try:

        # ----------------------------------------------------
        # DATAFRAME
        # ----------------------------------------------------

        if isinstance(
            input_data,
            pd.DataFrame
        ):

            table = input_data.copy()

        else:

            table = pd.DataFrame(
                input_data,
                columns=[
                    "Feature",
                    "Value"
                ]
            )

        if table.empty:

            return "❌ No input values provided."

        # ----------------------------------------------------
        # FEATURE-VALUE DICTIONARY
        # ----------------------------------------------------

        data = {}

        for _, row in table.iterrows():

            feature = str(
                row["Feature"]
            )

            value = row["Value"]

            if feature in feature_columns_global:

                data[feature] = value

        # ----------------------------------------------------
        # CREATE ROW
        # ----------------------------------------------------

        row_data = {}

        for feature in feature_columns_global:

            value = data.get(
                feature,
                np.nan
            )

            # Try numeric conversion
            if isinstance(
                value,
                str
            ):

                numeric_value = pd.to_numeric(
                    value,
                    errors="coerce"
                )

                if not pd.isna(
                    numeric_value
                ):

                    value = numeric_value

            row_data[feature] = value

        prediction_df = pd.DataFrame([
            row_data
        ])

        prediction = (
            pipeline_global
            .predict(
                prediction_df
            )
        )

        prediction_value = prediction[0]

        # ====================================================
        # CLASSIFICATION PROBABILITY
        # ====================================================

        if task_global == "Classification":

            result = f"""
## 🎯 Prediction Result

# **{prediction_value}**
"""

            try:

                probability = (
                    pipeline_global
                    .predict_proba(
                        prediction_df
                    )[0]
                )

                classes = (
                    pipeline_global
                    .named_steps[
                        "model"
                    ].classes_
                )

                probability_df = pd.DataFrame({

                    "Class":
                        classes,

                    "Probability":
                        (
                            probability * 100
                        ).round(2)
                })

                result += "\n### Probability\n"

                result += probability_df.to_markdown(
                    index=False
                )

            except:

                pass

            return result

        else:

            return f"""
## 🎯 Predicted Value

# **{prediction_value:.4f}**
"""

    except Exception as e:

        return f"""
## ❌ Prediction Error

`{str(e)}`

### Check

• Enter all required features
• Use numeric values for numeric columns
• Use valid categories for categorical columns
"""


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(
    target,
    task,
    test_size,
    scaling_method
):

    if df_working is None:

        return (
            pd.DataFrame(),
            "❌ Load dataset first."
        )

    if target not in df_working.columns:

        return (
            pd.DataFrame(),
            "❌ Select valid target."
        )

    df = df_working.copy()

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    X = X.dropna(
        axis=1,
        how="all"
    )

    if task == "Classification":

        y = y.astype(str)

    else:

        y = pd.to_numeric(
            y,
            errors="coerce"
        )

        valid = y.notna()

        X = X.loc[valid]

        y = y.loc[valid]

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if scaling_method == "StandardScaler":

        scaler = StandardScaler()

    elif scaling_method == "MinMaxScaler":

        scaler = MinMaxScaler()

    else:

        scaler = "passthrough"

    if scaler == "passthrough":

        numeric_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ])

    else:

        numeric_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                scaler
            )
        ])

    categorical_pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "num",
                numeric_pipe,
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "cat",
                categorical_pipe,
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y if task == "Classification" else None
        )

    except:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

    results = []

    for name, model in get_models(
        task
    ).items():

        try:

            pipe = Pipeline([
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ])

            pipe.fit(
                X_train,
                y_train
            )

            pred = pipe.predict(
                X_test
            )

            if task == "Classification":

                score = accuracy_score(
                    y_test,
                    pred
                )

                f1 = f1_score(
                    y_test,
                    pred,
                    average="weighted",
                    zero_division=0
                )

                results.append({

                    "Model": name,

                    "Accuracy":
                        round(score, 4),

                    "F1 Score":
                        round(f1, 4)
                })

            else:

                rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        pred
                    )
                )

                r2 = r2_score(
                    y_test,
                    pred
                )

                results.append({

                    "Model": name,

                    "RMSE":
                        round(rmse, 4),

                    "R²":
                        round(r2, 4)
                })

        except Exception as e:

            results.append({

                "Model": name,

                "Status":
                    "Failed"
            })

    result_df = pd.DataFrame(
        results
    )

    if task == "Classification":

        result_df = result_df.sort_values(
            "Accuracy",
            ascending=False
        )

        message = f"""
## 🏆 Model Comparison

Best model based on Accuracy:

**{result_df.iloc[0]['Model']}**

Accuracy:

**{result_df.iloc[0]['Accuracy']:.4f}**
"""

    else:

        result_df = result_df.sort_values(
            "R²",
            ascending=False
        )

        message = f"""
## 🏆 Model Comparison

Best model based on R²:

**{result_df.iloc[0]['Model']}**

R²:

**{result_df.iloc[0]['R²']:.4f}**
"""

    return (
        result_df,
        message
    )




# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="AI & ML Analytics Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Session-state persistence for Streamlit reruns.
def _init_state():
    defaults = {
        "df_original": None,
        "df_working": None,
        "model_global": None,
        "pipeline_global": None,
        "target_global": None,
        "task_global": None,
        "X_test_global": None,
        "y_test_global": None,
        "y_pred_global": None,
        "feature_columns_global": [],
        "loaded_name": None,
        "load_status": None,
        "clean_result": None,
        "clean_preview": None,
        "clean_report_df": None,
        "prediction_template": None,
        "training_result": None,
        "metrics_output": None,
        "evaluation_confusion": None,
        "evaluation_curve": None,
        "importance_table": None,
        "comparison_table": None,
        "comparison_result": None,
        "prediction_result": None,
        "target_table": None,
        "target_graph": None,
        "graph_output": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _sync_globals_from_state():
    global df_original, df_working
    global model_global, pipeline_global, target_global, task_global
    global X_test_global, y_test_global, y_pred_global, feature_columns_global

    df_original = st.session_state.df_original
    df_working = st.session_state.df_working
    model_global = st.session_state.model_global
    pipeline_global = st.session_state.pipeline_global
    target_global = st.session_state.target_global
    task_global = st.session_state.task_global
    X_test_global = st.session_state.X_test_global
    y_test_global = st.session_state.y_test_global
    y_pred_global = st.session_state.y_pred_global
    feature_columns_global = st.session_state.feature_columns_global


def _sync_state_from_globals():
    st.session_state.df_original = df_original
    st.session_state.df_working = df_working
    st.session_state.model_global = model_global
    st.session_state.pipeline_global = pipeline_global
    st.session_state.target_global = target_global
    st.session_state.task_global = task_global
    st.session_state.X_test_global = X_test_global
    st.session_state.y_test_global = y_test_global
    st.session_state.y_pred_global = y_pred_global
    st.session_state.feature_columns_global = feature_columns_global


def _load_uploaded_file(uploaded_file):
    global df_original, df_working
    try:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(uploaded_file)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Use CSV, XLSX or XLS.")

        df_original = df.copy()
        df_working = df.copy()
        _sync_state_from_globals()
        st.session_state.loaded_name = uploaded_file.name

        numeric_count = len(df.select_dtypes(include=np.number).columns)
        categorical_count = len(df.select_dtypes(exclude=np.number).columns)
        missing_count = int(df.isnull().sum().sum())
        duplicate_count = int(df.duplicated().sum())

        dtype_table = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing": df.isnull().sum().values,
            "Missing %": (df.isnull().mean() * 100).round(2).values,
            "Unique": df.nunique().values,
        })
        statistics = df.describe(include="all").T.reset_index()
        statistics.rename(columns={"index": "Column"}, inplace=True)
        missing_table = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": df.isnull().sum().values,
            "Missing %": (df.isnull().mean() * 100).round(2).values,
        })

        st.session_state.load_status = (
            f"### ✅ Dataset Loaded\n\n"
            f"**File:** `{uploaded_file.name}`  \n"
            f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)} | "
            f"**Numeric:** {numeric_count} | **Categorical:** {categorical_count}"
        )
        st.session_state.load_kpis = {
            "Rows": len(df), "Columns": len(df.columns),
            "Numeric": numeric_count, "Categorical": categorical_count,
            "Missing": missing_count, "Duplicates": duplicate_count,
        }
        st.session_state.dtype_table = dtype_table
        st.session_state.statistics = statistics
        st.session_state.missing_table = missing_table
        st.session_state.clean_result = None
        st.session_state.clean_preview = None
        st.session_state.clean_report_df = None
        st.session_state.training_result = None
        st.session_state.metrics_output = None
        st.session_state.evaluation_confusion = None
        st.session_state.evaluation_curve = None
        st.session_state.importance_table = None
        st.session_state.comparison_table = None
        st.session_state.comparison_result = None
        st.session_state.prediction_result = None
        return True
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return False


_init_state()
_sync_globals_from_state()

st.markdown("""
<div style='text-align:center; padding: 10px 0 16px 0;'>
<h1>🧠 AI & ML Analytics Dashboard</h1>
<p>End-to-End Data Science • Machine Learning • Prediction • Explainability</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📂 Upload Dataset",
    type=["csv", "xlsx", "xls"],
    help="Upload a CSV or Excel dataset."
)

if uploaded_file is not None:
    if st.button("Load Dataset", type="primary"):
        _load_uploaded_file(uploaded_file)
        st.rerun()

if st.session_state.load_status:
    st.markdown(st.session_state.load_status)

if st.session_state.get("load_kpis"):
    k = st.session_state.load_kpis
    cols = st.columns(6)
    for col, label in zip(cols, ["Rows", "Columns", "Numeric", "Categorical", "Missing", "Duplicates"]):
        col.metric(label, f"{k[label]:,}")

if st.session_state.df_working is None:
    st.info("Upload a CSV or Excel dataset and click **Load Dataset** to begin.")
    st.stop()

_sync_globals_from_state()
df = st.session_state.df_working

# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "📊 Overview", "📈 Statistics", "🧹 Cleaning", "🔎 EDA",
    "⚙️ Features", "🤖 Training", "📊 Evaluation", "🏆 Comparison",
    "🎯 Prediction", "🧠 Explainability"
])

with tabs[0]:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Data Types & Data Quality")
    st.dataframe(st.session_state.dtype_table, use_container_width=True)

with tabs[1]:
    st.subheader("Descriptive Statistics")
    st.dataframe(st.session_state.statistics, use_container_width=True)
    st.subheader("Missing Value Analysis")
    st.dataframe(st.session_state.missing_table, use_container_width=True)

with tabs[2]:
    st.subheader("🧹 Advanced Data Cleaning")
    c1, c2 = st.columns(2)
    with c1:
        missing_method = st.selectbox("Missing Value Treatment", [
            "Do Nothing", "Drop rows", "Drop columns >50% missing", "Mean imputation",
            "Median imputation", "Mode imputation", "Forward fill", "Backward fill", "Fill numeric 0"
        ])
        outlier_method = st.selectbox("Outlier Treatment", [
            "Do Nothing", "Remove IQR outliers", "Cap IQR outliers", "Remove Z-score >3"
        ])
        categorical_method = st.selectbox("Categorical Cleaning", [
            "Do Nothing", "Remove extra spaces", "Lowercase", "Uppercase"
        ])
    with c2:
        duplicate_method = st.selectbox("Duplicate Treatment", ["Do Nothing", "Remove duplicates"])
        numeric_method = st.selectbox("Numeric Cleaning", ["Do Nothing", "Convert numeric-looking text", "Round numeric values"])
        invalid_method = st.selectbox("Invalid Value Treatment", ["Do Nothing", "Replace negative numeric values"])

    b1, b2 = st.columns(2)
    with b1:
        apply_clean = st.button("🧹 Apply Cleaning", type="primary", use_container_width=True)
    with b2:
        reset_clean = st.button("🔄 Reset Dataset", use_container_width=True)

    if apply_clean:
        _sync_globals_from_state()
        result, preview, cleaned_df = clean_dataset(
            missing_method, duplicate_method, outlier_method,
            numeric_method, categorical_method, invalid_method
        )
        _sync_state_from_globals()
        st.session_state.clean_result = result
        st.session_state.clean_preview = preview
        st.session_state.clean_report_df = cleaned_df
        st.rerun()

    if reset_clean:
        _sync_globals_from_state()
        result, preview, cleaned_df = reset_cleaning()
        _sync_state_from_globals()
        st.session_state.clean_result = result
        st.session_state.clean_preview = preview
        st.session_state.clean_report_df = cleaned_df
        st.rerun()

    if st.session_state.clean_result:
        st.markdown(st.session_state.clean_result)
        st.dataframe(st.session_state.clean_preview, use_container_width=True)
        cleaned_df = st.session_state.clean_report_df
        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Cleaned CSV", csv_bytes, "cleaned_dataset.csv", "text/csv")

with tabs[3]:
    st.subheader("📊 Exploratory Data Analysis")
    columns = list(df.columns)
    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    graph_options = [
        "Histogram", "Box Plot", "Violin Plot", "Count Plot", "Bar Chart",
        "Scatter Plot", "Line Chart", "Area Chart", "KDE", "Pie Chart",
        "Correlation Heatmap", "Covariance Heatmap", "Missing Values", "Q-Q Plot"
    ]
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        graph_type = st.selectbox("📊 Graph Type", graph_options)
    with g2:
        x_column = st.selectbox("X / Numeric Column", columns)
    with g3:
        y_column = st.selectbox("Y Column", ["None"] + columns)
    with g4:
        hue_column = st.selectbox("Group / Hue", ["None"] + columns)

    if st.button("📊 Generate Visualization", type="primary"):
        _sync_globals_from_state()
        fig = create_graph(graph_type, x_column, None if y_column == "None" else y_column, hue_column)
        st.session_state.graph_output = fig
        _sync_state_from_globals()

    if st.session_state.graph_output is not None:
        st.pyplot(st.session_state.graph_output, clear_figure=False)

    st.markdown("### 🎯 Target Analysis")
    target_analysis_column = st.selectbox("Target Column", columns, key="target_analysis_col")
    if st.button("Analyze Target"):
        _sync_globals_from_state()
        table, fig = target_analysis(target_analysis_column)
        st.session_state.target_table = table
        st.session_state.target_graph = fig
    if st.session_state.target_table is not None:
        st.dataframe(st.session_state.target_table, use_container_width=True)
    if st.session_state.target_graph is not None:
        st.pyplot(st.session_state.target_graph, clear_figure=False)

with tabs[4]:
    st.markdown("### Feature Engineering")
    st.info("This section is prepared for advanced ML preprocessing.")
    feature_target = st.selectbox("Target Column", list(df.columns), key="feature_target")
    feature_method = st.selectbox("Feature Method", [
        "Select method", "Drop selected column", "Create log transformation", "Create squared feature"
    ])
    st.markdown("**Recommended future workflow:** numerical transformation → categorical encoding → scaling → feature selection → dimensionality reduction → train/test split.")

with tabs[5]:
    st.subheader("🤖 Model Training")
    columns = list(df.columns)
    training_target = st.selectbox("🎯 Target Column", columns, key="training_target")
    task_type = st.selectbox("ML Task", ["Classification", "Regression"], key="task_type")
    model_options = list(get_models(task_type).keys())
    model_name = st.selectbox("🤖 Model", model_options, key="model_name")
    scaling_method = st.selectbox("Scaling", ["StandardScaler", "MinMaxScaler", "No Scaling"], key="scaling_method")
    test_size = st.slider("Test Size", 0.10, 0.40, 0.20, 0.05, key="test_size")

    if st.button("🚀 Train Model", type="primary"):
        _sync_globals_from_state()
        result, metrics, fig_cm, fig_roc, prediction_template, importance_df = train_model(
            training_target, task_type, model_name, test_size, scaling_method
        )
        _sync_state_from_globals()
        st.session_state.training_result = result
        st.session_state.metrics_output = metrics
        st.session_state.evaluation_confusion = fig_cm
        st.session_state.evaluation_curve = fig_roc
        st.session_state.prediction_template = prediction_template
        st.session_state.importance_table = importance_df
        st.rerun()

    if st.session_state.training_result:
        st.markdown(st.session_state.training_result)
        st.dataframe(st.session_state.metrics_output, use_container_width=True)

with tabs[6]:
    st.subheader("Model Evaluation")
    if st.session_state.evaluation_confusion is None:
        st.info("Train a model first.")
    else:
        st.pyplot(st.session_state.evaluation_confusion, clear_figure=False)
        st.pyplot(st.session_state.evaluation_curve, clear_figure=False)

with tabs[7]:
    st.subheader("🏆 Compare All Models")
    comparison_target = st.selectbox("Target", list(df.columns), key="comparison_target")
    comparison_task = st.selectbox("Task", ["Classification", "Regression"], key="comparison_task")
    comparison_scaling = st.selectbox("Scaling", ["StandardScaler", "MinMaxScaler", "No Scaling"], key="comparison_scaling")
    comparison_test_size = st.slider("Test Size", 0.10, 0.40, 0.20, 0.05, key="comparison_test_size")
    if st.button("🏆 Compare All Models", type="primary"):
        _sync_globals_from_state()
        table, message = compare_models(comparison_target, comparison_task, comparison_test_size, comparison_scaling)
        st.session_state.comparison_table = table
        st.session_state.comparison_result = message
    if st.session_state.comparison_result:
        st.markdown(st.session_state.comparison_result)
        st.dataframe(st.session_state.comparison_table, use_container_width=True)

with tabs[8]:
    st.subheader("🎯 New Record Prediction")
    if st.session_state.pipeline_global is None:
        st.info("Train a model first.")
    else:
        template = st.session_state.prediction_template
        edited = st.data_editor(template, num_rows="dynamic", use_container_width=True, key="prediction_editor")
        if st.button("🎯 Predict", type="primary"):
            _sync_globals_from_state()
            result = make_prediction(edited)
            st.session_state.prediction_result = result
        if st.session_state.prediction_result:
            st.markdown(st.session_state.prediction_result)

with tabs[9]:
    st.subheader("🧠 Model Explainability")
    if st.session_state.importance_table is None or st.session_state.importance_table.empty:
        st.info("Train a model that supports feature importance/coefficient analysis first.")
    else:
        st.dataframe(st.session_state.importance_table, use_container_width=True)


# ============================================================
# 🧠 AI & ML END-TO-END ANALYTICS DASHBOARD
# Google Colab - COMPLETE SINGLE-CELL VERSION
# ============================================================

!pip install -q "gradio>=5,<6" pandas numpy matplotlib seaborn scikit-learn openpyxl scipy

# ============================================================
# IMPORTS
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import gradio as gr
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
# BUILD DASHBOARD
# ============================================================

with gr.Blocks(
    css=css,
    title="AI & ML Analytics Dashboard"
) as dashboard:

    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML("""
    <div id="main_header">

        <h1>🧠 AI & ML Analytics Dashboard</h1>

        <p>
        End-to-End Data Science • Machine Learning • Prediction • Explainability
        </p>

    </div>
    """)

    # ========================================================
    # UPLOAD
    # ========================================================

    with gr.Row():

        file_upload = gr.File(
            label="📂 Upload Dataset",
            file_types=[
                ".csv",
                ".xlsx",
                ".xls"
            ],
            type="filepath",
            scale=4
        )

        load_button = gr.Button(
            "Load Dataset",
            variant="primary",
            scale=1
        )

    status_output = gr.Markdown()

    kpi_output = gr.HTML()

    # ========================================================
    # TABS
    # ========================================================

    with gr.Tabs():

        # ====================================================
        # OVERVIEW
        # ====================================================

        with gr.Tab("📊 Overview"):

            preview_output = gr.Dataframe(
                label="Dataset Preview",
                interactive=False
            )

            dtype_output = gr.Dataframe(
                label="Data Types & Data Quality",
                interactive=False
            )

        # ====================================================
        # STATISTICS
        # ====================================================

        with gr.Tab("📈 Statistics"):

            statistics_output = gr.Dataframe(
                label="Descriptive Statistics",
                interactive=False
            )

            missing_output = gr.Dataframe(
                label="Missing Value Analysis",
                interactive=False
            )

        # ====================================================
        # CLEANING
        # ====================================================

        with gr.Tab("🧹 Cleaning"):

            gr.Markdown(
                "### Advanced Data Cleaning"
            )

            with gr.Row():

                missing_method = gr.Dropdown(
                    label="Missing Value Treatment",
                    choices=[
                        "Do Nothing",
                        "Drop rows",
                        "Drop columns >50% missing",
                        "Mean imputation",
                        "Median imputation",
                        "Mode imputation",
                        "Forward fill",
                        "Backward fill",
                        "Fill numeric 0"
                    ],
                    value="Do Nothing"
                )

                duplicate_method = gr.Dropdown(
                    label="Duplicate Treatment",
                    choices=[
                        "Do Nothing",
                        "Remove duplicates"
                    ],
                    value="Do Nothing"
                )

            with gr.Row():

                outlier_method = gr.Dropdown(
                    label="Outlier Treatment",
                    choices=[
                        "Do Nothing",
                        "Remove IQR outliers",
                        "Cap IQR outliers",
                        "Remove Z-score >3"
                    ],
                    value="Do Nothing"
                )

                numeric_method = gr.Dropdown(
                    label="Numeric Cleaning",
                    choices=[
                        "Do Nothing",
                        "Convert numeric-looking text",
                        "Round numeric values"
                    ],
                    value="Do Nothing"
                )

            with gr.Row():

                categorical_method = gr.Dropdown(
                    label="Categorical Cleaning",
                    choices=[
                        "Do Nothing",
                        "Remove extra spaces",
                        "Lowercase",
                        "Uppercase"
                    ],
                    value="Do Nothing"
                )

                invalid_method = gr.Dropdown(
                    label="Invalid Value Treatment",
                    choices=[
                        "Do Nothing",
                        "Replace negative numeric values"
                    ],
                    value="Do Nothing"
                )

            with gr.Row():

                clean_button = gr.Button(
                    "🧹 Apply Cleaning",
                    variant="primary"
                )

                reset_button = gr.Button(
                    "🔄 Reset Dataset"
                )

            cleaning_result = gr.Markdown()

            cleaned_preview = gr.Dataframe(
                label="Cleaned Dataset Preview"
            )

        # ====================================================
        # EDA
        # ====================================================

        with gr.Tab("🔎 EDA"):

            with gr.Row():

                graph_type = gr.Dropdown(
                    label="📊 Graph Type",
                    choices=[
                        "Histogram",
                        "Box Plot",
                        "Violin Plot",
                        "Count Plot",
                        "Bar Chart",
                        "Scatter Plot",
                        "Line Chart",
                        "Area Chart",
                        "KDE",
                        "Pie Chart",
                        "Correlation Heatmap",
                        "Covariance Heatmap",
                        "Missing Values",
                        "Q-Q Plot"
                    ],
                    value="Histogram"
                )

                x_column = gr.Dropdown(
                    label="X / Numeric Column",
                    choices=[]
                )

                y_column = gr.Dropdown(
                    label="Y Column",
                    choices=[]
                )

                hue_column = gr.Dropdown(
                    label="Group / Hue",
                    choices=["None"],
                    value="None"
                )

            graph_button = gr.Button(
                "📊 Generate Visualization",
                variant="primary"
            )

            graph_output = gr.Plot(
                label="Visualization"
            )

            # ------------------------------------------------
            # TARGET ANALYSIS
            # ------------------------------------------------

            gr.Markdown(
                "### 🎯 Target Analysis"
            )

            target_analysis_column = gr.Dropdown(
                label="Target Column",
                choices=[]
            )

            target_analysis_button = gr.Button(
                "Analyze Target"
            )

            target_table = gr.Dataframe(
                label="Target Distribution"
            )

            target_graph = gr.Plot(
                label="Target Visualization"
            )

        # ====================================================
        # FEATURES
        # ====================================================

        with gr.Tab("⚙️ Features"):

            gr.Markdown(
                """
                ### Feature Engineering

                This section is prepared for advanced ML preprocessing.
                """
            )

            with gr.Row():

                feature_target = gr.Dropdown(
                    label="Target Column",
                    choices=[]
                )

                feature_method = gr.Dropdown(
                    label="Feature Method",
                    choices=[
                        "Select method",
                        "Drop selected column",
                        "Create log transformation",
                        "Create squared feature"
                    ],
                    value="Select method"
                )

            gr.Markdown(
                """
                **Recommended future feature-engineering workflow:**

                Numerical transformation → categorical encoding →
                scaling → feature selection → dimensionality reduction →
                train/test split.
                """
            )

        # ====================================================
        # TRAINING
        # ====================================================

        with gr.Tab("🤖 Training"):

            with gr.Row():

                training_target = gr.Dropdown(
                    label="🎯 Target Column",
                    choices=[]
                )

                task_type = gr.Dropdown(
                    label="ML Task",
                    choices=[
                        "Classification",
                        "Regression"
                    ],
                    value="Classification"
                )

            with gr.Row():

                model_name = gr.Dropdown(
                    label="🤖 Model",
                    choices=[
                        "Logistic Regression",
                        "Decision Tree",
                        "Random Forest",
                        "KNN",
                        "SVM",
                        "Gradient Boosting"
                    ],
                    value="Logistic Regression"
                )

                scaling_method = gr.Dropdown(
                    label="Scaling",
                    choices=[
                        "StandardScaler",
                        "MinMaxScaler",
                        "No Scaling"
                    ],
                    value="StandardScaler"
                )

                test_size = gr.Slider(
                    label="Test Size",
                    minimum=0.10,
                    maximum=0.40,
                    value=0.20,
                    step=0.05
                )

            train_button = gr.Button(
                "🚀 Train Model",
                variant="primary"
            )

            training_result = gr.Markdown()

            metrics_output = gr.Dataframe(
                label="Model Performance"
            )

        # ====================================================
        # EVALUATION
        # ====================================================

        with gr.Tab("📊 Evaluation"):

            gr.Markdown(
                """
                ### Model Evaluation
                """

            )

            evaluation_confusion = gr.Plot(
                label="Confusion Matrix / Actual vs Predicted"
            )

            evaluation_curve = gr.Plot(
                label="ROC Curve / Residual Plot"
            )

        # ====================================================
        # COMPARISON
        # ====================================================

        with gr.Tab("🏆 Comparison"):

            with gr.Row():

                comparison_target = gr.Dropdown(
                    label="Target",
                    choices=[]
                )

                comparison_task = gr.Dropdown(
                    label="Task",
                    choices=[
                        "Classification",
                        "Regression"
                    ],
                    value="Classification"
                )

            with gr.Row():

                comparison_scaling = gr.Dropdown(
                    label="Scaling",
                    choices=[
                        "StandardScaler",
                        "MinMaxScaler",
                        "No Scaling"
                    ],
                    value="StandardScaler"
                )

                comparison_test_size = gr.Slider(
                    label="Test Size",
                    minimum=0.10,
                    maximum=0.40,
                    value=0.20,
                    step=0.05
                )

            comparison_button = gr.Button(
                "🏆 Compare All Models",
                variant="primary"
            )

            comparison_result = gr.Markdown()

            comparison_table = gr.Dataframe(
                label="Model Leaderboard"
            )

        # ====================================================
        # PREDICTION
        # ====================================================

        with gr.Tab("🎯 Prediction"):

            gr.Markdown(
                """
                ### 🎯 New Patient / New Record Prediction

                Enter one value for each feature below.
                The **Feature** column is automatically generated
                from the model used during training.
                """
            )

            prediction_input = gr.Dataframe(
                headers=[
                    "Feature",
                    "Value"
                ],
                datatype=[
                    "str",
                    "str"
                ],
                value=pd.DataFrame(
                    columns=[
                        "Feature",
                        "Value"
                    ]
                ),
                row_count=(1, "dynamic"),
                col_count=(2, "fixed"),
                interactive=True,
                label="Input Format"
            )

            predict_button = gr.Button(
                "🎯 Predict",
                variant="primary"
            )

            prediction_result = gr.Markdown(
                elem_classes="result-card"
            )

        # ====================================================
        # EXPLAINABILITY
        # ====================================================

        with gr.Tab("🧠 Explainability"):

            gr.Markdown(
                """
                ### 🧠 Model Explainability

                Feature importance helps understand which variables
                contribute most strongly to the trained model.
                """
            )

            importance_table = gr.Dataframe(
                label="Feature Importance"
            )

    # ========================================================
    # EVENTS
    # ========================================================

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    load_button.click(
        fn=load_dataset,
        inputs=file_upload,
        outputs=[
            status_output,
            kpi_output,
            preview_output,
            dtype_output,
            statistics_output,
            missing_output,
            x_column,
            hue_column,
            y_column,
            target_analysis_column
        ]
    ).then(
        fn=lambda: gr.update(
            choices=list(df_working.columns)
            if df_working is not None
            else []
        ),
        inputs=None,
        outputs=[
            training_target
        ]
    ).then(
        fn=lambda: gr.update(
            choices=list(df_working.columns)
            if df_working is not None
            else []
        ),
        inputs=None,
        outputs=[
            comparison_target
        ]
    ).then(
        fn=lambda: gr.update(
            choices=list(df_working.columns)
            if df_working is not None
            else []
        ),
        inputs=None,
        outputs=[
            feature_target
        ]
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    clean_button.click(
        fn=clean_dataset,
        inputs=[
            missing_method,
            duplicate_method,
            outlier_method,
            numeric_method,
            categorical_method,
            invalid_method
        ],
        outputs=[
            cleaning_result,
            cleaned_preview,
            preview_output
        ]
    )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    reset_button.click(
        fn=reset_cleaning,
        inputs=None,
        outputs=[
            cleaning_result,
            cleaned_preview,
            preview_output
        ]
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    graph_button.click(
        fn=create_graph,
        inputs=[
            graph_type,
            x_column,
            y_column,
            hue_column
        ],
        outputs=graph_output
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target_analysis_button.click(
        fn=target_analysis,
        inputs=[
            target_analysis_column
        ],
        outputs=[
            target_table,
            target_graph
        ]
    )

    # --------------------------------------------------------
    # TASK CHANGE
    # --------------------------------------------------------

    task_type.change(
        fn=change_task,
        inputs=task_type,
        outputs=model_name
    )

    comparison_task.change(
        fn=change_task,
        inputs=comparison_task,
        outputs=model_name
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_button.click(
        fn=train_model,
        inputs=[
            training_target,
            task_type,
            model_name,
            test_size,
            scaling_method
        ],
        outputs=[
            training_result,
            metrics_output,
            evaluation_confusion,
            evaluation_curve,
            prediction_input,
            importance_table
        ]
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predict_button.click(
        fn=make_prediction,
        inputs=[
            prediction_input
        ],
        outputs=[
            prediction_result
        ]
    )

    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    comparison_button.click(
        fn=compare_models,
        inputs=[
            comparison_target,
            comparison_task,
            comparison_test_size,
            comparison_scaling
        ],
        outputs=[
            comparison_table,
            comparison_result
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

dashboard.launch(
    share=True,
    debug=True
)
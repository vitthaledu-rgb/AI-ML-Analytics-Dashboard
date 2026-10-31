
# ============================================================
# AI & ML END-TO-END HEALTHCARE ANALYTICS DASHBOARD
# STREAMLIT VERSION - GOOGLE COLAB / STREAMLIT CLOUD
# ============================================================
# Features:
# 1. Dataset upload: CSV / XLSX / XLS
# 2. Overview + data quality KPIs
# 3. Descriptive statistics + missing-value analysis
# 4. Cleaning with before/after preview
# 5. EDA: 14 visualization types
# 6. Target analysis
# 7. Feature engineering
# 8. Classification + Regression
# 9. Preprocessing: imputation, one-hot encoding, scaling
# 10. Model training
# 11. Evaluation: confusion matrix / actual-vs-predicted + ROC / residual
# 12. Model comparison
# 13. Prediction for new records
# 14. Feature importance
# 15. Download cleaned data / statistics / comparison results
# ============================================================

import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
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
    roc_curve,
)

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI & ML Analytics Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

.main-header {
    padding: 18px 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #eef4ff, #f8fbff);
    border: 1px solid #dbe5f4;
    margin-bottom: 16px;
}

.main-header h1 {
    margin: 0;
    font-size: 2.1rem;
    font-weight: 800;
}

.main-header p {
    margin: 5px 0 0 0;
    color: #667085;
}

.kpi-card {
    background: white;
    border: 1px solid #e4e7ec;
    border-radius: 14px;
    padding: 14px 10px;
    text-align: center;
    min-height: 100px;
    box-shadow: 0 2px 8px rgba(16,24,40,.04);
}

.kpi-title {
    color: #667085;
    font-size: .75rem;
    text-transform: uppercase;
    letter-spacing: .04em;
}

.kpi-value {
    font-size: 1.55rem;
    font-weight: 800;
    margin-top: 6px;
}

.section-card {
    border: 1px solid #e4e7ec;
    border-radius: 14px;
    padding: 14px;
    background: #ffffff;
}

.small-note {
    color: #667085;
    font-size: .85rem;
}

div[data-testid="stMetric"] {
    border: 1px solid #e4e7ec;
    border-radius: 12px;
    padding: 10px;
    background: white;
}

button {
    border-radius: 9px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
}

.stTabs [data-baseweb="tab"] {
    padding-left: 12px;
    padding-right: 12px;
}

[data-testid="stDataFrame"] {
    border-radius: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "df_original": None,
    "df_working": None,
    "source_name": None,
    "model": None,
    "pipeline": None,
    "target": None,
    "task": None,
    "feature_columns": [],
    "X_test": None,
    "y_test": None,
    "y_pred": None,
    "metrics_df": None,
    "comparison_df": None,
    "importance_df": None,
    "evaluation_fig_1": None,
    "evaluation_fig_2": None,
    "prediction_template": None,
    "last_cleaning_summary": None,
    "feature_engineering_log": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def empty_fig(message="No result available"):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=14,
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


def dataframe_download(df, filename):
    if df is None:
        return None
    return df.to_csv(index=False).encode("utf-8")


def excel_download(df, filename):
    if df is None:
        return None
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return buffer.getvalue()


def load_dataframe(uploaded_file):
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    if name.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="xlrd")

    raise ValueError("Unsupported file format. Use CSV, XLSX or XLS.")


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(exclude=np.number).columns.tolist()


def get_models(task):
    if task == "Classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            ),
        }

    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "SVR": SVR(),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),
    }


def build_preprocessor(X, scaling_method):
    num_cols = numeric_columns(X)
    cat_cols = categorical_columns(X)

    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
    elif scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler is None:
        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                )
            ]
        )
    else:
        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                ("scaler", scaler),
            ]
        )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = []

    if num_cols:
        transformers.append(
            ("numeric", numeric_pipeline, num_cols)
        )

    if cat_cols:
        transformers.append(
            ("categorical", categorical_pipeline, cat_cols)
        )

    if not transformers:
        raise ValueError("No usable feature columns found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def safe_train_test_split(X, y, task, test_size):
    if task == "Classification":
        try:
            return train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y,
            )
        except Exception:
            return train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
            )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )


def prepare_target(df, target, task):
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    X = X.dropna(axis=1, how="all")

    if task == "Classification":
        y = y.astype(str)
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X = X.loc[valid].copy()
        y = y.loc[valid].copy()

    if len(X) < 5:
        raise ValueError("Not enough valid rows for machine learning.")

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")

    if task == "Classification" and y.nunique() < 2:
        raise ValueError("Classification requires at least 2 classes.")

    return X, y


def feature_names_from_pipeline(pipeline):
    try:
        pre = pipeline.named_steps["preprocessor"]
        return pre.get_feature_names_out()
    except Exception:
        return None


def get_feature_importance():
    pipeline = st.session_state.pipeline
    if pipeline is None:
        return pd.DataFrame()

    try:
        pre = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        names = pre.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            if coef.ndim > 1:
                importance = np.mean(np.abs(coef), axis=0)
            else:
                importance = np.abs(coef)
        else:
            return pd.DataFrame()

        n = min(len(names), len(importance))

        result = pd.DataFrame(
            {
                "Feature": names[:n],
                "Importance": importance[:n],
            }
        )

        return result.sort_values(
            "Importance",
            ascending=False,
        ).head(30)

    except Exception:
        return pd.DataFrame()


def target_analysis(df, target):
    values = df[target].value_counts(dropna=False)

    table = pd.DataFrame(
        {
            "Category": values.index.astype(str),
            "Count": values.values,
            "Percentage": (
                values.values / max(len(df), 1) * 100
            ).round(2),
        }
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))

    if pd.api.types.is_numeric_dtype(df[target]):
        sns.histplot(
            df[target].dropna(),
            kde=True,
            ax=ax,
        )
    else:
        values.head(15).plot(
            kind="bar",
            ax=ax,
        )
        ax.tick_params(
            axis="x",
            rotation=45,
        )

    ax.set_title(
        f"Target Analysis: {target}",
        fontweight="bold",
    )
    plt.tight_layout()

    return table, fig


def create_graph(df, graph_type, x_column, y_column, hue_column):
    if df is None or df.empty:
        return empty_fig("Load a dataset first")

    try:
        fig, ax = plt.subplots(figsize=(9, 5))

        hue = None if hue_column in [None, "None", ""] else hue_column

        if graph_type == "Histogram":
            sns.histplot(
                data=df,
                x=x_column,
                hue=hue,
                kde=True,
                ax=ax,
            )

        elif graph_type == "Box Plot":
            if y_column in [None, "None", ""]:
                sns.boxplot(
                    data=df,
                    y=x_column,
                    ax=ax,
                )
            else:
                sns.boxplot(
                    data=df,
                    x=x_column,
                    y=y_column,
                    hue=hue,
                    ax=ax,
                )

        elif graph_type == "Violin Plot":
            if y_column in [None, "None", ""]:
                sns.violinplot(
                    data=df,
                    y=x_column,
                    ax=ax,
                )
            else:
                sns.violinplot(
                    data=df,
                    x=x_column,
                    y=y_column,
                    hue=hue,
                    ax=ax,
                )

        elif graph_type == "Count Plot":
            sns.countplot(
                data=df,
                x=x_column,
                hue=hue,
                ax=ax,
            )
            ax.tick_params(
                axis="x",
                rotation=45,
            )

        elif graph_type == "Bar Chart":
            if y_column in [None, "None", ""]:
                counts = df[x_column].value_counts().head(15)
                counts.plot(kind="bar", ax=ax)
            else:
                temp = (
                    df.groupby(x_column)[y_column]
                    .mean()
                    .sort_values(ascending=False)
                    .head(20)
                )
                temp.plot(kind="bar", ax=ax)

            ax.tick_params(
                axis="x",
                rotation=45,
            )

        elif graph_type == "Scatter Plot":
            if y_column in [None, "None", ""]:
                return empty_fig(
                    "Scatter Plot requires X and Y columns."
                )
            sns.scatterplot(
                data=df,
                x=x_column,
                y=y_column,
                hue=hue,
                ax=ax,
            )

        elif graph_type == "Line Chart":
            if y_column in [None, "None", ""]:
                return empty_fig(
                    "Line Chart requires X and Y columns."
                )
            sns.lineplot(
                data=df,
                x=x_column,
                y=y_column,
                hue=hue,
                ax=ax,
            )

        elif graph_type == "Area Chart":
            if y_column in [None, "None", ""]:
                return empty_fig(
                    "Area Chart requires X and Y columns."
                )

            temp = df[[x_column, y_column]].dropna().head(200)

            if temp.empty:
                return empty_fig("No valid data for Area Chart.")

            if pd.api.types.is_numeric_dtype(temp[x_column]):
                temp = temp.sort_values(x_column)
                ax.fill_between(
                    temp[x_column],
                    temp[y_column],
                    alpha=0.4,
                )
                ax.plot(
                    temp[x_column],
                    temp[y_column],
                )
            else:
                ax.fill_between(
                    range(len(temp)),
                    temp[y_column],
                    alpha=0.4,
                )
                ax.plot(
                    range(len(temp)),
                    temp[y_column],
                )
                ax.set_xticks(
                    range(len(temp))
                )
                ax.set_xticklabels(
                    temp[x_column].astype(str),
                    rotation=45,
                    ha="right",
                )

        elif graph_type == "KDE":
            sns.kdeplot(
                data=df,
                x=x_column,
                hue=hue,
                fill=True,
                ax=ax,
            )

        elif graph_type == "Pie Chart":
            values = df[x_column].value_counts().head(10)
            if values.empty:
                return empty_fig("No values available.")
            ax.pie(
                values.values,
                labels=values.index.astype(str),
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title(f"{x_column} distribution")

        elif graph_type == "Correlation Heatmap":
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.shape[1] < 2:
                return empty_fig(
                    "At least 2 numeric columns required."
                )
            corr = numeric_df.corr()
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                ax=ax,
            )

        elif graph_type == "Covariance Heatmap":
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.shape[1] < 2:
                return empty_fig(
                    "At least 2 numeric columns required."
                )
            cov = numeric_df.cov()
            sns.heatmap(
                cov,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                ax=ax,
            )

        elif graph_type == "Missing Values":
            missing = (
                df.isnull()
                .sum()
                .sort_values(ascending=False)
            )
            missing = missing[missing > 0]

            if len(missing) == 0:
                return empty_fig("No missing values.")

            missing.plot(
                kind="bar",
                ax=ax,
            )
            ax.tick_params(
                axis="x",
                rotation=45,
            )
            ax.set_ylabel("Missing Count")

        elif graph_type == "Q-Q Plot":
            if not pd.api.types.is_numeric_dtype(df[x_column]):
                return empty_fig(
                    "Q-Q Plot requires a numeric X column."
                )

            values = pd.to_numeric(
                df[x_column],
                errors="coerce",
            ).dropna()

            if len(values) < 3:
                return empty_fig(
                    "Not enough numeric observations."
                )

            stats.probplot(
                values,
                dist="norm",
                plot=ax,
            )
            ax.set_title(
                f"Q-Q Plot: {x_column}"
            )

        ax.set_title(
            ax.get_title() or graph_type,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    except Exception as e:
        return empty_fig(
            f"Graph Error\n\n{str(e)}"
        )


def apply_cleaning(
    df,
    missing_method,
    duplicate_method,
    outlier_method,
    numeric_method,
    categorical_method,
    invalid_method,
):
    if df is None:
        raise ValueError("Load a dataset first.")

    work = df.copy()

    before_rows = len(work)
    before_cols = len(work.columns)
    before_missing = int(work.isnull().sum().sum())
    before_duplicates = int(work.duplicated().sum())

    # 1. Duplicates
    if duplicate_method == "Remove duplicates":
        work = work.drop_duplicates()

    # 2. Missing values
    if missing_method == "Drop rows":
        work = work.dropna()

    elif missing_method == "Drop columns >50% missing":
        threshold = int(len(work) * 0.50)
        threshold = max(threshold, 1)
        work = work.dropna(
            axis=1,
            thresh=threshold,
        )

    elif missing_method == "Mean imputation":
        for col in numeric_columns(work):
            work[col] = work[col].fillna(
                work[col].mean()
            )

    elif missing_method == "Median imputation":
        for col in numeric_columns(work):
            work[col] = work[col].fillna(
                work[col].median()
            )

    elif missing_method == "Mode imputation":
        for col in work.columns:
            mode = work[col].mode(dropna=True)
            if not mode.empty:
                work[col] = work[col].fillna(
                    mode.iloc[0]
                )

    elif missing_method == "Forward fill":
        work = work.ffill()

    elif missing_method == "Backward fill":
        work = work.bfill()

    elif missing_method == "Fill numeric 0":
        nums = numeric_columns(work)
        if nums:
            work[nums] = work[nums].fillna(0)

    # 3. Numeric cleaning
    if numeric_method == "Convert numeric-looking text":
        for col in work.columns:
            if work[col].dtype == "object":
                converted = pd.to_numeric(
                    work[col],
                    errors="coerce",
                )
                valid_ratio = converted.notna().mean()

                if valid_ratio >= 0.80:
                    work[col] = converted

    elif numeric_method == "Round numeric values":
        nums = numeric_columns(work)
        if nums:
            work[nums] = work[nums].round(2)

    # 4. Categorical cleaning
    cats = categorical_columns(work)

    if categorical_method == "Remove extra spaces":
        for col in cats:
            work[col] = (
                work[col]
                .astype(str)
                .str.strip()
            )

    elif categorical_method == "Lowercase":
        for col in cats:
            work[col] = (
                work[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    elif categorical_method == "Uppercase":
        for col in cats:
            work[col] = (
                work[col]
                .astype(str)
                .str.strip()
                .str.upper()
            )

    # 5. Invalid values
    if invalid_method == "Replace negative numeric values":
        for col in numeric_columns(work):
            work.loc[work[col] < 0, col] = np.nan

    # 6. Outliers
    if outlier_method != "Do Nothing":
        for col in numeric_columns(work):
            series = work[col].dropna()

            if series.empty:
                continue

            if outlier_method in [
                "Remove IQR outliers",
                "Cap IQR outliers",
            ]:
                q1 = work[col].quantile(0.25)
                q3 = work[col].quantile(0.75)
                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                if outlier_method == "Remove IQR outliers":
                    work = work[
                        (work[col] >= lower)
                        & (work[col] <= upper)
                    ].copy()

                else:
                    work[col] = work[col].clip(
                        lower,
                        upper,
                    )

            elif outlier_method == "Remove Z-score >3":
                if len(series) >= 3:
                    filled = work[col].fillna(
                        work[col].median()
                    )
                    z = np.abs(stats.zscore(filled))
                    work = work.loc[z < 3].copy()

    after_missing = int(work.isnull().sum().sum())
    after_duplicates = int(work.duplicated().sum())

    summary = pd.DataFrame(
        {
            "Quality Measure": [
                "Rows",
                "Columns",
                "Missing Values",
                "Duplicate Rows",
            ],
            "Before": [
                before_rows,
                before_cols,
                before_missing,
                before_duplicates,
            ],
            "After": [
                len(work),
                len(work.columns),
                after_missing,
                after_duplicates,
            ],
            "Change": [
                len(work) - before_rows,
                len(work.columns) - before_cols,
                after_missing - before_missing,
                after_duplicates - before_duplicates,
            ],
        }
    )

    return work, summary


def train_model(
    df,
    target,
    task,
    model_name,
    test_size,
    scaling_method,
):
    X, y = prepare_target(
        df,
        target,
        task,
    )

    preprocessor = build_preprocessor(
        X,
        scaling_method,
    )

    models = get_models(task)

    if model_name not in models:
        model_name = list(models.keys())[0]

    model = models[model_name]

    X_train, X_test, y_train, y_test = safe_train_test_split(
        X,
        y,
        task,
        test_size,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    y_pred = pipeline.predict(X_test)

    if task == "Classification":
        metrics = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                ],
                "Score": [
                    accuracy_score(y_test, y_pred),
                    precision_score(
                        y_test,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    ),
                    recall_score(
                        y_test,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    ),
                    f1_score(
                        y_test,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    ),
                ],
            }
        )

        if y_test.nunique() == 2:
            try:
                probabilities = pipeline.predict_proba(
                    X_test
                )[:, 1]
                auc = roc_auc_score(
                    y_test,
                    probabilities,
                )
                metrics = pd.concat(
                    [
                        metrics,
                        pd.DataFrame(
                            {
                                "Metric": ["ROC-AUC"],
                                "Score": [auc],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            except Exception:
                pass

        fig1, ax1 = plt.subplots(figsize=(7, 5))
        labels = np.unique(
            np.concatenate(
                [
                    np.asarray(y_test),
                    np.asarray(y_pred),
                ]
            )
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=labels,
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax1,
        )
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        ax1.set_title(
            "Confusion Matrix",
            fontweight="bold",
        )
        plt.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(7, 5))

        try:
            if len(labels) == 2:
                probabilities = pipeline.predict_proba(
                    X_test
                )[:, 1]

                positive_label = labels[1]
                y_binary = (
                    np.asarray(y_test)
                    == positive_label
                ).astype(int)

                fpr, tpr, _ = roc_curve(
                    y_binary,
                    probabilities,
                )

                ax2.plot(
                    fpr,
                    tpr,
                    label="ROC curve",
                )
                ax2.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--",
                )
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_title(
                    "ROC Curve",
                    fontweight="bold",
                )
                ax2.legend()

            else:
                ax2.text(
                    0.5,
                    0.5,
                    "Multiclass ROC requires\n"
                    "One-vs-Rest handling.",
                    ha="center",
                    va="center",
                )
                ax2.axis("off")

        except Exception:
            ax2.text(
                0.5,
                0.5,
                "ROC curve unavailable",
                ha="center",
                va="center",
            )
            ax2.axis("off")

        plt.tight_layout()

    else:
        mae = mean_absolute_error(
            y_test,
            y_pred,
        )
        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred,
            )
        )
        r2 = r2_score(
            y_test,
            y_pred,
        )

        metrics = pd.DataFrame(
            {
                "Metric": [
                    "MAE",
                    "RMSE",
                    "R²",
                ],
                "Score": [
                    mae,
                    rmse,
                    r2,
                ],
            }
        )

        fig1, ax1 = plt.subplots(figsize=(7, 5))

        ax1.scatter(
            y_test,
            y_pred,
            alpha=0.65,
        )

        minimum = min(
            np.min(y_test),
            np.min(y_pred),
        )
        maximum = max(
            np.max(y_test),
            np.max(y_pred),
        )

        ax1.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
        )

        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        ax1.set_title(
            "Actual vs Predicted",
            fontweight="bold",
        )
        plt.tight_layout()

        residuals = np.asarray(y_test) - np.asarray(y_pred)

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(
            y_pred,
            residuals,
            alpha=0.65,
        )
        ax2.axhline(
            0,
            linestyle="--",
        )
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residual")
        ax2.set_title(
            "Residual Plot",
            fontweight="bold",
        )
        plt.tight_layout()

    feature_columns = list(X.columns)

    prediction_template = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Value": ["" for _ in feature_columns],
        }
    )

    return {
        "pipeline": pipeline,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
        "fig1": fig1,
        "fig2": fig2,
        "feature_columns": feature_columns,
        "prediction_template": prediction_template,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": len(feature_columns),
        "model_name": model_name,
    }


def compare_models(
    df,
    target,
    task,
    test_size,
    scaling_method,
):
    X, y = prepare_target(
        df,
        target,
        task,
    )

    preprocessor = build_preprocessor(
        X,
        scaling_method,
    )

    X_train, X_test, y_train, y_test = safe_train_test_split(
        X,
        y,
        task,
        test_size,
    )

    results = []

    for name, model in get_models(task).items():
        try:
            pipe = Pipeline(
                [
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "model",
                        model,
                    ),
                ]
            )

            pipe.fit(
                X_train,
                y_train,
            )

            pred = pipe.predict(X_test)

            if task == "Classification":
                results.append(
                    {
                        "Model": name,
                        "Accuracy": round(
                            accuracy_score(
                                y_test,
                                pred,
                            ),
                            4,
                        ),
                        "Precision": round(
                            precision_score(
                                y_test,
                                pred,
                                average="weighted",
                                zero_division=0,
                            ),
                            4,
                        ),
                        "Recall": round(
                            recall_score(
                                y_test,
                                pred,
                                average="weighted",
                                zero_division=0,
                            ),
                            4,
                        ),
                        "F1 Score": round(
                            f1_score(
                                y_test,
                                pred,
                                average="weighted",
                                zero_division=0,
                            ),
                            4,
                        ),
                        "Status": "Success",
                    }
                )

            else:
                results.append(
                    {
                        "Model": name,
                        "MAE": round(
                            mean_absolute_error(
                                y_test,
                                pred,
                            ),
                            4,
                        ),
                        "RMSE": round(
                            np.sqrt(
                                mean_squared_error(
                                    y_test,
                                    pred,
                                )
                            ),
                            4,
                        ),
                        "R²": round(
                            r2_score(
                                y_test,
                                pred,
                            ),
                            4,
                        ),
                        "Status": "Success",
                    }
                )

        except Exception as e:
            results.append(
                {
                    "Model": name,
                    "Status": f"Failed: {str(e)[:100]}",
                }
            )

    result_df = pd.DataFrame(results)

    if task == "Classification":
        if "Accuracy" in result_df.columns:
            result_df = result_df.sort_values(
                "Accuracy",
                ascending=False,
            )
    else:
        if "R²" in result_df.columns:
            result_df = result_df.sort_values(
                "R²",
                ascending=False,
            )

    return result_df


def make_prediction_from_table(
    table,
):
    pipeline = st.session_state.pipeline
    features = st.session_state.feature_columns

    if pipeline is None:
        raise ValueError(
            "Train a model first."
        )

    if table is None or table.empty:
        raise ValueError(
            "Enter prediction values."
        )

    data = {}

    for _, row in table.iterrows():
        feature = str(row.get("Feature", "")).strip()
        value = row.get("Value", "")

        if feature in features:
            data[feature] = value

    row_data = {}

    for feature in features:
        value = data.get(
            feature,
            np.nan,
        )

        if isinstance(value, str):
            numeric_value = pd.to_numeric(
                value,
                errors="coerce",
            )
            if not pd.isna(numeric_value):
                value = numeric_value

        row_data[feature] = value

    prediction_df = pd.DataFrame([row_data])

    prediction = pipeline.predict(
        prediction_df
    )

    return prediction_df, prediction


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="main-header">
    <h1>🧠 AI & ML Analytics Dashboard</h1>
    <p>
    End-to-End Data Science • Data Cleaning • EDA • Feature Engineering •
    Machine Learning • Evaluation • Prediction • Explainability
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR - DATASET
# ============================================================

with st.sidebar:
    st.header("📂 Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV / XLSX / XLS",
        type=["csv", "xlsx", "xls"],
        help="Upload a healthcare, business, clinical, or general analytics dataset.",
    )

    if uploaded_file is not None:
        if st.button(
            "📥 Load Dataset",
            use_container_width=True,
            type="primary",
        ):
            try:
                df = load_dataframe(uploaded_file)

                st.session_state.df_original = df.copy()
                st.session_state.df_working = df.copy()
                st.session_state.source_name = uploaded_file.name

                # Reset model after loading a new dataset
                st.session_state.model = None
                st.session_state.pipeline = None
                st.session_state.target = None
                st.session_state.task = None
                st.session_state.feature_columns = []
                st.session_state.metrics_df = None
                st.session_state.comparison_df = None
                st.session_state.importance_df = None
                st.session_state.feature_engineering_log = []

                st.success(
                    f"Loaded: {uploaded_file.name}"
                )
                st.rerun()

            except Exception as e:
                st.error(
                    f"Dataset loading error: {e}"
                )

    if st.session_state.df_working is not None:
        st.divider()
        st.caption(
            f"Current dataset: {st.session_state.source_name}"
        )

        if st.button(
            "🔄 Reset to Original Dataset",
            use_container_width=True,
        ):
            st.session_state.df_working = (
                st.session_state.df_original.copy()
            )
            st.session_state.feature_engineering_log = []
            st.success("Dataset restored.")
            st.rerun()

        st.divider()
        st.subheader("📊 Quick Stats")

        df_sidebar = st.session_state.df_working

        st.metric(
            "Rows",
            f"{len(df_sidebar):,}",
        )
        st.metric(
            "Columns",
            f"{len(df_sidebar.columns):,}",
        )
        st.metric(
            "Missing",
            f"{int(df_sidebar.isnull().sum().sum()):,}",
        )
        st.metric(
            "Duplicates",
            f"{int(df_sidebar.duplicated().sum()):,}",
        )

# ============================================================
# NO DATA STATE
# ============================================================

if st.session_state.df_working is None:
    st.info(
        "👈 Upload a CSV/XLSX/XLS dataset from the sidebar and click "
        "**Load Dataset** to start."
    )

    st.markdown(
        """
### Dashboard workflow

**1. Upload → 2. Overview → 3. Statistics → 4. Cleaning → 5. EDA →  
6. Features → 7. Training → 8. Evaluation → 9. Comparison →  
10. Prediction → 11. Explainability**

This dashboard is designed for exploratory analytics and machine-learning
experimentation. It does **not** automatically diagnose a patient or replace
clinical judgment.
"""
    )

    st.stop()

# ============================================================
# GLOBAL DATA
# ============================================================

df = st.session_state.df_working.copy()
columns = df.columns.tolist()
num_cols = numeric_columns(df)
cat_cols = categorical_columns(df)

# ============================================================
# KPI ROW
# ============================================================

kpi_cols = st.columns(6)

kpi_data = [
    ("Rows", f"{len(df):,}"),
    ("Columns", f"{len(columns):,}"),
    ("Numeric", f"{len(num_cols):,}"),
    ("Categorical", f"{len(cat_cols):,}"),
    (
        "Missing",
        f"{int(df.isnull().sum().sum()):,}",
    ),
    (
        "Duplicates",
        f"{int(df.duplicated().sum()):,}",
    ),
]

for col, (title, value) in zip(
    kpi_cols,
    kpi_data,
):
    with col:
        st.markdown(
            f"""
<div class="kpi-card">
    <div class="kpi-title">{title}</div>
    <div class="kpi-value">{value}</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.write("")

# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "📊 Overview",
        "📈 Statistics",
        "🧹 Cleaning",
        "🔎 EDA",
        "⚙️ Features",
        "🤖 Training",
        "📊 Evaluation",
        "🏆 Comparison",
        "🎯 Prediction",
        "🧠 Explainability",
    ]
)

# ============================================================
# 1. OVERVIEW
# ============================================================

with tabs[0]:
    st.subheader("📊 Dataset Overview")

    st.caption(
        f"Source: {st.session_state.source_name}"
    )

    preview_rows = st.slider(
        "Preview rows",
        min_value=5,
        max_value=min(100, max(5, len(df))),
        value=min(10, max(5, len(df))),
        key="overview_preview_rows",
    )

    st.dataframe(
        df.head(preview_rows),
        use_container_width=True,
        height=360,
    )

    st.subheader("🔍 Data Types & Data Quality")

    dtype_table = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing": df.isnull().sum().values,
            "Missing %": (
                df.isnull().mean() * 100
            ).round(2).values,
            "Unique": df.nunique(
                dropna=True
            ).values,
        }
    )

    st.dataframe(
        dtype_table,
        use_container_width=True,
        height=380,
    )

    st.subheader("📦 Dataset Information")

    info_cols = st.columns(4)

    with info_cols[0]:
        st.metric(
            "Memory",
            f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB",
        )

    with info_cols[1]:
        st.metric(
            "Numeric Features",
            len(numeric_columns(df)),
        )

    with info_cols[2]:
        st.metric(
            "Categorical Features",
            len(categorical_columns(df)),
        )

    with info_cols[3]:
        st.metric(
            "Complete Rows",
            f"{int(df.notna().all(axis=1).sum()):,}",
        )

# ============================================================
# 2. STATISTICS
# ============================================================

with tabs[1]:
    st.subheader("📈 Descriptive Statistics")

    statistics = (
        df.describe(
            include="all"
        )
        .T
        .reset_index()
        .rename(
            columns={"index": "Column"}
        )
    )

    st.dataframe(
        statistics,
        use_container_width=True,
        height=450,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Missing Value Analysis")

        missing_table = pd.DataFrame(
            {
                "Column": df.columns,
                "Missing Count": df.isnull().sum().values,
                "Missing %": (
                    df.isnull().mean() * 100
                ).round(2).values,
            }
        ).sort_values(
            "Missing Count",
            ascending=False,
        )

        st.dataframe(
            missing_table,
            use_container_width=True,
            height=360,
        )

    with c2:
        st.subheader("Duplicate Analysis")

        duplicate_count = int(
            df.duplicated().sum()
        )

        duplicate_table = pd.DataFrame(
            {
                "Measure": [
                    "Total Rows",
                    "Duplicate Rows",
                    "Unique Rows",
                    "Duplicate %",
                ],
                "Value": [
                    len(df),
                    duplicate_count,
                    len(df) - duplicate_count,
                    round(
                        duplicate_count
                        / max(len(df), 1)
                        * 100,
                        2,
                    ),
                ],
            }
        )

        st.dataframe(
            duplicate_table,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("🔢 Numeric Distribution Summary")

    if num_cols:
        numeric_summary = df[num_cols].describe().T
        st.dataframe(
            numeric_summary,
            use_container_width=True,
        )
    else:
        st.info(
            "No numeric columns available."
        )

    st.subheader("⬇️ Download Statistics")

    stats_csv = statistics.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Descriptive Statistics CSV",
        data=stats_csv,
        file_name="descriptive_statistics.csv",
        mime="text/csv",
    )

# ============================================================
# 3. CLEANING
# ============================================================

with tabs[2]:
    st.subheader("🧹 Advanced Data Cleaning")

    st.caption(
        "Choose one treatment from each category. "
        "The cleaned dataset replaces the current working dataset."
    )

    c1, c2 = st.columns(2)

    with c1:
        missing_method = st.selectbox(
            "Missing Value Treatment",
            [
                "Do Nothing",
                "Drop rows",
                "Drop columns >50% missing",
                "Mean imputation",
                "Median imputation",
                "Mode imputation",
                "Forward fill",
                "Backward fill",
                "Fill numeric 0",
            ],
            key="missing_method",
        )

        duplicate_method = st.selectbox(
            "Duplicate Treatment",
            [
                "Do Nothing",
                "Remove duplicates",
            ],
            key="duplicate_method",
        )

        outlier_method = st.selectbox(
            "Outlier Treatment",
            [
                "Do Nothing",
                "Remove IQR outliers",
                "Cap IQR outliers",
                "Remove Z-score >3",
            ],
            key="outlier_method",
        )

    with c2:
        numeric_method = st.selectbox(
            "Numeric Cleaning",
            [
                "Do Nothing",
                "Convert numeric-looking text",
                "Round numeric values",
            ],
            key="numeric_method",
        )

        categorical_method = st.selectbox(
            "Categorical Cleaning",
            [
                "Do Nothing",
                "Remove extra spaces",
                "Lowercase",
                "Uppercase",
            ],
            key="categorical_method",
        )

        invalid_method = st.selectbox(
            "Invalid Value Treatment",
            [
                "Do Nothing",
                "Replace negative numeric values",
            ],
            key="invalid_method",
        )

    clean_col1, clean_col2 = st.columns(2)

    with clean_col1:
        apply_clean_button = st.button(
            "🧹 Apply Cleaning",
            type="primary",
            use_container_width=True,
        )

    with clean_col2:
        reset_clean_button = st.button(
            "🔄 Restore Original",
            use_container_width=True,
        )

    if reset_clean_button:
        st.session_state.df_working = (
            st.session_state.df_original.copy()
        )
        st.session_state.feature_engineering_log = []
        st.success(
            "Original dataset restored."
        )
        st.rerun()

    if apply_clean_button:
        try:
            cleaned, summary = apply_cleaning(
                df,
                missing_method,
                duplicate_method,
                outlier_method,
                numeric_method,
                categorical_method,
                invalid_method,
            )

            st.session_state.df_working = cleaned
            st.session_state.last_cleaning_summary = summary

            # A cleaning change invalidates an old model.
            st.session_state.pipeline = None
            st.session_state.model = None
            st.session_state.metrics_df = None
            st.session_state.comparison_df = None
            st.session_state.importance_df = None

            st.success(
                "Cleaning completed successfully."
            )

            st.rerun()

        except Exception as e:
            st.error(
                f"Cleaning error: {e}"
            )

    if st.session_state.last_cleaning_summary is not None:
        st.subheader("📋 Cleaning Before vs After")

        st.dataframe(
            st.session_state.last_cleaning_summary,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("👀 Current Dataset Preview")

    st.dataframe(
        st.session_state.df_working.head(25),
        use_container_width=True,
        height=400,
    )

    st.subheader("🔁 Original vs Current")

    compare_rows = min(
        10,
        len(st.session_state.df_original),
        len(st.session_state.df_working),
    )

    before_after_cols = st.columns(2)

    with before_after_cols[0]:
        st.markdown("**Original**")
        st.dataframe(
            st.session_state.df_original.head(
                compare_rows
            ),
            use_container_width=True,
        )

    with before_after_cols[1]:
        st.markdown("**Current / Cleaned**")
        st.dataframe(
            st.session_state.df_working.head(
                compare_rows
            ),
            use_container_width=True,
        )

    st.subheader("⬇️ Download Cleaned Dataset")

    cleaned_csv = st.session_state.df_working.to_csv(
        index=False
    ).encode("utf-8")

    cleaned_xlsx = excel_download(
        st.session_state.df_working,
        "cleaned_dataset.xlsx",
    )

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "Download CSV",
            data=cleaned_csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "Download Excel",
            data=cleaned_xlsx,
            file_name="cleaned_dataset.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

# ============================================================
# 4. EDA
# ============================================================

with tabs[3]:
    st.subheader("🔎 Exploratory Data Analysis")

    graph_types = [
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
        "Q-Q Plot",
    ]

    g1, g2, g3, g4 = st.columns(4)

    with g1:
        graph_type = st.selectbox(
            "📊 Graph Type",
            graph_types,
            key="graph_type",
        )

    with g2:
        x_column = st.selectbox(
            "X / Numeric Column",
            columns,
            key="x_column",
        )

    with g3:
        y_options = ["None"] + columns
        y_column = st.selectbox(
            "Y Column",
            y_options,
            key="y_column",
        )

    with g4:
        hue_column = st.selectbox(
            "Group / Hue",
            ["None"] + columns,
            key="hue_column",
        )

    if st.button(
        "📊 Generate Visualization",
        type="primary",
    ):
        fig = create_graph(
            df,
            graph_type,
            x_column,
            y_column,
            hue_column,
        )
        st.pyplot(
            fig,
            use_container_width=True,
        )
        plt.close(fig)

    st.divider()

    st.subheader("🎯 Target Analysis")

    target_col = st.selectbox(
        "Target Column",
        columns,
        key="eda_target_column",
    )

    if st.button(
        "Analyze Target",
        key="analyze_target",
    ):
        try:
            target_table, target_fig = target_analysis(
                df,
                target_col,
            )

            a1, a2 = st.columns(2)

            with a1:
                st.dataframe(
                    target_table,
                    use_container_width=True,
                    height=360,
                )

            with a2:
                st.pyplot(
                    target_fig,
                    use_container_width=True,
                )
                plt.close(target_fig)

        except Exception as e:
            st.error(
                f"Target analysis error: {e}"
            )

    st.divider()

    st.subheader("📌 Automatic EDA Checks")

    auto1, auto2, auto3 = st.columns(3)

    with auto1:
        st.metric(
            "Highly Missing Columns",
            int(
                (
                    df.isnull().mean() > 0.50
                ).sum()
            ),
        )

    with auto2:
        st.metric(
            "Constant Columns",
            int(
                (
                    df.nunique(dropna=False) <= 1
                ).sum()
            ),
        )

    with auto3:
        st.metric(
            "Numeric Columns",
            len(num_cols),
        )

# ============================================================
# 5. FEATURES
# ============================================================

with tabs[4]:
    st.subheader("⚙️ Feature Engineering")

    st.info(
        "Feature engineering changes the current working dataset. "
        "Use Reset to Original in the Cleaning tab if you want to undo "
        "all cleaning/feature changes."
    )

    feature_target = st.selectbox(
        "Target / Reference Column",
        columns,
        key="feature_target",
    )

    feature_method = st.selectbox(
        "Feature Method",
        [
            "Select method",
            "Drop selected column",
            "Create log transformation",
            "Create squared feature",
            "Create standardized feature",
            "Create missing indicator",
        ],
        key="feature_method",
    )

    new_feature_name = st.text_input(
        "New Feature Name",
        value="",
        help="Required for transformations that create a new column.",
    )

    if st.button(
        "⚙️ Apply Feature Engineering",
        type="primary",
    ):
        work = st.session_state.df_working.copy()

        try:
            if feature_method == "Select method":
                st.warning(
                    "Select a feature engineering method."
                )

            elif feature_method == "Drop selected column":
                work = work.drop(
                    columns=[feature_target]
                )

                st.session_state.feature_engineering_log.append(
                    f"Dropped column: {feature_target}"
                )

                st.session_state.df_working = work
                st.success(
                    f"Dropped: {feature_target}"
                )
                st.rerun()

            else:
                if not new_feature_name.strip():
                    raise ValueError(
                        "Enter a New Feature Name."
                    )

                name = new_feature_name.strip()

                if name in work.columns:
                    raise ValueError(
                        f"Feature '{name}' already exists."
                    )

                if not pd.api.types.is_numeric_dtype(
                    work[feature_target]
                ):
                    raise ValueError(
                        "Selected column must be numeric "
                        "for this transformation."
                    )

                if feature_method == "Create log transformation":
                    values = pd.to_numeric(
                        work[feature_target],
                        errors="coerce",
                    )
                    work[name] = np.log1p(
                        values.clip(lower=0)
                    )

                elif feature_method == "Create squared feature":
                    values = pd.to_numeric(
                        work[feature_target],
                        errors="coerce",
                    )
                    work[name] = values ** 2

                elif feature_method == "Create standardized feature":
                    values = pd.to_numeric(
                        work[feature_target],
                        errors="coerce",
                    )
                    mean = values.mean()
                    std = values.std()

                    if pd.isna(std) or std == 0:
                        raise ValueError(
                            "Standardization is not possible "
                            "because the column has zero variance."
                        )

                    work[name] = (
                        values - mean
                    ) / std

                elif feature_method == "Create missing indicator":
                    work[name] = (
                        work[feature_target]
                        .isna()
                        .astype(int)
                    )

                st.session_state.df_working = work
                st.session_state.feature_engineering_log.append(
                    f"{feature_method}: {feature_target} → {name}"
                )

                st.success(
                    f"Created feature: {name}"
                )
                st.rerun()

        except Exception as e:
            st.error(
                f"Feature engineering error: {e}"
            )

    st.subheader("📝 Feature Engineering History")

    if st.session_state.feature_engineering_log:
        for i, item in enumerate(
            st.session_state.feature_engineering_log,
            1,
        ):
            st.write(
                f"{i}. {item}"
            )
    else:
        st.caption(
            "No feature engineering operations yet."
        )

    st.subheader("🔍 Current Feature List")

    feature_info = pd.DataFrame(
        {
            "Feature": st.session_state.df_working.columns,
            "Data Type": st.session_state.df_working.dtypes.astype(
                str
            ).values,
            "Missing": st.session_state.df_working.isnull()
            .sum()
            .values,
            "Unique": st.session_state.df_working.nunique(
                dropna=True
            )
            .values,
        }
    )

    st.dataframe(
        feature_info,
        use_container_width=True,
        height=400,
    )

# ============================================================
# 6. TRAINING
# ============================================================

with tabs[5]:
    st.subheader("🤖 Machine Learning Training")

    training_target = st.selectbox(
        "🎯 Target Column",
        columns,
        key="training_target",
    )

    task_type = st.radio(
        "ML Task",
        [
            "Classification",
            "Regression",
        ],
        horizontal=True,
        key="task_type",
    )

    available_models = list(
        get_models(task_type).keys()
    )

    model_name = st.selectbox(
        "🤖 Model",
        available_models,
        key="model_name",
    )

    c1, c2 = st.columns(2)

    with c1:
        scaling_method = st.selectbox(
            "Scaling",
            [
                "StandardScaler",
                "MinMaxScaler",
                "No Scaling",
            ],
            key="training_scaling",
        )

    with c2:
        test_size = st.slider(
            "Test Size",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05,
            key="training_test_size",
        )

    st.caption(
        "Preprocessing automatically handles numeric imputation, "
        "categorical imputation, one-hot encoding and optional scaling."
    )

    if st.button(
        "🚀 Train Model",
        type="primary",
        use_container_width=True,
    ):
        try:
            result = train_model(
                df,
                training_target,
                task_type,
                model_name,
                test_size,
                scaling_method,
            )

            st.session_state.pipeline = result["pipeline"]
            st.session_state.model = result["model"]
            st.session_state.target = training_target
            st.session_state.task = task_type
            st.session_state.feature_columns = result[
                "feature_columns"
            ]
            st.session_state.X_test = result["X_test"]
            st.session_state.y_test = result["y_test"]
            st.session_state.y_pred = result["y_pred"]
            st.session_state.metrics_df = result["metrics"]
            st.session_state.evaluation_fig_1 = result["fig1"]
            st.session_state.evaluation_fig_2 = result["fig2"]
            st.session_state.prediction_template = result[
                "prediction_template"
            ]
            st.session_state.importance_df = get_feature_importance()

            st.success(
                f"Model trained: {result['model_name']}"
            )

        except Exception as e:
            st.error(
                f"Training error: {e}"
            )

    if st.session_state.metrics_df is not None:
        st.divider()

        st.subheader("📋 Training Result")

        summary_cols = st.columns(5)

        with summary_cols[0]:
            st.metric(
                "Model",
                st.session_state.model.__class__.__name__,
            )

        with summary_cols[1]:
            st.metric(
                "Task",
                st.session_state.task,
            )

        with summary_cols[2]:
            st.metric(
                "Target",
                str(st.session_state.target),
            )

        with summary_cols[3]:
            st.metric(
                "Training Rows",
                len(
                    st.session_state.X_test
                )
                if st.session_state.X_test is not None
                else 0,
            )

        with summary_cols[4]:
            st.metric(
                "Features",
                len(
                    st.session_state.feature_columns
                ),
            )

        st.subheader("Model Performance")

        st.dataframe(
            st.session_state.metrics_df.style.format(
                {"Score": "{:.4f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.success(
            "Training completed. Continue to Evaluation, "
            "Prediction and Explainability."
        )

# ============================================================
# 7. EVALUATION
# ============================================================

with tabs[6]:
    st.subheader("📊 Model Evaluation")

    if st.session_state.pipeline is None:
        st.info(
            "Train a model first from the Training tab."
        )
    else:
        st.write(
            f"**Model:** {st.session_state.model.__class__.__name__}"
        )
        st.write(
            f"**Task:** {st.session_state.task}"
        )
        st.write(
            f"**Target:** {st.session_state.target}"
        )

        e1, e2 = st.columns(2)

        with e1:
            st.markdown(
                "### Confusion Matrix / Actual vs Predicted"
            )
            if st.session_state.evaluation_fig_1 is not None:
                st.pyplot(
                    st.session_state.evaluation_fig_1,
                    use_container_width=True,
                )

        with e2:
            st.markdown(
                "### ROC Curve / Residual Plot"
            )
            if st.session_state.evaluation_fig_2 is not None:
                st.pyplot(
                    st.session_state.evaluation_fig_2,
                    use_container_width=True,
                )

        if st.session_state.task == "Classification":
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred

            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0,
            )

            report_df = (
                pd.DataFrame(report)
                .T
                .reset_index()
                .rename(
                    columns={"index": "Class"}
                )
            )

            st.dataframe(
                report_df,
                use_container_width=True,
            )

        else:
            st.subheader("Regression Diagnostics")

            y_test = np.asarray(
                st.session_state.y_test
            )
            y_pred = np.asarray(
                st.session_state.y_pred
            )

            residuals = y_test - y_pred

            diag = pd.DataFrame(
                {
                    "Actual": y_test,
                    "Predicted": y_pred,
                    "Residual": residuals,
                    "Absolute Error": np.abs(
                        residuals
                    ),
                }
            )

            st.dataframe(
                diag.head(100),
                use_container_width=True,
                height=350,
            )

# ============================================================
# 8. COMPARISON
# ============================================================

with tabs[7]:
    st.subheader("🏆 Model Comparison")

    comparison_target = st.selectbox(
        "Target",
        columns,
        key="comparison_target",
    )

    comparison_task = st.radio(
        "Task",
        [
            "Classification",
            "Regression",
        ],
        horizontal=True,
        key="comparison_task",
    )

    c1, c2 = st.columns(2)

    with c1:
        comparison_scaling = st.selectbox(
            "Scaling",
            [
                "StandardScaler",
                "MinMaxScaler",
                "No Scaling",
            ],
            key="comparison_scaling",
        )

    with c2:
        comparison_test_size = st.slider(
            "Test Size",
            0.10,
            0.40,
            0.20,
            0.05,
            key="comparison_test_size",
        )

    if st.button(
        "🏆 Compare All Models",
        type="primary",
    ):
        try:
            result_df = compare_models(
                df,
                comparison_target,
                comparison_task,
                comparison_test_size,
                comparison_scaling,
            )

            st.session_state.comparison_df = result_df

        except Exception as e:
            st.error(
                f"Comparison error: {e}"
            )

    if st.session_state.comparison_df is not None:
        result_df = st.session_state.comparison_df

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
        )

        successful = result_df[
            result_df["Status"]
            .astype(str)
            .str.startswith("Success")
        ]

        if not successful.empty:
            if comparison_task == "Classification":
                best = successful.iloc[0]
                st.success(
                    f"🏆 Best model by Accuracy: "
                    f"{best['Model']} — "
                    f"{best['Accuracy']:.4f}"
                )
            else:
                best = successful.iloc[0]
                st.success(
                    f"🏆 Best model by R²: "
                    f"{best['Model']} — "
                    f"{best['R²']:.4f}"
                )

        comparison_csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Model Comparison",
            data=comparison_csv,
            file_name="model_comparison.csv",
            mime="text/csv",
        )

# ============================================================
# 9. PREDICTION
# ============================================================

with tabs[8]:
    st.subheader("🎯 New Record / Patient Prediction")

    if st.session_state.pipeline is None:
        st.info(
            "Train a model first."
        )
    else:
        st.markdown(
            """
Enter one value for each feature used during training.
For categorical variables, enter a category present in the training data.
"""
        )

        features = st.session_state.feature_columns

        # Use a form so prediction only runs when requested.
        with st.form("prediction_form"):
            prediction_values = {}

            # Keep the UI compact by splitting features into columns.
            display_cols = st.columns(2)

            for i, feature in enumerate(features):
                with display_cols[i % 2]:
                    prediction_values[feature] = st.text_input(
                        feature,
                        value="",
                        key=f"prediction_{feature}",
                    )

            predict_submit = st.form_submit_button(
                "🎯 Predict",
                type="primary",
                use_container_width=True,
            )

        if predict_submit:
            try:
                table = pd.DataFrame(
                    {
                        "Feature": list(
                            prediction_values.keys()
                        ),
                        "Value": list(
                            prediction_values.values()
                        ),
                    }
                )

                prediction_df, prediction = (
                    make_prediction_from_table(
                        table
                    )
                )

                prediction_value = prediction[0]

                st.success(
                    f"Prediction: {prediction_value}"
                )

                st.subheader(
                    "Input Row Sent to Model"
                )

                st.dataframe(
                    prediction_df,
                    use_container_width=True,
                    hide_index=True,
                )

                if st.session_state.task == "Classification":
                    try:
                        probabilities = (
                            st.session_state.pipeline
                            .predict_proba(
                                prediction_df
                            )[0]
                        )

                        classes = (
                            st.session_state.pipeline
                            .named_steps["model"]
                            .classes_
                        )

                        probability_df = pd.DataFrame(
                            {
                                "Class": classes,
                                "Probability %": (
                                    probabilities * 100
                                ).round(2),
                            }
                        )

                        st.subheader(
                            "Class Probabilities"
                        )

                        st.dataframe(
                            probability_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                    except Exception:
                        st.caption(
                            "Probability output is not available "
                            "for this model."
                        )

            except Exception as e:
                st.error(
                    f"Prediction error: {e}"
                )

# ============================================================
# 10. EXPLAINABILITY
# ============================================================

with tabs[9]:
    st.subheader("🧠 Model Explainability")

    if st.session_state.pipeline is None:
        st.info(
            "Train a model first."
        )
    else:
        st.markdown(
            """
Feature importance shows which transformed variables contribute most
strongly to supported trained models. For linear models, absolute
coefficient magnitude is used; for tree-based models, model feature
importance is used.
"""
        )

        importance_df = get_feature_importance()

        if importance_df.empty:
            st.warning(
                "Feature importance is not available for this model."
            )
        else:
            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True,
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            plot_df = importance_df.sort_values(
                "Importance",
                ascending=True,
            ).tail(20)

            ax.barh(
                plot_df["Feature"].astype(str),
                plot_df["Importance"],
            )

            ax.set_xlabel(
                "Importance"
            )
            ax.set_title(
                "Top Feature Importance",
                fontweight="bold",
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True,
            )

            plt.close(fig)

            importance_csv = importance_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Feature Importance",
                data=importance_csv,
                file_name="feature_importance.csv",
                mime="text/csv",
            )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧠 AI & ML Analytics Dashboard • "
    "Streamlit • Pandas • NumPy • SciPy • Scikit-learn • "
    "Matplotlib • Seaborn • OpenPyXL"
)

st.caption(
    "For analytics and ML experimentation only. "
    "Healthcare predictions require appropriate validation, governance, "
    "privacy controls, and qualified clinical oversight."
)

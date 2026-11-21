"""
pipeline.py
===========
The PCOS-Prediction-Classifier notebook, ported into importable functions.

Rules followed while porting
----------------------------
* Every number is untouched: thresholds, seeds, split ratios, hyper-parameter
  grids, CV folds, figure sizes, colour palettes and plotting calls are exactly
  those of the notebook.
* Only the *plumbing* differs: functions return data / PNG bytes instead of
  calling ``plt.show()`` / ``print()``, and a few lines were adapted so they run
  on current library versions (``np.bool`` -> ``bool``, ``np.percentile``'s
  ``interpolation=`` -> ``method=``, chained ``fillna(inplace=True)``).
* ``sns.set(font_scale=0.6)`` was called mid-way through the notebook and then
  stayed active for every later plot.  ``plot_ctx(styled=True)`` re-creates that
  state for those plots (and restores matplotlib afterwards), so every figure
  looks the same as it does in the notebook.
"""
from __future__ import annotations

import io
import threading
import warnings
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from imblearn.combine import SMOTEENN  # noqa: E402
from imblearn.over_sampling import ADASYN, SMOTE  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    AdaBoostClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import RFE  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

warnings.filterwarnings("ignore")

TARGET = "PCOS (Y/N)"
_PLOT_LOCK = threading.RLock()  # matplotlib's pyplot state is global


# --------------------------------------------------------------------------- #
# Plot plumbing
# --------------------------------------------------------------------------- #
@contextmanager
def plot_ctx(styled: bool):
    """Thread-safe matplotlib context.

    ``styled=True`` reproduces the notebook state *after* ``sns.set(font_scale=0.6)``
    had been executed (cell 37 onwards).  rcParams are restored on exit.
    """
    with _PLOT_LOCK, plt.rc_context():
        if styled:
            sns.set(font_scale=0.6)
        yield


def _to_png(fig, dpi: int = 130) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render(builder, styled: bool) -> bytes:
    """Run a notebook-style plotting function and return the figure as PNG bytes."""
    with plot_ctx(styled):
        return _to_png(builder())


# --------------------------------------------------------------------------- #
# Utility functions (notebook cell 8)
# --------------------------------------------------------------------------- #
def printdf(data) -> pd.DataFrame:
    """for returing readable dataframes"""
    return pd.DataFrame(data)


def plot_missing_vals(data: pd.DataFrame):
    """for plotting missing values"""
    fig = plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    sns.heatmap(data.isnull(), cbar=True, vmin=0, vmax=1)
    plt.title("Missing Values Plot")
    return fig


def remove_outliers(data, feature):
    try:
        Q1 = np.percentile(data[feature], 25, method="midpoint")
        Q3 = np.percentile(data[feature], 75, method="midpoint")
    except TypeError:  # very old NumPy
        Q1 = np.percentile(data[feature], 25, interpolation="midpoint")
        Q3 = np.percentile(data[feature], 75, interpolation="midpoint")
    IQR = Q3 - Q1
    upper = data[feature] >= (Q3 + 1.5 * IQR)
    lower = data[feature] <= (Q1 - 1.5 * IQR)

    return data[~(upper | lower)]


def info_frame(df: pd.DataFrame) -> pd.DataFrame:
    """A tidy version of ``df.info()``."""
    return pd.DataFrame(
        {
            "Column": [str(c).strip() for c in df.columns],
            "Non-Null": df.notna().sum().to_numpy(),
            "Dtype": df.dtypes.astype(str).to_numpy(),
        }
    )


# --------------------------------------------------------------------------- #
# Stage 1 - loading, cleaning, EDA, preprocessing, resampling
# --------------------------------------------------------------------------- #
# Canonical column names (exactly as the notebook sees them after cleaning)
CANONICAL_COLUMNS = [
    "Sl. No", "Patient File No.", "PCOS (Y/N)", "Age (yrs)", "Weight (Kg)", "Height(Cm)", "BMI",
    "Blood Group", "Pulse rate(bpm)", "RR (breaths/min)", "Hb(g/dl)", "Cycle(R/I)", "Cycle length(days)",
    "Marraige Status (Yrs)", "Pregnant(Y/N)", "No. of aborptions", "I   beta-HCG(mIU/mL)",
    "II    beta-HCG(mIU/mL)", "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH", "Hip(inch)", "Waist(inch)",
    "Waist:Hip Ratio", "TSH (mIU/L)", "AMH(ng/mL)", "PRL(ng/mL)", "Vit D3 (ng/mL)", "PRG(ng/mL)",
    "RBS(mg/dl)", "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)", "Hair loss(Y/N)",
    "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)", "BP _Systolic (mmHg)", "BP _Diastolic (mmHg)",
    "Follicle No. (L)", "Follicle No. (R)", "Avg. F size (L) (mm)", "Avg. F size (R) (mm)", "Endometrium (mm)",
]
_IDENTIFIERS = {"Sl. No", "Patient File No."}


def _key(name) -> str:
    """Loose comparison key: lower-case, no whitespace, no underscores."""
    return "".join(str(name).lower().split()).replace("_", "")


_ALIASES = {  # spelling variants seen across versions of the Kaggle file
    _key("Marriage Status (Yrs)"): "Marraige Status (Yrs)",
    _key("Marraige Status(Yrs)"): "Marraige Status (Yrs)",
    _key("No. of abortions"): "No. of aborptions",
    _key("PCOS(Y/N)"): "PCOS (Y/N)",
}
_CANON_BY_KEY = {_key(c): c for c in CANONICAL_COLUMNS} | _ALIASES


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns that match a notebook column (ignoring spaces / case / known typos)."""
    mapping = {}
    for c in df.columns:
        canon = _CANON_BY_KEY.get(_key(c))
        if canon is not None:
            mapping[c] = canon
    return df.rename(columns=mapping)


def validate_columns(df: pd.DataFrame) -> None:
    import difflib

    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns and c not in _IDENTIFIERS]
    if missing:
        found = [str(c).strip() for c in df.columns]
        hints = []
        for m in missing:
            close = difflib.get_close_matches(m, found, n=1, cutoff=0.6)
            hints.append(f"{m!r}" + (f" (closest in your file: {close[0]!r})" if close else ""))
        raise ValueError(
            "The file does not look like the PCOS workbook. Missing columns: " + "; ".join(hints)
            + f". Columns found ({len(found)}): " + ", ".join(found)
        )


def read_dataset(source) -> pd.DataFrame:
    """Cell 11: ``pd.read_excel(..., sheet_name=1)``.

    The notebook reads the second sheet.  To be forgiving we pick the first sheet that actually
    contains the target column (this is the second sheet in the standard Kaggle file) and fall back
    to sheet 1.  Column names are then matched loosely (see ``standardize_columns``).
    """
    xl = pd.ExcelFile(source)
    chosen = None
    for i, name in enumerate(xl.sheet_names):
        head = xl.parse(name, nrows=2)
        if any(_key(c) == _key(TARGET) for c in head.columns):
            chosen = i
            break
    if chosen is None:
        if len(xl.sheet_names) < 2:
            raise ValueError(f"No sheet contains a '{TARGET}' column (sheets found: {xl.sheet_names}).")
        chosen = 1
    df = standardize_columns(xl.parse(xl.sheet_names[chosen]))
    validate_columns(df)
    return df


def prepare_data(raw: pd.DataFrame) -> dict:
    out: dict = {}
    data = raw.copy()

    # ---- cell 14: remove the unnamed column --------------------------------
    data.drop(
        data.columns[data.columns.str.contains("Unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    out["after_unnamed"] = data.copy()
    out["dtypes_before"] = info_frame(data)

    # ---- cell 19: drop Sl. No and Patient File No. (the first two columns) --
    ids = [c for c in data.columns if c in _IDENTIFIERS]
    data.drop(ids if len(ids) == 2 else data.columns[:2], axis=1, inplace=True)

    # ---- cell 21: object -> numeric ----------------------------------------
    converted = []
    for feature in [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])]:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
        converted.append(str(feature).strip())
    data = data.rename(columns=lambda x: x.strip())
    out["converted_columns"] = converted
    out["dtypes_after"] = info_frame(data)

    # ---- cells 24 / 26: missing values -------------------------------------
    nulls = data.isnull().sum()
    out["null_counts"] = nulls[nulls > 0]
    out["missing_before_png"] = render(lambda: plot_missing_vals(data), styled=False)

    fills = []
    for col, how in [
        ("Marraige Status (Yrs)", "mean"),
        ("Fast food (Y/N)", "mode"),
        ("II    beta-HCG(mIU/mL)", "mean"),
        ("AMH(ng/mL)", "mean"),
    ]:
        value = data[col].mean() if how == "mean" else data[col].mode()[0]
        n_missing = int(data[col].isnull().sum())
        data[col] = data[col].fillna(value)
        fills.append({"Column": col, "Strategy": how, "Fill value": float(value), "Rows filled": n_missing})
    out["fills"] = pd.DataFrame(fills)
    out["missing_after_png"] = render(lambda: plot_missing_vals(data), styled=False)

    out["describe"] = data.describe()
    out["data_filled"] = data.copy()

    # ---- cells 29 / 30: target, features, binary features ------------------
    target = TARGET
    all_features = list(data.columns[1:])
    features_df = data[all_features]
    binary_features = features_df.columns[(features_df.max() == 1) & (features_df.min() == 0)]
    out["binary_features"] = list(binary_features)
    out["class_counts_raw"] = {int(k): int(v) for k, v in data[target].value_counts().items()}

    # ---- cells 35 / 37: correlation matrix + heatmap -----------------------
    corr_matrix = data.corr().abs()
    out["corr_with_target"] = corr_matrix[target].sort_values()

    def _corr_heatmap():
        fig = plt.figure(figsize=(10, 8))
        sns.set(font_scale=0.6)
        indexes = corr_matrix.nlargest(17, target)[target].index
        cm = pd.DataFrame(np.corrcoef(data[indexes].values.T))
        sns.heatmap(
            data=cm,
            annot=True,
            yticklabels=indexes.values,
            xticklabels=indexes.values,
            cbar=True,
            cmap="YlGnBu",
            square=True,
        )
        plt.title(f"correlation heatmap with {target}")
        return fig

    out["corr_heatmap_png"] = render(_corr_heatmap, styled=False)  # sns.set happens inside, as in the notebook

    # ---- cell 39: highly-correlated feature pairs --------------------------
    correlation_threshold = 0.8  # threshold = 80%
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated_pairs = []
    for column in upper_triangle.columns:
        correlated_columns = upper_triangle.index[upper_triangle[column] >= correlation_threshold].tolist()
        for correlated_column in correlated_columns:
            highly_correlated_pairs.append((column, correlated_column))
    out["correlation_threshold"] = correlation_threshold
    out["highly_correlated_pairs"] = highly_correlated_pairs

    # ---- cell 41: drop redundant features ----------------------------------
    data.drop(["Weight (Kg)"], axis=1, inplace=True)
    data.drop(["Hip(inch)"], axis=1, inplace=True)

    # ---- cell 43: split binary / non-binary --------------------------------
    all_features = list(data.columns[1:])
    features_df = data[all_features]
    binary_features = features_df.columns[(features_df.max() == 1) & (features_df.min() == 0)]
    nonbinary_data = features_df.drop(binary_features, axis=1)
    out["pre_outlier"] = data.copy()
    out["nonbinary"] = nonbinary_data.columns.to_list()

    # ---- cell 45: selective outlier removal --------------------------------
    old_shape = data.shape
    features_to_filter = [
        "BP _Systolic (mmHg)",
        "Pulse rate(bpm)",
        "Waist:Hip Ratio",
        "BP _Systolic (mmHg)",
        "BP _Diastolic (mmHg)",
    ]
    log = []
    for feature in features_to_filter:
        data = remove_outliers(data, feature=feature)
        log.append({"Feature filtered": feature, "Rows remaining": int(data[feature].shape[0])})
    out["outlier_old_shape"] = old_shape
    out["outlier_log"] = pd.DataFrame(log)
    out["data"] = data  # post-outlier data (used for box/strip + violin plots)

    # ---- cell 47: MinMax scaling -------------------------------------------
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data)
    scaled_data.columns = data.columns
    out["scaled_data"] = scaled_data
    out["all_features"] = all_features
    out["class_counts_filtered"] = {int(k): int(v) for k, v in data[target].value_counts().items()}

    # ---- cells 52-54: resampling (ADASYN, SMOTE, SMOTE-ENN) ----------------
    X = scaled_data.drop("PCOS (Y/N)", axis=1)
    y = scaled_data["PCOS (Y/N)"]

    def _counts(s):
        return {int(k): int(v) for k, v in s.value_counts().sort_index().items()}

    resampling = {}

    # ADASYN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    resampling["ADASYN"] = (_counts(y_train), _counts(y_resampled))

    # SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    resampling["SMOTE"] = (_counts(y_train), _counts(y_resampled))

    # SMOTE-ENN  (the one carried forward in the notebook)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    resampling["SMOTE-ENN"] = (_counts(y_train), _counts(y_resampled))

    out["resampling"] = resampling
    out["X_resampled"] = X_resampled
    out["y_resampled"] = y_resampled
    out["n_train"] = len(X_train)
    out["n_test"] = len(X_test)
    return out


# --------------------------------------------------------------------------- #
# Per-feature plots (cells 32, 43, 48, 57, 58) - code is unchanged
# --------------------------------------------------------------------------- #
def feature_plot(kind: str, prep: dict, feature: str) -> bytes:
    target = TARGET

    if kind == "count":  # cell 32  (before sns.set -> default style)

        def build():
            fig = plt.figure(figsize=(5, 5))
            sns.countplot(data=prep["data_filled"], x=feature, hue=target)
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.legend()
            plt.title(f"Countplot for {feature}")
            return fig

        return render(build, styled=False)

    if kind == "box_raw":  # cell 43

        def build():
            fig = plt.figure(figsize=(5, 3))
            sns.boxplot(data=prep["pre_outlier"], x=target, y=feature, hue=target)
            plt.xlabel(target)
            plt.ylabel(feature)
            plt.title(f"Boxplot for {feature}")
            return fig

    elif kind == "box_scaled":  # cell 48

        def build():
            fig = plt.figure(figsize=(5, 3))
            sns.boxplot(data=prep["scaled_data"], x=target, y=feature, hue=target, palette="colorblind")
            plt.xlabel(target)
            plt.ylabel(feature)
            plt.title(f"Boxplot for {feature}")
            return fig

    elif kind == "box_strip":  # cell 57

        def build():
            fig = plt.figure(figsize=(5, 3))
            data = prep["data"]
            sns.boxplot(data=data, x=target, y=feature, hue=target)
            sns.stripplot(data=data, y=feature, x=target, jitter=True, size=3, hue=target)
            plt.xlabel(target)
            plt.ylabel(feature)
            plt.title(f"Boxplot and Stripplot for {feature}")
            return fig

    elif kind == "violin":  # cell 58

        def build():
            fig = plt.figure(figsize=(5, 3))
            data = prep["data"]
            sns.violinplot(data=data, x=target, y=feature, hue=target)
            sns.stripplot(data=data, y=feature, x=target, jitter=True, size=3, hue=target, palette="pastel")
            plt.xlabel(target)
            plt.ylabel(feature)
            plt.title(f"Violinplot and Stripplot for {feature}")
            return fig

    else:
        raise ValueError(kind)

    return render(build, styled=True)  # all of these ran after sns.set(font_scale=0.6)


# --------------------------------------------------------------------------- #
# Stage 2 - split + feature selection (cells 62-70)
# --------------------------------------------------------------------------- #
def run_feature_selection(prep: dict) -> dict:
    scaled_data = prep["scaled_data"]
    all_features = prep["all_features"]
    X_resampled, y_resampled = prep["X_resampled"], prep["y_resampled"]

    # cell 62: splitting the data  (no seed in the notebook -> none here)
    features_df = scaled_data[all_features]
    x_train_rfe, x_test_rfe, y_train_rfe, y_test_rfe = train_test_split(
        X_resampled, y_resampled, stratify=y_resampled, test_size=0.20
    )

    # cell 64: Feature selection using RFE
    model_lr = LogisticRegression(solver="liblinear", random_state=0)
    model_rfc = RandomForestClassifier()
    rfe_lr = RFE(model_lr, n_features_to_select=15)
    rfe_rfc = RFE(model_rfc, n_features_to_select=15)

    # Logistic regression
    x_train_rfe_lr = rfe_lr.fit_transform(x_train_rfe, y_train_rfe)
    features_lr_indices = [i for i, value in enumerate(rfe_lr.support_) if value]
    features_lr = features_df.columns[features_lr_indices]
    x_train_rfe_lr = printdf(x_train_rfe_lr)
    x_train_rfe_lr.columns = features_lr

    def _rank_lr():
        fig = plt.figure()
        sns.barplot(x=rfe_lr.ranking_, y=features_df.columns)
        plt.title("15 Logistic Regression - Feature Ranking")
        plt.ylabel("feature")
        return fig

    rank_lr_png = render(_rank_lr, styled=True)

    # random forest classifier
    x_train_rfe_rfc = rfe_rfc.fit_transform(x_train_rfe, y_train_rfe)
    features_rfc_indices = [i for i, value in enumerate(rfe_rfc.support_) if value]
    features_rfc = features_df.columns[features_rfc_indices]
    x_train_rfe_rfc = printdf(x_train_rfe_rfc)
    x_train_rfe_rfc.columns = features_rfc

    def _rank_rfc():
        fig = plt.figure()
        sns.barplot(x=rfe_rfc.ranking_, y=features_df.columns)
        plt.title("Random Forest Classifier - Feature Ranking")
        plt.ylabel("feature")
        return fig

    rank_rfc_png = render(_rank_rfc, styled=True)

    # cell 66: Feature contribution scores - Logistic Regression
    lr_model = rfe_lr.estimator_
    lr_model.fit(x_train_rfe_lr, y_train_rfe)
    lr_scores = lr_model.coef_[0]

    def _score_lr():
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=lr_scores, y=features_lr)
        plt.title("Logistic Regression - Feature Scores")
        plt.xlabel("Score")
        plt.ylabel("Feature")
        return fig

    score_lr_png = render(_score_lr, styled=True)

    # cell 68: Feature contribution scores - Random Forest
    rfc_model = rfe_rfc.estimator_
    rfc_model.fit(x_train_rfe_rfc, y_train_rfe)
    rfc_scores = rfc_model.feature_importances_

    def _score_rfc():
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=rfc_scores, y=features_rfc)
        plt.title("Random Forest Classifier - Feature Scores")
        plt.xlabel("Score")
        plt.ylabel("Feature")
        return fig

    score_rfc_png = render(_score_rfc, styled=True)

    # cells 69 / 70: quick accuracy check of both selectors
    model_lr.fit(x_train_rfe_lr, y_train_rfe)
    model_rfc.fit(x_train_rfe_rfc, y_train_rfe)
    y_pred_lr = model_lr.predict(x_test_rfe[features_lr])
    y_pred_rfc = model_rfc.predict(x_test_rfe[features_rfc])
    acc_lr = accuracy_score(y_test_rfe, y_pred_lr)
    acc_rfc = accuracy_score(y_test_rfe, y_pred_rfc)

    return {
        "features_lr": list(features_lr),
        "features_rfc": list(features_rfc),
        "rank_lr_png": rank_lr_png,
        "rank_rfc_png": rank_rfc_png,
        "score_lr_png": score_lr_png,
        "score_rfc_png": score_rfc_png,
        "lr_scores": dict(zip(features_lr, map(float, lr_scores))),
        "rfc_scores": dict(zip(features_rfc, map(float, rfc_scores))),
        "acc_lr": float(acc_lr),
        "acc_rfc": float(acc_rfc),
        "shape_lr": x_train_rfe_lr.shape,
        "shape_rfc": x_train_rfe_rfc.shape,
        "n_train": len(x_train_rfe),
        "n_test": len(x_test_rfe),
        # carried into the modelling stage
        "_x_train_rfe_lr": x_train_rfe_lr,
        "_x_train_rfe_rfc": x_train_rfe_rfc,
        "_y_train_rfe": y_train_rfe,
        "_x_test_rfe": x_test_rfe,
        "_y_test_rfe": y_test_rfe,
    }


# --------------------------------------------------------------------------- #
# Stage 3 - model stacking, optimisation and evaluation (cells 73-90)
# --------------------------------------------------------------------------- #
def getModels() -> dict:
    """returns the models required for use"""
    models = {
        "rfc": RandomForestClassifier(),
        "knn": KNeighborsClassifier(),
        # 'svm': SVC(), #  --- toggled off
        # 'lr': LogisticRegression(), #  --- toggled off
        "adb": AdaBoostClassifier(),
        # 'nb':GaussianNB(), #  --- toggled off
        "xgb": XGBClassifier(),
    }
    return models


def getBaseModels(models: dict) -> list:
    base_models = []
    for name, model in models.items():
        base_models.append((name, models[name]))
    return base_models


def getStackedML(models: dict, meta_model=RandomForestClassifier()) -> StackingClassifier:
    base_models = getBaseModels(models)
    model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    return model


# cell 76 - parameter grid (unchanged)
param_grid = {
    "rfc": {  # random forest classifier
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2"],
        "n_jobs": [-1],
    },
    "knn": {  # k nearest neighbours classifier
        "n_neighbors": [5, 10, 15],
        "p": [1, 2],
        # 'weights': ['uniform', 'distance'],
        "n_jobs": [-1],
    },
    "svm": {  # support vector machine classifier
        "C": [0.1, 1, 10],
        "kernel": ["poly", "sigmoid", "rbf"],
        "degree": [3, 4],
        "coef0": [0.0, 1.0],
        "random_state": [42],
    },
    "lr": {  # logistic regressor
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
    },
    "adb": {  # ada boost classifier
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5],
        "random_state": [42],
    },
    "nb": {},  # gaussian naive bayes classifier
    "xgb": {  # extreme gradient boost classifer
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "learning_rate": [0.1, 0.01, 0.001],
        "random_state": [42],
    },
    "stacked": {  # stacking classifier
        "cv": [3, 5, 7],
    },
}


def getBestModels(param_grid: dict, training_features, training_target) -> dict:
    """GridSearchCV (5-fold) over every model toggled on in getModels()."""
    models = getModels()
    best_models = {}
    for name, model in models.items():
        if param_grid[name] != {}:
            grid_search = GridSearchCV(model, param_grid[name], cv=5, n_jobs=-1)
            grid_search.fit(training_features, training_target)
            best_model = grid_search.best_estimator_
            best_models[name] = best_model
    return best_models


def calcPerfMetrics(models: dict, training_features, training_target, testing_features, testing_target) -> dict:
    """Same maths as the notebook; returns results instead of printing them."""
    results = {}
    with plot_ctx(True):  # runs after sns.set(font_scale=0.6) in the notebook
        fig = plt.figure()
        for name, model in models.items():
            model.fit(training_features, training_target)
            predictions = model.predict(testing_features)
            rep_text = classification_report(testing_target, predictions)
            rep_dict = classification_report(testing_target, predictions, output_dict=True)
            cm = confusion_matrix(testing_target, predictions)
            fpr, tpr, thresholds = roc_curve(testing_target, predictions)
            plt.plot(fpr, tpr, label=name)
            auc = roc_auc_score(testing_target, predictions)
            results[name] = {
                "report_text": rep_text,
                "report": pd.DataFrame(rep_dict).T,
                "cm": cm,
                "auc": float(auc),
                "accuracy": float(rep_dict["accuracy"]),
                "precision": float(rep_dict["1.0"]["precision"]),
                "recall": float(rep_dict["1.0"]["recall"]),
                "f1": float(rep_dict["1.0"]["f1-score"]),
                "support": int(len(testing_target)),
            }
        plt.legend()
        roc_png = _to_png(fig)
    return results, roc_png


def _best_params(models: dict) -> dict:
    out = {}
    for name, model in models.items():
        keys = param_grid[name].keys()
        p = model.get_params()
        out[name] = {k: p[k] for k in keys if k in p}
    return out


def run_modeling(fs: dict, progress=None) -> dict:
    say = progress or (lambda msg: None)
    x_train_rfe_lr = fs["_x_train_rfe_lr"]
    x_train_rfe_rfc = fs["_x_train_rfe_rfc"]
    y_train_rfe = fs["_y_train_rfe"]
    x_test_rfe = fs["_x_test_rfe"]
    y_test_rfe = fs["_y_test_rfe"]
    features_lr = fs["features_lr"]
    features_rfc = fs["features_rfc"]

    say("Grid-searching RFC, KNN, AdaBoost and XGBoost on the **Logistic-Regression** feature set...")
    best_models_lr = getBestModels(param_grid, x_train_rfe_lr, y_train_rfe)
    say("Grid-searching the same models on the **Random-Forest** feature set...")
    best_models_rfc = getBestModels(param_grid, x_train_rfe_rfc, y_train_rfe)
    params_lr, params_rfc = _best_params(best_models_lr), _best_params(best_models_rfc)

    say("Stacking and evaluating (Logistic-Regression features)...")
    stacked_model = getStackedML(best_models_lr)  # getting the stacked model
    best_models_lr["stacked"] = stacked_model  # adding the stacked model for iterative analysis
    res_lr, roc_lr = calcPerfMetrics(
        best_models_lr, x_train_rfe_lr, y_train_rfe, x_test_rfe[features_lr], y_test_rfe
    )

    say("Stacking and evaluating (Random-Forest features)...")
    stacked_model_rfc = getStackedML(best_models_rfc)  # noqa: F841  (built but unused in the notebook, too)
    # NOTE: exactly as in notebook cell 90, the stacked model added here is `stacked_model`
    # (the one built from the Logistic-Regression-tuned base models), not `stacked_model_rfc`.
    best_models_rfc["stacked"] = stacked_model
    res_rfc, roc_rfc = calcPerfMetrics(
        best_models_rfc, x_train_rfe_rfc, y_train_rfe, x_test_rfe[features_rfc], y_test_rfe
    )

    return {
        "lr": {"best_params": params_lr, "results": res_lr, "roc_png": roc_lr},
        "rfc": {"best_params": params_rfc, "results": res_rfc, "roc_png": roc_rfc},
    }

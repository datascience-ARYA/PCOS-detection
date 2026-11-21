"""
PCOS Detection Model - Streamlit showcase
Run with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

import pipeline as pl

st.set_page_config(
    page_title="PCOS Detection Model",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "PCOS_data_without_infertility.xlsx"
COLAB_URL = "https://colab.research.google.com/github/Fayyez/PCOS-Detection-Model/blob/main/pcos_prediction_classifier.ipynb"
REPO_URL = "https://github.com/Fayyez/PCOS-Detection-Model"
DATASET_URL = "https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos"

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="st-"], .stMarkdown { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.02em; }
header[data-testid="stHeader"] { background: transparent; }

.hero { background: linear-gradient(120deg, #0f766e 0%, #4f46e5 58%, #be185d 100%);
  border-radius: 22px; padding: 2.1rem 2.4rem; color: #fff; margin-bottom: 1.6rem;
  box-shadow: 0 18px 40px -18px rgba(79,70,229,.55); }
.hero h1 { color: #fff !important; font-size: 2.3rem; font-weight: 800; margin: .3rem 0 .4rem 0; padding: 0; }
.hero p { color: rgba(255,255,255,.88); font-size: 1.05rem; margin: 0; max-width: 56rem; line-height: 1.55; }
.hero .badge { display:inline-block; background: rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35);
  padding: .2rem .8rem; border-radius: 999px; font-size: .78rem; font-weight: 600; letter-spacing:.04em; text-transform: uppercase; }

.stat-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: .4rem 0 1.4rem 0; }
.stat { background:#fff; border:1px solid #e2e8f0; border-radius:16px; padding:1.1rem 1.25rem; box-shadow: 0 6px 18px -12px rgba(15,23,42,.25); }
.stat .v { font-size: 2rem; font-weight: 800; line-height:1.1; background: linear-gradient(90deg,#0f766e,#4f46e5);
  -webkit-background-clip:text; background-clip:text; color: transparent; }
.stat .l { font-weight: 600; color:#0f172a; margin-top:.35rem; font-size:.95rem; }
.stat .s { color:#64748b; font-size:.82rem; margin-top:.15rem; line-height:1.35; }
.stat.rose .v { background: linear-gradient(90deg,#be185d,#e11d48); -webkit-background-clip:text; background-clip:text; }

.card { background:#fff; border:1px solid #e2e8f0; border-radius:18px; padding:1.3rem 1.5rem; margin-bottom: 1rem;
  box-shadow: 0 6px 18px -12px rgba(15,23,42,.25); }
.card h4 { margin: 0 0 .5rem 0; font-size: 1.05rem; color:#0f172a; }
.card p, .card li { color:#334155; line-height:1.6; font-size:.95rem; }
.card ul { margin:.3rem 0 0 1.1rem; padding:0; }

.callout { border-left: 4px solid #0d9488; background:#f0fdfa; border-radius: 10px; padding: .9rem 1.1rem; color:#134e4a; margin: .6rem 0 1rem 0; font-size:.93rem; line-height:1.55; }
.callout.warn { border-color:#d97706; background:#fffbeb; color:#78350f; }
.callout.rose { border-color:#e11d48; background:#fff1f2; color:#881337; }

.sec { display:flex; align-items:center; gap:.6rem; margin: 1.6rem 0 .6rem 0; }
.sec .dot { width:10px; height:28px; border-radius:6px; background: linear-gradient(180deg,#0d9488,#4f46e5); }
.sec h3 { margin:0; padding:0; font-size:1.3rem; font-weight:700; }
.sec-sub { color:#64748b; margin: -.2rem 0 .8rem 1.1rem; font-size:.93rem; }

.steps { display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap:14px; margin-bottom:1rem; }
.step { background:#fff; border:1px solid #e2e8f0; border-radius:16px; padding:1rem 1.1rem; position:relative; }
.step .n { width:30px; height:30px; border-radius:50%; background: linear-gradient(135deg,#0d9488,#4f46e5); color:#fff;
  display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.9rem; margin-bottom:.55rem; }
.step b { color:#0f172a; } .step span { display:block; color:#64748b; font-size:.85rem; margin-top:.2rem; line-height:1.4; }

.chip { display:inline-block; padding:.28rem .75rem; margin:.18rem .25rem .18rem 0; border-radius:999px; font-size:.82rem; font-weight:500;
  background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe; }
.chip.teal { background:#f0fdfa; color:#115e59; border-color:#99f6e4; }
.chip.rose { background:#fff1f2; color:#9f1239; border-color:#fecdd3; }

.person { display:flex; align-items:center; gap:.8rem; background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:.8rem 1rem; }
.avatar { width:42px; height:42px; border-radius:50%; background: linear-gradient(135deg,#0d9488,#be185d); color:#fff; font-weight:700;
  display:flex; align-items:center; justify-content:center; }

.flow { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin:.6rem 0 1rem 0; }
.flow .node { background:#fff; border:1px solid #c7d2fe; border-radius:12px; padding:.55rem .9rem; font-weight:600; color:#3730a3; font-size:.9rem; }
.flow .node.meta { background: linear-gradient(135deg,#0d9488,#4f46e5); color:#fff; border:none; }
.flow .arrow { color:#94a3b8; font-size:1.2rem; }

[data-testid="stMetric"] { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:.85rem 1.1rem; box-shadow:0 6px 18px -14px rgba(15,23,42,.3); }
[data-testid="stMetricLabel"] p { color:#64748b; font-weight:600; }
[data-testid="stImage"] img { border-radius:12px; border:1px solid #e2e8f0; background:#fff; }
.stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom: 1px solid #e2e8f0; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: .55rem 1rem; font-weight:600; }
[data-testid="stSidebar"] { border-right: 1px solid #e2e8f0; }
.side-note { font-size:.78rem; color:#64748b; line-height:1.5; margin-top:.8rem; }
.footer { text-align:center; color:#94a3b8; font-size:.82rem; margin-top:3rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def H(markup: str) -> None:
    st.markdown(markup.strip(), unsafe_allow_html=True)


def hero(badge: str, title: str, subtitle: str) -> None:
    H(f'<div class="hero"><span class="badge">{html.escape(badge)}</span><h1>{html.escape(title)}</h1><p>{subtitle}</p></div>')


def section(title: str, sub: str | None = None) -> None:
    H(f'<div class="sec"><div class="dot"></div><h3>{html.escape(title)}</h3></div>')
    if sub:
        H(f'<div class="sec-sub">{sub}</div>')


def stat_cards(items: list[tuple]) -> None:
    cards = ""
    for it in items:
        value, label, sub = it[:3]
        cls = "stat rose" if len(it) > 3 and it[3] == "rose" else "stat"
        cards += f'<div class="{cls}"><div class="v">{html.escape(str(value))}</div><div class="l">{html.escape(label)}</div><div class="s">{sub}</div></div>'
    H(f'<div class="stat-grid">{cards}</div>')


def callout(text: str, kind: str = "") -> None:
    H(f'<div class="callout {kind}">{text}</div>')


def chips(items, kind: str = "") -> None:
    H("".join(f'<span class="chip {kind}">{html.escape(str(i))}</span>' for i in items))


def show_img(data: bytes, caption: str | None = None) -> None:
    st.image(data, caption=caption, width="stretch")


def table(df: pd.DataFrame, **kw) -> None:
    st.dataframe(df, width="stretch", hide_index=kw.pop("hide_index", True), **kw)


def fmt_counts(d: dict) -> str:
    return " · ".join(f"class {k}: {v}" for k, v in d.items())


# --------------------------------------------------------------------------- #
# Data access (cached)
# --------------------------------------------------------------------------- #
def get_file_bytes() -> bytes | None:
    up = st.session_state.get("uploaded_bytes")
    if up:
        return up
    if DEFAULT_DATA.exists():
        return DEFAULT_DATA.read_bytes()
    return None


@st.cache_data(show_spinner="Cleaning and preparing the dataset…")
def get_prep(file_bytes: bytes) -> dict:
    return pl.prepare_data(pl.read_dataset(BytesIO(file_bytes)))


@st.cache_data(show_spinner=False, max_entries=600)
def get_feature_png(kind: str, feature: str, key: str, _prep: dict) -> bytes:
    return pl.feature_plot(kind, _prep, feature)


@st.cache_resource(show_spinner=False)
def get_fs(key: str, _prep: dict) -> dict:
    return pl.run_feature_selection(_prep)


@st.cache_resource(show_spinner=False)
def get_models(key: str, _fs: dict, _progress=None) -> dict:
    return pl.run_modeling(_fs, _progress)


def require_prep() -> tuple[dict, str]:
    fb = get_file_bytes()
    if fb is None:
        hero("Dataset needed", "Upload the PCOS dataset to continue",
             "This page runs the real pipeline on your data. Upload <b>PCOS_data_without_infertility.xlsx</b> in the sidebar, "
             "or place it in a <code>data/</code> folder next to <code>app.py</code>.")
        st.link_button("Get the dataset on Kaggle", DATASET_URL, icon=":material/download:")
        st.stop()
    try:
        return get_prep(fb), hashlib.md5(fb).hexdigest()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not process the file: {e}")
        st.caption("Expected the Kaggle workbook, with the data on the second sheet (sheet_name=1).")
        st.stop()


def feature_gallery(kind: str, prep: dict, key: str, features: list[str], cols: int = 3) -> None:
    mode = st.radio("View", ["All features (grid)", "Single feature"], horizontal=True, key=f"{kind}_mode",
                    label_visibility="collapsed")
    if mode == "Single feature":
        f = st.selectbox("Feature", features, key=f"{kind}_sel")
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            show_img(get_feature_png(kind, f, key, prep))
        return
    with st.spinner("Rendering plots…"):
        grid = st.columns(cols)
        for i, f in enumerate(features):
            with grid[i % cols]:
                show_img(get_feature_png(kind, f, key, prep))


# --------------------------------------------------------------------------- #
# Static content taken from the notebook
# --------------------------------------------------------------------------- #
FEATURE_DICT = {
    "Vitals & demographics": [
        ("Age (yrs)", "Age of patient in years"),
        ("Blood Group", "A+ = 11, A- = 12, B+ = 13, B- = 14, O+ = 15, O- = 16, AB+ = 17, AB- = 18"),
        ("Pulse rate(bpm)", "Heart rate in beats per minute (adult resting range 60–100)"),
        ("RR (breaths/min)", "Respiration rate (adult resting range 12–16)"),
        ("Hb(g/dl)", "Hemoglobin level (normal for women 12.3–15.3 g/dL)"),
        ("BP _Systolic (mmHg)", "Pressure in the arteries when the heart beats"),
        ("BP _Diastolic (mmHg)", "Pressure in the arteries when the heart rests between beats"),
    ],
    "Body measurements": [
        ("Weight (Kg)", "Weight of patient in kg"),
        ("Height(Cm)", "Height of patient in centimetres"),
        ("BMI", "Body mass index"),
        ("Hip(inch)", "Hip size in inches"),
        ("Waist(inch)", "Waist size in inches"),
        ("Waist:Hip Ratio", "Waist divided by hip"),
    ],
    "Menstrual & reproductive": [
        ("Cycle(R/I)", "Cycle type, Regular / Irregular (numerically coded)"),
        ("Cycle length(days)", "Length of menstrual cycle; the average is about 28 days"),
        ("Marraige Status (Yrs)", "Years of marriage"),
        ("Pregnant(Y/N)", "Whether the patient is pregnant"),
        ("No. of aborptions", "Number of abortions, if any (437 of 541 patients had none)"),
    ],
    "Hormones & lab tests": [
        ("I beta-HCG(mIU/mL)", "Beta-hCG, first measurement"),
        ("II beta-HCG(mIU/mL)", "Beta-hCG, second measurement (< 5 negative, > 25 positive for pregnancy)"),
        ("FSH(mIU/mL)", "Follicle-stimulating hormone"),
        ("LH(mIU/mL)", "Luteinizing hormone"),
        ("FSH/LH", "Ratio of FSH to LH"),
        ("TSH (mIU/L)", "Thyroid-stimulating hormone (normal 0.4–4.0)"),
        ("AMH(ng/mL)", "Anti-Müllerian hormone"),
        ("PRL(ng/mL)", "Prolactin level"),
        ("Vit D3 (ng/mL)", "Vitamin D level (normal ≥ 20 ng/mL)"),
        ("PRG(ng/mL)", "Progesterone level"),
        ("RBS(mg/dl)", "Random blood sugar test"),
    ],
    "Symptoms & lifestyle": [
        ("Weight gain(Y/N)", "Has there been weight gain"),
        ("hair growth(Y/N)", "Has there been excess hair growth"),
        ("Skin darkening (Y/N)", "Skin darkening issues"),
        ("Hair loss(Y/N)", "Hair loss issues"),
        ("Pimples(Y/N)", "Pimple issues"),
        ("Fast food (Y/N)", "Is fast food part of the diet"),
        ("Reg.Exercise(Y/N)", "Regular exercise"),
    ],
    "Ultrasound": [
        ("Follicle No. (L)", "Number of follicles, left side"),
        ("Follicle No. (R)", "Number of follicles, right side"),
        ("Avg. F size (L) (mm)", "Average follicle size, left, in mm"),
        ("Avg. F size (R) (mm)", "Average follicle size, right, in mm"),
        ("Endometrium (mm)", "Endometrium thickness in mm"),
    ],
}

# Figures printed in the notebook (rounded as shown there); used only as a reference.
NOTEBOOK_REF = {
    "Logistic-Regression features": {"rfc": (0.99, 0.9848), "knn": (1.00, 1.0), "adb": (0.96, 0.9562), "xgb": (0.97, 0.9697)},
    "Random-Forest features": {"rfc": (0.97, 0.9697), "knn": (1.00, 1.0), "adb": (0.96, 0.9562), "xgb": (0.94, 0.9394)},
}
MODEL_LABEL = {"rfc": "Random Forest", "knn": "K-Nearest Neighbours", "adb": "AdaBoost", "xgb": "XGBoost", "stacked": "Stacked ensemble"}


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    hero("Machine learning · Women's health", "PCOS Detection Model",
         "A classification pipeline for the early, low-cost detection of Polycystic Ovary Syndrome from clinical, "
         "hormonal and lifestyle data, built to help close the gap of undiagnosed cases.")

    section("Why this matters")
    stat_cards([
        ("8–13%", "of women of reproductive age", "are affected by PCOS (WHO)"),
        ("70%", "of cases undiagnosed", "making it a leading cause of infertility"),
        ("52%", "prevalence in Pakistani women", "vs. 20–25% in the UK white population, about 2-fold higher", "rose"),
        ("541", "patient records", "43 clinical parameters from 10 hospitals in Kerala, India"),
    ])

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        H("""<div class="card"><h4>What is PCOS?</h4>
<p>Polycystic Ovary Syndrome is a common hormonal condition that usually starts in adolescence. It can cause hormonal imbalance,
irregular periods, excess androgen levels and cysts in the ovaries. Irregular periods, usually with a lack of ovulation, can make it
difficult to become pregnant.</p></div>
<div class="card"><h4>Project aim</h4>
<p>To address the undiagnosed-case problem we built a classification model that aims to improve diagnostic accuracy while making
it more cost-effective, using blood tests, menstrual-cycle data, lifestyle factors, age and more.</p></div>""")
    with c2:
        H('<div class="card"><h4>The team</h4></div>')
        for name in ["Muhammad Eehab Saadat", "Fayyez Farrukh", "Rijja Masood"]:
            initials = "".join(w[0] for w in name.split()[:2])
            H(f'<div class="person" style="margin-bottom:.6rem"><div class="avatar">{initials}</div><b>{name}</b></div>')
        st.link_button("Open the original notebook", COLAB_URL, icon=":material/open_in_new:", width="stretch")
        st.link_button("GitHub repository", REPO_URL, icon=":material/code:", width="stretch")

    section("The pipeline", "Each step has its own page in the sidebar, running the real notebook code.")
    steps = [
        ("Clean", "Drop ID columns, fix dtypes, fill 4 missing values"),
        ("Explore", "Class balance, correlations, redundant features"),
        ("Preprocess", "Selective outlier removal and MinMax scaling"),
        ("Resample", "ADASYN, SMOTE and SMOTE-ENN to fix imbalance"),
        ("Select features", "RFE with Logistic Regression and Random Forest"),
        ("Stack & tune", "RFC + KNN + AdaBoost + XGBoost, GridSearchCV"),
        ("Evaluate", "Accuracy, precision, recall, F1, ROC / AUC"),
    ]
    H('<div class="steps">' + "".join(
        f'<div class="step"><div class="n">{i}</div><b>{t}</b><span>{d}</span></div>' for i, (t, d) in enumerate(steps, 1)) + "</div>")

    callout("<b>Educational project.</b> This app showcases a research notebook. It is not a medical device and must not be used for diagnosis.", "warn")


def page_dataset() -> None:
    hero("Step 0", "The dataset",
         'The <a href="' + DATASET_URL + '" style="color:#fff;text-decoration:underline">PCOS dataset</a> by Prasoon Kottarathil (Kaggle) holds 43 physical '
         "and clinical parameters on 541 fertile female patients, collected across 10 hospitals in Kerala, India.")
    stat_cards([("541", "samples", "patient records"), ("43", "parameters", "1 target + 42 columns"),
                ("10", "hospitals", "across Kerala, India"), ("PCOS (Y/N)", "target", "does the patient have PCOS?", "rose")])

    t1, t2, t3 = st.tabs(["Feature dictionary", "Data preview", "Summary statistics"])
    with t1:
        for group, rows in FEATURE_DICT.items():
            with st.expander(f"{group}  ·  {len(rows)} features", expanded=group == "Hormones & lab tests"):
                table(pd.DataFrame(rows, columns=["Feature", "Description"]))
        callout("<b>Identifiers</b> (<i>Sl. No</i>, <i>Patient File No.</i>) are dropped during cleaning, as they only enumerate records.")
    prep = get_prep(get_file_bytes()) if get_file_bytes() else None
    with t2:
        if prep is None:
            callout("Upload the dataset in the sidebar to preview it here.", "warn")
            return
        st.caption(f"First rows of the sheet used by the notebook (after removing the empty unnamed column) · {prep['after_unnamed'].shape[0]} rows × {prep['after_unnamed'].shape[1]} columns")
        preview = prep["after_unnamed"].head(25).copy()
        for c in preview.columns:  # the raw sheet mixes strings into numeric columns
            if not pd.api.types.is_numeric_dtype(preview[c]):
                preview[c] = preview[c].astype(str)
        table(preview, hide_index=False)
    with t3:
        if prep is None:
            callout("Upload the dataset in the sidebar to see summary statistics.", "warn")
            return
        table(prep["describe"].T.reset_index().rename(columns={"index": "Feature"}))


def page_cleaning() -> None:
    prep, _ = require_prep()
    hero("Step 1", "Data cleaning & type normalisation",
         "Remove non-informative columns, convert numeric strings to numbers and fill the few missing values, "
         "keeping every record because information integrity matters in medicine.")

    section("1 · Drop non-informative columns")
    stat_cards([
        (f"{prep['after_unnamed'].shape[1]}", "columns after dropping 'Unnamed'", "an empty column with zero correspondence to the target"),
        (f"{prep['after_unnamed'].shape[1] - 2}", "columns after dropping IDs", "<i>Sl. No</i> and <i>Patient File No.</i> removed"),
    ])

    section("2 · Convert numeric strings to numbers",
            "These columns were numeric <i>strings</i>, so they were converted with <code>pd.to_numeric</code> rather than encoded.")
    if prep["converted_columns"]:
        chips(prep["converted_columns"], "teal")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Before")
        table(prep["dtypes_before"], height=320)
    with c2:
        st.caption("After")
        table(prep["dtypes_after"], height=320)

    section("3 · Fill the missing values",
            "Means for continuous columns and the mode for the binary one, instead of dropping records.")
    t1, t2 = st.tabs(["Before filling", "After filling"])
    with t1:
        show_img(prep["missing_before_png"])
    with t2:
        show_img(prep["missing_after_png"])
    table(prep["fills"].style.format({"Fill value": "{:.4f}"}))


def page_eda() -> None:
    prep, key = require_prep()
    hero("Step 2", "Exploratory data analysis",
         "Understand class balance, spot which features move with PCOS and find redundant, highly correlated features.")

    cc = prep["class_counts_raw"]
    total = sum(cc.values())
    stat_cards([(total, "records", "after cleaning"), (cc.get(0, 0), "PCOS negative", f"{cc.get(0, 0) / total:.1%} of records"),
                (cc.get(1, 0), "PCOS positive", f"{cc.get(1, 0) / total:.1%} of records", "rose")])

    t1, t2, t3 = st.tabs(["Binary features", "Correlation with target", "Redundant features"])
    with t1:
        st.caption("Count plots show how balanced each binary feature is across PCOS-positive and PCOS-negative patients.")
        feature_gallery("count", prep, key, prep["binary_features"], cols=4)
    with t2:
        st.caption("Absolute correlation of every feature with the target, and the heatmap of the 17 most correlated features.")
        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            s = prep["corr_with_target"].drop(pl.TARGET).sort_values(ascending=False)
            table(s.rename("Correlation").reset_index().rename(columns={"index": "Feature"}),
                  column_config={"Correlation": st.column_config.ProgressColumn("Correlation", min_value=0, max_value=1, format="%.3f")},
                  height=620)
        with c2:
            show_img(prep["corr_heatmap_png"])
    with t3:
        thr = prep["correlation_threshold"]
        st.caption(f"Feature pairs with mutual correlation of at least {thr:.0%}.")
        table(pd.DataFrame(prep["highly_correlated_pairs"], columns=["Feature A", "Feature B"]))
        callout("<b>Dropped:</b> <code>Weight (Kg)</code> (its information is already in <code>BMI</code>) and <code>Hip(inch)</code> "
                "(<code>Waist</code> is more readily available). <code>FSH</code> and <code>FSH/LH</code> were <b>kept</b>: both are closely tied to reproductive "
                "function and medically significant for PCOS diagnosis.")


def page_preprocess() -> None:
    prep, key = require_prep()
    hero("Step 3", "Preprocessing",
         "Selective outlier removal, then MinMax scaling to [0, 1] so distance-based models like KNN are not distorted.")
    t1, t2, t3, t4 = st.tabs(["Outliers", "Outlier removal", "Scaling", "Distributions"])
    nb = prep["nonbinary"]
    with t1:
        st.caption("Box plots of every continuous feature, split by PCOS status, before any outlier removal.")
        feature_gallery("box_raw", prep, key, nb)
    with t2:
        callout("<b>Caution:</b> medical outliers can be clinically significant, and blanket removal can introduce bias. Outliers were removed only from features "
                "where their distribution is even (symmetric) across PCOS-positive and PCOS-negative records.", "warn")
        o = prep["outlier_old_shape"]
        stat_cards([(f"{o[0]} × {o[1]}", "shape before", "rows × columns"),
                    (f"{prep['data'].shape[0]} × {prep['data'].shape[1]}", "shape after", "rows × columns")])
        table(prep["outlier_log"])
    with t3:
        st.caption("Box plots after outlier removal and MinMax scaling.")
        feature_gallery("box_scaled", prep, key, nb)
    with t4:
        kind = st.segmented_control("Plot type", ["Box + strip", "Violin + strip"], default="Box + strip")
        feature_gallery("box_strip" if kind != "Violin + strip" else "violin", prep, key, nb)


def page_resampling() -> None:
    prep, _ = require_prep()
    hero("Step 4", "Class balancing",
         "PCOS-positive cases are the minority class, which biases models. Three resampling methods were applied to the <b>training split only</b> "
         "(80/20 split, <code>random_state=42</code>) to avoid leakage.")
    cc = prep["class_counts_filtered"]
    stat_cards([(cc.get(0, 0), "PCOS negative", "after outlier removal"), (cc.get(1, 0), "PCOS positive", "after outlier removal", "rose"),
                (prep["n_train"], "training rows", "before resampling"), (prep["n_test"], "test rows", "held out")])

    cols = st.columns(3, gap="medium")
    desc = {"ADASYN": "Adaptive Synthetic Sampling", "SMOTE": "Synthetic Minority Over-sampling (strategy 1.0)",
            "SMOTE-ENN": "SMOTE + Edited Nearest Neighbours"}
    for col, (name, (before, after)) in zip(cols, prep["resampling"].items()):
        with col:
            chosen = name == "SMOTE-ENN"
            H(f'<div class="card"><h4>{name} {"✅" if chosen else ""}</h4><p style="margin:0 0 .6rem 0">{desc[name]}</p></div>')
            m1, m2 = st.columns(2)
            m1.metric("Original", f"{sum(before.values())}", fmt_counts(before), delta_color="off", delta_arrow="off")
            m2.metric("Resampled", f"{sum(after.values())}", fmt_counts(after), delta_color="off", delta_arrow="off")
    callout("<b>SMOTE-ENN</b> is carried forward. It oversamples the minority class and then removes noisy samples with Edited Nearest Neighbours, "
            "which was judged more medically appropriate after testing and literature review.")


def page_features() -> None:
    prep, key = require_prep()
    hero("Step 5", "Feature selection",
         "Wrapper-style <b>Recursive Feature Elimination (RFE)</b> picks the 15 most informative features, once with Logistic Regression "
         "and once with a Random Forest. The resampled data is split 80/20 again before selection.")
    with st.spinner("Running Recursive Feature Elimination…"):
        fs = get_fs(key, prep)

    stat_cards([(f"{fs['acc_lr']:.4f}", "Logistic Regression accuracy", "on RFE-selected features"),
                (f"{fs['acc_rfc']:.4f}", "Random Forest accuracy", "on RFE-selected features"),
                (fs["n_train"], "training rows", "after resampling"), (fs["n_test"], "test rows", "held out")])
    callout("The notebook does not seed this second split or the Random Forest, so figures vary slightly between runs. "
            "Results are cached for the session, so they stay consistent while you explore.", "warn")

    t1, t2, t3 = st.tabs(["Logistic Regression", "Random Forest", "Compare"])
    for tab, tag, name in [(t1, "lr", "Logistic Regression"), (t2, "rfc", "Random Forest")]:
        with tab:
            st.markdown(f"**15 features selected by {name}**")
            chips(fs[f"features_{tag}"], "teal" if tag == "lr" else "")
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.caption("Feature ranking (shorter bar = higher rank)")
                show_img(fs[f"rank_{tag}_png"])
            with c2:
                st.caption("Contribution scores of the selected features")
                show_img(fs[f"score_{tag}_png"])
    with t3:
        a, b = set(fs["features_lr"]), set(fs["features_rfc"])
        rows = [(f, f in a, f in b) for f in sorted(a | b)]
        df = pd.DataFrame(rows, columns=["Feature", "Logistic Regression", "Random Forest"])
        df["Chosen by both"] = df["Logistic Regression"] & df["Random Forest"]
        st.markdown(f"**{len(a & b)}** features were chosen by both selectors.")
        table(df.sort_values(["Chosen by both", "Feature"], ascending=[False, True]))


def page_modeling() -> None:
    prep, key = require_prep()
    hero("Step 6", "Model stacking & optimisation",
         "Four base learners are tuned with 5-fold <code>GridSearchCV</code> and combined by a stacking classifier "
         "with a Random Forest as the meta-model.")
    H('<div class="flow"><span class="node">Random Forest</span><span class="node">KNN</span><span class="node">AdaBoost</span>'
      '<span class="node">XGBoost</span><span class="arrow">➜</span><span class="node meta">Stacking · meta-model: Random Forest</span>'
      '<span class="arrow">➜</span><span class="node">PCOS / No PCOS</span></div>')

    with st.expander("Hyper-parameter grids searched", expanded=False):
        cols = st.columns(2)
        for i, name in enumerate(["rfc", "knn", "adb", "xgb"]):
            with cols[i % 2]:
                st.markdown(f"**{MODEL_LABEL[name]}**")
                table(pd.DataFrame([(k, ", ".join(map(str, v))) for k, v in pl.param_grid[name].items()], columns=["Parameter", "Values"]))

    fs = get_fs(key, prep)
    st.session_state.setdefault("trained", False)
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Train & evaluate models", type="primary", icon=":material/play_arrow:", width="stretch"):
            st.session_state["trained"] = True
    with c2:
        st.caption("Runs two grid searches and the stacked models. This can take several minutes the first time; afterwards it is cached.")

    if not st.session_state["trained"]:
        callout("Press <b>Train &amp; evaluate models</b> to run the search. Results then appear here and on the <b>Results</b> page.")
        return
    with st.status("Training…", expanded=True) as status:
        mod = get_models(key, fs, _progress=lambda m: st.write(m))
        status.update(label="Models trained", state="complete", expanded=False)

    section("Best parameters found")
    t1, t2 = st.tabs(["Logistic-Regression features", "Random-Forest features"])
    for tab, tag in [(t1, "lr"), (t2, "rfc")]:
        with tab:
            rows = [(MODEL_LABEL[m], ", ".join(f"{k}={v}" for k, v in p.items())) for m, p in mod[tag]["best_params"].items()]
            table(pd.DataFrame(rows, columns=["Model", "Best parameters"]))
    callout("Head to the <b>Results</b> page for metrics, confusion matrices and ROC curves.")


def _result_block(tag: str, res: dict, roc_png: bytes) -> None:
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.caption("ROC curves, all models")
        show_img(roc_png)
    with c2:
        name = st.segmented_control("Model", list(res), default=list(res)[0], format_func=lambda m: MODEL_LABEL[m],
                                    key=f"seg_{tag}") or list(res)[0]
        r = res[name]
        m = st.columns(5)
        m[0].metric("Accuracy", f"{r['accuracy']:.4f}")
        m[1].metric("Precision", f"{r['precision']:.4f}")
        m[2].metric("Recall", f"{r['recall']:.4f}")
        m[3].metric("F1-score", f"{r['f1']:.4f}")
        m[4].metric("AUC", f"{r['auc']:.4f}")
        st.caption("Precision, recall and F1 are for the PCOS-positive class.")
        cm = pd.DataFrame(r["cm"], index=["Actual: No PCOS", "Actual: PCOS"], columns=["Predicted: No PCOS", "Predicted: PCOS"])
        st.markdown("**Confusion matrix**")
        st.dataframe(cm.style.background_gradient(cmap="Blues"), width="stretch")
        st.markdown("**Classification report**")
        rep = r["report"].rename(index={"0.0": "No PCOS (0)", "1.0": "PCOS (1)"})
        st.dataframe(rep.style.format("{:.3f}", subset=["precision", "recall", "f1-score"]).format("{:.0f}", subset=["support"]), width="stretch")
        with st.expander("Raw report as printed by the notebook"):
            st.code(r["report_text"])


def page_results() -> None:
    prep, key = require_prep()
    hero("Step 7", "Results & conclusion", "How every model performed on the held-out test set, for both feature-selection strategies.")
    if not st.session_state.get("trained"):
        callout("No models trained yet. Run the training step first.", "warn")
        st.page_link(PG_MODEL, label="Go to Model Training", icon=":material/arrow_forward:")
        st.stop()
    fs = get_fs(key, prep)
    with st.spinner("Loading trained models…"):
        mod = get_models(key, fs)

    best = max(((t, m, r) for t in ("lr", "rfc") for m, r in mod[t]["results"].items()), key=lambda x: (x[2]["accuracy"], x[2]["auc"]))
    stat_cards([(f"{best[2]['accuracy']:.2%}", "best accuracy", f"{MODEL_LABEL[best[1]]} · {'Logistic-Regression' if best[0] == 'lr' else 'Random-Forest'} features"),
                (f"{best[2]['auc']:.4f}", "best AUC", "on the held-out test set"),
                (fs["n_test"], "test samples", "never seen during training")])

    t1, t2, t3, t4 = st.tabs(["Logistic-Regression features", "Random-Forest features", "Comparison", "Conclusion"])
    with t1:
        _result_block("lr", mod["lr"]["results"], mod["lr"]["roc_png"])
    with t2:
        _result_block("rfc", mod["rfc"]["results"], mod["rfc"]["roc_png"])
    with t3:
        rows = []
        for tag, label in [("lr", "Logistic Regression"), ("rfc", "Random Forest")]:
            for m, r in mod[tag]["results"].items():
                rows.append({"Features selected by": label, "Model": MODEL_LABEL[m], "Accuracy": r["accuracy"], "Precision": r["precision"],
                             "Recall": r["recall"], "F1": r["f1"], "AUC": r["auc"]})
        df = pd.DataFrame(rows)
        metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        table(df.style.format({c: "{:.4f}" for c in metrics}).background_gradient(subset=metrics, cmap="GnBu", vmin=0.85, vmax=1.0))
        with st.expander("Figures reported in the original notebook run"):
            ref = [{"Features selected by": k, "Model": MODEL_LABEL[m], "Accuracy": a, "AUC": u} for k, d in NOTEBOOK_REF.items() for m, (a, u) in d.items()]
            table(pd.DataFrame(ref))
            st.caption("The notebook's second split and several models are unseeded, so a fresh run can differ slightly from these.")
    with t4:
        H("""<div class="card"><h4>Conclusion</h4>
<p>The work met its objective of a high-precision, cost-effective model for the early detection of PCOS that supports medical professionals without
relying on more invasive methods.</p>
<ul>
<li>Recursive Feature Elimination with Logistic Regression and a Random Forest, plus selective outlier removal and scaling, improved efficiency and precision.</li>
<li>Removing outliers from the <b>entire</b> dataset indiscriminately hurt performance, likely through loss of crucial information.</li>
<li>In the notebook, KNN reached 1.0 on accuracy, precision, recall, F1 and AUC, whichever selector was used.</li>
<li>Models generally did better on features chosen by Logistic Regression than by the Random Forest.</li>
</ul></div>""")
        callout("<b>Interpretation note:</b> near-perfect scores on a single small test split should be read with care; they are not evidence of clinical readiness.", "warn")


# --------------------------------------------------------------------------- #
# Navigation + sidebar
# --------------------------------------------------------------------------- #
PG_HOME = st.Page(page_overview, title="Overview", icon=":material/home:", default=True)
PG_DATA = st.Page(page_dataset, title="Dataset", icon=":material/database:", url_path="dataset")
PG_CLEAN = st.Page(page_cleaning, title="Data Cleaning", icon=":material/cleaning_services:", url_path="cleaning")
PG_EDA = st.Page(page_eda, title="Exploration", icon=":material/query_stats:", url_path="eda")
PG_PRE = st.Page(page_preprocess, title="Preprocessing", icon=":material/tune:", url_path="preprocessing")
PG_RES = st.Page(page_resampling, title="Class Balancing", icon=":material/balance:", url_path="balancing")
PG_FEAT = st.Page(page_features, title="Feature Selection", icon=":material/checklist:", url_path="features")
PG_MODEL = st.Page(page_modeling, title="Model Training", icon=":material/hub:", url_path="training")
PG_RESULTS = st.Page(page_results, title="Results", icon=":material/emoji_events:", url_path="results")

nav = st.navigation({"Project": [PG_HOME, PG_DATA], "Pipeline": [PG_CLEAN, PG_EDA, PG_PRE, PG_RES, PG_FEAT, PG_MODEL], "Outcome": [PG_RESULTS]})

with st.sidebar:
    st.divider()
    st.markdown("**Dataset**")
    up = st.file_uploader("Upload PCOS_data_without_infertility.xlsx", type=["xlsx"], label_visibility="collapsed")
    if up is not None:
        st.session_state["uploaded_bytes"] = up.getvalue()
    elif "uploaded_bytes" in st.session_state and up is None:
        st.session_state.pop("uploaded_bytes")
    src = "uploaded file" if st.session_state.get("uploaded_bytes") else ("data/ folder" if DEFAULT_DATA.exists() else None)
    if src:
        st.success(f"Using {src}", icon=":material/check_circle:")
    else:
        st.warning("No dataset found yet", icon=":material/info:")
    st.link_button("Notebook on Colab", COLAB_URL, icon=":material/open_in_new:", width="stretch")
    H('<div class="side-note">Educational project · not a medical device.<br>Built by Eehab Saadat, Fayyez Farrukh and Rijja Masood.</div>')

nav.run()
H('<div class="footer">PCOS Detection Model · Streamlit showcase of the original notebook</div>')

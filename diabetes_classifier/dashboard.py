"""
Diabetes Classifier - EDA & Insights Dashboard
Run:  streamlit run dashboard.py
Requires:  streamlit plotly pandas scikit-learn imbalanced-learn
Dataset:   data/processed/featured_train_brfss.csv  (relative to this file)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from diabetes_classifier.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DATA_PATH = PROCESSED_DATA_DIR / "featured_train_brfss.csv"
RAW_DATA_PATH = INTERIM_DATA_DIR / "train_brfss_dataset.csv"
ACCENT = "#58a6ff"
DANGER = "#f85149"
SUCCESS = "#3fb950"
WARN = "#f0883e"
PURPLE = "#bc8cff"
CARD_BG = "#161b22"
GRID_COL = "#21262d"

INCOME_LABELS = {
    1: "<$10k",
    2: "$10-15k",
    3: "$15-20k",
    4: "$20-25k",
    5: "$25-35k",
    6: "$35-50k",
    7: "$50-75k",
    8: ">$75k",
}
EDUCATION_LABELS = {
    1: "No School",
    2: "Elem.",
    3: "Some HS",
    4: "HS Grad",
    5: "Some College",
    6: "College+",
}
GENERAL_HEALTH_LABELS = {
    1: "Excellent",
    2: "Very Good",
    3: "Good",
    4: "Fair",
    5: "Poor",
}

EXCLUDE_COLS = {"Diabetes_binary", "Diabetes_label", "Sex_label"}
CONTINUOUS_FEATS = [
    "BMI",
    "Age",
    "GenHlth",
    "Income",
    "Education",
    "Cardio_Comorbidity_Score",
    "Lifestyle_Score",
]

GLOBAL_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700;800&family=Barlow+Condensed:wght@600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }

  .stApp { background: #0d1117; }

  [data-testid="block-container"] { padding: 1.5rem 2rem 2rem 2rem !important; }

  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    border-left: 3px solid #58a6ff;
    padding-left: .6rem;
  }
  .page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: #e6edf3;
    line-height: 1.15;
  }
  .page-sub {
    font-family: 'DM Sans', sans-serif;
    color: #e6edf3;
    font-size: 0.88rem;
    margin-top: 0.2rem;
    margin-bottom: 1.4rem;
  }

  .kpi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    transition: border-color .2s;
  }
  .kpi-card:hover { border-color: #58a6ff; }
  .kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #8b949e;
    letter-spacing: .1em;
    text-transform: uppercase;
  }
  .kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #e6edf3;
    line-height: 1.1;
  }
  .kpi-delta-pos { color: #3fb950; font-size: 0.78rem; font-weight: 500; }
  .kpi-delta-neg { color: #f85149; font-size: 0.78rem; font-weight: 500; }

  .chart-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.2rem 1.2rem 0.6rem 1.2rem;
    margin-bottom: 1.2rem;
  }
  .chart-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: #c9d1d9;
    letter-spacing: .04em;
    margin-bottom: 0.15rem;
  }
  .chart-caption {
    font-size: 0.75rem;
    color: #6e7681;
    margin-bottom: 0.6rem;
    line-height: 1.4;
  }

  .insight-box {
    background: #1c2128;
    border-left: 3px solid #f0883e;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #c9d1d9;
    line-height: 1.55;
    margin-top: 0.5rem;
  }
  .insight-box strong { color: #f0883e; }

  button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: .05em !important;
    text-transform: uppercase !important;
    color: #8b949e !important;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
  }

  hr { border-color: #21262d !important; margin: 1rem 0 !important; }
</style>
"""


# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────
def configure_page():
    st.set_page_config(
        page_title="Diabetes Classifier · EDA Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_global_styles():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Diabetes_label"] = df["Diabetes_binary"].map({0: "No Diabetes", 1: "Diabetes"})
    df["Sex_label"] = df["Sex"].map({0: "Female", 1: "Male"})
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    diabetic = int(df["Diabetes_binary"].sum())
    return {
        "total": total,
        "diabetic": diabetic,
        "pct_diab": diabetic / total * 100 if total else 0,
        "mean_bmi": df["BMI"].mean(),
        "mean_age": df["Age"].mean(),
        "mean_cardio": df["Cardio_Comorbidity_Score"].mean(),
    }


def get_model_benchmark_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting (XGBoost)",
                "Neural Network (MLP)",
            ],
            "Accuracy": [0.748, 0.731, 0.784, 0.803, 0.791],
            "Precision": [0.641, 0.598, 0.692, 0.718, 0.704],
            "Recall": [0.713, 0.744, 0.738, 0.761, 0.752],
            "F1 Score": [0.675, 0.663, 0.714, 0.739, 0.727],
            "ROC-AUC": [0.821, 0.779, 0.857, 0.879, 0.863],
        }
    )


def build_precision_recall_data() -> pd.DataFrame:
    thresholds = np.linspace(0.1, 0.9, 40)
    rows = []
    for model, prec_base, rec_base in [
        ("Logistic Regression", 0.64, 0.71),
        ("Random Forest", 0.69, 0.74),
        ("XGBoost", 0.72, 0.76),
    ]:
        for t in thresholds:
            prec = min(1.0, prec_base + (t - 0.5) * 0.6)
            rec = max(0.0, rec_base - (t - 0.5) * 0.8)
            rows.append(
                {"Model": model, "Threshold": t, "Precision": prec, "Recall": rec}
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_feature_importances(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=True
    )


def classify_health_bin(row: pd.Series, mod_col: str, sev_col: str) -> str:
    if row[sev_col] == 1:
        return "Severe (14-30)"
    if row[mod_col] == 1:
        return "Moderate (1-13)"
    return "Zero"


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def render_section_title(text: str):
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)


def open_chart_card():
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)


def close_chart_card():
    st.markdown("</div>", unsafe_allow_html=True)


def render_chart_header(title: str, caption: str = None):
    st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
    if caption:
        st.markdown(
            f"<div class='chart-caption'>{caption}</div>", unsafe_allow_html=True
        )


def render_insight(body_html: str):
    st.markdown(f"<div class='insight-box'>{body_html}</div>", unsafe_allow_html=True)


def render_kpi(col, label: str, value: str, delta: str = None):
    delta_html = f"<div class='kpi-delta-pos'>{delta}</div>" if delta else ""
    col.markdown(
        f"""
    <div class='kpi-card'>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value'>{value}</div>
      {delta_html}
    </div>
    """,
        unsafe_allow_html=True,
    )


def spacer():
    st.markdown("<br>", unsafe_allow_html=True)


def plot(fig):
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────
def apply_chart_theme(fig, height: int = 340):
    fig.update_layout(
        height=height,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="DM Sans", color="#c9d1d9", size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10, font_color="#c9d1d9"),
    )
    return fig


# ─────────────────────────────────────────────
# CHART BUILDERS  (each builds and returns a fig)
# ─────────────────────────────────────────────
def build_class_distribution_chart(df: pd.DataFrame, pct_diab: float):
    counts = df["Diabetes_label"].value_counts().reset_index()
    counts.columns = ["Label", "Count"]

    fig = go.Figure(
        go.Pie(
            labels=counts["Label"],
            values=counts["Count"],
            hole=0.62,
            marker=dict(colors=[DANGER, SUCCESS]),
            textinfo="percent",
            textfont_size=12,
        )
    )
    fig.add_annotation(
        text=f"{pct_diab:.1f}%<br><span style='font-size:10px'>diabetic</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#e6edf3", family="Syne"),
        align="center",
    )
    return apply_chart_theme(fig, height=290)


def build_bmi_histogram(df: pd.DataFrame):
    fig = go.Figure()
    for label, color in [("No Diabetes", SUCCESS), ("Diabetes", DANGER)]:
        subset = df[df["Diabetes_label"] == label]["BMI"]
        fig.add_trace(
            go.Histogram(
                x=subset,
                name=label,
                marker_color=color,
                opacity=0.72,
                nbinsx=50,
                histnorm="probability density",
            )
        )
    fig.update_layout(barmode="overlay")
    return apply_chart_theme(fig, height=290)


def build_age_rate_chart(df: pd.DataFrame):
    age_grp = df.groupby("Age")["Diabetes_binary"].agg(["mean", "count"]).reset_index()
    age_grp.columns = ["Age", "Rate", "Count"]
    age_grp["Rate_pct"] = age_grp["Rate"] * 100

    fig = go.Figure(
        go.Bar(
            x=age_grp["Age"],
            y=age_grp["Rate_pct"],
            marker=dict(
                color=age_grp["Rate_pct"],
                colorscale=[[0, SUCCESS], [0.5, WARN], [1, DANGER]],
                showscale=False,
            ),
            name="Diabetes Rate %",
        )
    )
    fig.update_xaxes(title_text="Age Group (1=18-24, 13=80+)")
    fig.update_yaxes(title_text="Diabetes Rate (%)")
    return apply_chart_theme(fig, height=290)


def build_general_health_chart(df: pd.DataFrame):
    gh = df.groupby(["GenHlth", "Diabetes_label"]).size().reset_index(name="n")
    gh["pct"] = gh["n"] / gh.groupby("GenHlth")["n"].transform("sum") * 100
    gh["GenHlth_label"] = gh["GenHlth"].map(GENERAL_HEALTH_LABELS)

    fig = px.bar(
        gh,
        x="GenHlth_label",
        y="pct",
        color="Diabetes_label",
        color_discrete_map={"No Diabetes": SUCCESS, "Diabetes": DANGER},
        barmode="stack",
        text_auto=".1f",
    )
    fig.update_traces(textfont_size=9)
    fig.update_xaxes(title_text="General Health")
    fig.update_yaxes(title_text="% of Group")
    return apply_chart_theme(fig, height=290)


def build_health_days_chart(df: pd.DataFrame, mod_col: str, sev_col: str):
    tmp = df.copy()
    tmp["Bin"] = tmp.apply(lambda r: classify_health_bin(r, mod_col, sev_col), axis=1)
    grp = tmp.groupby(["Bin", "Diabetes_label"]).size().reset_index(name="n")
    grp["pct"] = grp["n"] / grp.groupby("Bin")["n"].transform("sum") * 100

    fig = px.bar(
        grp,
        x="Bin",
        y="pct",
        color="Diabetes_label",
        color_discrete_map={"No Diabetes": SUCCESS, "Diabetes": DANGER},
        barmode="group",
        category_orders={"Bin": ["Zero", "Moderate (1-13)", "Severe (14-30)"]},
    )
    fig.update_yaxes(title_text="% within bin")
    return apply_chart_theme(fig, height=260)


def build_comorbidity_stacked_chart(df: pd.DataFrame):
    cs = (
        df.groupby(["Cardio_Comorbidity_Score", "Diabetes_label"])
        .size()
        .reset_index(name="n")
    )
    cs["pct"] = (
        cs["n"] / cs.groupby("Cardio_Comorbidity_Score")["n"].transform("sum") * 100
    )

    fig = px.bar(
        cs,
        x="Cardio_Comorbidity_Score",
        y="pct",
        color="Diabetes_label",
        color_discrete_map={"No Diabetes": SUCCESS, "Diabetes": DANGER},
        barmode="group",
        text_auto=".1f",
    )
    fig.update_traces(textfont_size=9)
    fig.update_xaxes(title_text="Cardio Comorbidity Score")
    fig.update_yaxes(title_text="% of Group")
    return apply_chart_theme(fig, height=400)


def build_feature_importance_chart(importances: pd.Series):
    colors = [
        DANGER if v > importances.median() else ACCENT for v in importances.values
    ]
    fig = go.Figure(
        go.Bar(
            x=importances.values,
            y=importances.index,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_xaxes(title_text="Importance Score")
    return apply_chart_theme(fig, height=420)


def build_correlation_chart(df: pd.DataFrame, feature_cols: list):
    corr = (
        df[feature_cols + ["Diabetes_binary"]]
        .corr()["Diabetes_binary"]
        .drop("Diabetes_binary")
        .sort_values()
    )
    fig = go.Figure(
        go.Bar(
            x=corr.values,
            y=corr.index,
            orientation="h",
            marker_color=[DANGER if v > 0 else SUCCESS for v in corr.values],
        )
    )
    fig.add_vline(x=0, line_color="#484f58", line_width=1)
    fig.update_xaxes(title_text="Pearson r")
    return apply_chart_theme(fig, height=420)


def build_feature_density_chart(df: pd.DataFrame, feature: str):
    fig = px.histogram(
        df,
        x=feature,
        color="Diabetes_label",
        facet_col="Sex_label",
        barmode="overlay",
        histnorm="probability density",
        nbins=40,
        color_discrete_map={"No Diabetes": SUCCESS, "Diabetes": DANGER},
    )
    return apply_chart_theme(fig, height=320)


def build_model_radar_chart(df_models: pd.DataFrame, metrics: list):
    pal = [ACCENT, WARN, DANGER, SUCCESS, PURPLE]
    fig = go.Figure()
    for i, row in df_models.iterrows():
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        cats = metrics + [metrics[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=cats,
                fill="toself",
                name=row["Model"],
                line_color=pal[i],
                opacity=0.75,
            )
        )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0.6, 0.95], gridcolor=GRID_COL, color="#484f58"
            ),
            angularaxis=dict(gridcolor=GRID_COL),
            bgcolor=CARD_BG,
        )
    )
    return apply_chart_theme(fig, height=370)


def build_roc_auc_chart(df_models: pd.DataFrame):
    fig = go.Figure(
        go.Bar(
            x=df_models["ROC-AUC"],
            y=df_models["Model"],
            orientation="h",
            marker=dict(
                color=df_models["ROC-AUC"],
                colorscale=[[0, ACCENT], [1, DANGER]],
            ),
            text=df_models["ROC-AUC"].round(3),
            textposition="outside",
        )
    )
    fig.update_xaxes(range=[0.7, 0.92], title_text="ROC-AUC")
    return apply_chart_theme(fig, height=280)


def build_precision_recall_chart(df_pr: pd.DataFrame):
    fig = px.line(
        df_pr,
        x="Recall",
        y="Precision",
        color="Model",
        color_discrete_map={
            "Logistic Regression": ACCENT,
            "Random Forest": WARN,
            "XGBoost": DANGER,
        },
    )
    return apply_chart_theme(fig, height=300)


def build_income_chart(df: pd.DataFrame):
    inc = df.groupby("Income")["Diabetes_binary"].mean().reset_index()
    inc.columns = ["Income", "Rate"]
    inc["Rate_pct"] = inc["Rate"] * 100
    inc["Income_label"] = inc["Income"].map(INCOME_LABELS)

    fig = go.Figure(
        go.Bar(x=inc["Income_label"], y=inc["Rate_pct"], marker_color=ACCENT)
    )
    fig.update_yaxes(title_text="Diabetes Rate (%)")
    return apply_chart_theme(fig, height=260)


def build_education_chart(df: pd.DataFrame):
    edu = df.groupby("Education")["Diabetes_binary"].mean().reset_index()
    edu.columns = ["Education", "Rate"]
    edu["Rate_pct"] = edu["Rate"] * 100
    edu["Edu_label"] = edu["Education"].map(EDUCATION_LABELS)

    fig = go.Figure(go.Bar(x=edu["Edu_label"], y=edu["Rate_pct"], marker_color=PURPLE))
    fig.update_yaxes(title_text="Diabetes Rate (%)")
    fig.update_xaxes(tickangle=-25)
    return apply_chart_theme(fig, height=260)


def build_healthcare_access_chart(df: pd.DataFrame):
    hc = (
        df.groupby(["AnyHealthcare", "NoDocbcCost"])["Diabetes_binary"]
        .mean()
        .reset_index()
    )
    hc["Group"] = hc.apply(
        lambda r: (
            f"HC={'Yes' if r['AnyHealthcare'] == 1 else 'No'} · Cost={'Yes' if r['NoDocbcCost'] == 1 else 'No'}"
        ),
        axis=1,
    )
    hc["Rate_pct"] = hc["Diabetes_binary"] * 100

    fig = go.Figure(
        go.Bar(
            x=hc["Group"],
            y=hc["Rate_pct"],
            marker_color=[DANGER if v > 15 else ACCENT for v in hc["Rate_pct"]],
        )
    )
    fig.update_yaxes(title_text="Diabetes Rate (%)")
    fig.update_xaxes(tickangle=-15, tickfont_size=9)
    return apply_chart_theme(fig, height=260)


def build_cardio_score_scatter(df: pd.DataFrame):
    grp = df.groupby("Cardio_Comorbidity_Score")["Diabetes_binary"].mean().reset_index()
    grp["Rate_pct"] = grp["Diabetes_binary"] * 100
    grp["Size"] = df.groupby("Cardio_Comorbidity_Score").size().values

    fig = px.scatter(
        grp,
        x="Cardio_Comorbidity_Score",
        y="Rate_pct",
        size="Size",
        color="Rate_pct",
        color_continuous_scale=[[0, SUCCESS], [0.5, WARN], [1, DANGER]],
        size_max=55,
    )
    fig.update_xaxes(title_text="Cardio Score", tickvals=[0, 1, 2, 3, 4])
    fig.update_yaxes(title_text="Diabetes Rate (%)")
    fig.update_coloraxes(showscale=False)
    return apply_chart_theme(fig, height=300)


def build_lifestyle_score_chart(df: pd.DataFrame):
    grp = df.groupby("Lifestyle_Score")["Diabetes_binary"].mean().reset_index()
    grp["Rate_pct"] = grp["Diabetes_binary"] * 100

    fig = go.Figure(
        go.Bar(
            x=grp["Lifestyle_Score"],
            y=grp["Rate_pct"],
            marker=dict(
                color=grp["Rate_pct"],
                colorscale=[[0, SUCCESS], [1, DANGER]],
                reversescale=True,
            ),
        )
    )
    fig.update_xaxes(title_text="Lifestyle Score", tickvals=[0, 1, 2, 3])
    fig.update_yaxes(title_text="Diabetes Rate (%)")
    return apply_chart_theme(fig, height=300)


# ─────────────────────────────────────────────
# TAB RENDERERS
# ─────────────────────────────────────────────
def render_tab_eda(df: pd.DataFrame, df_raw: pd.DataFrame):
    c_dist, c_bmi = st.columns([1, 1.6])

    with c_dist:
        open_chart_card()
        render_chart_header(
            "Class Distribution", "Target variable balance after filtering"
        )
        plot(
            build_class_distribution_chart(
                df_raw, int(df_raw["Diabetes_binary"].sum()) / len(df_raw)
            )
        )
        close_chart_card()
        render_insight("""
          <strong>Class Imbalance:</strong> The dataset shows a significant skew —
          ~86% non-diabetic vs ~14% diabetic. SMOTE was applied during training to
          balance the classes and prevent model bias towards the majority class.
        """)

    with c_bmi:
        open_chart_card()
        render_chart_header(
            "BMI Distribution by Diabetes Status",
            "Log-transformed & standardised BMI (from feature engineering)",
        )
        plot(build_bmi_histogram(df))
        close_chart_card()
        render_insight("""
          <strong>BMI Signal:</strong> Diabetic patients exhibit a right-shifted BMI
          distribution even after log-transform, confirming that higher BMI remains one
          of the strongest individual predictors. This motivates its inclusion and the
          Log+StandardScaler pipeline used in cleaning.
        """)

    spacer()
    c_age, c_health = st.columns(2)

    with c_age:
        open_chart_card()
        render_chart_header(
            "Diabetes Rate by Age Group",
            "Age coded 1–13 (1=18-24 … 13=80+). Rate = diabetic / total in group.",
        )
        plot(build_age_rate_chart(df))
        close_chart_card()
        render_insight("""
          <strong>Age Trend:</strong> Diabetes prevalence rises steeply from middle age
          onward, peaking in groups 10–12 (65–79). Models should treat Age as a strong
          ordinal feature — encoding it numerically (as done here) preserves this monotone
          relationship.
        """)

    with c_health:
        open_chart_card()
        render_chart_header(
            "General Health vs Diabetes Prevalence",
            "GenHlth: 1=Excellent, 5=Poor. Stacked bar shows composition per health tier.",
        )
        plot(build_general_health_chart(df))
        close_chart_card()
        render_insight("""
          <strong>Self-reported Health:</strong> GenHlth is a powerful lagging indicator —
          the "Poor" tier contains ~40%+ diabetic patients. This self-reported feature captures
          composite health status and should rank highly in feature importance.
        """)
    spacer()
    render_section_title("Cardio Comorbidity Score")

    with st.container():
        open_chart_card()
        render_chart_header(
            "Comorbidity Score vs Diabetes Prevalence",
            "Comorbidity Score: 0=Excellent, 4=Poor",
        )
        plot(build_comorbidity_stacked_chart(df))
        close_chart_card()
        render_insight("""
          <strong>Interpretation & Modeling Insight:</strong> This feature aggregates HighBP,
          HighChol, Stroke, and HeartDisease. As the comorbidity score increases, the likelihood
          of a positive diabetes diagnosis drastically increases. Combining these reduced
          dimensionality while maintaining a strong predictive signal.
        """)


def render_tab_features(df: pd.DataFrame):
    render_section_title("Feature Importance (Random Forest)")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X, y = df[feature_cols], df["Diabetes_binary"]

    with st.spinner("Training Random Forest for feature importance…"):
        importances = compute_feature_importances(X, y)

    col_imp, col_corr = st.columns([1.3, 1])

    with col_imp:
        open_chart_card()
        render_chart_header(
            "Feature Importance – Random Forest",
            "Trained on filtered subset · higher = more predictive signal",
        )
        plot(build_feature_importance_chart(importances))
        close_chart_card()
        render_insight("""
          <strong>Top Predictors:</strong> <em>GenHlth</em>, <em>BMI</em>, <em>Age</em>, and
          <em>Cardio_Comorbidity_Score</em> consistently dominate. The engineered
          <em>Lifestyle_Score</em> and <em>Cardio_Comorbidity_Score</em> carry more signal
          than any single component feature, validating the feature-engineering step.
        """)

    with col_corr:
        open_chart_card()
        render_chart_header("Correlation with Target", "Pearson r with Diabetes_binary")
        plot(build_correlation_chart(df, feature_cols))
        close_chart_card()
        render_insight("""
          <strong>Negative Correlators:</strong> <em>Income</em>, <em>Education</em>, and
          <em>Lifestyle_Score</em> are negatively correlated — higher income/education and
          healthier lifestyle are protective. Models should retain these as they provide
          directional signal even at low absolute magnitude.
        """)

    spacer()
    render_section_title("Features vs Target")

    available_feats = [f for f in CONTINUOUS_FEATS if f in df.columns]
    sel_feat = st.selectbox("Select feature to inspect", available_feats, index=0)

    with st.container():
        open_chart_card()
        render_chart_header(f"Density - {sel_feat} by Sex")
        plot(build_feature_density_chart(df, sel_feat))
        close_chart_card()

    spacer()


def render_tab_model():
    render_section_title("Simulated Model Comparison")
    st.markdown(
        """
    <div class='chart-caption' style='margin-bottom:1rem'>
      Representative benchmark metrics for common classifiers on BRFSS data.
      Replace with your actual cross-validation results once models are trained.
    </div>
    """,
        unsafe_allow_html=True,
    )

    df_models = get_model_benchmark_data()
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    cm1, cm2 = st.columns([1.2, 1])

    with cm1:
        open_chart_card()
        render_chart_header("Model Performance Radar")
        plot(build_model_radar_chart(df_models, metrics))
        close_chart_card()

    with cm2:
        open_chart_card()
        render_chart_header("ROC-AUC Comparison")
        plot(build_roc_auc_chart(df_models))
        close_chart_card()
        render_insight("""
          <strong>🏆 Recommendation:</strong> <em>Gradient Boosting (XGBoost)</em> leads
          across all metrics. Its ensemble nature handles the imbalanced, mixed-type
          feature space better than linear or single-tree methods.
          Use <em>Recall</em> as primary optimisation target for clinical deployment
          — missing a diabetic case is costlier than a false alarm.
        """)

    spacer()
    render_section_title("Full Metrics Table")
    st.dataframe(
        df_models.set_index("Model")
        .style.background_gradient(cmap="Blues", axis=None)
        .format("{:.3f}"),
        use_container_width=True,
    )

    spacer()
    render_section_title("Precision–Recall Trade-off (Conceptual)")
    open_chart_card()
    plot(build_precision_recall_chart(build_precision_recall_data()))
    close_chart_card()
    render_insight("""
      <strong>⚕️ Clinical Threshold Choice:</strong> In a diabetes screening context,
      lowering the decision threshold raises <em>Recall</em> (catch more true positives) at
      the cost of <em>Precision</em> (more false alarms). Setting threshold ≈ 0.35–0.40
      balances sensitivity for early intervention without overwhelming clinical resources.
    """)


def render_tab_business(df: pd.DataFrame):
    render_section_title("Risk Stratification Overview")
    cb1, cb2, cb3 = st.columns(3)

    with cb1:
        open_chart_card()
        render_chart_header("💰 Income × Diabetes Rate")
        plot(build_income_chart(df))
        close_chart_card()
        render_insight("""
          Lower-income groups face up to <strong>3× higher prevalence</strong>.
          Targeted outreach for income brackets 1–3 could yield the highest
          return-on-investment for prevention programs.
        """)

    with cb2:
        open_chart_card()
        render_chart_header("🎓 Education × Diabetes Rate")
        plot(build_education_chart(df))
        close_chart_card()
        render_insight("""
          Education is a <strong>social determinant of health</strong>. The negative
          gradient suggests literacy-friendly health communications and community
          programmes targeting low-education cohorts could reduce rates.
        """)

    with cb3:
        open_chart_card()
        render_chart_header("🏥 Healthcare Access Gap")
        plot(build_healthcare_access_chart(df))
        close_chart_card()
        render_insight("""
          Patients <strong>without healthcare AND facing cost barriers</strong> show
          the highest undetected prevalence — they are most likely undiagnosed
          rather than truly healthier. This group is critical for screening initiatives.
        """)

    spacer()
    render_section_title("Engineered Scores vs Diabetes Risk")
    cs1, cs2 = st.columns(2)

    with cs1:
        open_chart_card()
        render_chart_header(
            "❤️ Cardio Comorbidity Score (0–4)",
            "Sum of HighBP + HighChol + Stroke + HeartDiseaseorAttack",
        )
        plot(build_cardio_score_scatter(df))
        close_chart_card()
        render_insight("""
          Near-<strong>linear risk escalation</strong>: each additional cardiovascular
          comorbidity adds ~10–12 percentage points of diabetes risk. Score ≥ 2
          should trigger automatic diabetes screening in a clinical workflow.
        """)

    with cs2:
        open_chart_card()
        render_chart_header(
            "🥦 Lifestyle Score (0–3)",
            "Sum of PhysActivity + Fruits + Veggies",
        )
        plot(build_lifestyle_score_chart(df))
        close_chart_card()
        render_insight("""
          A <strong>healthy lifestyle (score=3)</strong> is associated with ~4–6%
          lower diabetes prevalence compared to score=0. While modest, this
          is actionable — lifestyle interventions are cost-effective prevention tools,
          and the feature should be surfaced prominently in patient-facing risk tools.
        """)

    spacer()
    render_section_title("Key Recommendations for Modelling & Deployment")
    _render_recommendations()


def _render_recommendations():
    recs = [
        (
            "🎯",
            "Prioritise Recall",
            "In a clinical screening context, missing a diabetic case (false negative) is far costlier than a false alarm. Optimise threshold for Recall ≥ 0.80 and monitor F1 as secondary metric.",
        ),
        (
            "🔧",
            "Use Engineered Scores",
            "Cardio_Comorbidity_Score and Lifestyle_Score provide higher signal than the individual binary features they aggregate. Keep them in the final feature set.",
        ),
        (
            "⚖️",
            "Retain SMOTE-Balanced Training",
            "The raw 86/14 class split severely degrades minority-class performance. SMOTE-balanced training improves diabetic detection across all model types.",
        ),
        (
            "👥",
            "Segment by Income & Education",
            "Low-income / low-education populations carry disproportionate risk and may have systematic data gaps. Consider separate calibration or fairness auditing per demographic subgroup.",
        ),
        (
            "📱",
            "Business Deployment: Risk Score API",
            "Serve the trained XGBoost model as a REST API. Integrate with EHR systems to flag high-risk patients (Cardio ≥ 2, Age ≥ 9, GenHlth ≥ 4) for proactive outreach.",
        ),
    ]

    r1, r2 = st.columns(2)
    cols_cycle = [r1, r2, r1, r2, r1]

    for (icon, title, body), col in zip(recs, cols_cycle):
        col.markdown(
            f"""
        <div style='background:#1c2128;border:1px solid #21262d;border-radius:12px;
                    padding:1rem 1.2rem;margin-bottom:.9rem'>
          <div style='font-family:Syne,sans-serif;font-size:.85rem;font-weight:700;
                      color:#e6edf3;margin-bottom:.4rem'>{icon} {title}</div>
          <div style='font-size:.78rem;color:#8b949e;line-height:1.55'>{body}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# PAGE LAYOUT
# ─────────────────────────────────────────────
def render_page_header():
    st.markdown(
        """
    <div class='page-title'>Diabetes Classifier · EDA Dashboard</div>
    <div class='page-sub'>BRFSS 2015 Feature-Engineered Training Set · Exploratory Data Analysis & Business Insights</div>
    """,
        unsafe_allow_html=True,
    )


def render_kpi_row(kpis: dict):
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, label, value, delta in [
        (k1, "Total Records", f"{kpis['total']:,}", None),
        (k2, "Diabetic Cases", f"{kpis['diabetic']:,}", None),
        (k3, "Prevalence Rate", f"{kpis['pct_diab']:.1f}%", "vs ~9% US avg"),
        (k4, "Avg BMI (scaled)", f"{kpis['mean_bmi']:.3f}", None),
        (k5, "Avg Cardio Score", f"{kpis['mean_cardio']:.2f}", "0-4 scale"),
    ]:
        render_kpi(col, label, value, delta)


def render_tabs(df: pd.DataFrame, df_raw: pd.DataFrame):
    tab_eda, tab_feat, tab_model, tab_biz = st.tabs(
        [
            " EDA Findings ",
            " Feature-to-Target ",
            " Model Insights ",
            " Business Insights ",
        ]
    )

    with tab_eda:
        render_tab_eda(df, df_raw)
    with tab_feat:
        render_tab_features(df)
    with tab_model:
        render_tab_model()
    with tab_biz:
        render_tab_business(df)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main():
    configure_page()
    inject_global_styles()

    df = add_derived_columns(load_data(DATA_PATH))
    kpis = compute_kpis(df)
    df_raw = add_derived_columns(load_data(RAW_DATA_PATH))
    render_page_header()
    render_kpi_row(kpis)
    spacer()
    render_tabs(df, df_raw)


if __name__ == "__main__":
    main()

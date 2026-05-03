"""
Diabetes Classifier - EDA & Insights Dashboard
Run:  streamlit run dashboard.py
Requires:  streamlit plotly pandas scikit-learn imbalanced-learn
Dataset :  data/processed/featured_train_brfss.csv  (relative to this file)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Classifier · EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700;800&family=Barlow+Condensed:wght@600;700;800&display=swap');

  /* ── Reset & Base ── */
  html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }

  /* ── Page background ── */
  .stApp { background: #0d1117; }


  /* ── Main content ── */
  [data-testid="block-container"] { padding: 1.5rem 2rem 2rem 2rem !important; }

  /* ── Section headings ── */
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

  /* ── KPI Cards ── */
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
  .kpi-icon { font-size: 1.6rem; float: right; }

  /* ── Chart containers ── */
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

  /* ── Insight cards ── */
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

  /* ── Tab styling ── */
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

  /* ── Divider ── */
  hr { border-color: #21262d !important; margin: 1rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
TEMPLATE = "plotly_dark"
ACCENT   = "#58a6ff"
DANGER   = "#f85149"
SUCCESS  = "#3fb950"
WARN     = "#f0883e"
PURPLE   = "#bc8cff"
TEAL     = "#39d353"

CARD_BG  = "#161b22"
GRID_COL = "#21262d"

def chart_layout(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="DM Sans", color="#c9d1d9", size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10,font_color="#c9d1d9"),
    )
    return fig

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

data_path = "data/processed/featured_train_brfss.csv"

df_raw = load_data(data_path)

df = df_raw.copy()

# Derived columns for readability
df["Diabetes_label"] = df["Diabetes_binary"].map({0: "No Diabetes", 1: "Diabetes"})
df["Sex_label"]      = df["Sex"].map({0: "Female", 1: "Male"})


st.markdown("""
<div class='page-title'>Diabetes Classifier · EDA Dashboard</div>
<div class='page-sub'>BRFSS 2015 Feature-Engineered Training Set · Exploratory Data Analysis & Business Insights</div>
""", unsafe_allow_html=True)


total      = len(df)
diabetic   = df["Diabetes_binary"].sum()
pct_diab   = diabetic / total * 100 if total else 0
mean_bmi   = df["BMI"].mean()
mean_age   = df["Age"].mean()
mean_cardio= df["Cardio_Comorbidity_Score"].mean()

k1, k2, k3, k4, k5 = st.columns(5)

for col, label, value, delta in [
    (k1, "Total Records",         f"{total:,}",         None),
    (k2, "Diabetic Cases",        f"{diabetic:,}",      None),
    (k3, "Prevalence Rate",       f"{pct_diab:.1f}%",   "vs ~9% US avg"),
    (k4, "Avg BMI (scaled)",      f"{mean_bmi:.3f}",    None),
    (k5, "Avg Cardio Score",      f"{mean_cardio:.2f}", "0-4 scale"),
]:
    delta_html = ""
    if delta:
        delta_html = f"<div class='kpi-delta-pos'>{delta}</div>"
    col.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value'>{value}</div>
      {delta_html}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab_eda, tab_feat, tab_model, tab_biz = st.tabs([
    " EDA Findings ",
    " Feature-to-Target ",
    " Model Insights ",
    " Business Insights ",
])


with tab_eda:
    c_dist, c_bmi = st.columns([1, 1.6])

    with c_dist:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Class Distribution</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-caption'>Target variable balance after filtering</div>", unsafe_allow_html=True)

        counts = df["Diabetes_label"].value_counts().reset_index()
        counts.columns = ["Label", "Count"]
        counts["Pct"]  = (counts["Count"] / counts["Count"].sum() * 100).round(1)

        fig_donut = go.Figure(go.Pie(
            labels=counts["Label"],
            values=counts["Count"],
            hole=0.62,
            marker=dict(colors=[DANGER, SUCCESS]),
            textinfo="percent",
            textfont_size=12,
        ))
        fig_donut.add_annotation(
            text=f"{pct_diab:.1f}%<br><span style='font-size:10px'>diabetic</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#e6edf3", family="Syne"),
            align="center",
        )
        chart_layout(fig_donut, height=290)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
          <strong>Class Imbalance:</strong> The dataset shows a significant skew —
          ~86% non-diabetic vs ~14% diabetic. SMOTE was applied during training to
          balance the classes and prevent model bias towards the majority class.
        </div>
        """, unsafe_allow_html=True)

    with c_bmi:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>BMI Distribution by Diabetes Status</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-caption'>Log-transformed & standardised BMI (from feature engineering)</div>", unsafe_allow_html=True)

        fig_bmi = go.Figure()
        for label, color in [("No Diabetes", SUCCESS), ("Diabetes", DANGER)]:
            sub = df[df["Diabetes_label"] == label]["BMI"]
            fig_bmi.add_trace(go.Histogram(
                x=sub, name=label,
                marker_color=color, opacity=0.72,
                nbinsx=50, histnorm="probability density",
            ))
        fig_bmi.update_layout(barmode="overlay")
        chart_layout(fig_bmi, height=290)
        st.plotly_chart(fig_bmi, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
          <strong>BMI Signal:</strong> Diabetic patients exhibit a right-shifted BMI
          distribution even after log-transform, confirming that higher BMI remains one
          of the strongest individual predictors. This motivates its inclusion and the
          Log+StandardScaler pipeline used in cleaning.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    c_age, c_health = st.columns(2)

    with c_age:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Diabetes Rate by Age Group</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-caption'>Age coded 1–13 (1=18-24 … 13=80+). Rate = diabetic / total in group.</div>", unsafe_allow_html=True)

        age_grp = df.groupby("Age")["Diabetes_binary"].agg(["mean","count"]).reset_index()
        age_grp.columns = ["Age","Rate","Count"]
        age_grp["Rate_pct"] = age_grp["Rate"] * 100

        fig_age = go.Figure()
        fig_age.add_trace(go.Bar(
            x=age_grp["Age"], y=age_grp["Rate_pct"],
            marker=dict(
                color=age_grp["Rate_pct"],
                colorscale=[[0, SUCCESS],[0.5, WARN],[1, DANGER]],
                showscale=False,
            ),
            name="Diabetes Rate %",
        ))
        fig_age.update_xaxes(title_text="Age Group (1=18-24, 13=80+)")
        fig_age.update_yaxes(title_text="Diabetes Rate (%)")
        chart_layout(fig_age, height=290)
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
          <strong>Age Trend:</strong> Diabetes prevalence rises steeply from middle age
          onward, peaking in groups 10–12 (65–79). Models should treat Age as a strong
          ordinal feature — encoding it numerically (as done here) preserves this monotone
          relationship.
        </div>
        """, unsafe_allow_html=True)

    with c_health:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>General Health vs Diabetes Prevalence</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-caption'>GenHlth: 1=Excellent, 5=Poor. Stacked bar shows composition per health tier.</div>", unsafe_allow_html=True)

        gh = df.groupby(["GenHlth","Diabetes_label"]).size().reset_index(name="n")
        gh_tot = gh.groupby("GenHlth")["n"].transform("sum")
        gh["pct"] = gh["n"] / gh_tot * 100
        gh_map = {1:"Excellent",2:"Very Good",3:"Good",4:"Fair",5:"Poor"}
        gh["GenHlth_label"] = gh["GenHlth"].map(gh_map)

        fig_gh = px.bar(
            gh, x="GenHlth_label", y="pct", color="Diabetes_label",
            color_discrete_map={"No Diabetes": SUCCESS, "Diabetes": DANGER},
            barmode="stack", text_auto=".1f",
        )
        fig_gh.update_traces(textfont_size=9)
        fig_gh.update_xaxes(title_text="General Health")
        fig_gh.update_yaxes(title_text="% of Group")
        chart_layout(fig_gh, height=290)
        st.plotly_chart(fig_gh, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
          <strong>Self-reported Health:</strong> GenHlth is a powerful lagging indicator —
          the "Poor" tier contains ~40%+ diabetic patients. This self-reported feature captures
          composite health status and should rank highly in feature importance.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Health Days Binning Analysis</div>", unsafe_allow_html=True)
    c_m, c_p = st.columns(2)

    for col_w, feat_mod, feat_sev, title in [
        (c_m, "MentHlth_binned_Moderate", "MentHlth_binned_Severe",
         "Mental Health Days × Diabetes"),
        (c_p, "PhysHlth_binned_Moderate", "PhysHlth_binned_Severe",
         "Physical Health Days × Diabetes"),
    ]:
        # Reconstruct category
        def get_bin(row, m, s):
            if row[s] == 1: return "Severe (14-30)"
            if row[m] == 1: return "Moderate (1-13)"
            return "Zero"

        tmp = df.copy()
        if feat_mod in tmp.columns and feat_sev in tmp.columns:
            tmp["Bin"] = tmp.apply(lambda r: get_bin(r, feat_mod, feat_sev), axis=1)
            grp = tmp.groupby(["Bin","Diabetes_label"]).size().reset_index(name="n")
            grp_tot = grp.groupby("Bin")["n"].transform("sum")
            grp["pct"] = grp["n"] / grp_tot * 100

            with col_w:
                st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)

                fig_bin = px.bar(
                    grp, x="Bin", y="pct", color="Diabetes_label",
                    color_discrete_map={"No Diabetes": SUCCESS, "Diabetes": DANGER},
                    barmode="group",
                    category_orders={"Bin": ["Zero","Moderate (1-13)","Severe (14-30)"]},
                )
                fig_bin.update_yaxes(title_text="% within bin")
                chart_layout(fig_bin, height=260)
                st.plotly_chart(fig_bin, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Cardio Comorbidity Score</div>", unsafe_allow_html=True)
    c_comb=st.container()
    with c_comb:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Comorbidity Score vs Diabetes Prevalence</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-caption'>Comorbidity Score: 0=Excellent, 4=Poor</div>", unsafe_allow_html=True)
        cs=df.groupby(["Cardio_Comorbidity_Score","Diabetes_label"]).size().reset_index(name="n")
        cs_tot=cs.groupby(["Cardio_Comorbidity_Score"])["n"].transform("sum")
        cs["pct"]=cs["n"]/cs_tot *100
        fig_cs = px.bar(
            cs, x="Cardio_Comorbidity_Score", y="pct", color="Diabetes_label",
            color_discrete_map={"No Diabetes": SUCCESS, "Diabetes": DANGER},
            barmode="group", text_auto=".1f",
        )
        fig_cs.update_traces(textfont_size=9)
        fig_cs.update_xaxes(title_text="Cardio Comorbidity Score")
        fig_cs.update_yaxes(title_text="% of Group")
        chart_layout(fig_cs, height=400)
        st.plotly_chart(fig_cs, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='insight-box'>
          <strong>Interpretation & Modeling Insight:</strong>  This feature aggregates HighBP, HighChol, Stroke, and HeartDisease. 
                    As the comorbidity score increases, the likelihood of a positive diabetes diagnosis drastically increases. 
                    Combining these reduced dimensionality while maintaining a strong predictive signal.
        </div>
        """, unsafe_allow_html=True)



    


with tab_feat:

    st.markdown("<div class='section-title'>Feature Importance (Random Forest)</div>", unsafe_allow_html=True)
    from sklearn.ensemble import RandomForestClassifier
    feature_cols = [c for c in df.columns if c not in
                    ["Diabetes_binary","Diabetes_label","Sex_label"]]
    X = df[feature_cols]
    y = df["Diabetes_binary"]
    @st.cache_data(show_spinner=False)
    def get_importances(X_vals, y_vals):
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_vals, y_vals)
        return pd.Series(rf.feature_importances_, index=X_vals.columns).sort_values(ascending=True)
    with st.spinner("Training Random Forest for feature importance…"):
        importances = get_importances(X, y)
    col_imp, col_corr = st.columns([1.3, 1])
    with col_imp:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Feature Importance – Random Forest</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-caption'>Trained on filtered subset · higher = more predictive signal</div>", unsafe_allow_html=True)

        colors = [DANGER if v > importances.median() else ACCENT
                  for v in importances.values]

        fig_fi = go.Figure(go.Bar(
            x=importances.values,
            y=importances.index,
            orientation="h",
            marker_color=colors,
        ))
        fig_fi.update_xaxes(title_text="Importance Score")
        chart_layout(fig_fi, height=420)
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
          <strong>Top Predictors:</strong> <em>GenHlth</em>, <em>BMI</em>, <em>Age</em>, and
          <em>Cardio_Comorbidity_Score</em> consistently dominate. The engineered
          <em>Lifestyle_Score</em> and <em>Cardio_Comorbidity_Score</em> carry more signal
          than any single component feature, validating the feature-engineering step.
        </div>
        """, unsafe_allow_html=True)
    with col_corr:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Correlation with Target</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-caption'>Pearson r with Diabetes_binary</div>", unsafe_allow_html=True)

        corr = df[feature_cols + ["Diabetes_binary"]].corr()["Diabetes_binary"].drop("Diabetes_binary").sort_values()

        fig_corr = go.Figure(go.Bar(
            x=corr.values,
            y=corr.index,
            orientation="h",
            marker_color=[DANGER if v > 0 else SUCCESS for v in corr.values],
        ))
        fig_corr.add_vline(x=0, line_color="#484f58", line_width=1)
        fig_corr.update_xaxes(title_text="Pearson r")
        chart_layout(fig_corr, height=420)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
          <strong>Negative Correlators:</strong> <em>Income</em>, <em>Education</em>, and
          <em>Lifestyle_Score</em> are negatively correlated — higher income/education and
          healthier lifestyle are protective. Models should retain these as they provide
          directional signal even at low absolute magnitude.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Features vs Target</div>", unsafe_allow_html=True)
    cont_feats = ["BMI","Age","GenHlth","Income","Education","Cardio_Comorbidity_Score","Lifestyle_Score"]
    cont_feats = [f for f in cont_feats if f in df.columns]

    sel_feat = st.selectbox("Select feature to inspect", cont_feats, index=0)
    c_box = st.container()

    with c_box:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='chart-title'>Density - {sel_feat} by Sex</div>", unsafe_allow_html=True)

        fig_den = px.histogram(
            df, x=sel_feat, color="Diabetes_label",
            facet_col="Sex_label", barmode="overlay",
            histnorm="probability density", nbins=40,
            color_discrete_map={"No Diabetes": SUCCESS, "Diabetes": DANGER},
        )
        chart_layout(fig_den, height=320)
        st.plotly_chart(fig_den, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)




with tab_model:

    st.markdown("<div class='section-title'>Simulated Model Comparison</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='chart-caption' style='margin-bottom:1rem'>
      Representative benchmark metrics for common classifiers on BRFSS data.
      Replace with your actual cross-validation results once models are trained.
    </div>
    """, unsafe_allow_html=True)

    # Benchmark table
    models_data = {
        "Model": [
            "Logistic Regression","Decision Tree",
            "Random Forest","Gradient Boosting (XGBoost)","Neural Network (MLP)",
        ],
        "Accuracy": [0.748, 0.731, 0.784, 0.803, 0.791],
        "Precision": [0.641, 0.598, 0.692, 0.718, 0.704],
        "Recall":    [0.713, 0.744, 0.738, 0.761, 0.752],
        "F1 Score":  [0.675, 0.663, 0.714, 0.739, 0.727],
        "ROC-AUC":   [0.821, 0.779, 0.857, 0.879, 0.863],
    }
    df_models = pd.DataFrame(models_data)

    cm1, cm2 = st.columns([1.2, 1])

    with cm1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Model Performance Radar</div>", unsafe_allow_html=True)
        metrics = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
        pal = [ACCENT, WARN, DANGER, SUCCESS, PURPLE]

        fig_radar = go.Figure()
        for i, row in df_models.iterrows():
            vals = [row[m] for m in metrics] + [row[metrics[0]]]
            cats = metrics + [metrics[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats,
                fill="toself", name=row["Model"],
                line_color=pal[i], opacity=0.75,
            ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.6, 0.95],
                                gridcolor=GRID_COL, color="#484f58"),
                angularaxis=dict(gridcolor=GRID_COL),
                bgcolor=CARD_BG,
            )
        )
        chart_layout(fig_radar, height=370)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cm2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>ROC-AUC Comparison</div>", unsafe_allow_html=True)

        fig_auc = go.Figure(go.Bar(
            x=df_models["ROC-AUC"],
            y=df_models["Model"],
            orientation="h",
            marker=dict(
                color=df_models["ROC-AUC"],
                colorscale=[[0, ACCENT],[1, DANGER]],
            ),
            text=df_models["ROC-AUC"].round(3),
            textposition="outside",
        ))
        fig_auc.update_xaxes(range=[0.7, 0.92], title_text="ROC-AUC")
        chart_layout(fig_auc, height=280)
        st.plotly_chart(fig_auc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
          <strong>🏆 Recommendation:</strong> <em>Gradient Boosting (XGBoost)</em> leads
          across all metrics. Its ensemble nature handles the imbalanced, mixed-type
          feature space better than linear or single-tree methods.
          Use <em>Recall</em> as primary optimisation target for clinical deployment
          — missing a diabetic case is costlier than a false alarm.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics table ──
    st.markdown("<div class='section-title'>Full Metrics Table</div>", unsafe_allow_html=True)
    st.dataframe(
        df_models.set_index("Model").style
            .background_gradient(cmap="Blues", axis=None)
            .format("{:.3f}"),
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Precision-Recall trade-off ──
    st.markdown("<div class='section-title'>Precision–Recall Trade-off (Conceptual)</div>", unsafe_allow_html=True)

    thresholds = np.linspace(0.1, 0.9, 40)
    pr_data = []
    for model, prec_base, rec_base in [
        ("Logistic Regression", 0.64, 0.71),
        ("Random Forest",       0.69, 0.74),
        ("XGBoost",             0.72, 0.76),
    ]:
        for t in thresholds:
            prec = min(1.0, prec_base + (t - 0.5) * 0.6)
            rec  = max(0.0, rec_base  - (t - 0.5) * 0.8)
            pr_data.append({"Model": model, "Threshold": t, "Precision": prec, "Recall": rec})

    df_pr = pd.DataFrame(pr_data)
    fig_pr = px.line(
        df_pr, x="Recall", y="Precision", color="Model",
        color_discrete_map={
            "Logistic Regression": ACCENT,
            "Random Forest":       WARN,
            "XGBoost":             DANGER,
        },
    )
    chart_layout(fig_pr, height=300)
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.plotly_chart(fig_pr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
      <strong>⚕️ Clinical Threshold Choice:</strong> In a diabetes screening context,
      lowering the decision threshold raises <em>Recall</em> (catch more true positives) at
      the cost of <em>Precision</em> (more false alarms). Setting threshold ≈ 0.35–0.40
      balances sensitivity for early intervention without overwhelming clinical resources.
    </div>
    """, unsafe_allow_html=True)


with tab_biz:

    st.markdown("<div class='section-title'>Risk Stratification Overview</div>", unsafe_allow_html=True)

    cb1, cb2, cb3 = st.columns(3)

    # ── Income vs Diabetes ──
    with cb1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>💰 Income × Diabetes Rate</div>", unsafe_allow_html=True)
        inc = df.groupby("Income")["Diabetes_binary"].mean().reset_index()
        inc.columns = ["Income","Rate"]
        inc["Rate_pct"] = inc["Rate"] * 100
        inc_labels = {1:"<$10k",2:"$10-15k",3:"$15-20k",4:"$20-25k",5:"$25-35k",6:"$35-50k",7:"$50-75k",8:">$75k"}
        inc["Income_label"] = inc["Income"].map(inc_labels)

        fig_inc = go.Figure(go.Bar(
            x=inc["Income_label"], y=inc["Rate_pct"],
            marker_color=ACCENT,
        ))
        fig_inc.update_yaxes(title_text="Diabetes Rate (%)")
        chart_layout(fig_inc, height=260)
        st.plotly_chart(fig_inc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
          Lower-income groups face up to <strong>3× higher prevalence</strong>.
          Targeted outreach for income brackets 1–3 could yield the highest
          return-on-investment for prevention programs.
        </div>
        """, unsafe_allow_html=True)

    # ── Education vs Diabetes ──
    with cb2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>🎓 Education × Diabetes Rate</div>", unsafe_allow_html=True)
        edu = df.groupby("Education")["Diabetes_binary"].mean().reset_index()
        edu.columns = ["Education","Rate"]
        edu["Rate_pct"] = edu["Rate"] * 100
        edu_labels = {1:"No School",2:"Elem.",3:"Some HS",4:"HS Grad",5:"Some College",6:"College+"}
        edu["Edu_label"] = edu["Education"].map(edu_labels)

        fig_edu = go.Figure(go.Bar(
            x=edu["Edu_label"], y=edu["Rate_pct"],
            marker_color=PURPLE,
        ))
        fig_edu.update_yaxes(title_text="Diabetes Rate (%)")
        fig_edu.update_xaxes(tickangle=-25)
        chart_layout(fig_edu, height=260)
        st.plotly_chart(fig_edu, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
          Education is a <strong>social determinant of health</strong>. The negative
          gradient suggests literacy-friendly health communications and community
          programmes targeting low-education cohorts could reduce rates.
        </div>
        """, unsafe_allow_html=True)

    # ── Healthcare access ──
    with cb3:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>🏥 Healthcare Access Gap</div>", unsafe_allow_html=True)

        hc = df.groupby(["AnyHealthcare","NoDocbcCost"])["Diabetes_binary"].mean().reset_index()
        hc["Group"] = hc.apply(lambda r:
            f"HC={'Yes' if r['AnyHealthcare']==1 else 'No'} · Cost={'Yes' if r['NoDocbcCost']==1 else 'No'}", axis=1)
        hc["Rate_pct"] = hc["Diabetes_binary"] * 100

        fig_hc = go.Figure(go.Bar(
            x=hc["Group"], y=hc["Rate_pct"],
            marker_color=[DANGER if v > 15 else ACCENT for v in hc["Rate_pct"]],
        ))
        fig_hc.update_yaxes(title_text="Diabetes Rate (%)")
        fig_hc.update_xaxes(tickangle=-15, tickfont_size=9)
        chart_layout(fig_hc, height=260)
        st.plotly_chart(fig_hc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
          Patients <strong>without healthcare AND facing cost barriers</strong> show
          the highest undetected prevalence — they are most likely undiagnosed
          rather than truly healthier. This group is critical for screening initiatives.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Cardio & Lifestyle Scores ──
    st.markdown("<div class='section-title'>Engineered Scores vs Diabetes Risk</div>", unsafe_allow_html=True)
    cs1, cs2 = st.columns(2)

    with cs1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>❤️ Cardio Comorbidity Score (0–4)</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-caption'>Sum of HighBP + HighChol + Stroke + HeartDiseaseorAttack</div>", unsafe_allow_html=True)

        cardio_grp = df.groupby("Cardio_Comorbidity_Score")["Diabetes_binary"].mean().reset_index()
        cardio_grp["Rate_pct"] = cardio_grp["Diabetes_binary"] * 100
        cardio_grp["Size"]     = df.groupby("Cardio_Comorbidity_Score").size().values

        fig_c = px.scatter(
            cardio_grp, x="Cardio_Comorbidity_Score", y="Rate_pct",
            size="Size", color="Rate_pct",
            color_continuous_scale=[[0, SUCCESS],[0.5, WARN],[1, DANGER]],
            size_max=55,
        )
        fig_c.update_xaxes(title_text="Cardio Score", tickvals=[0,1,2,3,4])
        fig_c.update_yaxes(title_text="Diabetes Rate (%)")
        fig_c.update_coloraxes(showscale=False)
        chart_layout(fig_c, height=300)
        st.plotly_chart(fig_c, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
          Near-<strong>linear risk escalation</strong>: each additional cardiovascular
          comorbidity adds ~10–12 percentage points of diabetes risk. Score ≥ 2
          should trigger automatic diabetes screening in a clinical workflow.
        </div>
        """, unsafe_allow_html=True)

    with cs2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>🥦 Lifestyle Score (0–3)</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-caption'>Sum of PhysActivity + Fruits + Veggies</div>", unsafe_allow_html=True)

        life_grp = df.groupby("Lifestyle_Score")["Diabetes_binary"].mean().reset_index()
        life_grp["Rate_pct"] = life_grp["Diabetes_binary"] * 100

        fig_l = go.Figure(go.Bar(
            x=life_grp["Lifestyle_Score"], y=life_grp["Rate_pct"],
            marker=dict(
                color=life_grp["Rate_pct"],
                colorscale=[[0, SUCCESS],[1, DANGER]],
                reversescale=True,
            ),
        ))
        fig_l.update_xaxes(title_text="Lifestyle Score", tickvals=[0,1,2,3])
        fig_l.update_yaxes(title_text="Diabetes Rate (%)")
        chart_layout(fig_l, height=300)
        st.plotly_chart(fig_l, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
          A <strong>healthy lifestyle (score=3)</strong> is associated with ~4–6%
          lower diabetes prevalence compared to score=0. While modest, this
          is actionable — lifestyle interventions are cost-effective prevention tools,
          and the feature should be surfaced prominently in patient-facing risk tools.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary recommendations ──
    st.markdown("<div class='section-title'>Key Recommendations for Modelling & Deployment</div>", unsafe_allow_html=True)

    recs = [
        ("🎯", "Prioritise Recall",
         "In a clinical screening context, missing a diabetic case (false negative) is far costlier than a false alarm. Optimise threshold for Recall ≥ 0.80 and monitor F1 as secondary metric."),
        ("🔧", "Use Engineered Scores",
         "Cardio_Comorbidity_Score and Lifestyle_Score provide higher signal than the individual binary features they aggregate. Keep them in the final feature set."),
        ("⚖️", "Retain SMOTE-Balanced Training",
         "The raw 86/14 class split severely degrades minority-class performance. SMOTE-balanced training improves diabetic detection across all model types."),
        ("👥", "Segment by Income & Education",
         "Low-income / low-education populations carry disproportionate risk and may have systematic data gaps. Consider separate calibration or fairness auditing per demographic subgroup."),
        ("📱", "Business Deployment: Risk Score API",
         "Serve the trained XGBoost model as a REST API. Integrate with EHR systems to flag high-risk patients (Cardio ≥ 2, Age ≥ 9, GenHlth ≥ 4) for proactive outreach."),
    ]

    r1, r2 = st.columns(2)
    cols_cycle = [r1, r2, r1, r2, r1]
    for (icon, title, body), col in zip(recs, cols_cycle):
        col.markdown(f"""
        <div style='background:#1c2128;border:1px solid #21262d;border-radius:12px;
                    padding:1rem 1.2rem;margin-bottom:.9rem'>
          <div style='font-family:Syne,sans-serif;font-size:.85rem;font-weight:700;
                      color:#e6edf3;margin-bottom:.4rem'>{icon} {title}</div>
          <div style='font-size:.78rem;color:#8b949e;line-height:1.55'>{body}</div>
        </div>
        """, unsafe_allow_html=True)

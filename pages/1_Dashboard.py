import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from ml_system import load_and_merge_data

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

# ─── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .section-header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      padding: 0.6rem 1.2rem;
      border-radius: 10px;
      color: white;
      font-size: 1.15rem;
      font-weight: 700;
      margin: 1.2rem 0 0.8rem 0;
      letter-spacing: 0.5px;
  }
  .metric-card {
      background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
      border-radius: 12px;
      padding: 1rem 1.2rem;
      color: white;
      text-align: center;
      box-shadow: 0 4px 15px rgba(0,0,0,0.15);
  }
  .metric-card .value { font-size: 2rem; font-weight: 700; }
  .metric-card .label { font-size: 0.8rem; opacity: 0.85; margin-top: 0.2rem; }
  .chart-tip {
      font-size: 0.78rem;
      color: #888;
      margin-top: -0.4rem;
      margin-bottom: 0.5rem;
      font-style: italic;
  }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ────────────────────────────────────────────────────────────
with st.spinner("Loading student data..."):
    df = load_and_merge_data()

study_mapping = {1: "<2 hrs", 2: "2–5 hrs", 3: "5–10 hrs", 4: ">10 hrs"}
famrel_map    = {1: "Very Bad", 2: "Bad", 3: "Neutral", 4: "Good", 5: "Excellent"}

# ─── Sidebar Controls ────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Dashboard Controls")
selected_subject = st.sidebar.selectbox("📚 Subject", ["Math", "Portuguese"])
suffix = "_mat" if selected_subject == "Math" else "_por"

df["studytime_label"] = df[f"studytime{suffix}"].map(study_mapping)
df["famrel_label"]    = df[f"famrel{suffix}"].map(famrel_map)

# ─── Grade filter ────────────────────────────────────────────────────────────
g3_min, g3_max = int(df[f"G3{suffix}"].min()), int(df[f"G3{suffix}"].max())
grade_range = st.sidebar.slider(
    f"🎯 Filter Final Grade (G3 – {selected_subject})",
    g3_min, g3_max, (g3_min, g3_max)
)
df_f = df[(df[f"G3{suffix}"] >= grade_range[0]) & (df[f"G3{suffix}"] <= grade_range[1])].copy()

# ─── Gender filter ───────────────────────────────────────────────────────────
gender_opts = ["All"] + sorted(df_f["sex"].unique().tolist())
sel_gender = st.sidebar.selectbox("👤 Gender", gender_opts)
if sel_gender != "All":
    df_f = df_f[df_f["sex"] == sel_gender]

# ─── Internet filter ─────────────────────────────────────────────────────────
internet_opts = ["All", "yes", "no"]
sel_internet = st.sidebar.selectbox("🌐 Internet Access", internet_opts)
if sel_internet != "All":
    df_f = df_f[df_f["internet"] == sel_internet]

st.sidebar.markdown(f"**Students in view:** `{len(df_f)}`")

# ─── Page Header ────────────────────────────────────────────────────────────
st.title("📊 Exploratory Data Analysis Dashboard")
st.markdown(
    f"Interactive visualisations of the **{selected_subject}** dataset  "
    f"({len(df_f)} students shown after filters)."
)

# ─── KPI Cards ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    ("Total Students", len(df_f)),
    ("Features", df.shape[1]),
    (f"Avg G3 ({selected_subject})", round(df_f[f"G3{suffix}"].mean(), 2)),
    (f"Pass Rate (≥10)", f"{(df_f[f'G3{suffix}']>=10).mean()*100:.1f}%"),
    ("Avg Absences", round(df_f[f"absences{suffix}"].mean(), 1)),
]
for col, (label, value) in zip([k1, k2, k3, k4, k5], kpis):
    col.markdown(
        f'<div class="metric-card"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – UNIVARIATE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 1 · Univariate Analysis</div>', unsafe_allow_html=True)
st.markdown('<p class="chart-tip">Hover over elements for details. Use the legend to toggle series.</p>', unsafe_allow_html=True)

u_col1, u_col2 = st.columns(2)

# 1-A  Histogram – Final Grade Distribution (G3)
with u_col1:
    st.subheader(f"Final Grade Distribution – G3 ({selected_subject})")
    fig_u1 = px.histogram(
        df_f, x=f"G3{suffix}", nbins=20,
        color_discrete_sequence=["#667eea"],
        marginal="violin",
        hover_data=[f"G3{suffix}"],
        labels={f"G3{suffix}": "Final Grade (G3)"},
    )
    fig_u1.add_vline(
        x=df_f[f"G3{suffix}"].mean(), line_dash="dash", line_color="#f97316",
        annotation_text=f"Mean = {df_f[f'G3{suffix}'].mean():.1f}",
        annotation_position="top right"
    )
    fig_u1.update_layout(bargap=0.05, template="plotly_white")
    st.plotly_chart(fig_u1, use_container_width=True)

# 1-B  Pie chart – Gender Distribution
with u_col2:
    st.subheader("Gender Distribution")
    fig_u2 = px.pie(
        df_f, names="sex",
        color_discrete_sequence=["#764ba2", "#a78bfa"],
        hole=0.45,
    )
    fig_u2.update_traces(textposition="inside", textinfo="percent+label",
                         pull=[0.04, 0])
    fig_u2.update_layout(template="plotly_white")
    st.plotly_chart(fig_u2, use_container_width=True)

u_col3, u_col4 = st.columns(2)

# 1-C  Violin – Age Distribution by Gender
with u_col3:
    st.subheader("Age Distribution by Gender")
    fig_u3 = px.violin(
        df_f, y="age", x="sex", color="sex", box=True, points="all",
        color_discrete_sequence=["#f97316", "#06b6d4"],
        labels={"age": "Age", "sex": "Gender"},
        hover_data=["age"],
    )
    fig_u3.update_layout(template="plotly_white")
    st.plotly_chart(fig_u3, use_container_width=True)

# 1-D  Bar chart – Weekly Study Time
with u_col4:
    st.subheader(f"Weekly Study Time ({selected_subject})")
    study_counts = (
        df_f[f"studytime{suffix}"]
        .map(study_mapping)
        .value_counts()
        .reindex(study_mapping.values())
        .reset_index()
    )
    study_counts.columns = ["Study Time", "Count"]
    fig_u4 = px.bar(
        study_counts, x="Study Time", y="Count",
        color="Count", color_continuous_scale="Viridis",
        text="Count",
        labels={"Study Time": "Weekly Study Time", "Count": "No. of Students"},
    )
    fig_u4.update_traces(textposition="outside")
    fig_u4.update_layout(coloraxis_showscale=False, template="plotly_white")
    st.plotly_chart(fig_u4, use_container_width=True)

# 1-E  Funnel – Parental Education Levels (mother)
st.subheader("Mother's Education Level")
medu_map = {0: "None", 1: "Primary (4th)", 2: "5th–9th", 3: "Secondary", 4: "Higher"}
medu_counts = df_f["Medu"].map(medu_map).value_counts().reset_index()
medu_counts.columns = ["Education Level", "Count"]
medu_counts = medu_counts.sort_values("Count", ascending=False)
fig_u5 = px.funnel(
    medu_counts, x="Count", y="Education Level",
    color_discrete_sequence=["#667eea"],
    labels={"Count": "No. of Students", "Education Level": "Mother's Education"},
)
fig_u5.update_layout(template="plotly_white")
st.plotly_chart(fig_u5, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – BIVARIATE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔗 2 · Bivariate Analysis</div>', unsafe_allow_html=True)
st.markdown('<p class="chart-tip">Drag to zoom in any scatter plot. Double-click to reset.</p>', unsafe_allow_html=True)

b_col1, b_col2 = st.columns(2)

# 2-A  Strip – Address Type vs Final Grade
with b_col1:
    st.subheader(f"Home Address vs Final Grade ({selected_subject})")
    addr_df = df_f.copy()
    addr_df["address"] = addr_df["address"].map({"U": "Urban", "R": "Rural"})
    fig_b1 = px.strip(
        addr_df,
        x="address", y=f"G3{suffix}",
        color="sex",
        stripmode="overlay",
        color_discrete_sequence=["#667eea", "#f97316"],
        labels={
            "address": "Home Address",
            f"G3{suffix}": "Final Grade (G3)",
            "sex": "Gender",
        },
        hover_data=[f"G3{suffix}", "address", "sex"],
    )
    fig_b1.update_layout(template="plotly_white")
    st.plotly_chart(fig_b1, use_container_width=True)

# 2-B  Box plot – G3 by Parental Cohabitation
with b_col2:
    st.subheader(f"Final Grade by Parental Status ({selected_subject})")
    pstatus_label = df_f["Pstatus"].map({"T": "Together", "A": "Apart"})
    fig_b2 = go.Figure()
    for pstatus, color in zip(["Together", "Apart"], ["#667eea", "#f97316"]):
        subset = df_f[pstatus_label == pstatus][f"G3{suffix}"]
        fig_b2.add_trace(go.Box(
            y=subset, name=pstatus,
            marker_color=color, boxmean="sd",
            hovertemplate="<b>%{x}</b><br>Grade: %{y}<extra></extra>",
        ))
    fig_b2.update_layout(
        xaxis_title="Parental Status", yaxis_title="Final Grade (G3)",
        template="plotly_white",
    )
    st.plotly_chart(fig_b2, use_container_width=True)

b_col3, b_col4 = st.columns(2)

# 2-C  Bar – Study Time vs Avg Grade
with b_col3:
    st.subheader(f"Study Time vs Avg Grade ({selected_subject})")
    agg = (
        df_f.groupby("studytime_label")[f"G3{suffix}"]
        .mean()
        .reset_index()
    )
    agg.columns = ["Study Time", "Avg Final Grade"]
    agg["Study Time"] = pd.Categorical(
        agg["Study Time"], categories=list(study_mapping.values()), ordered=True
    )
    agg = agg.sort_values("Study Time")
    fig_b3 = px.bar(
        agg, x="Study Time", y="Avg Final Grade",
        color="Avg Final Grade",
        color_continuous_scale="Viridis",
        text="Avg Final Grade",
        labels={"Avg Final Grade": "Avg Final Grade (G3)"},
    )
    fig_b3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_b3.update_layout(template="plotly_white", coloraxis_showscale=False)
    st.plotly_chart(fig_b3, use_container_width=True)

# 2-D  Violin – G3 Distribution by School
with b_col4:
    st.subheader(f"Grade Distribution by School ({selected_subject})")
    fig_b7 = px.violin(
        df_f,
        x="school", y=f"G3{suffix}",
        box=True, points="all",
        color_discrete_sequence=["#a78bfa"],
        labels={
            "school": "School",
            f"G3{suffix}": "Final Grade (G3)",
        },
    )
    fig_b7.update_layout(template="plotly_white")
    st.plotly_chart(fig_b7, use_container_width=True)

# 2-E  Box – G3 by Guardian Type
st.subheader(f"Final Grade by Guardian Type ({selected_subject})")
guardian_order = ["mother", "father", "other"]
guard_df = df_f[df_f[f"guardian{suffix}"].isin(guardian_order)].copy()
fig_b8 = px.box(
    guard_df,
    x=f"guardian{suffix}", y=f"G3{suffix}",
    color_discrete_sequence=["#667eea"],
    category_orders={f"guardian{suffix}": guardian_order},
    labels={
        f"guardian{suffix}": "Guardian",
        f"G3{suffix}": "Final Grade (G3)",
    },
    points="outliers",
    notched=True,
)
fig_b8.update_layout(template="plotly_white")
st.plotly_chart(fig_b8, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – MULTIVARIATE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🧩 3 · Multivariate Analysis</div>', unsafe_allow_html=True)
st.markdown('<p class="chart-tip">Use the colour legend to isolate groups. Zoom and pan freely.</p>', unsafe_allow_html=True)


m_col3, m_col4 = st.columns(2)

# 3-C  Bubble chart – Absences vs G3, size = failures, colour = sex
with m_col3:
    st.subheader(f"Absences & Final Grade ({selected_subject})")
    bub_df = df_f.copy()
    bub_df["bubble_size"] = bub_df[f"failures{suffix}"].apply(lambda x: max(x * 8, 5))
    fig_m3 = px.scatter(
        bub_df,
        x=f"absences{suffix}", y=f"G3{suffix}",
        size="bubble_size", color="sex",
        color_discrete_sequence=["#667eea", "#f97316"],
        hover_data=[f"failures{suffix}", f"studytime{suffix}", "sex"],
        labels={
            f"absences{suffix}": "Absences",
            f"G3{suffix}": "Final Grade (G3)",
            "sex": "Gender",
        },
        opacity=0.75,
    )
    fig_m3.update_layout(template="plotly_white")
    st.plotly_chart(fig_m3, use_container_width=True)

# 3-D  Facet bar – Avg G3 by Family Relationship Rating and Gender
with m_col4:
    st.subheader(f"Family Relations & Gender vs Avg Grade ({selected_subject})")
    fam_agg = (
        df_f.groupby(["famrel_label", "sex"])[f"G3{suffix}"]
        .mean()
        .reset_index()
    )
    fam_agg.columns = ["Family Relation", "Gender", "Avg Grade"]
    order = ["Very Bad", "Bad", "Neutral", "Good", "Excellent"]
    fam_agg["Family Relation"] = pd.Categorical(
        fam_agg["Family Relation"], categories=order, ordered=True
    )
    fam_agg = fam_agg.sort_values("Family Relation")
    fig_m4 = px.line(
        fam_agg, x="Family Relation", y="Avg Grade", color="Gender",
        markers=True,
        color_discrete_sequence=["#a78bfa", "#34d399"],
        labels={"Avg Grade": "Avg Final Grade (G3)"},
    )
    fig_m4.update_traces(line_width=2.5, marker_size=9)
    fig_m4.update_layout(template="plotly_white")
    st.plotly_chart(fig_m4, use_container_width=True)

# 3-E  Interactive Correlation Heatmap (full numeric)
st.subheader("Feature Correlation Heatmap (Interactive)")

# Dynamic column selector
num_cols_all = df_f.select_dtypes(include="number").columns.tolist()
default_cols = [
    f"G1{suffix}", f"G2{suffix}", f"G3{suffix}",
    f"studytime{suffix}", f"absences{suffix}", f"failures{suffix}",
    "age", "Medu", "Fedu",
]
default_cols = [c for c in default_cols if c in num_cols_all]

selected_corr_cols = st.multiselect(
    "Choose features for the correlation matrix",
    options=num_cols_all,
    default=default_cols,
)
if len(selected_corr_cols) >= 2:
    corr_matrix = df_f[selected_corr_cols].corr().round(2)
    fig_m5 = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        labels={"color": "Correlation"},
    )
    fig_m5.update_layout(height=550, template="plotly_white")
    st.plotly_chart(fig_m5, use_container_width=True)
else:
    st.info("Please select at least 2 features for the correlation matrix.")

st.markdown("---")
st.caption("📊 EDA Dashboard · Student Performance AI System · Data is filtered using the sidebar controls.")

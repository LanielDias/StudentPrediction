import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

from ml_system import (
    load_and_merge_data,
    run_regression_pipelines,
    run_classification_pipelines,
    generate_rich_recommendations,
    RECOMMENDATION_LIBRARY,
)

st.set_page_config(page_title="AI Recommendations", page_icon="🎯", layout="wide")

# ── Custom CSS for recommendation cards ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.rec-card {
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 16px;
    border-left: 5px solid;
    background: rgba(255,255,255,0.04);
    box-shadow: 0 2px 12px rgba(0,0,0,0.12);
    transition: transform 0.15s ease;
}
.rec-card:hover { transform: translateX(3px); }

.rec-card.critical  { border-color: #ef4444; background: rgba(239,68,68,0.06); }
.rec-card.important { border-color: #f59e0b; background: rgba(245,158,11,0.06); }
.rec-card.helpful   { border-color: #22c55e; background: rgba(34,197,94,0.06); }

.badge-critical  { background:#ef4444; color:black; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:700; margin-right:8px; }
.badge-important { background:#f59e0b; color:black; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:700; margin-right:8px; }
.badge-helpful   { background:#22c55e; color:black; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:700; margin-right:8px; }

.card-title { font-size:16px; font-weight:700; margin-bottom:10px; }
.card-step  { font-size:13.5px; color:black; margin-bottom:6px; line-height:1.55; }
.shap-bar   { height:4px; border-radius:4px; margin-top:10px; background:linear-gradient(90deg,#6366f1,#ec4899); }

.risk-badge-hr { background:#ef4444; color:black; border-radius:12px; padding:6px 18px; font-size:15px; font-weight:700; display:inline-block; }
.risk-badge-mr { background:#f59e0b; color:black; border-radius:12px; padding:6px 18px; font-size:15px; font-weight:700; display:inline-block; }
.risk-badge-lr { background:#22c55e; color:black; border-radius:12px; padding:6px 18px; font-size:15px; font-weight:700; display:inline-block; }

/* Section headers */
.section-header {
    font-size: 22px;
    font-weight: 800;
    margin-top: 28px;
    margin-bottom: 4px;
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subject-header-math { font-size:18px; font-weight:700; color:#60a5fa; margin-bottom:12px; }
.subject-header-por  { font-size:18px; font-weight:700; color:#f472b6; margin-bottom:12px; }

.metric-box {
    border-radius:12px;
    padding:16px;
    text-align:center;
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1);
}
.metric-val { font-size:32px; font-weight:800; }
.metric-lbl { font-size:12px; color:#9ca3af; font-weight:500; margin-top:4px; }

.potential-pill {
    display:inline-block;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    color:black;
    padding:4px 14px;
    border-radius:20px;
    font-size:13px;
    font-weight:600;
    margin-top:6px;
}
</style>
""", unsafe_allow_html=True)


# ── Guard: redirect if no prediction ─────────────────────────────────────────
if not st.session_state.get("pred_ready"):
    st.title("🎯 AI-Driven Recommendations")
    st.warning("⚠️ No student prediction found. Please go to the **Student Prediction Form** first and submit a prediction.")
    st.page_link("pages/4_Student_Prediction_Form.py", label="→ Go to Student Prediction Form", icon="🎓")
    st.stop()

# ── Load data from session ────────────────────────────────────────────────────
student_df   = st.session_state["student_df"]
test_student = st.session_state["test_student"]
pred_math    = float(st.session_state["pred_math"])
pred_por     = float(st.session_state["pred_por"])
risk_str     = st.session_state["risk_str"]
risk_icon    = st.session_state["risk_icon"]
risk_color   = st.session_state["risk_color"]

@st.cache_data
def _load():
    return load_and_merge_data()

with st.spinner("Loading models..."):
    df = _load()
    reg = run_regression_pipelines(df)
    cls = run_classification_pipelines(df)

avg_features = df.select_dtypes(include="number").mean()

# ── Compute SHAP values (Math & Portuguese) ───────────────────────────────────
@st.cache_resource
def _train_rf_mat(_X_proc, _y):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(_X_proc, _y)
    return rf

@st.cache_resource
def _train_rf_por(_X_proc, _y):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(_X_proc, _y)
    return rf

def _feat_names(preprocessor, X_orig):
    try:
        c_cols = X_orig.select_dtypes(include="object").columns.tolist()
        n_cols = X_orig.select_dtypes(exclude="object").columns.tolist()
        ohe = preprocessor.named_transformers_["cat"]
        return n_cols + list(ohe.get_feature_names_out(c_cols))
    except Exception:
        return [f"f{i}" for i in range(preprocessor.transform(X_orig).shape[1])]

shap_mat_df = None
shap_por_df = None

with st.spinner("Computing SHAP feature importance..."):
    try:
        import shap

        prep_mat   = reg["preprocessor_mat"]
        X_mat_tr   = reg["X_mat_train"]
        y_mat_tr   = reg["y_mat_train"]
        sample_mat = student_df[reg["common_cols"] + reg["math_cols"]]

        X_mat_proc     = prep_mat.transform(X_mat_tr)
        student_mat_pr = prep_mat.transform(sample_mat)
        feat_names_mat = _feat_names(prep_mat, X_mat_tr)

        rf_mat    = _train_rf_mat(X_mat_proc, y_mat_tr.values)
        exp_mat   = shap.TreeExplainer(rf_mat, feature_perturbation="tree_path_dependent")
        sv_mat    = exp_mat.shap_values(student_mat_pr)

        shap_mat_df = pd.DataFrame({
            "Feature":    [feat_names_mat[i] for i in range(len(sv_mat[0]))],
            "SHAP Value": sv_mat[0],
        }).sort_values("SHAP Value", key=abs, ascending=False)

        prep_por   = reg["preprocessor_por"]
        X_por_tr   = reg["X_por_train"]
        y_por_tr   = reg["y_por_train"]
        sample_por = student_df[reg["common_cols"] + reg["por_cols"]]

        X_por_proc     = prep_por.transform(X_por_tr)
        student_por_pr = prep_por.transform(sample_por)
        feat_names_por = _feat_names(prep_por, X_por_tr)

        rf_por    = _train_rf_por(X_por_proc, y_por_tr.values)
        exp_por   = shap.TreeExplainer(rf_por, feature_perturbation="tree_path_dependent")
        sv_por    = exp_por.shap_values(student_por_pr)

        shap_por_df = pd.DataFrame({
            "Feature":    [feat_names_por[i] for i in range(len(sv_por[0]))],
            "SHAP Value": sv_por[0],
        }).sort_values("SHAP Value", key=abs, ascending=False)

    except Exception as e:
        st.warning(f"SHAP computation skipped (fallback to avg comparison): {e}")

# ── Generate rich recommendations ─────────────────────────────────────────────
rec = generate_rich_recommendations(
    student_df.iloc[0],
    avg_features,
    shap_mat_df=shap_mat_df,
    shap_por_df=shap_por_df,
)

# ════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ════════════════════════════════════════════════════════════════════════════

st.title("🎯 AI-Driven Personalized Recommendations")
st.markdown("*SHAP-powered, subject-specific action plans tailored to this student's unique risk profile.*")

# ── Section 1: Student Snapshot ───────────────────────────────────────────────
st.markdown('<div class="section-header">📌 Student Snapshot</div>', unsafe_allow_html=True)
st.markdown("")

badge_cls = {"High Risk": "risk-badge-hr", "Medium Risk": "risk-badge-mr", "Low Risk": "risk-badge-lr"}.get(risk_str, "risk-badge-lr")

snap1, snap2, snap3, snap4 = st.columns(4)
with snap1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val" style="color:#60a5fa;">{pred_math:.1f}</div>
        <div class="metric-lbl">📐 Predicted Math G3</div>
    </div>""", unsafe_allow_html=True)
with snap2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val" style="color:#f472b6;">{pred_por:.1f}</div>
        <div class="metric-lbl">📚 Predicted Portuguese G3</div>
    </div>""", unsafe_allow_html=True)
with snap3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val" style="color:#a78bfa;">{((pred_math+pred_por)/2):.1f}</div>
        <div class="metric-lbl">⚡ Average Score</div>
    </div>""", unsafe_allow_html=True)
with snap4:
    st.markdown(f"""
    <div class="metric-box">
        <div style="margin-top:4px;"><span class="{badge_cls}">{risk_icon} {risk_str}</span></div>
        <div class="metric-lbl" style="margin-top:8px;">🔰 Risk Classification</div>
    </div>""", unsafe_allow_html=True)

# ── Summary Callout ───────────────────────────────────────────────────────────
st.markdown("")
if risk_str == "High Risk":
    st.error(rec["summary_text"])
elif risk_str == "Medium Risk":
    st.warning(rec["summary_text"])
else:
    st.success(rec["summary_text"])

st.markdown("---")

# ── Section 2: Radar Chart ────────────────────────────────────────────────────
st.markdown('<div class="section-header">📡 Student vs Dataset Average — Key Factor Radar</div>', unsafe_allow_html=True)
st.markdown("*See at a glance where this student stands across the 8 most predictive behavioural and academic factors.*")

RADAR_FEATURES = {
    "studytime_mat": ("Study Time (Math)", True, 4),
    "studytime_por": ("Study Time (Por)", True, 4),
    "absences_mat":  ("Absences (Math)", False, 93),
    "absences_por":  ("Absences (Por)", False, 93),
    "G1_mat":        ("G1 Math", True, 20),
    "G2_mat":        ("G2 Math", True, 20),
    "G1_por":        ("G1 Por", True, 20),
    "G2_por":        ("G2 Por", True, 20),
}

categories = []
student_vals = []
avg_vals = []

for feat, (label, higher_better, scale) in RADAR_FEATURES.items():
    if feat in student_df.columns and feat in avg_features.index:
        categories.append(label)
        s_raw = float(student_df[feat].iloc[0])
        a_raw = float(avg_features[feat])
        # Normalise to 0–1; for lower-is-better, invert
        if higher_better:
            student_vals.append(s_raw / scale)
            avg_vals.append(a_raw / scale)
        else:
            student_vals.append(1 - s_raw / scale)
            avg_vals.append(1 - a_raw / scale)

if categories:
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=student_vals + [student_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="This Student",
        line={"color": "#6366f1", "width": 2.5},
        fillcolor="rgba(99,102,241,0.18)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=avg_vals + [avg_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Dataset Average",
        line={"color": "#f59e0b", "width": 2, "dash": "dash"},
        fillcolor="rgba(245,158,11,0.10)",
    ))
    fig_radar.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 1], "tickfont": {"size": 10}, "showticklabels": False},
            "angularaxis": {"tickfont": {"size": 12, "color": "#d1d5db"}},
            "bgcolor": "rgba(0,0,0,0)",
        },
        legend={"font": {"size": 12}, "bgcolor": "rgba(0,0,0,0)"},
        paper_bgcolor="rgba(0,0,0,0)",
        height=430,
        margin={"t": 20, "b": 20},
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ── Section 3: SHAP-Backed Subject Action Plans ───────────────────────────────
st.markdown('<div class="section-header">🧠 SHAP-Driven Action Plans</div>', unsafe_allow_html=True)
st.markdown("*Each recommendation below is ranked by how strongly the factor hurt this student's predicted grade, according to SHAP analysis.*")
st.markdown("")

PRIORITY_CONFIG = {
    "critical":  ("🔴 Critical",  "critical"),
    "important": ("🟡 Important", "important"),
    "helpful":   ("🟢 Helpful",   "helpful"),
}

def render_action_card(action):
    p_label, p_cls = PRIORITY_CONFIG.get(action["priority"], ("🟢 Helpful", "helpful"))
    badge_html = f'<span class="badge-{p_cls}">{p_label}</span>'
    steps_html = "".join(f'<div class="card-step">• {step}</div>' for step in action["steps"])
    shap_pct   = min(100, int(action["shap"] * 200))
    st.markdown(f"""
    <div class="rec-card {p_cls}">
        <div class="card-title">{badge_html} {action["label"]}</div>
        {steps_html}
        <div class="shap-bar" style="width:{shap_pct}%;"></div>
        <div style="font-size:11px;color:black;margin-top:4px;">SHAP impact magnitude: {action['shap']:.3f}</div>
    </div>""", unsafe_allow_html=True)

col_math, col_por = st.columns(2)

with col_math:
    st.markdown('<div class="subject-header-math">📐 Mathematics Recommendations</div>', unsafe_allow_html=True)
    if rec["math_actions"]:
        for action in rec["math_actions"]:
            render_action_card(action)
        # Improvement potential pill
        pot = rec["improvement_potential"]["math"]
        st.markdown(f'<div class="potential-pill">📈 Estimated improvement potential: +{pot:.1f} pts</div>', unsafe_allow_html=True)
    else:
        st.success("✅ No critical Math issues detected — keep up the great work!")

with col_por:
    st.markdown('<div class="subject-header-por">📚 Portuguese Recommendations</div>', unsafe_allow_html=True)
    if rec["por_actions"]:
        for action in rec["por_actions"]:
            render_action_card(action)
        pot = rec["improvement_potential"]["por"]
        st.markdown(f'<div class="potential-pill">📈 Estimated improvement potential: +{pot:.1f} pts</div>', unsafe_allow_html=True)
    else:
        st.success("✅ No critical Portuguese issues detected — keep up the great work!")

st.markdown("---")

# ── Section 4: What-If Grade Simulator ────────────────────────────────────────
st.markdown('<div class="section-header">🔮 What-If Grade Simulator</div>', unsafe_allow_html=True)
st.markdown("*Adjust the key factors below and see how your predicted Math and Portuguese grades would change.*")
st.info(
    "**ℹ️ How the Simulator Works:** Adjust the sliders below. Each change rebuilds the student profile "
    "and **re-runs the same trained Deep Learning model** used on the Dataset Predictions page to produce "
    "a live, accurate simulated grade.  \n\n"
    "Move any slider to a different value and the predicted Math and Portuguese grades will update instantly "
    "based on the model's actual learned behaviour — not a simplified formula."
)
st.markdown("")

sim_c1, sim_c2, sim_c3 = st.columns(3)
with sim_c1:
    sim_studytime_mat = st.slider("📐 Study Time (Math)",   1, 4, int(test_student.get("studytime_mat", 2)))
    sim_absences_mat  = st.slider("📐 Absences (Math)",     0, 30, int(test_student.get("absences_mat", 6)))
with sim_c2:
    sim_studytime_por = st.slider("📚 Study Time (Por)",    1, 4, int(test_student.get("studytime_por", 2)))
    sim_absences_por  = st.slider("📚 Absences (Por)",      0, 30, int(test_student.get("absences_por", 4)))
with sim_c3:
    # Use per-subject keys — fall back to the _mat variant, then a sensible default
    default_goout = int(test_student.get("goout_mat", test_student.get("goout_por", 3)))
    default_Walc  = int(test_student.get("Walc_mat",  test_student.get("Walc_por",  1)))
    sim_goout  = st.slider("🎉 Going-Out Frequency",  1, 5, default_goout)
    sim_Walc   = st.slider("🍺 Weekend Alcohol",       1, 5, default_Walc)

sim_student: dict = dict(test_student)
sim_student["studytime_mat"] = sim_studytime_mat
sim_student["absences_mat"]  = sim_absences_mat
sim_student["studytime_por"] = sim_studytime_por
sim_student["absences_por"]  = sim_absences_por
# Update ALL per-subject goout / Walc keys so the model receives the simulated values
for _sfx in ("_mat", "_por"):
    sim_student[f"goout{_sfx}"] = sim_goout
    sim_student[f"Walc{_sfx}"]  = sim_Walc

try:
    # ── Re-run the real trained ML pipelines with the simulated student profile ──
    final_math_pipeline = reg["final_math_pipeline"]
    final_por_pipeline  = reg["final_por_pipeline"]
    common_cols         = reg["common_cols"]
    math_cols           = reg["math_cols"]
    por_cols            = reg["por_cols"]

    sim_df     = pd.DataFrame([sim_student])
    sample_mat = sim_df[common_cols + math_cols]
    sample_por = sim_df[common_cols + por_cols]

    sim_math   = float(np.clip(final_math_pipeline.predict(sample_mat)[0], 0, 20))
    sim_por    = float(np.clip(final_por_pipeline.predict(sample_por)[0],  0, 20))
    delta_math = sim_math - pred_math
    delta_por  = sim_por  - pred_por

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("🔮 Simulated Math G3",       f"{sim_math:.2f}",  f"{delta_math:+.2f}")
    r2.metric("🔮 Simulated Portuguese G3", f"{sim_por:.2f}",   f"{delta_por:+.2f}")
    r3.metric("📐 Math Change",  f"{delta_math:+.2f} pts", help="Change vs original ML prediction")
    r4.metric("📚 Por Change",   f"{delta_por:+.2f} pts",  help="Change vs original ML prediction")

    # Visual bar comparison
    bar_df = pd.DataFrame({
        "Scenario":  ["Original", "Simulated", "Original", "Simulated"],
        "Subject":   ["Math", "Math", "Portuguese", "Portuguese"],
        "Grade":     [pred_math, sim_math, pred_por, sim_por],
    })
    fig_bar = px.bar(
        bar_df, x="Subject", y="Grade", color="Scenario", barmode="group",
        color_discrete_map={"Original": "#6366f1", "Simulated": "#22c55e"},
        text_auto=".1f",
        labels={"Grade": "Predicted G3 (0–20)"},
    )
    fig_bar.update_layout(
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"range": [0, 22], "gridcolor": "rgba(255,255,255,0.06)"},
        legend={"bgcolor": "rgba(0,0,0,0)"},
        margin={"t": 10, "b": 10},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

except Exception as e:
    st.error(f"Simulation error: {e}")

st.markdown("---")

# ── Section 5: Improvement Timeline Projection ─────────────────────────────────
st.markdown('<div class="section-header">📅 Improvement Timeline Projection</div>', unsafe_allow_html=True)
st.markdown("*If this student consistently follows the recommended actions, here's a projected grade improvement trajectory over 12 weeks.*")

weeks       = list(range(0, 13, 2))  # 0 2 4 6 8 10 12
# Assumes roughly linear improvement using potential as max headroom
math_pot    = rec["improvement_potential"]["math"]
por_pot     = rec["improvement_potential"]["por"]
math_proj   = [min(20.0, pred_math + math_pot * (w / 12)) for w in weeks]
por_proj    = [min(20.0, pred_por  + por_pot  * (w / 12)) for w in weeks]

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=weeks, y=math_proj,
    mode="lines+markers", name="Math (Projected)",
    line={"color": "#60a5fa", "width": 3},
    marker={"size": 8},
))
fig_line.add_trace(go.Scatter(
    x=weeks, y=por_proj,
    mode="lines+markers", name="Portuguese (Projected)",
    line={"color": "#f472b6", "width": 3},
    marker={"size": 8},
))
# Pass threshold line
fig_line.add_hline(y=10, line_dash="dash", line_color="#ef4444", opacity=0.5,
                   annotation_text="Pass Threshold (10)", annotation_position="top left")
fig_line.add_hline(y=14, line_dash="dot", line_color="#22c55e", opacity=0.5,
                   annotation_text="Good Grade (14)", annotation_position="top left")

fig_line.update_layout(
    xaxis_title="Weeks of Consistent Effort",
    yaxis_title="Projected Grade (0–20)",
    yaxis={"range": [max(0.0, min(pred_math, pred_por)-1.0), 21],
           "gridcolor": "rgba(255,255,255,0.06)"},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend={"bgcolor": "rgba(0,0,0,0)"},
    height=360,
    margin={"t": 10, "b": 10},
)
st.plotly_chart(fig_line, use_container_width=True)

st.caption("⚠️ *Projection is an optimistic estimate based on the improvement potential computed from the model's grade gap. Actual progress depends on individual effort and consistency.*")

st.markdown("---")

# ── Section 6: Cross-Subject Lifestyle Actions ────────────────────────────────
if rec["common_actions"]:
    st.markdown('<div class="section-header">🌱 Cross-Subject Lifestyle Recommendations</div>', unsafe_allow_html=True)
    st.markdown("*These actions benefit both Math and Portuguese performance simultaneously.*")
    st.markdown("")
    n_cols = min(3, len(rec["common_actions"]))
    if n_cols > 0:
        cols = st.columns(n_cols)
        for i, act in enumerate(rec["common_actions"]):
            with cols[i % n_cols]:
                steps_md = "\n".join(f"- {s}" for s in act["steps"])
                st.info(f"**{act['label']}**\n\n{steps_md}")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6b7280;font-size:13px;'>"
    "🤖 Powered by SHAP feature importance · Random Forest surrogate · Deep Learning grade prediction"
    "</div>",
    unsafe_allow_html=True
)

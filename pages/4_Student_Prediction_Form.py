import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from ml_system import load_and_merge_data, run_regression_pipelines, run_classification_pipelines, generate_recommendations, generate_rich_recommendations

st.set_page_config(page_title="Student Form", page_icon="🎓", layout="wide")

st.title("🎓 Student Prediction Form")
st.markdown("Enter a student's profile to predict their **Mathematics** and **Portuguese** final grades, calculate **Risk Level**, and view personalized recommendations.")

with st.spinner("Loading engine..."):
    df = load_and_merge_data()
    reg = run_regression_pipelines(df)
    cls = run_classification_pipelines(df)

with st.sidebar:
    st.markdown("### ⚙️ Quick Tools")
    load_sample = st.button("📋 Load Sample Student[0]", use_container_width=True)
    st.markdown("---")
    st.markdown("""
**📘 Scale Guide:**
| Range | Used For |
|---|---|
| 1–4 | Study time, Travel time |
| 1–5 | Social, Alcohol, Family, Health |
| 0–20 | Grades (G1, G2) |
| 0–93 | Absences |
""")

if load_sample:
    d = df.iloc[0].to_dict()
    # Persist the full sample dict so the predict run can also read from it
    st.session_state["loaded_sample_d"] = d
    # ── Directly write slider/number_input values into session_state ──────────
    # This is necessary because Streamlit ignores the `value` param if the key
    # already exists in session_state. Setting it explicitly forces the update.
    st.session_state["age_sl"]       = int(d.get("age", 18))
    st.session_state["medu_sl"]      = int(d.get("Medu", 4))
    st.session_state["fedu_sl"]      = int(d.get("Fedu", 4))
    # Math sliders
    st.session_state["st_mat"]       = int(d.get("studytime_mat", 2))
    st.session_state["fail_mat"]     = int(d.get("failures_mat", 0))
    st.session_state["abs_mat"]      = int(d.get("absences_mat", 6))
    st.session_state["ttm"]          = int(d.get("traveltime_mat", 2))
    st.session_state["g1_mat"]       = int(d.get("G1_mat", 5))
    st.session_state["g2_mat"]       = int(d.get("G2_mat", 6))
    st.session_state["famrel_mat"]   = int(d.get("famrel_mat", 4))
    st.session_state["freetime_mat"] = int(d.get("freetime_mat", 3))
    st.session_state["goout_mat"]    = int(d.get("goout_mat", 4))
    st.session_state["dalc_mat"]     = int(d.get("Dalc_mat", 1))
    st.session_state["walc_mat"]     = int(d.get("Walc_mat", 1))
    st.session_state["health_mat"]   = int(d.get("health_mat", 3))
    # Portuguese sliders
    st.session_state["st_por"]       = int(d.get("studytime_por", 2))
    st.session_state["fail_por"]     = int(d.get("failures_por", 0))
    st.session_state["abs_por"]      = int(d.get("absences_por", 4))
    st.session_state["ttp"]          = int(d.get("traveltime_por", 1))
    st.session_state["g1_por"]       = int(d.get("G1_por", 0))
    st.session_state["g2_por"]       = int(d.get("G2_por", 11))
    st.session_state["famrel_por"]   = int(d.get("famrel_por", 4))
    st.session_state["freetime_por"] = int(d.get("freetime_por", 3))
    st.session_state["goout_por"]    = int(d.get("goout_por", 3))
    st.session_state["dalc_por"]     = int(d.get("Dalc_por", 1))
    st.session_state["walc_por"]     = int(d.get("Walc_por", 1))
    st.session_state["health_por"]   = int(d.get("health_por", 3))
    # ── Clear selectbox keys so their index param takes effect ────────────────
    for k in ["school_sel","addr_sel","sex_sel","famsz_sel","pstat_sel",
              "reason_sel","guardian_sel","mjob_sel","fjob_sel","nursery_sel",
              "internet_sel","ssup_mat","fsup_mat","paid_mat_sel","gm_sel",
              "higher_mat","act_mat","rom_mat","ssup_por","fsup_por",
              "paid_por_sel","gp_sel","higher_por","act_por","rom_por"]:
        st.session_state.pop(k, None)
else:
    # Re-use the persisted sample dict (survives across the predict-run)
    d = st.session_state.get("loaded_sample_d", {})

# ── Page CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.form-hero {
    background: linear-gradient(135deg,#1e3a5f 0%,#2d6a9f 55%,#5ba4cf 100%);
    border-radius:16px; padding:28px 32px 22px; margin-bottom:24px; color:white;
}
.form-hero h2 { margin:0 0 6px; font-size:1.65rem; }
.form-hero p  { margin:0; opacity:.85; font-size:.9rem; }

.card-header {
    font-size:1.1rem; font-weight:800; color:#1e293b;
    border-bottom:2px solid #e2e8f0; padding-bottom:8px; margin-bottom:16px;
}
.card-math  { border-left:4px solid #4c72b0; padding-left:12px; }
.card-por   { border-left:4px solid #dd8452; padding-left:12px; }
.card-demo  { border-left:4px solid #55a868; padding-left:12px; }

.fl { font-weight:700; font-size:.82rem; color:#1e293b; margin-bottom:1px; }
.fd { font-size:.73rem; color:#64748b; margin-bottom:3px; line-height:1.3; }
.fc { font-size:.72rem; color:#0369a1; font-style:italic; }

.sub-head {
    font-size:.78rem; font-weight:700; letter-spacing:.07em;
    color:#94a3b8; text-transform:uppercase; margin:14px 0 6px;
}
</style>

<div class="form-hero">
  <h2>🎓 Student Prediction Form</h2>
  <p>Fill in the student's profile below. Every field has a full description and scale meaning shown live.<br>
  The AI will predict <strong>Math &amp; Portuguese final grades</strong>, <strong>Risk Level</strong>, and personalised recommendations.</p>
</div>
""", unsafe_allow_html=True)

# ── Human-readable option maps (display label → stored value) ────────────────
SCHOOL_OPTS   = {"GP – Gabriel Pereira": "GP", "MS – Mousinho da Silveira": "MS"}
SEX_OPTS      = {"F – Female": "F", "M – Male": "M"}
ADDR_OPTS     = {"U – Urban": "U", "R – Rural": "R"}
FAMSZ_OPTS    = {"LE3 – 3 or fewer members": "LE3", "GT3 – More than 3 members": "GT3"}
PSTAT_OPTS    = {"T – Living together": "T", "A – Living apart": "A"}
EDU_LABELS    = {0: "0 – None", 1: "1 – Primary (up to 4th grade)", 2: "2 – 5th–9th grade",
                 3: "3 – Secondary education", 4: "4 – Higher education"}
MJOB_OPTS     = {"teacher": "Teacher", "health": "Health care", "services": "Civil services",
                 "at_home": "Stay at home", "other": "Other"}
FJOB_OPTS     = MJOB_OPTS
REASON_OPTS   = {"home": "Close to home", "reputation": "School reputation",
                 "course": "Course preference", "other": "Other"}
GUARD_OPTS    = {"mother": "Mother", "father": "Father", "other": "Other"}
YESNO_OPTS    = {"yes": "Yes", "no": "No"}
TRAVEL_LABELS = {1: "1 – Under 15 min", 2: "2 – 15 to 30 min",
                 3: "3 – 30 min to 1 hour", 4: "4 – Over 1 hour"}
STUDY_LABELS  = {1: "1 – Under 2 hrs/week", 2: "2 – 2 to 5 hrs/week",
                 3: "3 – 5 to 10 hrs/week", 4: "4 – Over 10 hrs/week"}
SCALE_LABELS  = {1: "1 – Very low", 2: "2 – Low", 3: "3 – Moderate",
                 4: "4 – High", 5: "5 – Very high"}
FAMREL_LABELS = {1: "1 – Very bad", 2: "2 – Bad", 3: "3 – Average",
                 4: "4 – Good", 5: "5 – Excellent"}
HEALTH_LABELS = {1: "1 – Very bad", 2: "2 – Poor", 3: "3 – Fair",
                 4: "4 – Good", 5: "5 – Very good"}

def _sel(col, opts_map: dict[str, str], default_val: str):
    """Selectbox backed by a display→value dict; returns the raw stored value."""
    inv = {v: k for k, v in opts_map.items()}
    labels = list(opts_map.keys())
    default_label = str(inv.get(default_val, labels[0]))
    chosen = labels.index(default_label) if default_label in labels else 0
    return opts_map[col.selectbox("", labels, index=chosen)]

def _slider_labeled(col, label, mapping, default_val, help_text=""):
    """Slider that shows human-readable tick labels in the widget help."""
    mn, mx = min(mapping), max(mapping)
    tick_str = " · ".join(f"{k} = {v}" for k, v in mapping.items())
    # st.slider doesn't support custom tick labels — put them in help
    val = col.slider(label, mn, mx, int(default_val),
                     help=f"{help_text}\n\n**Scale:** {tick_str}" if help_text else f"**Scale:** {tick_str}")
    return val

# ── THE UI FORM ──────────────────────────────────────────────────────────────
with st.form("student_form"):

    # ── SECTION 1: Demographics ──────────────────────────────────────────────
    st.subheader("🏫 Demographic & Background Information")
    st.caption("Basic personal and family background of the student.")
    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        st.markdown("**School**")
        st.caption("Which school does the student attend?")
        school_label = st.selectbox("", list(SCHOOL_OPTS.keys()), key="school_sel",
            index=list(SCHOOL_OPTS.values()).index(d.get("school", "GP")))
        school = SCHOOL_OPTS[school_label]

        st.markdown("**Home Address Type**")
        st.caption("Is the student's home in an urban or rural area?")
        addr_label = st.selectbox("", list(ADDR_OPTS.keys()), key="addr_sel",
            index=list(ADDR_OPTS.values()).index(d.get("address", "U")))
        address = ADDR_OPTS[addr_label]

        st.markdown("**Mother's Education Level** `Medu`")
        st.caption("0 = None · 1 = Primary (4th grade) · 2 = 5th–9th grade · 3 = Secondary · 4 = Higher")
        Medu = st.slider("", 0, 4, int(d.get("Medu", 4)), key="medu_sl",
            help="Mother's highest completed education level (0–4).")

        st.markdown("**Mother's Job** `Mjob`")
        st.caption("Occupation category of the student's mother.")
        mjob_label = st.selectbox("", list(MJOB_OPTS.values()), key="mjob_sel",
            index=list(MJOB_OPTS.keys()).index(d.get("Mjob", "at_home")))
        Mjob = {v: k for k, v in MJOB_OPTS.items()}[mjob_label]

        st.markdown("**Attended Nursery School?** `nursery`")
        st.caption("Did the student attend nursery school as a child?")
        nursery_label = st.selectbox("", ["Yes", "No"], key="nursery_sel",
            index=0 if d.get("nursery", "yes") == "yes" else 1)
        nursery = "yes" if nursery_label == "Yes" else "no"

    with sc2:
        st.markdown("**Sex**")
        st.caption("Student's biological sex.")
        sex_label = st.selectbox("", list(SEX_OPTS.keys()), key="sex_sel",
            index=list(SEX_OPTS.values()).index(d.get("sex", "F")))
        sex = SEX_OPTS[sex_label]

        st.markdown("**Family Size** `famsize`")
        st.caption("How many people are in the student's household?")
        famsz_label = st.selectbox("", list(FAMSZ_OPTS.keys()), key="famsz_sel",
            index=list(FAMSZ_OPTS.values()).index(d.get("famsize", "GT3")))
        famsize = FAMSZ_OPTS[famsz_label]

        st.markdown("**Father's Education Level** `Fedu`")
        st.caption("0 = None · 1 = Primary (4th grade) · 2 = 5th–9th grade · 3 = Secondary · 4 = Higher")
        Fedu = st.slider("", 0, 4, int(d.get("Fedu", 4)), key="fedu_sl",
            help="Father's highest completed education level (0–4).")

        st.markdown("**Father's Job** `Fjob`")
        st.caption("Occupation category of the student's father.")
        fjob_label = st.selectbox("", list(FJOB_OPTS.values()), key="fjob_sel",
            index=list(FJOB_OPTS.keys()).index(d.get("Fjob", "teacher")))
        Fjob = {v: k for k, v in FJOB_OPTS.items()}[fjob_label]

        st.markdown("**Internet Access at Home?** `internet`")
        st.caption("Does the student have internet access at home?")
        internet_label = st.selectbox("", ["Yes", "No"], key="internet_sel",
            index=0 if d.get("internet", "yes") == "yes" else 1)
        internet = "yes" if internet_label == "Yes" else "no"

    with sc3:
        st.markdown("**Age**")
        st.caption("Student's age in years (15–22).")
        age = st.slider("", 15, 22, int(d.get("age", 16)), key="age_sl")

        st.markdown("**Parent Cohabitation Status** `Pstatus`")
        st.caption("Are the student's parents currently living together?")
        pstat_label = st.selectbox("", list(PSTAT_OPTS.keys()), key="pstat_sel",
            index=list(PSTAT_OPTS.values()).index(d.get("Pstatus", "T")))
        Pstatus = PSTAT_OPTS[pstat_label]

        st.markdown("**Reason for Choosing This School** `reason`")
        st.caption("What was the primary reason the student chose this school?")
        reason_opts_disp = list(REASON_OPTS.values())
        reason_keys = list(REASON_OPTS.keys())
        reason_label = st.selectbox("", reason_opts_disp, key="reason_sel",
            index=reason_keys.index(d.get("reason", "course")))
        reason = reason_keys[reason_opts_disp.index(reason_label)]

        st.markdown("**Student's Guardian** `guardian`")
        st.caption("Who is primarily responsible for the student?")
        guard_opts_disp = list(GUARD_OPTS.values())
        guard_keys = list(GUARD_OPTS.keys())
        guard_def = d.get("guardian", "mother")
        guard_label = st.selectbox("", guard_opts_disp, key="guardian_sel",
            index=guard_keys.index(guard_def) if guard_def in guard_keys else 0)
        guardian = guard_keys[guard_opts_disp.index(guard_label)]

    # ── SECTION 2: Math Variables ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📐 Math Course Variables")
    st.caption("Inputs specific to the student's Mathematics course performance and context.")

    st.markdown('<p class="sub-head">📊 Academic Performance</p>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown("**Weekly Study Time** `studytime_mat`")
        st.caption("Hours/week studying Math")
        studytime_mat = st.slider("", 1, 4, int(d.get("studytime_mat", 2)), key="st_mat",
            help="1=Under 2 hrs · 2=2–5 hrs · 3=5–10 hrs · 4=Over 10 hrs")
    with mc2:
        st.markdown("**Past Failures** `failures_mat`")
        st.caption("Math classes previously failed (0–3)")
        failures_mat = st.slider("", 0, 3, int(d.get("failures_mat", 0)), key="fail_mat",
            help="0=None, 1=One, 2=Two, 3=Three or more")
    with mc3:
        st.markdown("**Absences** `absences_mat`")
        st.caption("Total Math class absences (0–30)")
        absences_mat = st.number_input("", 0, 30, int(d.get("absences_mat", 6)), key="abs_mat")
    with mc4:
        st.markdown("**Travel Time to School** `traveltime_mat`")
        st.caption("Daily commute to school")
        traveltime_mat = st.slider("", 1, 4, int(d.get("traveltime_mat", 1)), key="ttm",
            help="1=Under 15 min · 2=15–30 min · 3=30 min–1 hr · 4=Over 1 hr")

    mg1, mg2 = st.columns(2)
    with mg1:
        st.markdown("**1st Period Math Grade (G1)** `G1_mat`")
        st.caption("First assessment period grade — 0 to 20")
        G1_mat = st.number_input("", 0, 20, int(d.get("G1_mat", 5)), key="g1_mat")
    with mg2:
        st.markdown("**2nd Period Math Grade (G2)** `G2_mat`")
        st.caption("Second assessment period grade — 0 to 20 · ⭐ Strongest predictor of G3")
        G2_mat = st.number_input("", 0, 20, int(d.get("G2_mat", 6)), key="g2_mat")

    st.markdown('<p class="sub-head">🏠 Support & Lifestyle (Math)</p>', unsafe_allow_html=True)
    sl1, sl2, sl3, sl4 = st.columns(4)
    with sl1:
        st.markdown("**School Extra Support?** `schoolsup_mat`")
        st.caption("Extra educational support from school for Math")
        schoolsup_label = st.selectbox("", ["Yes", "No"], key="ssup_mat",
            index=0 if d.get("schoolsup_mat", "yes") == "yes" else 1)
        schoolsup_mat = "yes" if schoolsup_label == "Yes" else "no"

        st.markdown("**Family Support?** `famsup_mat`")
        st.caption("Family provides extra educational support for Math")
        famsup_label = st.selectbox("", ["Yes", "No"], key="fsup_mat",
            index=0 if d.get("famsup_mat", "no") == "yes" else 1)
        famsup_mat = "yes" if famsup_label == "Yes" else "no"

        st.markdown("**Paid Tutoring?** `paid_mat`")
        st.caption("Extra paid classes for Math subject")
        paid_mat_label = st.selectbox("", ["Yes", "No"], key="paid_mat_sel",
            index=0 if d.get("paid_mat", "no") == "yes" else 1)
        paid_mat = "yes" if paid_mat_label == "Yes" else "no"

    with sl2:
        st.markdown("**Guardian** `guardian_mat`")
        st.caption("Who is primarily responsible for this student?")
        gm_disp = list(GUARD_OPTS.values()); gm_keys = list(GUARD_OPTS.keys())
        gm_def = d.get("guardian_mat", "mother")
        gm_label = st.selectbox("", gm_disp, key="gm_sel",
            index=gm_keys.index(gm_def) if gm_def in gm_keys else 0)
        guardian_mat = gm_keys[gm_disp.index(gm_label)]

        st.markdown("**Wants Higher Education?** `higher_mat`")
        st.caption("Does the student aspire to pursue higher education?")
        hm = st.selectbox("", ["Yes", "No"], key="higher_mat",
            index=0 if d.get("higher_mat", "yes") == "yes" else 1)
        higher_mat = "yes" if hm == "Yes" else "no"

        st.markdown("**Extra-Curricular Activities?** `activities_mat`")
        st.caption("Participates in extra-curricular activities")
        am = st.selectbox("", ["Yes", "No"], key="act_mat",
            index=0 if d.get("activities_mat", "no") == "yes" else 1)
        activities_mat = "yes" if am == "Yes" else "no"

        st.markdown("**Romantic Relationship?** `romantic_mat`")
        st.caption("Currently in a romantic relationship")
        rm = st.selectbox("", ["Yes", "No"], key="rom_mat",
            index=0 if d.get("romantic_mat", "no") == "yes" else 1)
        romantic_mat = "yes" if rm == "Yes" else "no"

    with sl3:
        st.markdown("**Family Relationship Quality** `famrel_mat`")
        st.caption("Quality of family relationships (1=Very bad → 5=Excellent)")
        famrel_mat = st.slider("", 1, 5, int(d.get("famrel_mat", 4)), key="famrel_mat",
            help="1=Very bad · 2=Bad · 3=Average · 4=Good · 5=Excellent")

        st.markdown("**Free Time After School** `freetime_mat`")
        st.caption("Amount of free time after school (1=Very low → 5=Very high)")
        freetime_mat = st.slider("", 1, 5, int(d.get("freetime_mat", 3)), key="freetime_mat",
            help="1=Very low · 2=Low · 3=Moderate · 4=High · 5=Very high")

        st.markdown("**Going Out with Friends** `goout_mat`")
        st.caption("Frequency of going out socially (1=Very low → 5=Very high)")
        goout_mat = st.slider("", 1, 5, int(d.get("goout_mat", 3)), key="goout_mat",
            help="1=Very low · 2=Low · 3=Moderate · 4=High · 5=Very high")

    with sl4:
        st.markdown("**Weekday Alcohol** `Dalc_mat`")
        st.caption("Workday alcohol consumption (1=Very low → 5=Very high)")
        Dalc_mat = st.slider("", 1, 5, int(d.get("Dalc_mat", 1)), key="dalc_mat",
            help="1=Very low · 2=Low · 3=Moderate · 4=High · 5=Very high")

        st.markdown("**Weekend Alcohol** `Walc_mat`")
        st.caption("Weekend alcohol consumption (1=Very low → 5=Very high)")
        Walc_mat = st.slider("", 1, 5, int(d.get("Walc_mat", 1)), key="walc_mat",
            help="1=Very low · 2=Low · 3=Moderate · 4=High · 5=Very high")

        st.markdown("**Health Status** `health_mat`")
        st.caption("Current health status (1=Very bad → 5=Very good)")
        health_mat = st.slider("", 1, 5, int(d.get("health_mat", 3)), key="health_mat",
            help="1=Very bad · 2=Poor · 3=Fair · 4=Good · 5=Very good")

    math_data_dict = {
        "paid_mat": paid_mat, "guardian_mat": guardian_mat, "higher_mat": higher_mat,
        "activities_mat": activities_mat, "romantic_mat": romantic_mat,
        "traveltime_mat": traveltime_mat, "famrel_mat": famrel_mat,
        "freetime_mat": freetime_mat, "goout_mat": goout_mat,
        "Dalc_mat": Dalc_mat, "Walc_mat": Walc_mat, "health_mat": health_mat,
    }

    # ── SECTION 4: Portuguese Variables ──────────────────────────────────────
    st.markdown("---")
    st.subheader("📚 Portuguese Course Variables")
    st.caption("Inputs specific to the student's Portuguese Language course performance and context.")

    st.markdown('<p class="sub-head">📊 Academic Performance</p>', unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        st.markdown("**Weekly Study Time** `studytime_por`")
        st.caption("Hours/week studying Portuguese")
        studytime_por = st.slider("", 1, 4, int(d.get("studytime_por", 2)), key="st_por",
            help="1=Under 2 hrs · 2=2–5 hrs · 3=5–10 hrs · 4=Over 10 hrs")
    with pc2:
        st.markdown("**Past Failures** `failures_por`")
        st.caption("Portuguese classes previously failed (0–3)")
        failures_por = st.slider("", 0, 3, int(d.get("failures_por", 0)), key="fail_por",
            help="0=None, 1=One, 2=Two, 3=Three or more")
    with pc3:
        st.markdown("**Absences** `absences_por`")
        st.caption("Total Portuguese class absences (0–30)")
        absences_por = st.number_input("", 0, 30, int(d.get("absences_por", 4)), key="abs_por")
    with pc4:
        st.markdown("**Travel Time to School** `traveltime_por`")
        st.caption("Daily commute to school")
        traveltime_por = st.slider("", 1, 4, int(d.get("traveltime_por", 1)), key="ttp",
            help="1=Under 15 min · 2=15–30 min · 3=30 min–1 hr · 4=Over 1 hr")

    pg1, pg2 = st.columns(2)
    with pg1:
        st.markdown("**1st Period Portuguese Grade (G1)** `G1_por`")
        st.caption("First assessment period grade — 0 to 20")
        G1_por = st.number_input("", 0, 20, int(d.get("G1_por", 0)), key="g1_por")
    with pg2:
        st.markdown("**2nd Period Portuguese Grade (G2)** `G2_por`")
        st.caption("Second assessment period grade — 0 to 20 · ⭐ Strongest predictor of G3")
        G2_por = st.number_input("", 0, 20, int(d.get("G2_por", 11)), key="g2_por")

    st.markdown('<p class="sub-head">🏠 Support & Lifestyle (Portuguese)</p>', unsafe_allow_html=True)
    pl1, pl2, pl3, pl4 = st.columns(4)
    with pl1:
        st.markdown("**School Extra Support?** `schoolsup_por`")
        st.caption("Extra educational support from school for Portuguese")
        ssup_por_label = st.selectbox("", ["Yes", "No"], key="ssup_por",
            index=0 if d.get("schoolsup_por", "yes") == "yes" else 1)
        schoolsup_por = "yes" if ssup_por_label == "Yes" else "no"

        st.markdown("**Family Support?** `famsup_por`")
        st.caption("Family provides extra educational support for Portuguese")
        fsup_por_label = st.selectbox("", ["Yes", "No"], key="fsup_por",
            index=0 if d.get("famsup_por", "no") == "yes" else 1)
        famsup_por = "yes" if fsup_por_label == "Yes" else "no"

        st.markdown("**Paid Tutoring?** `paid_por`")
        st.caption("Extra paid classes for Portuguese subject")
        paid_por_label = st.selectbox("", ["Yes", "No"], key="paid_por_sel",
            index=0 if d.get("paid_por", "no") == "yes" else 1)
        paid_por = "yes" if paid_por_label == "Yes" else "no"

    with pl2:
        st.markdown("**Guardian** `guardian_por`")
        st.caption("Who is primarily responsible for this student?")
        gp_disp = list(GUARD_OPTS.values()); gp_keys = list(GUARD_OPTS.keys())
        gp_def = d.get("guardian_por", "mother")
        gp_label = st.selectbox("", gp_disp, key="gp_sel",
            index=gp_keys.index(gp_def) if gp_def in gp_keys else 0)
        guardian_por = gp_keys[gp_disp.index(gp_label)]

        st.markdown("**Wants Higher Education?** `higher_por`")
        st.caption("Does the student aspire to pursue higher education?")
        hp = st.selectbox("", ["Yes", "No"], key="higher_por",
            index=0 if d.get("higher_por", "yes") == "yes" else 1)
        higher_por = "yes" if hp == "Yes" else "no"

        st.markdown("**Extra-Curricular Activities?** `activities_por`")
        st.caption("Participates in extra-curricular activities")
        ap = st.selectbox("", ["Yes", "No"], key="act_por",
            index=0 if d.get("activities_por", "no") == "yes" else 1)
        activities_por = "yes" if ap == "Yes" else "no"

        st.markdown("**Romantic Relationship?** `romantic_por`")
        st.caption("Currently in a romantic relationship")
        rp = st.selectbox("", ["Yes", "No"], key="rom_por",
            index=0 if d.get("romantic_por", "no") == "yes" else 1)
        romantic_por = "yes" if rp == "Yes" else "no"

    with pl3:
        st.markdown("**Family Relationship Quality** `famrel_por`")
        st.caption("Quality of family relationships (1=Very bad → 5=Excellent)")
        famrel_por = st.slider("", 1, 5, int(d.get("famrel_por", 4)), key="famrel_por",
            help="1=Very bad · 2=Bad · 3=Average · 4=Good · 5=Excellent")

        st.markdown("**Free Time After School** `freetime_por`")
        st.caption("Amount of free time after school (1=Very low → 5=Very high)")
        freetime_por = st.slider("", 1, 5, int(d.get("freetime_por", 3)), key="freetime_por",
            help="1=Very low · 2=Low · 3=Moderate · 4=High · 5=Very high")

        st.markdown("**Going Out with Friends** `goout_por`")
        st.caption("Frequency of going out socially (1=Very low → 5=Very high)")
        goout_por = st.slider("", 1, 5, int(d.get("goout_por", 3)), key="goout_por",
            help="1=Very low · 2=Low · 3=Moderate · 4=High · 5=Very high")

    with pl4:
        st.markdown("**Weekday Alcohol** `Dalc_por`")
        st.caption("Workday alcohol consumption (1=Very low → 5=Very high)")
        Dalc_por = st.slider("", 1, 5, int(d.get("Dalc_por", 1)), key="dalc_por",
            help="1=Very low · 2=Low · 3=Moderate · 4=High · 5=Very high")

        st.markdown("**Weekend Alcohol** `Walc_por`")
        st.caption("Weekend alcohol consumption (1=Very low → 5=Very high)")
        Walc_por = st.slider("", 1, 5, int(d.get("Walc_por", 1)), key="walc_por",
            help="1=Very low · 2=Low · 3=Moderate · 4=High · 5=Very high")

        st.markdown("**Health Status** `health_por`")
        st.caption("Current health status (1=Very bad → 5=Very good)")
        health_por = st.slider("", 1, 5, int(d.get("health_por", 3)), key="health_por",
            help="1=Very bad · 2=Poor · 3=Fair · 4=Good · 5=Very good")

    por_data_dict = {
        "paid_por": paid_por, "guardian_por": guardian_por, "higher_por": higher_por,
        "activities_por": activities_por, "romantic_por": romantic_por,
        "traveltime_por": traveltime_por, "famrel_por": famrel_por,
        "freetime_por": freetime_por, "goout_por": goout_por,
        "Dalc_por": Dalc_por, "Walc_por": Walc_por, "health_por": health_por,
    }


    submit = st.form_submit_button("Predict & Generate Recommendations", type="primary", use_container_width=True)

if submit:
    with st.spinner("Processing Prediction..."):
        test_student = {
            # ── Demographics (common to both subjects) ────────────────
            "school": school, "sex": sex, "age": age, "address": address,
            "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
            "Mjob": Mjob, "Fjob": Fjob, "reason": reason,
            "nursery": nursery, "internet": internet,
            # ── Math course ───────────────────────────────────────────
            "studytime_mat": studytime_mat, "failures_mat": failures_mat,
            "famsup_mat": famsup_mat, "schoolsup_mat": schoolsup_mat,
            "absences_mat": absences_mat, "G1_mat": G1_mat, "G2_mat": G2_mat,
            **math_data_dict,
            # ── Portuguese course ─────────────────────────────────────
            "studytime_por": studytime_por, "failures_por": failures_por,
            "famsup_por": famsup_por, "schoolsup_por": schoolsup_por,
            "absences_por": absences_por, "G1_por": G1_por, "G2_por": G2_por,
            **por_data_dict,
        }
        
        student_df = pd.DataFrame([test_student])
        
        final_math_pipeline = reg["final_math_pipeline"]
        final_por_pipeline = reg["final_por_pipeline"]
        final_classifier = cls["final_classifier"]
        
        sample_mat = student_df[reg["common_cols"] + reg["math_cols"]]
        sample_por = student_df[reg["common_cols"] + reg["por_cols"]]
        
        pred_math = float(np.clip(final_math_pipeline.predict(sample_mat)[0], 0, 20))
        pred_por  = float(np.clip(final_por_pipeline.predict(sample_por)[0], 0, 20))
        
        student_df["Pred_G3_Math"] = pred_math
        student_df["Pred_G3_Portuguese"] = pred_por
        
        # 3-class prediction: 0=Low Risk, 1=Medium Risk, 2=High Risk
        class_input = student_df.copy()
        pred_risk_num = final_classifier.predict(class_input)[0]
        
        risk_map = {0: ("Low Risk", "🟢", "#28a745"), 1: ("Medium Risk", "🟡", "#ffc107"), 2: ("High Risk", "🔴", "#dc3545")}
        risk_str, risk_icon, risk_color = risk_map.get(int(pred_risk_num), ("Unknown", "⚪", "gray"))
        
        student_df["Pred_Risk_Level"] = risk_str
        
        # Store in session_state for XAI tab to use
        st.session_state["student_df"] = student_df
        st.session_state["test_student"] = test_student
        st.session_state["pred_math"] = pred_math
        st.session_state["pred_por"] = pred_por
        st.session_state["risk_str"] = risk_str
        st.session_state["risk_icon"] = risk_icon
        st.session_state["risk_color"] = risk_color
        st.session_state["class_input"] = class_input
        st.session_state["pred_ready"] = True

# Show results and XAI tabs once prediction is done
if st.session_state.get("pred_ready"):
    student_df = st.session_state["student_df"]
    pred_math = st.session_state["pred_math"]
    pred_por = st.session_state["pred_por"]
    risk_str = st.session_state["risk_str"]
    risk_icon = st.session_state["risk_icon"]
    risk_color = st.session_state["risk_color"]
    class_input = st.session_state["class_input"]

    st.markdown("---")
    tab_results, tab_xai = st.tabs(["📊 Prediction Results", "🔍 Explainable AI"])

    # ─── Tab 1: Prediction Results ───────────────────────────────────────────────
    with tab_results:
        st.subheader("Prediction Results")
        
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Predicted Math (G3)", f"{pred_math:.2f}")
        rc2.metric("Predicted Portuguese (G3)", f"{pred_por:.2f}")
        with rc3:
            st.markdown(f"""
            <div style='border-radius:10px; padding:12px 16px; background:rgba(0,0,0,0.05); border-left:5px solid {risk_color};'>
                <div style='font-size:13px; color:gray; font-weight:600;'>Risk Classification</div>
                <div style='font-size:26px; font-weight:800; color:{risk_color};'>{risk_icon} {risk_str}</div>
            </div>
            """, unsafe_allow_html=True)
        
        numeric_df   = df.select_dtypes(include="number")
        avg_features = numeric_df.mean()

        rec = generate_rich_recommendations(student_df.iloc[0], avg_features)

        st.markdown("### 💡 AI-Driven Recommendations")

        PRIORITY_COLORS = {"critical": "#ef4444", "important": "#f59e0b", "helpful": "#22c55e"}
        PRIORITY_LABELS = {"critical": "🔴 Critical", "important": "🟡 Important", "helpful": "🟢 Helpful"}
        BG_COLORS       = {"critical": "rgba(239,68,68,0.07)", "important": "rgba(245,158,11,0.07)", "helpful": "rgba(34,197,94,0.07)"}

        def mini_card(action):
            prio   = action["priority"]
            color  = PRIORITY_COLORS.get(prio, "#6b7280")
            bg     = BG_COLORS.get(prio, "rgba(255,255,255,0.04)")
            badge  = PRIORITY_LABELS.get(prio, prio)
            steps_html = "".join(f"<div style='font-size:12.5px;color:black;margin-bottom:4px;'>• {s}</div>" for s in action["steps"][:3])
            st.markdown(f"""
            <div style='border-radius:12px;padding:14px 16px;margin-bottom:12px;
                        border-left:4px solid {color};background:{bg};'>
              <div style='font-size:13px;font-weight:700;margin-bottom:8px;'>
                <span style='background:{color};color:#fff;font-size:11px;padding:2px 8px;
                             border-radius:20px;margin-right:8px;'>{badge}</span>
                {action["label"]}
              </div>
              {steps_html}
            </div>""", unsafe_allow_html=True)

        math_col, por_col = st.columns(2)

        with math_col:
            st.markdown("**📐 Mathematics**")
            if rec["math_actions"]:
                for act in rec["math_actions"][:2]:
                    mini_card(act)
                pot = rec["improvement_potential"]["math"]
                st.markdown(f"<span style='background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;"
                            f"padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600;'>"
                            f"📈 Up to +{pot:.1f} pts potential</span>", unsafe_allow_html=True)
            else:
                st.success("✅ Math performance is on track!")

        with por_col:
            st.markdown("**📚 Portuguese**")
            if rec["por_actions"]:
                for act in rec["por_actions"][:2]:
                    mini_card(act)
                pot = rec["improvement_potential"]["por"]
                st.markdown(f"<span style='background:linear-gradient(135deg,#db2777,#9d174d);color:#fff;"
                            f"padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600;'>"
                            f"📈 Up to +{pot:.1f} pts potential</span>", unsafe_allow_html=True)
            else:
                st.success("✅ Portuguese performance is on track!")

        st.markdown("")
        st.page_link(
            "pages/6_Recommendations.py",
            label="🎯 View Full AI Recommendations Report (with Radar Chart, What-If Simulator & Timeline)",
            icon="🔍",
        )

    # ─── Tab 2: Explainable AI (InterpretML EBM) ─────────────────────────────────
    with tab_xai:
        st.subheader("🔍 Explainable AI — Why this prediction?")
        st.markdown("Using **InterpretML Explainable Boosting Machine (EBM)** — a glass-box model — to explain what factors influenced the **Risk Level** and **Math/Portuguese Grades** for this student.")

        with st.expander("🛠️ Under the Hood: How EBM Explainability Works"):
            st.markdown("""
            **InterpretML EBM (Explainable Boosting Machine):**
            - EBM is a **Generalized Additive Model (GAM)** — its prediction is the exact additive sum of individual feature contributions (scores).
            - Unlike post-hoc methods (e.g. SHAP), EBM scores are **intrinsically exact** — no approximation needed.
            - A positive score means that feature pushed the prediction **toward** the predicted class/grade; negative means it pushed it **away**.
            - **Section 1** uses `ExplainableBoostingClassifier` trained on Risk labels (Low/Medium/High Risk).
            - **Section 2** uses `ExplainableBoostingRegressor` trained on Math and Portuguese G3 grades to explain the predicted grade for this individual student.
            """)

        X_train_cls      = cls["X_train_cls"]
        y_train_cls      = cls["y_train_cls"]
        X_class          = cls["X_class"]
        preprocessor_class = cls["preprocessor_class"]

        # Regression data for grade EBM
        y_mat_train      = reg["y_mat_train"]
        y_por_train      = reg["y_por_train"]
        X_mat_train      = reg["X_mat_train"]
        X_por_train      = reg["X_por_train"]
        preprocessor_mat = reg["preprocessor_mat"]
        preprocessor_por = reg["preprocessor_por"]

        def _feat_names(preprocessor, X_orig, X_proc):
            try:
                cat = X_orig.select_dtypes(include="object").columns.tolist()
                num = X_orig.select_dtypes(exclude="object").columns.tolist()
                ohe = preprocessor.named_transformers_["cat"]
                return num + list(ohe.get_feature_names_out(cat))
            except Exception:
                return [f"f{i}" for i in range(X_proc.shape[1])]

        # ── Section 1: Risk Classification EBM ───────────────────────────────────
        st.markdown("---")
        st.markdown("### 1. 🔵 Risk Classification — EBM Local Explanation")
        st.markdown("The EBM glass-box model shows exactly which features pushed this student toward the predicted risk class.")

        with st.spinner("Training Risk EBM..."):
            try:
                # pyre-ignore[21]
                from interpret.glassbox import ExplainableBoostingClassifier

                X_train_proc_cls = preprocessor_class.transform(X_train_cls)
                student_proc_cls = preprocessor_class.transform(class_input)
                feat_names_cls   = _feat_names(preprocessor_class, X_train_cls,
                                               X_train_proc_cls)

                @st.cache_resource
                def _train_risk_ebm(_X, _y, _feat_names):
                    ebm = ExplainableBoostingClassifier(
                        random_state=42, max_rounds=200, n_jobs=1)
                    ebm.fit(pd.DataFrame(_X, columns=_feat_names), _y)
                    return ebm

                risk_ebm = _train_risk_ebm(X_train_proc_cls, y_train_cls.values, feat_names_cls)

                risk_map_num = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}
                pred_class_idx = risk_map_num.get(risk_str, 2)

                student_df_cls = pd.DataFrame(student_proc_cls, columns=feat_names_cls)
                ebm_local = risk_ebm.explain_local(student_df_cls, [pred_class_idx])
                data = ebm_local.data(0)

                names = list(data["names"])
                raw_scores = data["scores"]

                def _extract_score(s, idx):
                    try:
                        arr = np.asarray(s).flatten()
                        return float(arr[idx]) if arr.size > 1 else float(arr[0])
                    except Exception:
                        return float(s)

                flat_scores = [_extract_score(s, pred_class_idx) for s in raw_scores]
                ebm_risk_df = pd.DataFrame({"Feature": names, "Score": flat_scores})
                ebm_risk_df["AbsScore"] = ebm_risk_df["Score"].abs()
                ebm_risk_df = ebm_risk_df.sort_values("AbsScore", ascending=False).head(15).drop(columns="AbsScore")

                fig_r, ax_r = plt.subplots(figsize=(10, 6))
                clrs_r = ["#1f77b4" if v > 0 else "#ff7f0e" for v in ebm_risk_df["Score"]]
                ax_r.barh(ebm_risk_df["Feature"].iloc[::-1], ebm_risk_df["Score"].iloc[::-1],
                          color=list(reversed(clrs_r)), edgecolor="black", linewidth=0.5)
                ax_r.axvline(0, color="black", linestyle="--", linewidth=0.8)
                ax_r.set_xlabel("EBM Score (log-odds contribution to predicted risk class)")
                ax_r.set_title(f"EBM Local Explanation → {risk_str}")
                ax_r.tick_params(axis='y', labelsize=8)
                plt.tight_layout()
                st.pyplot(fig_r)
                plt.close(fig_r)

                top_inc = ebm_risk_df[ebm_risk_df["Score"] > 0].head(2)
                top_dec = ebm_risk_df[ebm_risk_df["Score"] < 0].head(2)
                total_pos = ebm_risk_df[ebm_risk_df["Score"] > 0]["Score"].sum()
                total_neg = ebm_risk_df[ebm_risk_df["Score"] < 0]["Score"].sum()

                inc_text = ", ".join(f"`{r['Feature']}` (+{r['Score']:.3f})"
                                     for _, r in top_inc.iterrows()) or "none"
                dec_text = ", ".join(f"`{r['Feature']}` ({r['Score']:.3f})"
                                     for _, r in top_dec.iterrows()) or "none"

                risk_icon_map = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}
                r_icon = risk_icon_map.get(risk_str, "⚪")

                st.info(f"""
**{r_icon} Why did the EBM predict '{risk_str}'?**

The EBM assigns each feature an exact log-odds score for this student:
- **Blue bars (positive scores)** → that feature made the model *more confident* in **{risk_str}**
- **Orange bars (negative scores)** → that feature pulled the prediction *away from* **{risk_str}**

**Top risk-increasing features:** {inc_text}

**Top risk-reducing features (working in the student's favour):** {dec_text}

**Net picture:** Positive contributions sum to **+{total_pos:.3f}** log-odds vs protective contributions of **{total_neg:.3f}**.
The positive balance is why the model confidently assigns **{risk_str}**.

The single biggest lever for this student is `{ebm_risk_df.iloc[0]['Feature']}` —
{'addressing' if ebm_risk_df.iloc[0]['Score'] > 0 else 'maintaining'} this factor would have the largest impact on changing the risk classification.
                """)

            except Exception as e:
                st.error(f"Risk EBM failed: {e}")
                st.exception(e)

        # ── Section 2: Grade EBM Local Explanations ──────────────────────────────
        st.markdown("---")
        st.markdown("### 2. 📊 Grade Predictions — EBM Local Explanations")
        st.markdown("Below we show exactly which features pushed this student's predicted **Math** and **Portuguese** grades up or down from the model's baseline.")

        grade_c1, grade_c2 = st.columns(2)

        def run_grade_ebm(preprocessor, X_train_raw, y_train_raw,
                          student_raw, pred_grade, avg_grade, subject_label, color):
            # pyre-ignore[21]
            from interpret.glassbox import ExplainableBoostingRegressor

            X_proc     = preprocessor.transform(X_train_raw)
            stu_proc   = preprocessor.transform(student_raw)
            f_names    = _feat_names(preprocessor, X_train_raw, X_proc)

            @st.cache_resource
            def _train_grade_ebm(_X, _y, _label, _feat_names):
                ebm = ExplainableBoostingRegressor(
                    random_state=42, max_rounds=200, n_jobs=1)
                ebm.fit(pd.DataFrame(_X, columns=_feat_names), _y.values)
                return ebm

            ebm_g = _train_grade_ebm(X_proc, y_train_raw, subject_label, f_names)

            stu_df = pd.DataFrame(stu_proc, columns=f_names)
            local  = ebm_g.explain_local(stu_df)
            d      = local.data(0)

            feat_n = list(d["names"])
            scores = [float(np.asarray(s).flatten()[0]) for s in d["scores"]]

            df_g = pd.DataFrame({"Feature": feat_n, "Score": scores})
            df_g["Abs"] = df_g["Score"].abs()
            df_g = df_g.sort_values("Abs", ascending=False).head(12).drop(columns="Abs")

            fig_g, ax_g = plt.subplots(figsize=(7, 5))
            clrs_g = ["#28a745" if v > 0 else "#dc3545" for v in df_g["Score"]]
            ax_g.barh(df_g["Feature"].iloc[::-1], df_g["Score"].iloc[::-1],
                      color=list(reversed(clrs_g)), edgecolor="black", linewidth=0.5)
            ax_g.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax_g.set_xlabel(f"EBM contribution to {subject_label} grade")
            ax_g.set_title(f"{subject_label} Grade EBM — Local Explanation")
            ax_g.tick_params(axis='y', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_g)
            plt.close(fig_g)

            pos_f = df_g[df_g["Score"] > 0].head(2)
            neg_f = df_g[df_g["Score"] < 0].head(2)
            diff  = pred_grade - avg_grade
            dir_w = "above" if diff >= 0 else "below"

            pos_lines = "\n".join(
                f"  • `{r['Feature']}` pushed the grade **up by {r['Score']:.2f} pts**"
                for _, r in pos_f.iterrows()) or "  • No strong positive contributors"
            neg_lines = "\n".join(
                f"  • `{r['Feature']}` pulled the grade **down by {abs(r['Score']):.2f} pts**"
                for _, r in neg_f.iterrows()) or "  • No strong negative contributors"

            st.success(f"""
**📖 {subject_label} EBM — Why {pred_grade:.1f}/20?**

The predicted {subject_label} grade is **{pred_grade:.1f}/20**, which is **{abs(diff):.1f} pts {dir_w}** the dataset average of {avg_grade:.1f}.

**Green bars — what boosted the grade:**
{pos_lines}

**Red bars — what held the grade back:**
{neg_lines}

The single biggest drag is `{neg_f.iloc[0]['Feature'] if not neg_f.empty else 'N/A'}` — addressing this would have the most direct impact on improving the {subject_label} prediction.
            """)

        with grade_c1:
            st.markdown("#### 📐 Mathematics EBM")
            with st.spinner("Computing Math Grade EBM..."):
                try:
                    sample_mat = student_df[reg["common_cols"] + reg["math_cols"]]
                    avg_math   = float(df["G3_mat"].mean())
                    run_grade_ebm(preprocessor_mat, X_mat_train, y_mat_train,
                                  sample_mat, pred_math, avg_math, "Math", "#4c72b0")
                except Exception as e:
                    st.warning("Math Grade EBM failed.")
                    st.exception(e)

        with grade_c2:
            st.markdown("#### 📚 Portuguese EBM")
            with st.spinner("Computing Portuguese Grade EBM..."):
                try:
                    sample_por = student_df[reg["common_cols"] + reg["por_cols"]]
                    avg_por    = float(df["G3_por"].mean())
                    run_grade_ebm(preprocessor_por, X_por_train, y_por_train,
                                  sample_por, pred_por, avg_por, "Portuguese", "#dd8452")
                except Exception as e:
                    st.warning("Portuguese Grade EBM failed.")
                    st.exception(e)


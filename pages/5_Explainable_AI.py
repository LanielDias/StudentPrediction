import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_system import load_and_merge_data, run_regression_pipelines, run_classification_pipelines

st.set_page_config(page_title="Explainable AI", page_icon="🧠", layout="wide")

st.title("🧠 Explainable AI — InterpretML EBM")
st.markdown(
    "Using **InterpretML Explainable Boosting Machine (EBM)** — a glass-box model — "
    "to explain what drives Math and Portuguese grade predictions across the entire dataset."
)

with st.expander("🛠️ How EBM Works"):
    st.markdown("""
**Explainable Boosting Machine (EBM)** is a Generalized Additive Model (GAM):

`prediction = intercept + f₁(x₁) + f₂(x₂) + … + fₙ(xₙ)`

- Each `fᵢ` is a **learned shape function** — an exact curve showing how one feature alone affects the prediction.
- EBM is trained by gradient boosting, **one feature at a time**, so each contribution is perfectly isolated.
- Scores are **intrinsically exact** — no post-hoc approximation (unlike SHAP on a black-box model).
- **Global importance** = mean absolute contribution of each feature across all training students.
- **Shape functions** = the model's learned non-linear relationship between a feature and the grade.
    """)

with st.spinner("Loading models and data..."):
    df  = load_and_merge_data()
    reg = run_regression_pipelines(df)
    cls = run_classification_pipelines(df)

X_mat_train      = reg["X_mat_train"]
X_por_train      = reg["X_por_train"]
y_mat_train      = reg["y_mat_train"]
y_por_train      = reg["y_por_train"]
preprocessor_mat = reg["preprocessor_mat"]
preprocessor_por = reg["preprocessor_por"]
X_train_cls      = cls["X_train_cls"]
y_train_cls      = cls["y_train_cls"]
preprocessor_cls = cls["preprocessor_class"]

# ── Feature name helper ────────────────────────────────────────────────────────
def _feat_names(preprocessor, X_orig, X_proc):
    try:
        cat = X_orig.select_dtypes(include="object").columns.tolist()
        num = X_orig.select_dtypes(exclude="object").columns.tolist()
        ohe = preprocessor.named_transformers_["cat"]
        return num + list(ohe.get_feature_names_out(cat))
    except Exception:
        return [f"f{i}" for i in range(X_proc.shape[1])]

FEATURE_DESCRIPTIONS = {
    "G2_mat": "2nd-period Math grade", "G2_por": "2nd-period Portuguese grade",
    "G1_mat": "1st-period Math grade", "G1_por": "1st-period Portuguese grade",
    "absences_mat": "Math absences", "absences_por": "Portuguese absences",
    "studytime_mat": "Math study time (hrs/week)", "studytime_por": "Portuguese study time (hrs/week)",
    "failures_mat": "Previous Math failures", "failures_por": "Previous Portuguese failures",
    "Medu": "Mother's education level", "Fedu": "Father's education level",
    "age": "Student's age", "traveltime_mat": "Travel time (Math)",
    "traveltime_por": "Travel time (Portuguese)",
}

FEATURE_DIRECTION = {
    "G2_mat": True, "G2_por": True, "G1_mat": True, "G1_por": True,
    "studytime_mat": True, "studytime_por": True, "Medu": True, "Fedu": True,
    "absences_mat": False, "absences_por": False,
    "failures_mat": False, "failures_por": False,
}

def describe(feat):
    return FEATURE_DESCRIPTIONS.get(feat, f"`{feat}`")

try:
    from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier

    # ── Train EBM helpers ──────────────────────────────────────────────────────
    @st.cache_resource
    def train_ebm_reg(_X_proc, _y, _feat_names, _label):
        ebm = ExplainableBoostingRegressor(random_state=42, max_rounds=200, n_jobs=1)
        ebm.fit(pd.DataFrame(_X_proc, columns=_feat_names), _y.values)
        return ebm

    @st.cache_resource
    def train_ebm_cls(_X_proc, _y, _feat_names, _label):
        ebm = ExplainableBoostingClassifier(random_state=42, max_rounds=200, n_jobs=1)
        ebm.fit(pd.DataFrame(_X_proc, columns=_feat_names), _y)
        return ebm

    with st.spinner("Training EBM models... (first run ~30 s, cached afterwards)"):
        X_mat_proc = preprocessor_mat.transform(X_mat_train)
        X_por_proc = preprocessor_por.transform(X_por_train)
        X_cls_proc = preprocessor_cls.transform(X_train_cls)

        mat_fn = _feat_names(preprocessor_mat, X_mat_train, X_mat_proc)
        por_fn = _feat_names(preprocessor_por, X_por_train, X_por_proc)
        cls_fn = _feat_names(preprocessor_cls, X_train_cls, X_cls_proc)

        ebm_mat = train_ebm_reg(X_mat_proc, y_mat_train, mat_fn, "math")
        ebm_por = train_ebm_reg(X_por_proc, y_por_train, por_fn, "por")
        ebm_cls = train_ebm_cls(X_cls_proc, y_train_cls, cls_fn, "risk")

    # ── Global importance extractor ────────────────────────────────────────────
    def ebm_global_importance(ebm):
        gexp  = ebm.explain_global()
        data  = gexp.data()
        df_i  = pd.DataFrame({"Feature": list(data["names"]),
                               "Importance": list(data["scores"])})
        df_i  = df_i.sort_values("Importance", ascending=False).reset_index(drop=True)
        return df_i, gexp

    # ── Shape function plotter ─────────────────────────────────────────────────
    def plot_shape(ebm, gexp, imp_df, color, ax, subject):
        top_feat = imp_df.iloc[0]["Feature"]
        feat_names = list(ebm.feature_names_in_)
        if top_feat not in feat_names:
            ax.text(0.5, 0.5, "Shape not available", ha="center", va="center",
                    transform=ax.transAxes)
            return top_feat
        idx       = feat_names.index(top_feat)
        feat_data = gexp.data(idx)
        x_vals    = feat_data.get("names", [])
        y_vals    = feat_data.get("scores", [])
        if len(x_vals) == 0 or len(y_vals) == 0:
            ax.text(0.5, 0.5, "Shape not available", ha="center", va="center",
                    transform=ax.transAxes)
            return top_feat
        try:
            x_num = [float(v) for v in x_vals]
        except (TypeError, ValueError):
            x_num = list(range(len(x_vals)))
        y_num = [float(v) for v in y_vals]
        # EBM continuous: n+1 bin edges → compute midpoints → length n
        if len(x_num) == len(y_num) + 1:
            x_num = [(x_num[i] + x_num[i + 1]) / 2 for i in range(len(x_num) - 1)]
        else:
            n = min(len(x_num), len(y_num))
            x_num, y_num = [x_num[i] for i in range(n)], [y_num[i] for i in range(n)]
        ax.plot(x_num, y_num, color=color, linewidth=2.5)
        ax.fill_between(x_num, y_num, alpha=0.15, color=color)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(top_feat, fontsize=10)
        ax.set_ylabel(f"EBM contribution to {subject} grade", fontsize=9)
        ax.set_title(f"{subject}: EBM Shape — {top_feat}", fontsize=11)
        ax.grid(True, alpha=0.3)
        return top_feat

    with st.spinner("Computing global EBM importance..."):
        imp_mat, gexp_mat = ebm_global_importance(ebm_mat)
        imp_por, gexp_por = ebm_global_importance(ebm_por)
        imp_cls, gexp_cls = ebm_global_importance(ebm_cls)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Global Feature Importance
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("1. 📊 Global Feature Importance (EBM)")
    st.markdown(
        "Each bar represents the **mean absolute EBM score** for that feature across all students — "
        "the higher the bar, the more that feature moves the predicted grade on average."
    )

    imp_c1, imp_c2 = st.columns(2)

    with imp_c1:
        st.subheader("📐 Mathematics")
        fig_im, ax_im = plt.subplots(figsize=(8, 5))
        top12m = imp_mat.head(12)
        ax_im.barh(top12m["Feature"][::-1], top12m["Importance"][::-1],
                   color="#4c72b0", edgecolor="black")
        ax_im.set_xlabel("Mean Absolute EBM Score (grade points)")
        ax_im.set_title("Math EBM — Global Importance")
        ax_im.tick_params(axis="y", labelsize=8)
        plt.tight_layout(); st.pyplot(fig_im); plt.close(fig_im)

        t1m, t2m, t3m = imp_mat.iloc[0], imp_mat.iloc[1], imp_mat.iloc[2]
        st.info(f"""
**🔵 Math EBM Insight:**

The single most influential feature for predicting Math grades is **{describe(t1m['Feature'])}** (`{t1m['Feature']}`),
contributing an average of **{t1m['Importance']:.3f} grade points** per student — exact, not an approximation.

Top 3:
1. `{t1m['Feature']}` — **{t1m['Importance']:.3f} pts**
2. `{t2m['Feature']}` — **{t2m['Importance']:.3f} pts**
3. `{t3m['Feature']}` — **{t3m['Importance']:.3f} pts**
        """)

    with imp_c2:
        st.subheader("📚 Portuguese")
        fig_ip, ax_ip = plt.subplots(figsize=(8, 5))
        top12p = imp_por.head(12)
        ax_ip.barh(top12p["Feature"][::-1], top12p["Importance"][::-1],
                   color="#dd8452", edgecolor="black")
        ax_ip.set_xlabel("Mean Absolute EBM Score (grade points)")
        ax_ip.set_title("Portuguese EBM — Global Importance")
        ax_ip.tick_params(axis="y", labelsize=8)
        plt.tight_layout(); st.pyplot(fig_ip); plt.close(fig_ip)

        t1p, t2p, t3p = imp_por.iloc[0], imp_por.iloc[1], imp_por.iloc[2]
        st.info(f"""
**🔵 Portuguese EBM Insight:**

**{describe(t1p['Feature'])}** (`{t1p['Feature']}`) dominates Portuguese predictions,
contributing **{t1p['Importance']:.3f} pts** on average per student.

Top 3:
1. `{t1p['Feature']}` — **{t1p['Importance']:.3f} pts**
2. `{t2p['Feature']}` — **{t2p['Importance']:.3f} pts**
3. `{t3p['Feature']}` — **{t3p['Importance']:.3f} pts**
        """)

    # Risk classifier importance
    st.markdown("#### 🔴 Risk Classification (EBM)")
    st.markdown("Feature importance for the Risk Level classifier (Low / Medium / High risk).")
    fig_ic, ax_ic = plt.subplots(figsize=(10, 4))
    top10c = imp_cls.head(10)
    ax_ic.barh(top10c["Feature"][::-1], top10c["Importance"][::-1],
               color="#9467bd", edgecolor="black")
    ax_ic.set_xlabel("Mean Absolute EBM Score (log-odds)")
    ax_ic.set_title("Risk EBM — Global Feature Importance")
    ax_ic.tick_params(axis="y", labelsize=8)
    plt.tight_layout(); st.pyplot(fig_ic); plt.close(fig_ic)

   

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Per-Student Local EBM (ICE-style)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("3. 🎯 Per-Student Local EBM Explanation")
    st.markdown(
        "Select a student from the dataset to see an **individual-level EBM explanation** — "
        "exactly which features pushed *that student's* predicted grade up or down from the model baseline."
    )

    sample_index = st.sidebar.number_input(
        "Student index for local EBM", min_value=0, max_value=len(df) - 1, value=0
    )

    X_mat_full = reg["X_mat"]
    X_por_full = reg["X_por"]

    local_c1, local_c2 = st.columns(2)

    def run_local_ebm(ebm, preprocessor, X_full, X_train_raw, X_proc_train,
                      fn, sample_idx, pred_col, avg_grade, subject, color, ax):
        student_raw  = X_full.iloc[[sample_idx]]
        student_proc = preprocessor.transform(student_raw)
        stu_df       = pd.DataFrame(student_proc, columns=fn)
        local        = ebm.explain_local(stu_df)
        d            = local.data(0)
        names        = list(d["names"])
        scores       = [float(np.asarray(s).flatten()[0]) for s in d["scores"]]
        df_l = pd.DataFrame({"Feature": names, "Score": scores})
        df_l["Abs"] = df_l["Score"].abs()
        df_l = df_l.sort_values("Abs", ascending=False).head(12).drop(columns="Abs")

        clrs = ["#28a745" if v > 0 else "#dc3545" for v in df_l["Score"]]
        ax.barh(df_l["Feature"].iloc[::-1], df_l["Score"].iloc[::-1],
                color=list(reversed(clrs)), edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"EBM contribution to {subject} grade")
        ax.set_title(f"{subject} EBM — Student #{sample_idx}")
        ax.tick_params(axis="y", labelsize=8)

        pred_g = float(ebm.predict(stu_df)[0])
        diff   = pred_g - avg_grade
        dir_w  = "above" if diff >= 0 else "below"
        pos_f  = df_l[df_l["Score"] > 0].head(2)
        neg_f  = df_l[df_l["Score"] < 0].head(2)
        pos_lines = "\n".join(f"  • `{r['Feature']}` **+{r['Score']:.2f} pts**"
                              for _, r in pos_f.iterrows()) or "  • None"
        neg_lines = "\n".join(f"  • `{r['Feature']}` **{r['Score']:.2f} pts**"
                              for _, r in neg_f.iterrows()) or "  • None"
        return pred_g, diff, dir_w, pos_lines, neg_lines, avg_grade

    with local_c1:
        st.subheader(f"📐 Math — Student #{sample_index}")
        fig_lm, ax_lm = plt.subplots(figsize=(7, 5))
        with st.spinner("Computing Math local EBM..."):
            try:
                avg_m = float(df["G3_mat"].mean())
                pred_m, diff_m, dir_m2, pos_m, neg_m, _ = run_local_ebm(
                    ebm_mat, preprocessor_mat, X_mat_full, X_mat_train,
                    X_mat_proc, mat_fn, sample_index, "G3_mat", avg_m, "Math", "#4c72b0", ax_lm)
                plt.tight_layout(); st.pyplot(fig_lm); plt.close(fig_lm)
                st.success(f"""
**Student #{sample_index} — Predicted Math: {pred_m:.1f}/20** ({abs(diff_m):.1f} pts {dir_m2} avg of {avg_m:.1f})

**Boosted by:** {pos_m}

**Held back by:** {neg_m}
                """)
            except Exception as e:
                st.warning("Math local EBM failed."); st.exception(e)

    with local_c2:
        st.subheader(f"📚 Portuguese — Student #{sample_index}")
        fig_lp, ax_lp = plt.subplots(figsize=(7, 5))
        with st.spinner("Computing Portuguese local EBM..."):
            try:
                avg_p = float(df["G3_por"].mean())
                pred_p, diff_p, dir_p2, pos_p, neg_p, _ = run_local_ebm(
                    ebm_por, preprocessor_por, X_por_full, X_por_train,
                    X_por_proc, por_fn, sample_index, "G3_por", avg_p, "Portuguese", "#dd8452", ax_lp)
                plt.tight_layout(); st.pyplot(fig_lp); plt.close(fig_lp)
                st.success(f"""
**Student #{sample_index} — Predicted Portuguese: {pred_p:.1f}/20** ({abs(diff_p):.1f} pts {dir_p2} avg of {avg_p:.1f})

**Boosted by:** {pos_p}

**Held back by:** {neg_p}
                """)
            except Exception as e:
                st.warning("Portuguese local EBM failed."); st.exception(e)
except Exception as e:
    st.error(f"InterpretML EBM failed to load: {e}")
    st.exception(e)

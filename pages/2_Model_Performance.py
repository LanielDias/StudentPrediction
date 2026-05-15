import streamlit as st
import plotly.express as px
from ml_system import load_and_merge_data, run_regression_pipelines, run_classification_pipelines

st.set_page_config(page_title="Model Performance", page_icon="⚙️", layout="wide")

st.title("⚙️ AI Models Performance")
st.markdown("Detailed breakdown of Deep Learning model performances across Regression (Grade Prediction) and Classification (Risk Level) tasks.")

with st.spinner("Loading and Training Models... (This may take a moment on first run)"):
    df = load_and_merge_data()
    reg_results = run_regression_pipelines(df)
    cls_results = run_classification_pipelines(df)

st.header("1. Academic Risk Classification")
st.markdown("Performance of various Deep Learning classifiers at identifying **High Risk** students.")

cls_df = cls_results["classification_results_df"]
best_cls = cls_df.iloc[0]["Model"]

col1, col2 = st.columns([1, 2])
with col1:
    st.info(f"🏆 Best Risk Classifier: **{best_cls}**")
    st.dataframe(cls_df.style.highlight_max(axis=0, color='lightgreen', subset=["Accuracy"]), use_container_width=True)

with col2:
    fig = px.bar(cls_df, x="Accuracy", y="Model", orientation='h', 
                 title="Classification Accuracy Comparison", 
                 color="Accuracy", color_continuous_scale="Viridis", text_auto='.4f')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


st.divider()
st.header("2. Final Grade Regression Models")
st.markdown("Testing multiple network architectures to predict continuous final scores (G3) for Math and Portuguese.")

tab1, tab2 = st.tabs(["Mathematics", "Portuguese"])

math_df = reg_results["math_comparison"]
por_df = reg_results["por_comparison"]

with tab1:
    st.subheader("Mathematics Evaluation")
    
    col_m1, col_m2 = st.columns([1, 1])
    with col_m1:
        st.info(f"🏆 Best Math Regressor: **{math_df.iloc[0]['Model']}**")
        st.dataframe(math_df.style.highlight_max(axis=0, color='lightgreen', subset=["Accuracy (±2)", "R2"])
                            .highlight_min(axis=0, color='lightgreen', subset=["RMSE", "MAE"]), 
                     use_container_width=True)
    with col_m2:
        fig = px.bar(math_df, x="Model", y="Accuracy (±2)", 
                     title="Math Accuracy (±2 points) Comparison",
                     color="Accuracy (±2)", color_continuous_scale="Blues", text_auto='.2f')
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Portuguese Evaluation")
    
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        st.info(f"🏆 Best Portuguese Regressor: **{por_df.iloc[0]['Model']}**")
        st.dataframe(por_df.style.highlight_max(axis=0, color='lightgreen', subset=["Accuracy (±2)", "R2"])
                            .highlight_min(axis=0, color='lightgreen', subset=["RMSE", "MAE"]), 
                     use_container_width=True)
    with col_p2:
        fig = px.bar(por_df, x="Model", y="Accuracy (±2)", 
                     title="Portuguese Accuracy (±2 points) Comparison",
                     color="Accuracy (±2)", color_continuous_scale="Blues", text_auto='.2f')
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)

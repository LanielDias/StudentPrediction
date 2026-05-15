import streamlit as st
import pandas as pd
from ml_system import load_and_merge_data, run_regression_pipelines, run_classification_pipelines, generate_recommendations

st.set_page_config(page_title="Dataset Predictions", page_icon="📝", layout="wide")

st.title("📝 Dataset Deep Learning Predictions")
st.markdown("This table provides an overview of predicted scores, calculated risk levels, and AI-generated personalized learning recommendations for the entire student dataset.")

with st.spinner("Extracting predictions..."):
    df = load_and_merge_data()
    reg = run_regression_pipelines(df)
    cls = run_classification_pipelines(df)

final_math_pipeline = reg['final_math_pipeline']
final_por_pipeline = reg['final_por_pipeline']
X_mat = reg["X_mat"]
X_por = reg["X_por"]

final_classifier = cls["final_classifier"]
X_class = cls["X_class"]

# 1. Run Predictions
pred_math = final_math_pipeline.predict(X_mat)
pred_por = final_por_pipeline.predict(X_por)

pred_risk_level = final_classifier.predict(X_class)

# 2. Append to View Dataframe
view_df = df.copy()
view_df["Pred_G3_Math"] = pred_math
view_df["Pred_G3_Portuguese"] = pred_por

view_df["Pred_Risk_Level"] = pred_risk_level
# 3-class mapping: 0 = Low Risk, 1 = Medium Risk, 2 = High Risk
risk_label_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
view_df["Pred_Risk_Level"] = view_df["Pred_Risk_Level"].apply(
    lambda x: risk_label_map.get(int(x), str(x))
)

view_df["Pred_Avg_Score"] = (view_df["Pred_G3_Math"] + view_df["Pred_G3_Portuguese"]) / 2

# 3. Recommendations
numeric_df = df.select_dtypes(include='number')
avg_features = numeric_df.mean()

# Apply the personalized recommendations logic row by row
view_df["Personalized_Recommendations"] = view_df.apply(
     generate_recommendations,
     axis=1,
     args=(avg_features,)
 )

cols_to_show = [
    "school", "sex", "age",
    "G3_mat", "Pred_G3_Math", 
    "G3_por", "Pred_G3_Portuguese",
    "Pred_Avg_Score", "Pred_Risk_Level",
    "Personalized_Recommendations"
]

def color_risk(val):
    if val == "High Risk":
        return "color: #dc3545; font-weight: bold;"
    elif val == "Medium Risk":
        return "color: #ffc107; font-weight: bold;"
    elif val == "Low Risk":
        return "color: #28a745; font-weight: bold;"
    return ""

st.dataframe(view_df[cols_to_show].style.format({
    "G3_mat": "{:.0f}",
    "Pred_G3_Math": "{:.2f}",
    "G3_por": "{:.0f}",
    "Pred_G3_Portuguese": "{:.2f}",
    "Pred_Avg_Score": "{:.2f}"
}).map(color_risk, subset=["Pred_Risk_Level"]),
use_container_width=True, height=600)

st.markdown("### 💡 What do these recommendations mean?")
st.info("""
The **Intelligent Personalized Learning Recommendation Engine** analyzes features where a student performs below the overall dataset average.
- 🔴 **High Risk** — Student is at serious risk of failing. Top 2 problematic features are identified and targeted interventions are provided.
- 🟡 **Medium Risk** — Student may need improvement in specific subjects. Targeted subject advice is given.
- 🟢 **Low Risk** — Student is on track. No major improvements required.
""")

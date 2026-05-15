import streamlit as st

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.25rem;
        color: #4b5563;
        margin-bottom: 2rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        background-color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 Smart Student Performance Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Personalized Learning & Risk Assessment System</div>', unsafe_allow_html=True)

st.markdown("""
This system uses advanced Machine Learning and Deep Learning models to predict student academic performance, identify at-risk students, and provide tailored learning recommendations.

### 🌟 System Capabilities
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="feature-icon">📊</div>
        <h4>Interactive Dashboard</h4>
        <p>Explore the dataset through univariate, bivariate, and multivariate analysis visualizations.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="feature-icon">🤖</div>
        <h4>AI Models Prediction</h4>
        <p>Review the performance metrics of Deep Learning Classifiers and Regression models.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="feature-icon">💡</div>
        <h4>Personalized Recommendations</h4>
        <p>Generate precise, per-student academic advice and identify risk levels early.</p>
    </div>
    """, unsafe_allow_html=True)

st.info("👈 Please navigate using the sidebar to explore the different sections of the application.")

st.markdown("""
---
*Built as a multi-page analytical Streamlit web application.*
""")

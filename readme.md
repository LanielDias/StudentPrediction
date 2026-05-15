# 🎓 Smart Student Performance Prediction and Personalized Learning Recommendation

A student-focused educational analytics project built to predict academic performance, identify students who may need support, and generate personalized learning recommendations using Machine Learning and Explainable AI.

---

## 🚀 Project Overview

This project was developed as part of my MSc Big Data Analytics work to explore how AI can be used in education for early academic intervention and personalized learning support.

The system combines Machine Learning, Deep Learning, and Explainable AI techniques to analyze student-related factors such as:
- Academic records
- Study habits
- Attendance
- Lifestyle and social behavior
- Family and demographic information

Based on these factors, the application predicts final student performance, classifies academic risk levels, and provides personalized recommendations that can help improve learning outcomes.

The project is implemented as a multi-page Streamlit application with interactive dashboards, prediction systems, and explainable AI visualizations.

---

## ✨ Main Features

### 📊 Interactive Analytics Dashboard
- Univariate, bivariate, and multivariate analysis
- Interactive visualizations using Plotly
- Real-time filtering and exploration

### 🤖 AI-Based Grade Prediction
Predicts final student grades for Mathematics and Portuguese using Deep Learning models.

### ⚠️ Academic Risk Classification
Automatically classifies students into:
- Low Risk
- Medium Risk
- High Risk

### 🎯 Personalized Learning Recommendations
Provides recommendation strategies based on predicted performance, risk level, and feature analysis.

### 🧠 Explainable AI
Provides transparent model interpretation using:
- SHAP
- Explainable Boosting Machine (EBM)

### 🔮 What-If Grade Simulation
Allows experimentation with student parameters to observe predicted academic outcomes.

---

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Web Framework | Streamlit |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Explainable AI | SHAP, InterpretML |
| Deployment Ready | Streamlit Multi-Page App |

---

## 🧠 Deep Learning Models

### Regression Models
Used for final grade prediction:
- Multi-Layer Perceptron (MLP)
- Autoencoder Regressor
- Denoising Autoencoder
- Wide & Deep Neural Network
- Attention Neural Network

### Classification Models
Used for student risk classification:
- Deep Learning MLP
- CNN
- LSTM
- RNN
- Autoencoder Classifier

---

## 📂 Dataset

Dataset Source:
https://archive.ics.uci.edu/dataset/320/student+performance

The dataset contains:
- Student grades
- Attendance
- Family background
- Study habits
- Social and lifestyle attributes
- School-related information

Subjects Included:
- Mathematics
- Portuguese

---

## 📁 Project Structure

```bash
Smart-Student-Performance-Prediction/
│
├── app.py
├── ml_system.py
├── student-mat.csv
├── student-por.csv
│
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Model_Performance.py
│   ├── 3_Dataset_Predictions.py
│   ├── 4_Student_Prediction_Form.py
│   ├── 5_Explainable_AI.py
│   └── 6_Recommendations.py
│
└── README.md
```

---

## 📸 Application Modules

### 📊 Dashboard
Interactive EDA dashboard with visual analytics.

### ⚙️ Model Performance
Displays evaluation metrics for all Deep Learning models.

### 📝 Dataset Predictions
Shows dataset-wide predictions and AI-generated recommendations.

### 🎓 Student Prediction Form
Allows users to enter student details and generate predictions.

### 🧠 Explainable AI
Provides transparent explanations for model predictions.

### 🎯 AI Recommendations
Generates personalized academic improvement strategies.

---

## 🧪 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/LanielDias/Smart-Student-Performance-Prediction-and-Personalized-Learning-Recommendation.git
```

### 2️⃣ Navigate to the Project Folder

```bash
cd Smart-Student-Performance-Prediction-and-Personalized-Learning-Recommendation
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 📈 Explainable AI

The project includes Explainable AI techniques to make prediction results easier to understand and interpret.

### SHAP
Used to identify how individual features influence predictions.

### Explainable Boosting Machine (EBM)
Provides interpretable and transparent AI decision-making.

This enables educators to understand:
- Why a student is classified as high risk
- Which features most impact performance
- How interventions can improve outcomes

---

## 🎯 Project Objectives

- Predict student academic performance accurately
- Detect at-risk students early
- Support personalized learning strategies
- Improve transparency using Explainable AI
- Enable data-driven academic intervention

---

## 🔮 Future Improvements

- Real-time student monitoring
- Cloud deployment
- Multi-subject support
- Parent and teacher dashboards
- Adaptive learning integration
- Mobile application support

---

## 👨‍💻 Author

Laniel Charles Dias  
MSc Big Data Analytics  
St Agnes College (Autonomous), Mangaluru

GitHub: https://github.com/LanielDias

---

## 📜 License

This project is developed for academic, research, and educational purposes.


